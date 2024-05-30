import pandas as pd
import os
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve,auc
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedShuffleSplit

def get_matrix(path,assay):
    '''
    get fingerprints
    '''
    df_pubchem=pd.read_csv(os.path.join(path,'pubchem',assay+'_pubchemfps.csv'),header=0,index_col=0)
    df_morgan=pd.read_csv(os.path.join(path,'morgan',assay+'_morganfps.csv'),header=0,index_col=0,dtype='object')
    df_maccs=pd.read_csv(os.path.join(path,'maccs',assay+'_maccsfps.csv'),header=0,index_col=0)
    df_morgan=df_morgan.iloc[:,1:]; df_maccs=df_maccs.iloc[:,1:]
    df_fps=pd.concat([df_pubchem,df_morgan,df_maccs],axis=1) #2073（one col is assayoutcome）
    df_fps.dropna(axis=0,how='any',inplace=True)
    return(df_fps)

def xgb_model(train_x,train_y,test_x,test_y,params,model_path):
    model=XGBClassifier(booster='gbtree',tree_method='gpu_hist',objective='binary:logistic',**params)
    model.fit(train_x.values, train_y)
    model.save_model(model_path)
    ypred = model.predict_proba(test_x.values)[:,1]
    return ypred

def get_values(test_y,ypred): #print test results
    precision, recall, thresholds = precision_recall_curve(test_y, ypred)
    print('AUPRC: %.4F' %auc(recall,precision))
    print ('AUROC: %.4f' % metrics.roc_auc_score(test_y,ypred))

def objective(trial):
    param = {
        'booster':'gbtree',
        'tree_method':'gpu_hist', 
        "objective": "binary:logistic",
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 256),
        'gamma': trial.suggest_float('gamma', 1e-7, 10.0,log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0,log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0,log=True),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
    }
    sss=StratifiedShuffleSplit(n_splits=5,train_size=0.75,test_size=0.25,random_state=2)
    fiveval_scores=np.zeros(5)

    idx=0
    for train_idx,test_idx in sss.split(tv_x,tv_y):
        train_x, train_y = tv_x.iloc[train_idx], tv_y.iloc[train_idx]
        test_x, test_y = tv_x.iloc[test_idx], tv_y.iloc[test_idx]
        #===========
        smote=SMOTE(random_state=3)
        train_x, train_y = smote.fit_resample(train_x, train_y) 
        #===========
        model = XGBClassifier(**param)
        model.fit(train_x.values.astype('float64'), train_y.values)
        y_pred = model.predict_proba(test_x.values.astype('float64'))[:,1]
        fiveval_scores[idx] = roc_auc_score(test_y.values,y_pred)
        idx += 1

    return np.mean(fiveval_scores)


namelist=['0_aromatase_anta','1_ahr_ago','2_ar_ago','3_er_ago','4_gr_ago','5_tshr_ago','6_tr_anta']


#========================
for index_assay,assay in enumerate(namelist):
    print('assay = ',assay)
    path=r"\comparing with fingerprints and descriptors\generate_fps_and_dsc\files\2_fingerprints" #the file path of the fingerprints fold
    df=get_matrix(path,assay) #get the feature matrix of descriptors
    #========================
    X=df.iloc[:,1:]
    Y=df.iloc[:,0]

    if assay=='3_er_ago' or assay=='4_gr_ago':
        a=5
    else:
        a=7

    tv_x, test_x, tv_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=a,stratify=Y,shuffle=True) #split out the test set, the random_state is fixed for each assay (the same as MSFragTox)
    #=========
    print('df.shape',df.shape)
    print('tv_x.shape',tv_x.shape)
    print('test_x.shape',test_x.shape)
    df1=df.drop_duplicates(keep='first',inplace=False)
    print('df_dropduplicates.shape',df1.shape)
    tv_x1 = tv_x.drop_duplicates(keep='first',inplace=False)
    print('tv_x1.shape',tv_x1.shape)
    test_x1 = test_x.drop_duplicates(keep='first',inplace=False)
    print('test_x1.shape',test_x1.shape)

    print(tv_y.value_counts())
    print(test_y.value_counts())
    #=========

    #parameter tuning, using the optuna and bayesian optimization method
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
    n_trials=50
    study.optimize(objective, n_trials=n_trials)
    print('Number of finished trials:', len(study.trials))
    print("------------------------------------------------")
    print('Best trial:', study.best_trial.params)
    print("------------------------------------------------")

    param=study.best_params

    #to get five AUROC and AUPRC values when training with optimal params
    #use the optimal params to retrain the training set, and evaluate on validation set (five times)
    #=================
    sss=StratifiedShuffleSplit(n_splits=5,train_size=0.75,test_size=0.25,random_state=2)
    auroc_fiveval_scores=np.zeros(5)
    auprc_fiveval_scores=np.zeros(5)
    
    idx=0
    for train_idx,test_idx in sss.split(tv_x,tv_y):
        train_x, train_y = tv_x.iloc[train_idx], tv_y.iloc[train_idx]
        test_x, test_y = tv_x.iloc[test_idx], tv_y.iloc[test_idx]
        #===========
        smote=SMOTE(random_state=3)
        train_x, train_y = smote.fit_resample(train_x, train_y)
        #===========
        model = XGBClassifier(booster='gbtree',tree_method='gpu_hist',objective='binary:logistic',**param)
        model.fit(train_x.values.astype('float64'), train_y.values)
        y_pred = model.predict_proba(test_x.values.astype('float64'))[:,1]
        auroc_fiveval_scores[idx] = roc_auc_score(test_y.values,y_pred)
        precision, recall, thresholds = precision_recall_curve(test_y.values, y_pred)
        auprc_fiveval_scores[idx] =auc(recall,precision)
        
        idx += 1

    print('AUROC five-time validation scores',auroc_fiveval_scores)
    print('AUROC mean score',np.mean(auroc_fiveval_scores))

    print('AUPRC five-time validation scores',auprc_fiveval_scores)
    print('AUPRC mean score',np.mean(auprc_fiveval_scores))

    #use training and validation sets to train the optimal model and use the test set to see performance 
    #========
    smote=SMOTE(random_state=3)
    tv_x, tv_y = smote.fit_resample(tv_x, tv_y)
    #========
    ypred = xgb_model(tv_x, tv_y, test_x, test_y,param,model_path=os.path.join(r"\comparing with fingerprints and descriptors\fingerprints_model\models",assay+'.model'))
    get_values(test_y,ypred)
    print('\n=======================\n')


#optimal params for 7 assays
'''
param_list=[
    {'max_depth': 7, 'learning_rate': 0.16669703803685532, 'n_estimators': 136, 'min_child_weight': 1, 'gamma': 0.004486107043264892, 'subsample': 0.7543650219695988, 'colsample_bytree': 0.6740080284038519, 'reg_alpha': 2.215726728409855, 'reg_lambda': 0.01187744304300704, 'random_state': 24},
    {'max_depth': 9, 'learning_rate': 0.11805454402314801, 'n_estimators': 260, 'min_child_weight': 2, 'gamma': 0.0838108312264574, 'subsample': 0.945818960911283, 'colsample_bytree': 0.8843400696196121, 'reg_alpha': 0.07800863761411342, 'reg_lambda': 2.6860006100146645, 'random_state': 2020},
    {'max_depth': 9, 'learning_rate': 0.019176524307943213, 'n_estimators': 66, 'min_child_weight': 4, 'gamma': 6.876334314209999, 'subsample': 0.7913263744661241, 'colsample_bytree': 0.9469943355857618, 'reg_alpha': 0.009957841767547364, 'reg_lambda': 0.0056252005137967164, 'random_state': 24},
    {'max_depth': 9, 'learning_rate': 0.051919724233437325, 'n_estimators': 313, 'min_child_weight': 1, 'gamma': 1.2441580643867275e-06, 'subsample': 0.5693901175856368, 'colsample_bytree': 0.571446155215107, 'reg_alpha': 0.11721195433785853, 'reg_lambda': 2.869836343154137, 'random_state': 2020},
    {'max_depth': 5, 'learning_rate': 0.07775468944211074, 'n_estimators': 98, 'min_child_weight': 26, 'gamma': 1.455033249350299, 'subsample': 0.6546147173153014, 'colsample_bytree': 0.5527679454472382, 'reg_alpha': 0.004414474575704913, 'reg_lambda': 0.0016314808179897645, 'random_state': 2020},
    {'max_depth': 5, 'learning_rate': 0.087174585147424, 'n_estimators': 174, 'min_child_weight': 10, 'gamma': 1.4819226693632948e-07, 'subsample': 0.5186706733570097, 'colsample_bytree': 0.931296173396766, 'reg_alpha': 0.04104133306416034, 'reg_lambda': 9.536688332803275, 'random_state': 24},
    {'max_depth': 6, 'learning_rate': 0.1997551213313506, 'n_estimators': 403, 'min_child_weight': 1, 'gamma': 0.0004951494197556952, 'subsample': 0.8826356014475892, 'colsample_bytree': 0.9422784298764051, 'reg_alpha': 0.012457102000708745, 'reg_lambda': 0.20002718795627422, 'random_state': 48}
]
'''