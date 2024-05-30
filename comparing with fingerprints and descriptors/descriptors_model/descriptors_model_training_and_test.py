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

from sklearn.metrics import precision_recall_curve,auc

def get_matrix(path,assay):
    '''
    get descriptors
    '''
    df_des=pd.read_csv(os.path.join(path,assay+'_descriptor.csv'),header=0,index_col=0)

    return(df_des)

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
        model.fit(train_x.values, train_y.values) 
        y_pred = model.predict_proba(test_x.values)[:,1]
        fiveval_scores[idx] = roc_auc_score(test_y.values,y_pred)
        idx += 1

    return np.mean(fiveval_scores) #AUROC


namelist=['0_aromatase_anta','1_ahr_ago','2_ar_ago','3_er_ago','4_gr_ago','5_tshr_ago','6_tr_anta']

#========================
for index_assay,assay in enumerate(namelist):
    print('assay = ',assay)
    path=r"\comparing with fingerprints and descriptors\generate_fps_and_dsc\files\3_descriptors"
    df=get_matrix(path,assay) #get the feature matrix of descriptors

    #========================
    X=df.iloc[:,1:]
    Y=df.iloc[:,0]
    if assay=='3_er_ago' or assay=='4_gr_ago':
        a=5
    else:
        a=7
    tv_x, test_x, tv_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=a,stratify=Y,shuffle=True) #split out the test set, the random_state is fixed for each assay (and is the same as MSFragTox)

    print('df.shape',df.shape)
    print('tv_x.shape',tv_x.shape)
    print('test_x.shape',test_x.shape)
    #======
    tv=pd.concat([tv_y,tv_x],axis=1)
    tv1=tv.dropna(axis=0,how='any',inplace=False)

    tv_x=tv1.iloc[:,1:]
    tv_y=tv1.iloc[:,0]
    print('tv_x.shape after dropna',tv_x.shape)
    #======

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
        model.fit(train_x.values, train_y.values)
        y_pred = model.predict_proba(test_x.values)[:,1]
        auroc_fiveval_scores[idx] = roc_auc_score(test_y.values,y_pred)
        precision, recall, thresholds = precision_recall_curve(test_y.values, y_pred)
        auprc_fiveval_scores[idx] =auc(recall,precision)

        idx += 1

    print('AUROC five-time validation scores',auroc_fiveval_scores)
    print('AUROC mean score',np.mean(auroc_fiveval_scores))

    print('AUPRC five-time validation scores',auprc_fiveval_scores)
    print('AUPRC mean score',np.mean(auprc_fiveval_scores))
    #=================

    #use training and validation sets to train the optimal model and use the test set to see performance 
    smote=SMOTE(random_state=3)
    tv_x, tv_y = smote.fit_resample(tv_x, tv_y)
    ypred = xgb_model(tv_x, tv_y, test_x, test_y,param,model_path=os.path.join(r'\comparing with fingerprints and descriptors\descriptors_model\models',assay+'.model'))
    get_values(test_y,ypred)
    print('\n=======================\n')




#optimal params for 7 assays
'''
param_list=[
    {'max_depth': 7, 'learning_rate': 0.058409513144205415, 'n_estimators': 77, 'min_child_weight': 1, 'gamma': 4.620048752549471e-05, 'subsample': 0.7156738183264166, 'colsample_bytree': 0.8233761897652457, 'reg_alpha': 0.009314242286269791, 'reg_lambda': 0.07588980167615542, 'random_state': 2020},
    {'max_depth': 8, 'learning_rate': 0.08829524572824876, 'n_estimators': 174, 'min_child_weight': 1, 'gamma': 0.4389523079407237, 'subsample': 0.8020616199483908, 'colsample_bytree': 0.5401740882920967, 'reg_alpha': 0.07932696020911502, 'reg_lambda': 0.016760811597353126, 'random_state': 24},
    {'max_depth': 4, 'learning_rate': 0.19716463548574426, 'n_estimators': 435, 'min_child_weight': 2, 'gamma': 2.6106088761365362, 'subsample': 0.6099435631744179, 'colsample_bytree': 0.8591552353563187, 'reg_alpha': 0.0030453582943158757, 'reg_lambda': 0.07643313959944203, 'random_state': 48},
    {'max_depth': 10, 'learning_rate': 0.07705353491380929, 'n_estimators': 255, 'min_child_weight': 2, 'gamma': 0.0007807479844292372, 'subsample': 0.7480418995277142, 'colsample_bytree': 0.7120150508532364, 'reg_alpha': 1.3400170162226528, 'reg_lambda': 0.0018325378183966857, 'random_state': 2020},
    {'max_depth': 9, 'learning_rate': 0.014010632203818481, 'n_estimators': 302, 'min_child_weight': 160, 'gamma': 4.0092323684512633e-07, 'subsample': 0.502112626362271, 'colsample_bytree': 0.8598023369861064, 'reg_alpha': 3.4094868050712073, 'reg_lambda': 0.00828049676502811, 'random_state': 48},
    {'max_depth': 9, 'learning_rate': 0.17816597697549674, 'n_estimators': 420, 'min_child_weight': 1, 'gamma': 0.953534784159156, 'subsample': 0.7529927339972597, 'colsample_bytree': 0.7104246972950375, 'reg_alpha': 0.6013075859116179, 'reg_lambda': 0.0026359793988950967, 'random_state': 48},
    {'max_depth': 10, 'learning_rate': 0.11098967577526404, 'n_estimators': 328, 'min_child_weight': 1, 'gamma': 0.0004654292834547522, 'subsample': 0.8344090818556894, 'colsample_bytree': 0.8055729811221874, 'reg_alpha': 0.0751541474156902, 'reg_lambda': 3.3442095185464566, 'random_state': 48}
]
'''