import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import descriptor
from maccskeys import GetMACCSFPs  #167 features
from morganfp import GetMorganFPs  #1024 features
from pubchemfp import GetPubChemFPs  #881 features

def get_fps(path,fp_function,pathout):
	#read stan_smiles activity list
	df=pd.read_csv(path,header=0,index_col=None)
	# df=df.iloc[:10,:] #for test
	smiles=df['stan_smiles']
	fp_list=[]
	if fp_function=='pubchem':
		for idx,smi in enumerate(smiles):
			mol=Chem.MolFromSmiles(smi)
			if mol==None: 
				fp_list.append(np.full(881,'x'))
				continue
			fp_pubchem=GetPubChemFPs(mol)
			fp_list.append(fp_pubchem)
		fp_df=pd.DataFrame(fp_list,index=None,columns=['pubchemfp_'+str(x) for x in range(881)],dtype=object)

	elif fp_function=='morgan':
		bitinfodict={}
		for idx,smi in enumerate(smiles):
			mol=Chem.MolFromSmiles(smi)
			if mol==None: 
				fp_list.append(np.full(1024,'x'))
				continue
			fp_morgan, bitinfo =GetMorganFPs(mol,nBits=1024,return_bitInfo=True)
			fp_list.append(fp_morgan)
			bitinfodict[smi]=bitinfo
		fp_df=pd.DataFrame(fp_list,index=None,columns=['morganfp_'+str(x) for x in range(1024)],dtype=object)

		# path1=os.path.join(os.path.split(pathout)[0],'bitinfo',os.path.split(pathout)[1].split('.csv')[0]+'.pickle')
		# fout = open(path1,'wb')
		# pickle.dump(bitinfodict,fout)
		# fout.close()
		

	elif fp_function=='maccs':
		for idx,smi in enumerate(smiles):
			mol=Chem.MolFromSmiles(smi)
			if mol==None: 
				fp_list.append(np.full(167,np.nan))
				continue
			fp_maccs=GetMACCSFPs(mol)
			fp_list.append(fp_maccs)
		fp_df=pd.DataFrame(fp_list,index=None,columns=['maccsfp_'+str(x) for x in range(167)],dtype=object)


	fp_df=pd.concat([df,fp_df],axis=1) #to include the stan_smiles and activity columns
	print(fp_df.head())
	fp_df.to_csv(pathout,index=None)

def get_descriptors(path,pathout):
	df=pd.read_csv(path)
	# df=df.iloc[:10,:] #for test
	smiles=df['stan_smiles'].tolist()
	value = descriptor.Extraction().batch_transform(smiles)
	MolD = descriptor.Extraction().bitsinfo['IDs'].tolist()
	data1 = pd.DataFrame(value,columns=MolD)
	data = pd.concat([df,data1],axis=1)
	data.to_csv(pathout,index=None)


namelist=['0_aromatase_anta','1_ahr_ago','2_ar_ago','3_er_ago','4_gr_ago','5_tshr_ago','6_tr_anta']
path1=r'files\1_smiles_list' #the path for the smiles_list folder
#=================
#generate fingerprints
for idx,f in enumerate(namelist):
	pathf=os.path.join(path1,f+'_stan_smi_activity.csv')
	for func in ['pubchem','morgan','maccs']:
		pathout=os.path.join(r"files\2_fingerprints",func,namelist[idx]+'_'+func+'fps.csv') 
		get_fps(pathf,func,pathout)
#=================
#generate descriptors
for idx,f in enumerate(namelist):
	pathf=os.path.join(path1,f+'_stan_smi_activity.csv')
	pathout=os.path.join(r"files\3_descriptors",namelist[idx]+'_'+'descriptor.csv')
	get_descriptors(pathf,pathout)
