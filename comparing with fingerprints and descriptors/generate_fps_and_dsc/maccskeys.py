from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import  DataStructs
import numpy as np
import pandas as pd
import os

_type = 'SMARTS-based'

file_path = os.path.dirname(__file__)

def GetMACCSFPs(mol):

    '''
    166 bits
    '''

    fp =  AllChem.GetMACCSKeysFingerprint(mol)

    # arr = np.zeros((0,),  dtype=np.bool)
    arr = np.zeros((0,),dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def GetMACCSFPInfos():
    return pd.read_excel(os.path.join(file_path, 'maccskeys.xlsx'))


if __name__ == '__main__':
    print('-'*10+'START'+'-'*10)
    SMILES = 'C1=NC2NC3=CNCC3=CC2CC1'
    mol = Chem.MolFromSmiles(SMILES)
    result = GetMACCSFPs(mol)
    print('Molecule: %s'%SMILES)
    print('-'*25)
    print('Results: %s'%result)
    print('-'*10+'END'+'-'*10)