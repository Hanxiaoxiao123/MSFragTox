from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import  DataStructs
import numpy as np

def GetMorganFPs(mol, nBits=2048, radius = 2, return_bitInfo = False):
    
    """
    ECFP4: radius=2
    """
    bitInfo={}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, 
                                               bitInfo=bitInfo, nBits = nBits)
    # arr = np.zeros((0,),  dtype=np.bool)
    arr = np.zeros((0,),dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    if return_bitInfo:
        return arr, bitInfo
    return arr


if __name__ == '__main__':
    print('-'*10+'START'+'-'*10)
    SMILES = 'C1=NC2NC3=CNCC3=CC2CC1'
    mol = Chem.MolFromSmiles(SMILES)
    result = GetMorganFPs(mol,return_bitInfo=True)
    print('Molecule: %s'%SMILES)
    print('-'*25)
    print('Results: %s'%result[0])
    print('Results: %s'%result[1])
    print('-'*10+'END'+'-'*10)