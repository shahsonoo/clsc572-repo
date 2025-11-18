import numpy as np
from typing import Iterable, List
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs

def _to_mols(smiles_list: Iterable[str]):
    mols=[]
    for s in smiles_list:
        mols.append(Chem.MolFromSmiles(str(s)))
    return mols

def generate_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    mols=_to_mols(smiles_list)
    fps=[]
    for mol in mols:
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=int)); continue
        fp=AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr=np.zeros((n_bits,),dtype=int)
        DataStructs.ConvertToNumpyArray(fp,arr)
        fps.append(arr)
    return np.vstack(fps)

def generate_maccs_keys(smiles_list):
    mols=_to_mols(smiles_list)
    fps=[]
    for mol in mols:
        if mol is None:
            fps.append(np.zeros(167,dtype=int)); continue
        fp=MACCSkeys.GenMACCSKeys(mol)
        arr=np.zeros((fp.GetNumBits(),),dtype=int)
        DataStructs.ConvertToNumpyArray(fp,arr)
        fps.append(arr)
    return np.vstack(fps)
