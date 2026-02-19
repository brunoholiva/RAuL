import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
uncharger = rdMolStandardize.Uncharger()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def is_valid_smiles(smiles:str) -> bool:
    """Check if a SMILES string is valid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def standardize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        clean_molecule = rdMolStandardize.Cleanup(mol)
        parent_clean_molecule = rdMolStandardize.FragmentParent(clean_molecule)
        uncharged_clean_molecule = uncharger.uncharge(parent_clean_molecule)
        tautomer_uncharged_clean_molecule = tautomer_enumerator.Canonicalize(uncharged_clean_molecule)
        Chem.RemoveStereochemistry(tautomer_uncharged_clean_molecule)
        standardized_smiles = Chem.MolToSmiles(tautomer_uncharged_clean_molecule, isomericSmiles=False)
        return standardized_smiles
    except Exception:
        return None


def process_smiles(smi):
    if is_valid_smiles(smi):
        std_smi = standardize_smiles(smi)
        if std_smi:
            return std_smi
    return None


def preprocess_smiles_list(smiles_list):
    """
    Preprocess a list of SMILES strings by standardizing them and converting to RDKit Mol objects.
    
    :param smiles_list: List of SMILES strings
    :return: Tuple containing standardized SMILES, RDKit Mol objects, and a validity mask
    """
    std_smiles = [process_smiles(s) for s in smiles_list]
    valid_mask = [s is not None for s in std_smiles]
    std_smiles = [s if s is not None else "" for s in std_smiles]
    mols = [Chem.MolFromSmiles(s) if v else None for s, v in zip(std_smiles, valid_mask)]
    return std_smiles, mols, valid_mask


_SASCORER = None
_AD_NN = None
_AD_META = {"radius": 2, "n_bits": 2048}

def _mol_to_fp(mol, r=None, bits=None):
    """
    Convert a RDKit Mol object to a Morgan fingerprint numpy array.
    
    :param mol: RDKit Mol object
    :param r: Radius for the Morgan fingerprint
    :param bits: Number of bits for the fingerprint
    :return: Numpy array representing the Morgan fingerprint
    """
    if mol is None:
        return None
    r = _AD_META["radius"] if r is None else r
    bits = _AD_META["n_bits"] if bits is None else bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=bits)
    arr = np.zeros((bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _smiles_to_fp(smiles, r=None, bits=None):
    """
    Convert a SMILES string to a Morgan fingerprint numpy array.

    :param smiles: SMILES string
    :param r: Radius for the Morgan fingerprint
    :param bits: Number of bits for the fingerprint
    :return: Numpy array representing the Morgan fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    return _mol_to_fp(mol, r=r, bits=bits)


def randomize_smiles(smiles):
    # randomize SMILES for data augmentation
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    ans = list(range(mol.GetNumAtoms()))
    if not ans:
        return smiles
    np.random.shuffle(ans)
    new_mol = Chem.RenumberAtoms(mol, ans)
    return Chem.MolToSmiles(new_mol, canonical=False)


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except:
        return None
