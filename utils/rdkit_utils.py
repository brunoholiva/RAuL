"""
Utility functions for processing and standardizing molecules using RDKit.
"""

from typing import List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc import Evaluator

UNCHARGER = rdMolStandardize.Uncharger()
TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.

    Parameters
    ----------
    smiles: str
        The SMILES string to check.
    Returns
    -------
    bool
        True if SMILES string can be parsed by RDKit, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def standardize_smiles(smiles: str):
    """
    Standardize a SMILES string by uncharging, canonicalizing tautomers, and removing stereochemistry.

    Parameters
    ----------
    smiles: str
        The SMILES string to standardize.

    Returns
    -------
    Optional[str]
        The standardized SMILES string, or None if standardization fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharged_mol = UNCHARGER.uncharge(parent_clean_mol)
        tautomer_mol = TAUTOMER_ENUMERATOR.Canonicalize(uncharged_mol)

        Chem.RemoveStereochemistry(tautomer_mol)

        return Chem.MolToSmiles(tautomer_mol, isomericSmiles=False)
    except Exception:
        return None


def process_smiles(smi):
    """
    Validate and standardize a SMILES string in one step.

    Parameters
    ----------
    smi: str
        The SMILES string.

    Returns
    -------
    Optional[str]
        The standardized SMILES string if valid, otherwise None.
    """
    if is_valid_smiles(smi):
        std_smi = standardize_smiles(smi)
        if std_smi:
            return std_smi
    return None


def preprocess_smiles_list(
    smiles_list: List[str],
) -> Tuple[List[str], List[Optional[Chem.rdchem.Mol]], npt.NDArray[np.bool_]]:
    """
    Preprocess a list of SMILES by standardizing and converting to Mol objects.

    Parameters:
    -----------
    smiles_list: List[str]
        A list of raw SMILES strings.

    Returns:
    --------
    Tuple[List[str], List[Optional[Chem.rdchem.Mol]], npt.NDArray[np.bool_]]
        A tuple containing:
        - List of standardized SMILES strings (empty string if invalid).
        - List of RDKit Mol objects (None if invalid).
        - A 1D numpy boolean array indicating valid SMILES.
    """
    std_smiles: List[str] = []
    mols: List[Optional[Chem.rdchem.Mol]] = []
    valid_mask = np.zeros(len(smiles_list), dtype=np.bool_)

    for i, smi in enumerate(smiles_list):
        processed_smi = process_smiles(smi)

        if processed_smi is not None:
            std_smiles.append(processed_smi)
            mols.append(Chem.MolFromSmiles(processed_smi))
            valid_mask[i] = True
        else:
            std_smiles.append("")
            mols.append(None)
            valid_mask[i] = False

    return std_smiles, mols, valid_mask


def mol_to_fp(
    mol: Optional[Chem.rdchem.Mol], radius: int = 2, n_bits: int = 2048
) -> Optional[npt.NDArray[np.float32]]:
    """
    Convert an RDKit Mol object to a Morgan fingerprint numpy array.

    Parameters
    ----------
    mol : Optional[Chem.rdchem.Mol]
        The RDKit Mol object.
    radius : int, optional
        Radius for the Morgan fingerprint (default is 2).
    n_bits : int, optional
        Number of bits for the fingerprint vector (default is 2048).

    Returns
    -------
    Optional[npt.NDArray[np.float32]]
        A 1D numpy array representing the fingerprint, or None if mol is None.
    """

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fp(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> Optional[npt.NDArray[np.float32]]:
    """
    Convert a SMILES string to a Morgan fingerprint numpy array.

    Parameters
    ----------
    smiles : str
        The SMILES string.
    radius : int, optional
        Radius for the Morgan fingerprint (default is 2).
    n_bits : int, optional
        Number of bits for the fingerprint vector (default is 2048).

    Returns
    -------
    Optional[npt.NDArray[np.float32]]
        A 1D numpy array representing the fingerprint, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_fp(mol, radius=radius, n_bits=n_bits)


def randomize_smiles(smiles: str) -> str:
    """
    Randomize a SMILES string representation for data augmentation.

    Parameters
    ----------
    smiles : str
        The canonical or raw SMILES string.

    Returns
    -------
    str
        A randomized, non-canonical SMILES string. Returns the original string
        if RDKit fails to parse it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    ans = list(range(mol.GetNumAtoms()))
    if not ans:
        return smiles

    np.random.shuffle(ans)
    new_mol = Chem.RenumberAtoms(mol, ans)
    return Chem.MolToSmiles(new_mol, canonical=False)


def get_scaffold(
    mol: Optional[Chem.rdchem.Mol],
) -> Optional[Chem.rdchem.Mol]:
    """
    Extract the Murcko scaffold from an RDKit Mol object.

    Parameters
    ----------
    mol : Optional[Chem.rdchem.Mol]
        The input RDKit Mol object.

    Returns
    -------
    Optional[Chem.rdchem.Mol]
        The Murcko scaffold as an RDKit Mol object, or None if
        extraction fails or the input is None.
    """
    if mol is None:
        return None

    try:
        return MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None


def model_uniqueness(smiles_list: List[str]) -> float:
    """
    Calculate the proportion of unique, valid SMILES in a list.

    Parameters
    ----------
    smiles_list : List[str]
        A list of generated SMILES strings.

    Returns
    -------
    float
        The uniqueness score (0.0 to 1.0).
    """
    valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
    unique_smiles = set(valid_smiles)
    return len(unique_smiles) / max(1, len(valid_smiles))


def model_novelty(smiles_list: List[str], train_smiles_set: Set[str]) -> float:
    """
    Calculate the proportion of valid SMILES not present in training data.

    Parameters
    ----------
    smiles_list : List[str]
        A list of generated SMILES strings.
    train_smiles_set : Set[str]
        A set containing all training SMILES for fast lookup.

    Returns
    -------
    float
        The novelty score (0.0 to 1.0).
    """
    valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
    unique_smiles = set(valid_smiles)
    novel_smiles = [s for s in unique_smiles if s not in train_smiles_set]
    return len(novel_smiles) / max(1, len(unique_smiles))


def calculate_average_tanimoto(
    fps_matrix: npt.NDArray[np.float32],
) -> float:
    """
    Calculate the average pairwise Tanimoto similarity for a matrix.

    Parameters:
    -----------
    fps_matrix: npt.NDArray[np.float32]
        A 2D numpy array of shape (n_molecules, n_bits).

    Returns:
    --------
    float
        The average Tanimoto similarity across all unique pairs.
    """
    intersection = np.dot(fps_matrix, fps_matrix.T)

    sum_on_bits = fps_matrix.sum(axis=1)
    union = sum_on_bits[:, None] + sum_on_bits[None, :] - intersection

    tanimoto_matrix = intersection / (union + 1e-8)

    i, j = np.triu_indices(len(fps_matrix), k=1)
    return float(np.mean(tanimoto_matrix[i, j]))


def model_diversity(smiles_list: List[str]) -> float:
    """
    Calculate the internal diversity of a generated set of molecules.

    Parameters:
    -----------
    smiles_list: List[str]
        A list of generated SMILES strings.

    Returns:
    --------
    float
        The average diversity score (1.0 - average similarity).
    """
    fps_list = [smiles_to_fp(s) for s in smiles_list]

    valid_fps = [fp for fp in fps_list if fp is not None]

    if len(valid_fps) < 2:
        return 0.0

    fps_matrix = np.stack(valid_fps)

    avg_similarity = calculate_average_tanimoto(fps_matrix)

    return 1.0 - avg_similarity


def model_validity(smiles_list: List[str]) -> float:
    """
    Calculate the proportion of valid SMILES using TDC Evaluator.

    Parameters:
    -----------
    smiles_list: List[str]
        A list of generated SMILES strings.

    Returns:
    --------
    float
        The validity score (0.0 to 1.0).
    """
    evaluator = Evaluator(name="Validity")
    return float(evaluator(smiles_list))
