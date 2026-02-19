"""This module contains functions for scoring molecules based on molecular properties (QED and SA)."""
import sys
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import joblib
from utils.rdkit_utils import (
    _smiles_to_fp
)
import torch

_SASCORER = None
def _get_sascorer():
    """
    Lazy load the sascorer module from RDKit contrib.
    """
    global _SASCORER
    if _SASCORER is None:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        _SASCORER = sascorer
    return _SASCORER


_AD_NN = None
_AD_META = {"radius": 2, "n_bits": 2048}


def sa_score(smiles=None, mol=None):
    """
    Compute the SA (synthetic accessibility) score for a given SMILES string or RDKit Mol object.
    
    :param smiles: SMILES string
    :param mol: RDKit Mol object
    :return: SA score (lower is better)
    """
    sascorer = _get_sascorer()
    if mol is None:
        if not smiles:
            return 10.0
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return 10.0


def qed_score(smiles=None, mol=None):
    """
    Compute the QED score for a given SMILES string or RDKit Mol object.
    
    :param smiles: SMILES string
    :param mol: RDKit Mol object
    :return: QED score
    """
    if mol is None:
        if not smiles:
            return 0.0
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0

_AD_TRAIN_FPS_TENSOR = None
_AD_TRAIN_FPS_SUM = None


def load_ad_nn(path):
    """
    Load the AD nearest neighbor model from the given path.
    """
    global _AD_NN, _AD_META, _AD_TRAIN_FPS_TENSOR, _AD_TRAIN_FPS_SUM
    obj = joblib.load(path)
    
    if isinstance(obj, dict):
        _AD_NN = obj.get("nn")
        _AD_META["radius"] = obj.get("radius", 2)
        _AD_META["n_bits"] = obj.get("n_bits", 2048)
        
        # Extract raw fingerprints and send them to the GPU
        raw_fps = obj.get("fps")
        if raw_fps is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Convert to float32 for fast matrix multiplication
            t_fps = torch.tensor(raw_fps, dtype=torch.float32, device=device)
            _AD_TRAIN_FPS_TENSOR = t_fps
            _AD_TRAIN_FPS_SUM = t_fps.sum(dim=1)
            
    else:
        _AD_NN = obj
    return _AD_NN


def ad_domain_score(smiles_list, fps=None, default_score=1.0, n_neighbors=5):
    if _AD_TRAIN_FPS_TENSOR is None and _AD_NN is None:
        return np.full(len(smiles_list), default_score, dtype=np.float32)

    scores = np.full(len(smiles_list), default_score, dtype=np.float32)

    valid_idx = []
    fps_list = []
    if fps is None:
        for i, s in enumerate(smiles_list):
            fp = _smiles_to_fp(s)
            if fp is not None:
                fps_list.append(fp)
                valid_idx.append(i)
    else:
        for i, fp in enumerate(fps):
            if fp is not None:
                fps_list.append(fp)
                valid_idx.append(i)

    if len(fps_list) == 0:
        return scores

    fps_arr = np.stack(fps_list, axis=0)


    if _AD_TRAIN_FPS_TENSOR is not None:
        device = _AD_TRAIN_FPS_TENSOR.device
        with torch.no_grad():
            batch_t = torch.tensor(fps_arr, dtype=torch.float32, device=device)
            batch_sum = batch_t.sum(dim=1) 
            
            # 1. Intersection (A AND B) = Dot Product
            intersection = torch.matmul(batch_t, _AD_TRAIN_FPS_TENSOR.T) 
            
            # 2. Union (A OR B) = Sum(A) + Sum(B) - Intersection
            union = batch_sum.unsqueeze(1) + _AD_TRAIN_FPS_SUM.unsqueeze(0) - intersection
            
            # 3. Tanimoto Similarity
            tanimoto = intersection / (union + 1e-8)
            
            # 4. Jaccard Distance = 1 - Tanimoto
            # We want the 5 SMALLEST distances, which means the 5 LARGEST similarities
            topk_sim, _ = torch.topk(tanimoto, k=n_neighbors, dim=1, largest=True)
            topk_dist = 1.0 - topk_sim
            
            mean_dist = topk_dist.mean(dim=1).cpu().numpy()
            
        for i, d in zip(valid_idx, mean_dist):
            scores[i] = float(d)
            
        return scores

    fps_arr = fps_arr.astype(bool)
    distances, _ = _AD_NN.kneighbors(fps_arr, n_neighbors=n_neighbors, return_distance=True)
    mean_dist = distances.mean(axis=1).astype(np.float32)

    for i, d in zip(valid_idx, mean_dist):
        scores[i] = d
    return scores


def get_mw(smiles_list=None, mols=None):
    if mols is None and smiles_list is not None:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    if mols is None:
        return np.array([], dtype=np.float32)


    mw_vals = np.array([Descriptors.MolWt(m) if m is not None else 600.0 for m in mols], dtype=np.float32)
    return mw_vals