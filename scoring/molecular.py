"""This module contains functions for scoring molecules based on molecular properties (QED and SA)."""

import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import numpy.typing as npt
import torch
from rdkit import Chem
from rdkit.Chem import QED, FilterCatalog

_SASCORER = None


def _get_sascorer() -> Any:
    """Lazy load the sascorer module from RDKit contrib."""
    global _SASCORER
    if _SASCORER is None:
        from rdkit.Chem import RDConfig

        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer  # type: ignore

        _SASCORER = sascorer
    return _SASCORER


_PAINS_CATALOG = None


def _get_pains_catalog() -> FilterCatalog.FilterCatalog:
    """Lazy load the PAINS filter catalog."""
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS
        )
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def sa_score(mol: Optional[Chem.rdchem.Mol]) -> float:
    """
    Compute the Synthetic Accessibility (SA) score for a molecule.

    Parameters
    ----------
    mol : Optional[Chem.rdchem.Mol]
        The RDKit Mol object.

    Returns
    -------
    float
        The SA score (lower is better). Returns 10.0 if mol is invalid.
    """
    if mol is None:
        return 10.0
    try:
        sascorer = _get_sascorer()
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 10.0


def qed_score(mol: Optional[Chem.rdchem.Mol]) -> float:
    """
    Compute the Quantitative Estimate of Druglikeness (QED) score.

    Parameters
    ----------
    mol : Optional[Chem.rdchem.Mol]
        The RDKit Mol object.

    Returns
    -------
    float
        The QED score. Returns 0.0 if mol is invalid.
    """
    if mol is None:
        return 0.0
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def load_ad_model(path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load the Applicability Domain (AD) nearest neighbor model.

    Parameters
    ----------
    path : str
        The file path to the saved joblib model.
    device : torch.device
        The device to load the PyTorch tensors onto.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the Scikit-Learn NN model, metadata,
        and pre-loaded PyTorch tensors for fast matrix multiplication.
    """
    obj = joblib.load(path)

    ad_model = {
        "nn": None,
        "radius": 2,
        "n_bits": 2048,
        "train_fps_tensor": None,
        "train_fps_sum": None,
    }

    if isinstance(obj, dict):
        ad_model["nn"] = obj.get("nn")
        ad_model["radius"] = obj.get("radius", 2)
        ad_model["n_bits"] = obj.get("n_bits", 2048)

        raw_fps = obj.get("fps")
        if raw_fps is not None:
            t_fps = torch.tensor(raw_fps, dtype=torch.float32, device=device)
            ad_model["train_fps_tensor"] = t_fps
            ad_model["train_fps_sum"] = t_fps.sum(dim=1)
    else:
        ad_model["nn"] = obj

    return ad_model


def ad_domain_score(
    fps: List[Optional[npt.NDArray[np.float32]]],
    ad_model: Dict[str, Any],
    n_neighbors: int = 5,
    default_score: float = 1.0,
) -> npt.NDArray[np.float32]:
    """
    Compute the Applicability Domain (AD) score for a list of fingerprints.

    The higher the score, the farther the molecule is from the training set.

    Parameters
    ----------
    fps : List[Optional[npt.NDArray[np.float32]]]
        A list of Morgan fingerprint numpy arrays.
    ad_model : Dict[str, Any]
        The dictionary containing the loaded AD model and tensors.
    n_neighbors : int, optional
        The number of nearest neighbors to consider (default is 5).
    default_score : float, optional
        The fallback score if calculation fails (default is 1.0).

    Returns
    -------
    npt.NDArray[np.float32]
        A 1D numpy array of AD scores matching the length of the input.
    """
    scores = np.full(len(fps), default_score, dtype=np.float32)

    valid_idx = []
    valid_fps = []

    for i, fp in enumerate(fps):
        if fp is not None:
            valid_fps.append(fp)
            valid_idx.append(i)

    if not valid_fps:
        return scores

    fps_arr = np.stack(valid_fps, axis=0)
    train_tensor = ad_model.get("train_fps_tensor")

    if train_tensor is not None:
        device = train_tensor.device
        train_sum = ad_model["train_fps_sum"]

        with torch.no_grad():
            batch_t = torch.tensor(fps_arr, dtype=torch.float32, device=device)
            batch_sum = batch_t.sum(dim=1)

            intersection = torch.matmul(batch_t, train_tensor.T)
            union = (
                batch_sum.unsqueeze(1) + train_sum.unsqueeze(0) - intersection
            )
            tanimoto = intersection / (union + 1e-8)

            topk_sim, _ = torch.topk(
                tanimoto, k=n_neighbors, dim=1, largest=True
            )
            topk_dist = 1.0 - topk_sim

            mean_dist = topk_dist.mean(dim=1).cpu().numpy()

        for i, d in zip(valid_idx, mean_dist):
            scores[i] = float(d)
        return scores

    nn_model = ad_model.get("nn")
    if nn_model is not None:
        fps_arr_bool = fps_arr.astype(bool)
        distances, _ = nn_model.kneighbors(
            fps_arr_bool, n_neighbors=n_neighbors, return_distance=True
        )
        mean_dist_sklearn = distances.mean(axis=1).astype(np.float32)

        for i, d in zip(valid_idx, mean_dist_sklearn):
            scores[i] = float(d)

    return scores


def passes_pains(mol: Optional[Chem.rdchem.Mol]) -> bool:
    """
    Check if a molecule passes the PAINS filter.

    Parameters
    ----------
    mol : Optional[Chem.rdchem.Mol]
        The RDKit Mol object.

    Returns
    -------
    bool
        True if the molecule has no PAINS matches, False otherwise.
    """
    if mol is None:
        return False
    catalog = _get_pains_catalog()
    return not catalog.HasMatch(mol)
