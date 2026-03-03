"""
Reward calculation, replay buffer management, and diversity filtering for the reinforcement learning loop
"""

import concurrent.futures
from typing import Any, Dict, List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from scoring.activity import predict_activity_proba
from scoring.molecular import (
    ad_domain_score,
    passes_pains,
    qed_score,
    sa_score,
)
from utils.rdkit_utils import get_scaffold, mol_to_fp, process_smiles
from utils.utils import reverse_sigmoid, sigmoid


def process_single_molecule(smi: str) -> Dict[str, Any]:
    """
    Process a single SMILES string and compute RDKit properties.
    """
    invalid_dict = {
        "valid": False,
        "smi": smi,
        "std_smi": "",
        "canon_smi": None,
        "qed": 0.0,
        "sa": 10.0,
        "mw": 600.0,
        "fp": None,
        "scaffold": None,
        "scaffold_smi": None,
        "scaffold_fp": None,
        "pains_ok": 0.0,
    }

    std_smi = process_smiles(smi)
    if not std_smi:
        return invalid_dict

    mol = Chem.MolFromSmiles(std_smi)
    if not mol:
        return invalid_dict

    canon_smi = Chem.MolToSmiles(mol)

    scaffold_mol = get_scaffold(mol)
    scaffold_smi = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else None

    scaffold_fp = (
        AllChem.GetMorganFingerprintAsBitVect(
            scaffold_mol, radius=2, nBits=2048
        )
        if scaffold_mol
        else None
    )

    return {
        "valid": True,
        "smi": smi,
        "std_smi": std_smi,
        "canon_smi": canon_smi,
        "qed": qed_score(mol=mol),
        "sa": sa_score(mol=mol),
        "mw": Descriptors.MolWt(mol),
        "fp": mol_to_fp(mol),
        "scaffold": scaffold_smi,
        "scaffold_smi": scaffold_smi,
        "scaffold_fp": scaffold_fp,
        "pains_ok": 1.0 if passes_pains(mol) else 0.0,
    }


def parallel_process_batch(smiles_list, max_workers=None):
    """
    Takes a list of 300 SMILES and splits them across all available CPU cores.
    """
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        results = list(executor.map(process_single_molecule, smiles_list))
    return results


def make_rf_scorer(rf_model, weight: float):
    def scorer(processed_data) -> np.ndarray:
        fps = [d["fp"] for d in processed_data]
        rf_probs = predict_activity_proba(fps, rf_model=rf_model)
        rf_probs_clipped = np.clip(rf_probs, 0.0, 0.7)
        return weight * np.array(
            [sigmoid(v, low=0.6, high=0.85) for v in rf_probs_clipped],
            dtype=np.float32,
        )

    return scorer


def make_ad_scorer(ad_model, weight: float = 1.0):
    def scorer(processed_data) -> np.ndarray:
        fps = [d["fp"] for d in processed_data]
        ad_dists = ad_domain_score(fps=fps, ad_model=ad_model)
        return weight * np.array(
            [reverse_sigmoid(v, low=0.5, high=0.7) for v in ad_dists],
            dtype=np.float32,
        )

    return scorer


def make_qed_scorer(weight: float):
    def scorer(processed_data) -> np.ndarray:
        qed_vals = np.array(
            [d.get("qed", 0.0) for d in processed_data], dtype=np.float32
        )
        return weight * np.array(
            [sigmoid(v, low=0.4, high=0.8) for v in qed_vals], dtype=np.float32
        )

    return scorer


def make_sa_scorer(weight: float):
    def scorer(processed_data) -> np.ndarray:
        sa_vals = np.array(
            [d.get("sa", 10.0) for d in processed_data], dtype=np.float32
        )
        return weight * np.array(
            [reverse_sigmoid(v, low=3.0, high=6.0) for v in sa_vals],
            dtype=np.float32,
        )

    return scorer


def make_mw_penalty():
    def scorer(processed_data) -> np.ndarray:
        mw_weights = np.array(
            [d.get("mw", 600.0) for d in processed_data], dtype=np.float32
        )
        return np.array(
            [reverse_sigmoid(v, low=500, high=600) for v in mw_weights],
            dtype=np.float32,
        )

    return scorer


def make_valid_mask():
    def scorer(processed_data) -> np.ndarray:
        return np.array(
            [1.0 if d["valid"] else 0.0 for d in processed_data],
            dtype=np.float32,
        )

    return scorer


def make_pains_filter():
    def scorer(processed_data) -> np.ndarray:
        return np.array(
            [d.get("pains_ok", 1.0) for d in processed_data], dtype=np.float32
        )

    return scorer


def compute_reward(processed_data, additive_scorers, multiplier_scorers):
    """
    Calculates rewards using pre-configured scoring functions.
    """
    batch_size = len(processed_data)
    total_reward = np.zeros(batch_size, dtype=np.float32)

    for scorer in additive_scorers:
        total_reward += scorer(processed_data)

    for multiplier in multiplier_scorers:
        total_reward *= multiplier(processed_data)

    return total_reward


def apply_diversity_filter(
    processed_data: List[Dict[str, Any]],
    raw_scores: Any,
    global_memory: Dict[str, Any],
    bucket_size: int = 25,
    min_score: float = None,
    min_similarity: float = 0.4,
) -> List[float]:
    """
    Applies a diversity filter to the raw scores based on scaffold similarity.

    Parameters
    ----------
    processed_data : List[Dict[str, Any]]
        The list of dictionaries pre-computed by parallel_process_batch.
    raw_scores : Any
        The raw reward scores corresponding to each item in processed_data.
    global_memory : Dict[str, Any]
        A dictionary that keeps track of seen scaffolds and their counts.
    bucket_size : int, optional
        The maximum count for a scaffold before it gets penalized (default is 25).
    min_score : float, optional
        The minimum score threshold to consider for diversity filtering (default is 0.4).
    min_similarity : float, optional
        The minimum Tanimoto similarity to consider two scaffolds as the same (default is 0.4).

    Returns
    -------
    List[float]
        The list of final scores after applying the diversity filter.
    """
    final_scores = []
    local_smiles_seen = set()
    MAX_MEMORY_SIZE = 5000

    if min_score is None:
        min_score = float(np.median(raw_scores))

    mem_keys = list(global_memory.keys())
    mem_fps = [global_memory[k]["fp"] for k in mem_keys]

    new_batch_keys = []
    new_batch_fps = []

    for i, data in enumerate(processed_data):
        score = float(raw_scores[i])

        if not data["valid"]:
            final_scores.append(0.0)
            continue

        canon_smiles = data["canon_smi"]
        if canon_smiles in local_smiles_seen:
            final_scores.append(0.0)
            continue
        local_smiles_seen.add(canon_smiles)

        if score < min_score:
            final_scores.append(score)
            continue

        scaffold_smi = data.get("scaffold_smi")
        scaffold_fp = data.get("scaffold_fp")
        if scaffold_smi is None or scaffold_fp is None:
            final_scores.append(0.0)
            continue

        best_match_smi = None

        if mem_fps:
            sims = DataStructs.BulkTanimotoSimilarity(scaffold_fp, mem_fps)
            max_sim_idx = int(np.argmax(sims))
            if sims[max_sim_idx] >= min_similarity:
                best_match_smi = mem_keys[max_sim_idx]

        if best_match_smi is None and new_batch_fps:
            sims = DataStructs.BulkTanimotoSimilarity(
                scaffold_fp, new_batch_fps
            )
            max_sim_idx = int(np.argmax(sims))
            if sims[max_sim_idx] >= min_similarity:
                best_match_smi = new_batch_keys[max_sim_idx]

        if best_match_smi is not None:
            if best_match_smi in global_memory:
                if global_memory[best_match_smi]["count"] >= bucket_size:
                    final_scores.append(0.0)
                else:
                    final_scores.append(score)
                    global_memory[best_match_smi]["count"] += 1
            else:
                final_scores.append(0.0)
        else:
            final_scores.append(score)
            global_memory[scaffold_smi] = {"count": 1, "fp": scaffold_fp}

            new_batch_keys.append(scaffold_smi)
            new_batch_fps.append(scaffold_fp)

            if len(global_memory) > MAX_MEMORY_SIZE:
                global_memory.popitem(last=False)

    return final_scores
