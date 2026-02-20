import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from utils.utils import sigmoid, reverse_sigmoid
from utils.rdkit_utils import get_scaffold, process_smiles, _mol_to_fp
from scoring.activity import predict_activity_proba
from scoring.molecular import (
    sa_score,
    qed_score,
    ad_domain_score,
    get_mw
)
from pretraining.vocabulary import SMILESTokenizer
import torch
import concurrent.futures

def process_single_molecule(smi):
    """
    This function does all the heavy RDKit lifting for ONE molecule.
    It returns a dictionary of basic data types so Python can easily 
    send it back to the main process.
    """
    std_smi = process_smiles(smi)
    if not std_smi:
        return {"valid": False, "smi": smi, "fp": None, "canon_smi": None, "scaffold": None}
    
    mol = Chem.MolFromSmiles(std_smi)
    if not mol:
        return {"valid": False, "smi": smi, "fp": None, "canon_smi": None, "scaffold": None}

    canon_smi = Chem.MolToSmiles(mol)
    
    return {
        "valid": True,
        "smi": smi,
        "std_smi": std_smi,
        "canon_smi": canon_smi,
        "qed": qed_score(mol=mol),
        "sa": sa_score(mol=mol),
        "mw": Descriptors.MolWt(mol),
        "fp": _mol_to_fp(mol),
        "scaffold": get_scaffold(canon_smi)
    }


def parallel_process_batch(smiles_list, max_workers=None):
    """
    Takes a list of 300 SMILES and splits them across all available CPU cores.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_molecule, smiles_list))
    return results


def compute_reward(processed_data, w_rf=4.0, w_qed=1.0, w_sa=1.0):
    """
    Calculates rewards using the pre-computed data from the parallel workers.
    """
    n_mols = len(processed_data)
    
    # Extract lists from our parallel results
    valid_mask = np.array([1.0 if d["valid"] else 0.0 for d in processed_data], dtype=np.float32)
    fps = [d["fp"] for d in processed_data]
    std_smiles = [d["std_smi"] if d["valid"] else "" for d in processed_data]
    
    # Default values for invalid molecules (will be zeroed out by valid_mask anyway)
    qed_vals = np.array([d.get("qed", 0.0) for d in processed_data], dtype=np.float32)
    sa_vals = np.array([d.get("sa", 10.0) for d in processed_data], dtype=np.float32)
    mw_weights = np.array([d.get("mw", 600.0) for d in processed_data], dtype=np.float32)

    # ML Models run incredibly fast on vectorized Numpy arrays in the main thread
    rf_probs = predict_activity_proba(fps)
    ad_dists = ad_domain_score(std_smiles, fps=fps)

    # Apply Transforms
    score_rf = np.array([sigmoid(v, low=0.5, high=0.85) for v in rf_probs], dtype=np.float32)
    score_ad_trust = np.array([reverse_sigmoid(v, low=0.4, high=0.6) for v in ad_dists], dtype=np.float32)
    score_qed = np.array([sigmoid(v, low=0.4, high=0.8) for v in qed_vals], dtype=np.float32)
    score_sa = np.array([reverse_sigmoid(v, low=3.0, high=6.0) for v in sa_vals], dtype=np.float32)
    score_mw = np.array([reverse_sigmoid(v, low=500, high=600) for v in mw_weights], dtype=np.float32)

    # Weighted Sum with GATING
    term_rf = w_rf * (score_rf * score_ad_trust)
    term_qed = w_qed * score_qed
    term_sa = w_sa * score_sa

    raw_total_reward = term_rf + term_qed + term_sa
    total_reward = raw_total_reward * score_mw  # Penalize if MW is out of range

    # Zero out invalid molecules
    final_rewards = total_reward * valid_mask

    return final_rewards

def apply_diversity_filter(processed_data, raw_scores, global_memory, bucket_size=25):
    """
    Uses the pre-computed scaffolds so we don't recalculate them here.
    """
    final_scores = []
    local_smiles_seen = set()

    for i, data in enumerate(processed_data):
        score = raw_scores[i]
        
        if not data["valid"]:
            final_scores.append(0.0)
            continue

        canon_smiles = data["canon_smi"]

        # Step B: Check for exact SMILES repeat in current batch
        if canon_smiles in local_smiles_seen:
            final_scores.append(0.0)
            continue
        local_smiles_seen.add(canon_smiles)

        # Step C: Scaffold Bucket check
        scaffold = data["scaffold"]
        if scaffold and scaffold in global_memory and global_memory[scaffold] >= bucket_size:
            final_scores.append(0.0)
        else:
            final_scores.append(score)
            if scaffold:
                global_memory[scaffold] = global_memory.get(scaffold, 0) + 1
            
    return final_scores


def update_replay_buffer(buffer, smiles_list, scores, prior_logprobs, buffer_size=100, max_per_scaffold=3):
    """
    Updates the buffer with new experience, enforcing scaffold diversity.
    Structure of buffer item: (score, smiles, prior_logprob)
    """
    # 1. Zip the current batch into tuples
    new_items = []
    for sm, sc, plp in zip(smiles_list, scores, prior_logprobs):
        if sc > 0.0:
            new_items.append((float(sc), sm, float(plp.item())))

    # 2. Add to existing buffer and sort by score (descending)
    combined = buffer + new_items
    combined.sort(key=lambda x: x[0], reverse=True)

    # 3. Filter for intra-buffer diversity
    diverse_buffer = []
    scaffold_counts = {}

    for item in combined:
        score, smi, prior = item
        scaffold = get_scaffold(smi)
        
        # If it's a valid scaffold, check its count
        if scaffold:
            count = scaffold_counts.get(scaffold, 0)
            if count < max_per_scaffold:
                diverse_buffer.append(item)
                scaffold_counts[scaffold] = count + 1
        else:
            # If get_scaffold fails, just add it (or you can choose to skip)
            diverse_buffer.append(item)

        # Stop once we hit the buffer size
        if len(diverse_buffer) >= buffer_size:
            break

    return diverse_buffer

import random

def sample_replay_buffer(buffer, batch_size, voc, device, max_length):
    """
    Samples 'batch_size' items from the buffer and converts them 
    into tensors ready for the model.
    """
    if len(buffer) == 0:
        return None, None, None

    # REINVENT strategy: Sample uniformly from the top-k buffer
    # If buffer is smaller than requested batch, take everything
    n_samples = min(len(buffer), batch_size)
    sampled_items = random.sample(buffer, n_samples)

    # Unzip the items
    scores, smiles, prior_logprobs = zip(*sampled_items)

    # Tokenize the SMILES back into tensors
    # We need to manually handle the padding here since we are outside the standard sampler
    tokenizer = SMILESTokenizer()
    tokenized = [tokenizer.tokenize(s) for s in smiles]
    
    # Pad to max_length and add start/end tokens
    # Note: Your model expects ^ at start and $ at end/padding
    codes = torch.full((n_samples, max_length), voc["$"], dtype=torch.long, device=device)
    
    for i, tokens in enumerate(tokenized):
        # Convert tokens to indices
        token_indices = [voc[t] for t in tokens if t in voc] 
        # Add Start Token
        full_seq = [voc["^"]] + token_indices + [voc["$"]]
        
        # Truncate if too long (rare, but safe)
        if len(full_seq) > max_length:
            full_seq = full_seq[:max_length]
            
        codes[i, :len(full_seq)] = torch.tensor(full_seq, dtype=torch.long, device=device)

    # Convert scores and priors to tensors
    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    priors_tensor = torch.tensor(prior_logprobs, dtype=torch.float32, device=device)

    return codes, scores_tensor, priors_tensor
