import numpy as np
from rdkit import Chem
from utils.utils import sigmoid, reverse_sigmoid
from utils.rdkit_utils import get_scaffold, preprocess_smiles_list, _mol_to_fp
from scoring.activity import predict_activity_proba
from scoring.molecular import (
    sa_score,
    qed_score,
    ad_domain_score,
    get_mw
)
from pretraining.vocabulary import SMILESTokenizer
import torch


def compute_property_scores(std_smiles, mols):
    """
    Compute the property scores (RF activity, QED, SA, AD) for a list of standardized SMILES 
    and their corresponding Mol objects.
    
    :param std_smiles: List of standardized SMILES strings
    :param mols: List of RDKit Mol objects
    :return: Tuple containing RF scores, QED scores, SA scores, and AD scores
    """
    rf_scores = predict_activity_proba(std_smiles)

    qed_scores = np.array(
        [qed_score(mol=m) for m in mols],
        dtype=np.float32,
    )

    sa_raw = np.array(
        [sa_score(mol=m) for m in mols],
        dtype=np.float32,
    )

    fps = [_mol_to_fp(m) for m in mols]
    ad_scores = ad_domain_score(std_smiles, fps=fps)

    return rf_scores, qed_scores, sa_raw, ad_scores



def compute_reward(smiles_list, w_rf=4.0, w_qed=1.0, w_sa=1.0):
    """
    Reward function with GATED Activity Score.
    AD Score acts as a trust gate: Reward = (RF * AD_Trust) + QED + SA
    """
    # 1. Preprocess
    std_smiles, mols, valid_mask = preprocess_smiles_list(smiles_list)

    # 2. Compute Raw Values
    rf_probs = predict_activity_proba(std_smiles)
    qed_vals = np.array([qed_score(mol=m) for m in mols], dtype=np.float32)
    sa_vals = np.array([sa_score(mol=m) for m in mols], dtype=np.float32)
    mw_weights = get_mw(mols=mols)

    fps = [_mol_to_fp(m) for m in mols]
    ad_dists = ad_domain_score(std_smiles, fps=fps)

    # 3. Apply Transforms
    score_rf = np.array([sigmoid(v, low=0.5, high=0.85) for v in rf_probs], dtype=np.float32)
    score_ad_trust = np.array([reverse_sigmoid(v, low=0.3, high=0.6) for v in ad_dists], dtype=np.float32)
    score_qed = np.array([sigmoid(v, low=0.4, high=0.8) for v in qed_vals], dtype=np.float32)
    score_sa = np.array([reverse_sigmoid(v, low=3.0, high=6.0) for v in sa_vals], dtype=np.float32)
    score_mw = np.array([reverse_sigmoid(v, low=500, high=600) for v in mw_weights], dtype=np.float32)
    # 4. Weighted Sum with GATING
    term_rf = w_rf * (score_rf * score_ad_trust)
    term_qed = w_qed * score_qed
    term_sa = w_sa * score_sa

    raw_total_reward = term_rf + term_qed + term_sa

    total_reward = raw_total_reward * score_mw  # Penalize if MW is out of range

    final_rewards = total_reward * np.array(valid_mask, dtype=np.float32)

    return final_rewards


def apply_diversity_filter(smiles_list, raw_scores, global_memory, bucket_size=25):
    final_scores = []
    local_smiles_seen = set()  # Global SMILES memory size 1 logic

    for i, (smiles, score) in enumerate(zip(smiles_list, raw_scores)):
        # Step A: Canonicalize for exact duplicate check
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            final_scores.append(0.0)
            continue
        canon_smiles = Chem.MolToSmiles(mol)

        # Step B: Check for exact SMILES repeat (Global SMILES Memory size 1)
        if canon_smiles in local_smiles_seen:
            final_scores.append(0.0)
            continue
        local_smiles_seen.add(canon_smiles)

        # Step C: Scaffold Bucket check
        scaffold = get_scaffold(canon_smiles)
        if scaffold in global_memory and global_memory[scaffold] >= bucket_size:
            final_scores.append(0.0)
        else:
            final_scores.append(score)
            # Update global memory
            global_memory[scaffold] = global_memory.get(scaffold, 0) + 1
            
    return final_scores


def update_replay_buffer(buffer, smiles_list, scores, prior_logprobs, buffer_size=100):
    """
    Updates the buffer with new experience.
    Keeps the buffer sorted by score (highest first) and truncated to buffer_size.
    
    Structure of buffer item: (score, smiles, prior_logprob)
    """
    # 1. Zip the current batch into tuples
    # We cast score to float for sorting
    new_items = []
    for sm, sc, plp in zip(smiles_list, scores, prior_logprobs):
        # Only add valid molecules with non-zero score
        if sc > 0.0:
            new_items.append((float(sc), sm, float(plp.item())))

    # 2. Add to existing buffer
    buffer.extend(new_items)

    # 3. Sort by score (descending) -> Best molecules first
    # This is the "Prioritized" part of REINVENT
    buffer.sort(key=lambda x: x[0], reverse=True)

    # 4. Keep only the top N
    return buffer[:buffer_size]

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
