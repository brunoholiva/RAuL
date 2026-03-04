"""
Utility functions for SMILES generation, sampling, and reward transformations.Base code used from ACARL
"""

import random
from typing import Any, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn

from config import ModelConfig, TrainingConfig
from pretraining.vocabulary import SMILESTokenizer, Vocabulary


def sample_smiles_nograd(
    model: nn.Module,
    voc: Vocabulary,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
) -> Tuple[List[str], torch.Tensor]:
    """
    Sample SMILES strings from a trained model without tracking gradients.

    This function generates SMILES strings autoregressively by sampling tokens
    from the model's output distribution. It uses temperature scaling and top-k
    filtering to control the randomness of sampling. The generation stops when
    all sequences reach the end-of-sequence token ('$') or hit the block_size limit.

    Parameters:
    -----------
    model: nn.Module
        The trained PyTorch model for SMILES generation (e.g., GPT).
    voc: Vocabulary
        The vocabulary object containing token-to-index mappings and decode methods.
    model_cfg: ModelConfig
        Configuration for the model architecture (e.g., n_layer, n_head, n_embd, max_length).
    train_cfg: TrainingConfig
        Configuration for training parameters (e.g., learning_rate, temperature, top_k).
    Returns:
    --------
    Tuple[List[str], torch.Tensor]
        A tuple containing:
        - smiles (List[str]): List of generated SMILES strings (untokenized).
        - token_ids (torch.Tensor): Tensor of shape (batch_size, max_length) containing
          the token indices for all generated sequences.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        token_ids = torch.full(
            (train_cfg.batch_size, model_cfg.max_length),
            fill_value=voc["$"],
            dtype=torch.long,
            device=device,
        )
        token_ids[:, 0] = voc["^"]
        finished = torch.zeros(
            train_cfg.batch_size, dtype=torch.bool, device=device
        )

        for i in range(1, model_cfg.max_length):
            logits, _, _ = model(token_ids[:, :i])
            logits = logits[:, -1, :] / train_cfg.temperature

            if train_cfg.top_k is not None:
                logits = top_k_logits(logits, k=train_cfg.top_k)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token = torch.where(
                finished, torch.full_like(next_token, voc["$"]), next_token
            )

            token_ids[:, i] = next_token
            finished |= next_token == voc["$"]

            if finished.all():
                break

    tokenizer = SMILESTokenizer()
    smiles = []
    for i in range(train_cfg.batch_size):
        tokens = voc.decode(token_ids[i].cpu().numpy())
        smiles.append(tokenizer.untokenize(tokens))

    return smiles, token_ids


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across numpy, random, and PyTorch.

    Parameters:
    -----------
    seed: int
        The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits so that only the top k tokens have non-negative infinity values.

    Parameters:
    -----------
    logits: torch.Tensor
        The raw logits from the model.
    k: int
        The number of top tokens to keep.

    Returns:
    --------
    torch.Tensor
        The masked logits.
    """
    v, ix = torch.topk(logits, k)
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(dim=-1, index=ix, src=v)
    return out


def to_tensor(data: Any, device: torch.device) -> torch.Tensor:
    """
    Convert data (numpy array or list) to a PyTorch tensor on a specific device.

    Parameters:
    -----------
    data: Any
        The input data to convert (e.g., numpy array, list).
    device: torch.device
        The device to place the tensor on.

    Returns:
    --------
    torch.Tensor
        The resulting PyTorch tensor.
    """
    return torch.as_tensor(data, device=device)


def reverse_sigmoid(
    value: float, low: float, high: float, k: float = 0.25
) -> float:
    """
    Apply a reverse sigmoid transformation.

    High values are BAD (0.0), Low values are GOOD (1.0).
    Example: Penalizing Molecular Weight.

    Parameters:
    -----------
    value: float
        The raw value to be transformed.
    low: float
        The lower bound of the transformation window.
    high: float
        The upper bound of the transformation window.
    k: float, default=0.25
        The slope of the sigmoid curve.

    Returns:
    --------
    float
        The transformed score between 0.0 and 1.0.
    """
    if value < low:
        return 1.0
    if value > high:
        return 0.0

    exponent = k * (value - (high + low) / 2) / (high - low)
    return 1.0 / (1.0 + 10.0**exponent)


def sigmoid(value: float, low: float, high: float, k: float = 0.25) -> float:
    """
    Apply a standard sigmoid transformation.

    High values are GOOD (1.0), Low values are BAD (0.0).
    Example: QED or Activity Probability.

    Parameters:
    -----------
    value: float
        The raw value to be transformed.
    low: float
        The lower bound of the transformation window.
    high: float
        The upper bound of the transformation window.
    k: float, default=0.25
        The slope of the sigmoid curve.

    Returns:
    --------
    float
        The transformed score between 0.0 and 1.0.
    """
    if value < low:
        return 0.0
    if value > high:
        return 1.0

    exponent = k * ((high + low) / 2 - value) / (high - low)
    return 1.0 / (1.0 + 10.0**exponent)


def load_smiles_set(path: Optional[str]) -> Optional[Set[str]]:
    """
    Load a set of SMILES strings from a text file for fast lookup.

    Parameters
    ----------
    path : Optional[str]
        The file path to the SMILES list.

    Returns
    -------
    Optional[Set[str]]
        A set containing the SMILES strings, or None if path is None.
    """
    if path is None:
        return None
    smiles_set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles_set.add(s)
    return smiles_set
