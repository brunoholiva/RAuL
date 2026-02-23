"""
Utility functions for SMILES generation, sampling, and reward transformations.Base code used from ACARL
"""

import random
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from pretraining.vocabulary import SMILESTokenizer, Vocabulary


def sample_smiles_nograd(
    model: nn.Module,
    voc: Vocabulary,
    n_mols: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int = 10,
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
    n_mols: int
        The number of SMILES strings to generate.
    block_size: int
        The maximum length of the generated SMILES strings (including start/end tokens).
    temperature: float, default=1.0
        The temperature for sampling. Higher values (>1.0) lead to more random samples,
        lower values (<1.0) make the distribution sharper and more deterministic.
    top_k: int, default=10
        The number of top tokens to consider for sampling at each step. Limits the
        sampling pool to the k most likely tokens.

    Returns:
    --------
    Tuple[List[str], torch.Tensor]
        A tuple containing:
        - smiles (List[str]): List of generated SMILES strings (untokenized).
        - codes (torch.Tensor): Tensor of shape (n_mols, block_size) containing
          the token indices for all generated sequences.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        codes = torch.full(
            (n_mols, block_size),
            fill_value=voc["$"],
            dtype=torch.long,
            device=device,
        )
        codes[:, 0] = voc["^"]
        finished = torch.zeros(n_mols, dtype=torch.bool, device=device)

        for i in range(1, block_size):
            logits, _, _ = model(codes[:, :i])
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = top_k_logits(logits, k=top_k)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token = torch.where(
                finished, torch.full_like(next_token, voc["$"]), next_token
            )

            codes[:, i] = next_token
            finished |= next_token == voc["$"]

            if finished.all():
                break

    tokenizer = SMILESTokenizer()
    smiles = []
    for i in range(n_mols):
        tokens = voc.decode(codes[i].cpu().numpy())
        smiles.append(tokenizer.untokenize(tokens))

    return smiles, codes


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
