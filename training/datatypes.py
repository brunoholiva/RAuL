from dataclasses import dataclass
from typing import List

import torch

@dataclass
class GeneratedExperience:
    """Holds the outputs of a single generation step."""

    smiles: List[str]
    token_ids: torch.Tensor
    logprobs: torch.Tensor
    logprobs_ref: torch.Tensor

