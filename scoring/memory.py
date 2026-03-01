import random

import torch
from rdkit import DataStructs

from pretraining.vocabulary import SMILESTokenizer


class ReplayBuffer:
    def __init__(
        self,
        voc,
        device,
        max_length=200,
        buffer_size=100,
        max_per_scaffold=3,
        similarity_threshold=0.65,
    ):
        self.voc = voc
        self.device = device
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.max_per_scaffold = max_per_scaffold
        self.similarity_threshold = similarity_threshold
        self._items = []

    def __len__(self):
        return len(self._items)

    def add_experience(
        self,
        processed_data: list,
        scores: torch.Tensor,
        prior_logprobs: torch.Tensor,
    ):
        """Adds new valid molecules and filters them to maintain diversity."""
        new_items = self._parse_new_data(
            processed_data, scores, prior_logprobs
        )

        combined = self._items + new_items
        combined.sort(key=lambda x: x[0], reverse=True)

        self._items = self._enforce_diversity(combined)

    def sample(self, batch_size: int):
        """Samples random molecules and converts them to PyTorch tensors."""
        if not self._items:
            return None, None, None

        n_samples = min(len(self._items), batch_size)
        sampled_items = random.sample(self._items, n_samples)

        return self._convert_to_tensors(sampled_items, n_samples)

    def _parse_new_data(self, processed_data, scores, prior_logprobs):
        """Extracts valid molecules into a simple tuple format."""
        new_items = []
        for data, sc, plp in zip(processed_data, scores, prior_logprobs):
            score_val = (
                float(sc.item()) if isinstance(sc, torch.Tensor) else float(sc)
            )
            if score_val > 0.0 and data["valid"]:
                new_items.append(
                    (
                        score_val,
                        data["smi"],
                        float(plp.item()),
                        data.get("scaffold_fp"),
                    )
                )
        return new_items

    def _enforce_diversity(self, combined_items):
        """Loops through items and ensures we don't have too many of the same scaffold."""
        diverse_buffer = []
        diverse_fps = []

        for item in combined_items:
            score, smi, prior, scaffold_fp = item

            if scaffold_fp is not None:
                if diverse_fps:
                    sims = DataStructs.BulkTanimotoSimilarity(
                        scaffold_fp, diverse_fps
                    )
                    similar_count = sum(
                        1 for sim in sims if sim >= self.similarity_threshold
                    )

                    if similar_count < self.max_per_scaffold:
                        diverse_buffer.append(item)
                        diverse_fps.append(scaffold_fp)
                else:
                    diverse_buffer.append(item)
                    diverse_fps.append(scaffold_fp)
            else:
                diverse_buffer.append(item)

            if len(diverse_buffer) >= self.buffer_size:
                break

        return diverse_buffer

    def _convert_to_tensors(self, sampled_items, n_samples):
        scores, smiles, prior_logprobs, _ = zip(*sampled_items)
        tokenizer = SMILESTokenizer()

        codes = torch.full(
            (n_samples, self.max_length),
            self.voc["$"],
            dtype=torch.long,
            device=self.device,
        )

        for i, smi in enumerate(smiles):
            tokens = tokenizer.tokenize(smi)
            token_indices = [self.voc[t] for t in tokens if t in self.voc]
            full_seq = [self.voc["^"]] + token_indices + [self.voc["$"]]

            if len(full_seq) > self.max_length:
                full_seq = full_seq[: self.max_length]

            codes[i, : len(full_seq)] = torch.tensor(
                full_seq, dtype=torch.long, device=self.device
            )

        scores_tensor = torch.tensor(
            scores, dtype=torch.float32, device=self.device
        )
        priors_tensor = torch.tensor(
            prior_logprobs, dtype=torch.float32, device=self.device
        )

        return codes, scores_tensor, priors_tensor
