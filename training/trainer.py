import os
from collections import OrderedDict
from typing import Any, Tuple

import torch
from tqdm import tqdm

from config import ExperimentConfig
from pretraining.model import GPT
from scoring.evaluator import MetricsCalculator, TensorBoardLogger
from scoring.memory import ReplayBuffer
from scoring.reward import (
    apply_diversity_filter,
    compute_reward,
    parallel_process_batch,
)
from training.datatypes import GeneratedExperience
from utils.utils import sample_smiles_nograd


class RLTrainer:
    """Handles the execution of the Reinforcement Learning loop."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        model: GPT,
        ref_model: GPT,
        optimizer: torch.optim.Optimizer,
        voc: Any,
        experience_buffer: ReplayBuffer,
        metrics_calculator: MetricsCalculator,
        logger: TensorBoardLogger,
        additive_scorers: list,
        multiplier_scorers: list,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.voc = voc
        self.experience_buffer = experience_buffer
        self.metrics_calculator = metrics_calculator
        self.logger = logger
        self.additive_scorers = additive_scorers
        self.multiplier_scorers = multiplier_scorers
        self.device = device
        self.global_scaffold_memory = OrderedDict()

    def train(self):
        """The core RL loop."""
        for p in self.ref_model.parameters():
            p.requires_grad = False

        for step in tqdm(range(self.cfg.training.max_steps)):
            self.model.train()

            experience = self._generate_experience()

            processed_data = parallel_process_batch(
                experience.smiles, max_workers=self.cfg.training.max_workers
            )

            reward = compute_reward(
                processed_data,
                additive_scorers=self.additive_scorers,
                multiplier_scorers=self.multiplier_scorers,
            )

            filtered_reward = apply_diversity_filter(
                processed_data,
                reward,
                self.global_scaffold_memory,
                bucket_size=25,
            )

            score = torch.tensor(
                filtered_reward, dtype=torch.float32, device=self.device
            )

            self.experience_buffer.add_experience(
                processed_data, filtered_reward, experience.logprobs_ref
            )

            total_logprobs, total_priors, total_scores = (
                self._combine_current_and_replay(experience, score)
            )
            loss = self._compute_DAP_loss(total_logprobs, total_priors, total_scores)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if (
                self.cfg.evaluation.eval_every > 0
                and step % self.cfg.evaluation.eval_every == 0
            ):
                self.model.eval()
                metrics, img_np = self.metrics_calculator.calculate(processed_data)
                self.logger.log_metrics(step, metrics, img_np)
            if (
                self.cfg.training.save_every > 0
                and step % self.cfg.training.save_every == 0
            ):
                self._save_checkpoint(step)

    def _generate_experience(self) -> GeneratedExperience:
        """
        Generate a batch of SMILES sequences and compute their log probabilities
        under both the current model and the reference model.
        """
        smiles_list, token_ids = sample_smiles_nograd(
            model=self.model,
            voc=self.voc,
            model_cfg=self.cfg.model,
            train_cfg=self.cfg.training,
        )

        logprobs, _, _ = logprobs_from_token_ids(
            self.model, token_ids, self.voc, train_mode=True
        )

        with torch.no_grad():
            logprobs_ref, _, _ = logprobs_from_token_ids(
                self.ref_model, token_ids, self.voc, train_mode=False
            )

        return GeneratedExperience(
            smiles=smiles_list,
            token_ids=token_ids,
            logprobs=logprobs,
            logprobs_ref=logprobs_ref,
        )

    def _combine_current_and_replay(
        self, experience: GeneratedExperience, score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine current experience logprobs and scores with replay buffer samples."""
        replay_token_ids, replay_scores, replay_priors = self.experience_buffer.sample(
            batch_size=max(1, self.cfg.training.batch_size // 10)
        )

        if replay_token_ids is not None:
            replay_logprobs, _, _ = logprobs_from_token_ids(
                self.model, replay_token_ids, self.voc, train_mode=True
            )
            total_logprobs = torch.cat([experience.logprobs, replay_logprobs])
            total_priors = torch.cat([experience.logprobs_ref, replay_priors])
            total_scores = torch.cat([score, replay_scores])
        else:
            total_logprobs = experience.logprobs
            total_priors = experience.logprobs_ref
            total_scores = score

        return total_logprobs, total_priors, total_scores

    def _compute_DAP_loss(
        self,
        logprobs: torch.Tensor,
        priors: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the Difference between Augmented and Posterior (DAP) loss."""
        augmented_likelihood = priors + (self.cfg.reward.sigma * scores)
        loss = 0.5 * ((logprobs - augmented_likelihood) ** 2).mean()
        return loss

    def _save_checkpoint(self, step: int) -> None:
        """Save the model checkpoint at the specified training step."""
        save_dir = os.path.join(self.cfg.paths.ckpt_save_path, self.cfg.run.run_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")

        torch.save(self.model.state_dict(), save_path)


def _sequence_mask(target: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Create a binary mask identifying valid tokens before the EOS token.

    Parameters
    ----------
    target : torch.Tensor
        The target sequence tensor.
    eos_token_id : int
        The integer ID of the End-Of-Sequence token.

    Returns
    -------
    torch.Tensor
        A binary float tensor where 1.0 indicates a valid token and 0.0
        indicates padding/EOS.
    """
    eos = target == eos_token_id
    return (eos.cumsum(dim=1) == 0).float()


def _token_logprobs(
    logits: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the log probabilities of the target tokens from logits.

    Parameters
    ----------
    logits : torch.Tensor
        The raw logits output from the model.
    target : torch.Tensor
        The target token indices.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The full log-softmax tensor and the specific log probabilities
        gathered for the target tokens.
    """
    log_probs = logits.log_softmax(dim=-1)
    gathered_logprobs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return log_probs, gathered_logprobs


def _sequence_entropy(logits: torch.Tensor, not_finished: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean Shannon entropy of the generated sequence.

    Parameters
    ----------
    logits : torch.Tensor
        The raw logits output from the model.
    not_finished : torch.Tensor
        The binary mask identifying valid tokens.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the mean entropy of valid tokens.
    """
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)
    token_entropy = token_entropy * not_finished
    seq_lengths = not_finished.sum(dim=1) + 1e-8
    return (token_entropy.sum(dim=1) / seq_lengths).mean()


def logprobs_from_token_ids(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    voc: Any,
    train_mode: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute log probabilities and entropy for a batch of token token_ids.

    Parameters
    ----------
    model : torch.nn.Module
        The GPT model.
    token_ids : torch.Tensor
        The token sequences to evaluate.
    voc : Any
        The vocabulary object.
    train_mode : bool, default=True
        Whether to set the model to train mode.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Sequence log probabilities, sequence entropy, and value estimates.
    """
    model.train() if train_mode else model.eval()
    logits, values, _ = model(token_ids[:, :-1])

    target = token_ids[:, 1:]
    not_finished = _sequence_mask(target, voc["$"])

    _, token_logprobs = _token_logprobs(logits, target)
    token_logprobs = token_logprobs * not_finished
    logprobs = token_logprobs.sum(dim=1)

    entropy = _sequence_entropy(logits, not_finished)

    return logprobs, entropy, values

