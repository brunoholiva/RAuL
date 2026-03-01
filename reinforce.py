import argparse
import os
from collections import OrderedDict
from typing import Any, Optional, Set, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ExperimentConfig, ModelConfig, PathsConfig, TrainingConfig
from pretraining.model import GPT, GPTConfig
from pretraining.vocabulary import read_vocabulary
from scoring.activity import load_rf_model
from scoring.evaluator import MetricsCalculator, TensorBoardLogger
from scoring.memory import ReplayBuffer
from scoring.molecular import load_ad_model
from scoring.reward import (
    apply_diversity_filter,
    compute_reward,
    make_ad_scorer,
    make_mw_penalty,
    make_pains_filter,
    make_qed_scorer,
    make_rf_scorer,
    make_sa_scorer,
    make_valid_mask,
    parallel_process_batch,
)
from utils.utils import sample_smiles_nograd, set_seed


def create_model(
    voc: Any,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    paths_cfg: PathsConfig,
    device: str,
) -> Tuple[GPT, torch.optim.Optimizer]:
    """
    Creates the GPT model and optimizer.

    Parameters
    ----------
    voc : Any
        The vocabulary object containing token mappings.
    model_cfg : ModelConfig
        The model configuration object.
    training_cfg : TrainingConfig
        The training configuration object.
    paths_cfg : PathsConfig
        The paths configuration object.
    device : str
        The device to run the model on (e.g., "cuda" or "cpu").


    Returns
    -------
    Tuple[GPT, torch.optim.Optimizer]
        The created GPT model and its optimizer.
    """
    model_config = GPTConfig(
        voc.__len__(),
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        block_size=model_cfg.max_length,
    )
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=training_cfg.learning_rate,
        betas=(0.9, 0.95),
    )
    if paths_cfg.ckpt_load_path is not None:
        print(f"Loading checkpoint from {paths_cfg.ckpt_load_path}")
        model.load_state_dict(
            torch.load(paths_cfg.ckpt_load_path, map_location=device),
            strict=False,
        )
    return model, optimizer


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


def _sequence_entropy(
    logits: torch.Tensor, not_finished: torch.Tensor
) -> torch.Tensor:
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


def logprobs_from_codes(
    model: torch.nn.Module,
    codes: torch.Tensor,
    voc: Any,
    train_mode: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute log probabilities and entropy for a batch of token codes.

    Parameters
    ----------
    model : torch.nn.Module
        The GPT model.
    codes : torch.Tensor
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
    logits, values, _ = model(codes[:, :-1])

    target = codes[:, 1:]
    not_finished = _sequence_mask(target, voc["$"])

    _, token_logprobs = _token_logprobs(logits, target)
    token_logprobs = token_logprobs * not_finished
    logprobs = token_logprobs.sum(dim=1)

    entropy = _sequence_entropy(logits, not_finished)

    return logprobs, entropy, values


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


def main() -> None:
    """
    Main training loop for the RAuL reinforcement learning pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to TOML config file"
    )

    args = parser.parse_args()

    cfg = ExperimentConfig.from_toml(args.config)

    global_scaffold_memory = OrderedDict()
    rf_model = load_rf_model(path=cfg.paths.rf_model_path)
    ad_model = load_ad_model(path=cfg.paths.ad_nn_path, device=device)
    train_smiles_set = load_smiles_set(path=cfg.paths.train_smiles_path)

    writer = SummaryWriter("runs/logging/" + cfg.run.run_name)
    ckpt_save_dir = cfg.paths.ckpt_save_path + cfg.run.run_name
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    hparams_text = "\n".join(
        [
            f"sigma={cfg.reward.sigma}",
            f"w_rf={cfg.reward.w_rf}",
            f"w_qed={cfg.reward.w_qed}",
            f"w_sa={cfg.reward.w_sa}",
            f"learning_rate={cfg.training.learning_rate}",
            f"batch_size={cfg.training.batch_size}",
            f"max_length={cfg.model.max_length}",
            f"temperature={cfg.training.temperature}",
            f"top_k={cfg.training.top_k}",
            f"n_layer={cfg.model.n_layer}",
            f"n_head={cfg.model.n_head}",
            f"n_embd={cfg.model.n_embd}",
        ]
    )
    writer.add_text("hparams", hparams_text, 0)

    set_seed(42)

    voc = read_vocabulary(cfg.paths.vocab_path)
    model, optimizer = create_model(
        voc=voc,
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        paths_cfg=cfg.paths,
        device=device,
    )

    ref_model = GPT(
        GPTConfig(
            voc.__len__(),
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            block_size=cfg.model.max_length,
        )
    ).to(device)

    ref_model.load_state_dict(
        torch.load(cfg.paths.ckpt_load_path, map_location=device),
        strict=False,
    )
    ref_model.eval()

    active_additive_scorers = [
        make_rf_scorer(rf_model, weight=cfg.reward.w_rf),
        make_ad_scorer(ad_model, weight=1.0),
        make_qed_scorer(weight=cfg.reward.w_qed),
        make_sa_scorer(weight=cfg.reward.w_sa),
    ]

    active_multiplier_scorers = [
        make_mw_penalty(),
        make_valid_mask(),
        make_pains_filter(),
    ]

    experience_buffer = ReplayBuffer(
        voc=voc,
        device=device,
        max_length=cfg.model.max_length,
        buffer_size=100,
        max_per_scaffold=3,
        similarity_threshold=0.65,
    )

    metrics_calculator = MetricsCalculator(
        model=model,
        voc=voc,
        rf_model=rf_model,
        ad_model=ad_model,
        train_smiles_set=train_smiles_set,
        model_cfg=cfg.model,
        train_cfg=cfg.training,
    )

    logger = TensorBoardLogger(writer)

    for p in ref_model.parameters():
        p.requires_grad = False

    for step in tqdm(range(cfg.training.max_steps)):
        model.train()
        smiles_list, codes = sample_smiles_nograd(
            model,
            voc=voc,
            model_cfg=cfg.model,
            train_cfg=cfg.training,
        )

        logprobs, _, _ = logprobs_from_codes(
            model, codes, voc, train_mode=True
        )

        with torch.no_grad():
            logprobs_ref, _, _ = logprobs_from_codes(
                ref_model, codes, voc, train_mode=False
            )

        processed_data = parallel_process_batch(
            smiles_list, max_workers=cfg.training.max_workers
        )

        reward = compute_reward(
            processed_data,
            additive_scorers=active_additive_scorers,
            multiplier_scorers=active_multiplier_scorers,
        )

        filtered_reward = apply_diversity_filter(
            processed_data, reward, global_scaffold_memory, bucket_size=25
        )

        score = torch.tensor(
            filtered_reward, dtype=torch.float32, device=device
        )

        experience_buffer.add_experience(
            processed_data, filtered_reward, logprobs_ref
        )

        replay_codes, replay_scores, replay_priors = experience_buffer.sample(
            batch_size=max(1, cfg.training.batch_size // 10)
        )

        if replay_codes is not None:
            replay_logprobs, _, _ = logprobs_from_codes(
                model, replay_codes, voc, train_mode=True
            )

            total_logprobs = torch.cat([logprobs, replay_logprobs])
            total_priors = torch.cat([logprobs_ref, replay_priors])
            total_scores = torch.cat([score, replay_scores])
        else:
            total_logprobs = logprobs
            total_priors = logprobs_ref
            total_scores = score

        # REINVENT DAP loss
        augmented_likelihood = total_priors + (cfg.reward.sigma * total_scores)
        loss = 0.5 * ((total_logprobs - augmented_likelihood) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (
            cfg.evaluation.eval_every > 0
            and step % cfg.evaluation.eval_every == 0
        ):
            model.eval()
            metrics, img_np = metrics_calculator.calculate(processed_data)
            logger.log_metrics(step, metrics, img_np)


if __name__ == "__main__":
    main()
    main()
