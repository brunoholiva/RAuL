import argparse
import os
from collections import OrderedDict
from typing import Any, Optional, Set, Tuple

import numpy as np
import toml
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pretraining.model import GPT, GPTConfig
from pretraining.vocabulary import read_vocabulary
from scoring.activity import load_rf_model, predict_activity_proba
from scoring.molecular import ad_domain_score, load_ad_model
from scoring.reward import (
    apply_diversity_filter,
    compute_reward,
    parallel_process_batch,
    sample_replay_buffer,
    update_replay_buffer,
)
from utils.rdkit_utils import (
    model_diversity,
    model_novelty,
    model_uniqueness,
    model_validity,
)
from utils.utils import sample_smiles_nograd, set_seed


def create_model(
    voc: Any,
    n_layer: int,
    n_head: int,
    n_embd: int,
    max_length: int,
    learning_rate: float,
    device: str,
    ckpt_load_path: Optional[str] = None,
) -> Tuple[GPT, torch.optim.Optimizer]:
    """
    Creates the GPT model and optimizer.

    Parameters
    ----------
    voc : Any
        The vocabulary object containing token mappings.
    n_layer : int
        The number of transformer layers.
    n_head : int
        The number of attention heads.
    n_embd : int
        The embedding dimension.
    max_length : int
        The maximum sequence length (block size).
    learning_rate : float
        The learning rate for the optimizer.
    device : str
        The device to run the model on (e.g., "cuda" or "cpu").
    ckpt_load_path : Optional[str], optional
        The path to a checkpoint to load model weights from (default is None).

    Returns
    -------
    Tuple[GPT, torch.optim.Optimizer]
        The created GPT model and its optimizer.
    """
    model_config = GPTConfig(
        voc.__len__(),
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=max_length,
    )
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95)
    )
    if ckpt_load_path is not None:
        print(f"Loading checkpoint from {ckpt_load_path}")
        model.load_state_dict(
            torch.load(ckpt_load_path, map_location=device), strict=False
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

    with open(args.config, "r") as f:
        config = toml.load(f)

    model_cfg = config["model"]
    training_cfg = config["training"]
    reward_cfg = config["reward"]
    eval_cfg = config["evaluation"]
    paths_cfg = config["paths"]
    run_cfg = config["run"]

    global_scaffold_memory = OrderedDict()
    rf_model = load_rf_model(paths_cfg["rf_model_path"])
    ad_model = load_ad_model(path=paths_cfg["ad_nn_path"], device=device)
    train_smiles_set = load_smiles_set(paths_cfg.get("train_smiles_path"))

    writer = SummaryWriter("runs/logging/" + run_cfg["run_name"])
    ckpt_save_dir = paths_cfg["ckpt_save_path"] + run_cfg["run_name"]
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    # Log hyperparameters to track
    hparams_text = "\n".join(
        [
            f"sigma={reward_cfg['sigma']}",
            f"w_rf={reward_cfg['w_rf']}",
            f"w_qed={reward_cfg['w_qed']}",
            f"w_sa={reward_cfg['w_sa']}",
            f"learning_rate={training_cfg['learning_rate']}",
            f"batch_size={training_cfg['batch_size']}",
            f"max_length={model_cfg['max_length']}",
            f"temperature={training_cfg['temperature']}",
            f"top_k={training_cfg['top_k']}",
            f"n_layer={model_cfg['n_layer']}",
            f"n_head={model_cfg['n_head']}",
            f"n_embd={model_cfg['n_embd']}",
        ]
    )
    writer.add_text("hparams", hparams_text, 0)

    set_seed(42)

    voc = read_vocabulary(paths_cfg["vocab_path"])
    model, optimizer = create_model(
        voc=voc,
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        max_length=model_cfg["max_length"],
        learning_rate=training_cfg["learning_rate"],
        device=device,
        ckpt_load_path=paths_cfg["ckpt_load_path"],
    )

    ref_model = GPT(
        GPTConfig(
            voc.__len__(),
            n_layer=model_cfg["n_layer"],
            n_head=model_cfg["n_head"],
            n_embd=model_cfg["n_embd"],
            block_size=model_cfg["max_length"],
        )
    ).to(device)

    ref_model.load_state_dict(
        torch.load(paths_cfg["ckpt_load_path"], map_location=device),
        strict=False,
    )
    ref_model.eval()
    replay_buffer = []

    from scoring.reward import (
        make_ad_scorer,
        make_mw_penalty,
        make_pains_filter,
        make_qed_scorer,
        make_rf_scorer,
        make_sa_scorer,
        make_valid_mask,
    )

    active_additive_scorers = [
        make_rf_scorer(rf_model, weight=reward_cfg["w_rf"]),
        make_ad_scorer(ad_model, weight=1.0),
        make_qed_scorer(weight=reward_cfg["w_qed"]),
        make_sa_scorer(weight=reward_cfg["w_sa"]),
    ]

    active_multiplier_scorers = [
        make_mw_penalty(),
        make_valid_mask(),
        make_pains_filter(),
    ]

    for p in ref_model.parameters():
        p.requires_grad = False

    for step in tqdm(range(training_cfg["max_steps"])):
        model.train()
        smiles_list, codes = sample_smiles_nograd(
            model,
            voc=voc,
            n_mols=training_cfg["batch_size"],
            block_size=model_cfg["max_length"],
            temperature=training_cfg["temperature"],
            top_k=training_cfg["top_k"],
        )

        logprobs, _, _ = logprobs_from_codes(
            model, codes, voc, train_mode=True
        )

        with torch.no_grad():
            logprobs_ref, _, _ = logprobs_from_codes(
                ref_model, codes, voc, train_mode=False
            )

        processed_data = parallel_process_batch(
            smiles_list, max_workers=training_cfg["max_workers"]
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

        replay_buffer = update_replay_buffer(
            replay_buffer,
            processed_data,
            filtered_reward,
            logprobs_ref,
            buffer_size=100,
        )

        replay_codes, replay_scores, replay_priors = sample_replay_buffer(
            replay_buffer,
            batch_size=max(1, training_cfg["batch_size"] // 10),
            voc=voc,
            device=device,
            max_length=model_cfg["max_length"],
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
        augmented_likelihood = total_priors + (
            reward_cfg["sigma"] * total_scores
        )
        loss = 0.5 * ((total_logprobs - augmented_likelihood) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        writer.add_scalar("loss/dap", float(loss.item()), step)
        writer.add_scalar(
            "reward/mean", float(total_scores.mean().item()), step
        )
        writer.add_scalar("reward/max", float(total_scores.max().item()), step)
        writer.add_scalar("reward/min", float(total_scores.min().item()), step)

        if (step + 1) % 500 == 0:
            ckpt_path = os.path.join(ckpt_save_dir, f"step_{step+1}.pt")
            torch.save(model.state_dict(), ckpt_path)

        if eval_cfg["eval_every"] > 0 and step % eval_cfg["eval_every"] == 0:
            mask = np.array([d["valid"] for d in processed_data], dtype=bool)

            qed_scores = np.array(
                [d.get("qed", 0.0) for d in processed_data], dtype=np.float32
            )
            sa_raw = np.array(
                [d.get("sa", 10.0) for d in processed_data], dtype=np.float32
            )

            fps = [d["fp"] for d in processed_data]
            std_smiles = [d.get("std_smi", "") for d in processed_data]

            rf_probs = predict_activity_proba(fps, rf_model=rf_model)
            ad_dists = ad_domain_score(fps, ad_model=ad_model)

            rf_valid = rf_probs[mask]
            ad_valid = ad_dists[mask]
            qed_valid = qed_scores[mask]
            sa_valid = sa_raw[mask]

            if rf_valid.size > 0:
                writer.add_histogram("rf/prob_dist", rf_valid, step)
            if ad_valid.size > 0:
                writer.add_histogram("ad/dist_dist", ad_valid, step)
            if qed_valid.size > 0:
                writer.add_histogram("qed/score_dist", qed_valid, step)
            if sa_valid.size > 0:
                writer.add_histogram("sa/score_dist", sa_valid, step)

            sampled_smiles, _ = sample_smiles_nograd(
                model, voc=voc, n_mols=100, block_size=200, top_k=10
            )
            validity = model_validity(sampled_smiles)
            uniqueness = model_uniqueness(sampled_smiles)
            diversity = model_diversity(sampled_smiles)
            novelty = model_novelty(sampled_smiles, train_smiles_set)

            writer.add_scalar("metrics/validity", float(validity), step)
            writer.add_scalar("metrics/uniqueness", float(uniqueness), step)
            writer.add_scalar("metrics/diversity", float(diversity), step)
            writer.add_scalar("metrics/novelty", float(novelty), step)

            mols = [Chem.MolFromSmiles(s) for s in sampled_smiles]
            valid_mols = [m for m in mols if m is not None]
            if len(valid_mols) > 0:
                k = min(8, len(valid_mols))
                idx = np.random.choice(len(valid_mols), size=k, replace=False)
                mols_subset = [valid_mols[i] for i in idx]
                img = Draw.MolsToGridImage(
                    mols_subset, molsPerRow=4, subImgSize=(200, 200)
                )
                img_np = np.array(img)
                writer.add_image(
                    "samples/molecules", img_np, step, dataformats="HWC"
                )


if __name__ == "__main__":
    main()
