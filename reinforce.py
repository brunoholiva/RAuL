import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import toml
from pretraining.vocabulary import SMILESTokenizer, read_vocabulary
from pretraining.model import GPT, GPTConfig
from utils.utils import (
    set_seed,
    top_k_logits,
    model_validity,
    model_uniqueness,
    model_novelty,
    model_diversity,
    sample_SMILES
)
from utils.rdkit_utils import preprocess_smiles_list
from scoring.molecular import load_ad_nn
from scoring.activity import init_rf_model
from scoring.reward import (
    compute_reward,
    compute_property_scores,
    apply_diversity_filter,
    update_replay_buffer,
    sample_replay_buffer
)
from rdkit import Chem
from rdkit.Chem import Draw


def create_model(
    voc, n_layer, n_head, n_embd, max_length, learning_rate, device, ckpt_load_path=None
):
    """
    Create the GPT model and optimizer.
    
    :param voc: Vocabulary object
    :param n_layer: Number of layers in the model
    :param n_head: Number of attention heads
    :param n_embd: Embedding dimension
    :param max_length: Maximum sequence length
    :param learning_rate: Learning rate for the optimizer
    :param device: Device to run the model on
    :param ckpt_load_path: Path to load checkpoint from
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
        model.load_state_dict(torch.load(ckpt_load_path, map_location=device), strict=False)
    return model, optimizer


def sample_smiles_nograd(model, voc, n_mols, block_size, temperature=1.0, top_k=10):
    """
    Sample SMILES strings from the model without computing gradients.
    
    :param model: The GPT model to sample from
    :param voc: Vocabulary object
    :param n_mols: Number of molecules (SMILES) to sample
    :param block_size: Maximum sequence length for sampling
    :param temperature: Sampling temperature (higher means more random)
    :param top_k: Number of top tokens to consider for sampling
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        codes = torch.full(
            (n_mols, block_size), fill_value=voc["$"], dtype=torch.long, device=device
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


def _sequence_mask(target, eos_token_id):
    eos = target == eos_token_id
    return (eos.cumsum(dim=1) == 0).float()


def _token_logprobs(logits, target):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs, log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)


def _sequence_entropy(logits, not_finished):
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)
    token_entropy = token_entropy * not_finished
    seq_lengths = not_finished.sum(dim=1) + 1e-8
    return (token_entropy.sum(dim=1) / seq_lengths).mean()


def logprobs_from_codes(model, codes, voc, train_mode=True):
    model.train() if train_mode else model.eval()
    logits, values, _ = model(codes[:, :-1])

    target = codes[:, 1:]
    not_finished = _sequence_mask(target, voc["$"])

    log_probs, token_logprobs = _token_logprobs(logits, target)
    token_logprobs = token_logprobs * not_finished
    logprobs = token_logprobs.sum(dim=1)

    entropy = _sequence_entropy(logits, not_finished)

    return logprobs, entropy, values


def load_smiles_set(path):
    if path is None:
        return None
    smiles_set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles_set.add(s)
    return smiles_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    
    args = parser.parse_args()
    
    # Load config from TOML file
    with open(args.config, "r") as f:
        config = toml.load(f)
    
    # Extract sections
    model_cfg = config["model"]
    training_cfg = config["training"]
    reward_cfg = config["reward"]
    eval_cfg = config["evaluation"]
    paths_cfg = config["paths"]
    run_cfg = config["run"]

    global_scaffold_memory = {}
    init_rf_model(paths_cfg["rf_model_path"])
    load_ad_nn(paths_cfg["ad_nn_path"])
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
            f"n_embd={model_cfg['n_embd']}"
        ]
    )
    writer.add_text("hparams", hparams_text, 0)

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    ref_model.load_state_dict(torch.load(paths_cfg["ckpt_load_path"], map_location=device), strict=False)
    ref_model.eval()
    replay_buffer = []

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

        # Agent log-probs
        logprobs, _, _ = logprobs_from_codes(model, codes, voc, train_mode=True)

        # Prior log-probs
        with torch.no_grad():
            logprobs_ref, _, _ = logprobs_from_codes(ref_model, codes, voc, train_mode=False)

        # 1. Compute Reward for the ONLINE batch
        reward = compute_reward(
            smiles_list,
            w_rf=reward_cfg["w_rf"],
            w_qed=reward_cfg["w_qed"],
            w_sa=reward_cfg["w_sa"],
        )
        
        # 2. Apply Filter
        filtered_reward = apply_diversity_filter(
            smiles_list, 
            reward, 
            global_scaffold_memory, 
            bucket_size=25 
        )
        
        # Convert to tensor
        score = torch.tensor(filtered_reward, dtype=torch.float32, device=device)
        
        replay_buffer = update_replay_buffer(
            replay_buffer,
            smiles_list,  
            filtered_reward, 
            logprobs_ref, 
            buffer_size=100  
        )

        replay_codes, replay_scores, replay_priors = sample_replay_buffer(
            replay_buffer, 
            batch_size=max(1, training_cfg["batch_size"] // 10),
            voc=voc, 
            device=device, 
            max_length=model_cfg["max_length"]
        )

        if replay_codes is not None:
            replay_logprobs, _, _ = logprobs_from_codes(model, replay_codes, voc, train_mode=True)
            
            total_logprobs = torch.cat([logprobs, replay_logprobs])
            total_priors = torch.cat([logprobs_ref, replay_priors])
            total_scores = torch.cat([score, replay_scores])
        else:
            total_logprobs = logprobs
            total_priors = logprobs_ref
            total_scores = score

        # REINVENT DAP loss
        augmented_likelihood = total_priors + (reward_cfg["sigma"] * total_scores)
        loss = 0.5 * ((total_logprobs - augmented_likelihood) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # logging
        writer.add_scalar("loss/dap", float(loss.item()), step)
        writer.add_scalar("reward/mean", float(total_scores.mean().item()), step)
        writer.add_scalar("reward/max", float(total_scores.max().item()), step)
        writer.add_scalar("reward/min", float(total_scores.min().item()), step)
        
        # log AD and RF distributions from current batch
        if eval_cfg["eval_every"] > 0 and step % eval_cfg["eval_every"] == 0:
            std_smiles, mols, valid_mask = preprocess_smiles_list(smiles_list)
            rf_probs, qed_scores, sa_raw, ad_dists = compute_property_scores(std_smiles, mols)

            mask = np.array(valid_mask, dtype=bool)
            rf_valid = np.array(rf_probs, dtype=np.float32)[mask]
            ad_valid = np.array(ad_dists, dtype=np.float32)[mask]
            qed_valid = np.array(qed_scores, dtype=np.float32)[mask]
            sa_valid = np.array(sa_raw, dtype=np.float32)[mask]

            if rf_valid.size > 0:
                writer.add_histogram("rf/prob_dist", rf_valid, step)
            if ad_valid.size > 0:
                writer.add_histogram("ad/dist_dist", ad_valid, step)
            if qed_valid.size > 0:
                writer.add_histogram("qed/score_dist", qed_valid, step)
            if sa_valid.size > 0:
                writer.add_histogram("sa/score_dist", sa_valid, step)

        if step % 500 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_save_dir, f"step{step}.pt"),
            )

        if eval_cfg["eval_every"] > 0 and step % eval_cfg["eval_every"] == 0:
            sampled_smiles, _, _ = sample_SMILES(model, voc=voc, n_mols=100, block_size=200, top_k=10)
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
                img = Draw.MolsToGridImage(mols_subset, molsPerRow=4, subImgSize=(200, 200))
                img_np = np.array(img)  # HWC RGB
                writer.add_image("samples/molecules", img_np, step, dataformats="HWC")


if __name__ == "__main__":
    main()
