import argparse
from typing import Any, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from config import ExperimentConfig, ModelConfig, PathsConfig, TrainingConfig
from pretraining.model import GPT, GPTConfig
from pretraining.vocabulary import read_vocabulary
from scoring.activity import load_rf_model
from scoring.evaluator import MetricsCalculator, TensorBoardLogger
from scoring.memory import ReplayBuffer
from scoring.molecular import load_ad_model
from scoring.reward import (
    make_ad_scorer,
    make_mw_penalty,
    make_pains_filter,
    make_qed_scorer,
    make_rf_scorer,
    make_sa_scorer,
    make_valid_mask,
)
from utils.utils import set_seed, load_smiles_set
from training.trainer import RLTrainer


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
    set_seed(42)

    voc = read_vocabulary(cfg.paths.vocab_path)
    rf_model = load_rf_model(path=cfg.paths.rf_model_path)
    ad_model = load_ad_model(path=cfg.paths.ad_nn_path, device=device)
    train_smiles_set = load_smiles_set(cfg.paths.train_smiles_path)

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
        torch.load(cfg.paths.ckpt_load_path, map_location=device), strict=False
    )

    writer = SummaryWriter("runs/logging/" + cfg.run.run_name)
    logger = TensorBoardLogger(writer)

    active_additive_scorers = [
        make_rf_scorer(rf_model, weight=cfg.reward.w_rf),
        #make_ad_scorer(ad_model, weight=1.0),
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

    trainer = RLTrainer(
        cfg=cfg,
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        voc=voc,
        experience_buffer=experience_buffer,
        metrics_calculator=metrics_calculator,
        logger=logger,
        additive_scorers=active_additive_scorers,
        multiplier_scorers=active_multiplier_scorers,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
