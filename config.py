from dataclasses import dataclass
from typing import Optional

import toml


@dataclass
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    max_length: int


@dataclass
class TrainingConfig:
    batch_size: int
    max_steps: int
    learning_rate: float
    temperature: float
    top_k: int
    max_workers: int


@dataclass
class RewardConfig:
    w_rf: float
    w_qed: float
    w_sa: float
    sigma: float


@dataclass
class EvaluationConfig:
    eval_every: int


@dataclass
class PathsConfig:
    vocab_path: str
    ckpt_save_path: str
    rf_model_path: str
    ad_nn_path: str
    ckpt_load_path: Optional[str] = None
    train_smiles_path: Optional[str] = None


@dataclass
class RunConfig:
    run_name: str


@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    reward: RewardConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    run: RunConfig

    @classmethod
    def from_toml(cls, file_path: str) -> "ExperimentConfig":
        """
        Method to read a TOML file and convert it into heavily typed dataclasses.
        """
        with open(file_path, "r") as f:
            raw_dict = toml.load(f)

        return cls(
            model=ModelConfig(**raw_dict.get("model", {})),
            training=TrainingConfig(**raw_dict.get("training", {})),
            reward=RewardConfig(**raw_dict.get("reward", {})),
            evaluation=EvaluationConfig(**raw_dict.get("evaluation", {})),
            paths=PathsConfig(**raw_dict.get("paths", {})),
            run=RunConfig(**raw_dict.get("run", {})),
        )
