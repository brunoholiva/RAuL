# RAuL

Reinforcement Antibiotic Learner

Reinforcement learning and pretraining for SMILES-based antimicrobial molecule generation.

## Layout

- reinforce.py: main RL entrypoint (reads config.toml)
- pretraining/: pretraining utilities and model
- scoring/: reward and property scoring
- utils/: helpers
- data/: data for pretraining, vocab, and predictor model necesseties

## Usage

- Pretraining script: pretraining/pretrain.py
- RL training script: reinforce.py (uses config.toml)

## Important

- Model architecture and code was based on [ACARL](https://github.com/HXYfighter/ACARL)'s

- Pretraining was employed using chEMBL dataset v36.

- Molecules were standardized in the exact same way as the predictor model training molecules were. Details are in the predictor model [repo](https://github.com/brunoholiva/antimicrobial_collection)

- Reward and loss functions were based on [REINVENT4](https://github.com/MolecularAI/REINVENT4)
