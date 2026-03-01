from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from scoring.activity import predict_activity_proba
from scoring.molecular import ad_domain_score
from utils.rdkit_utils import (
    get_diversity,
    get_novelty,
    get_uniqueness,
    get_validity,
)
from utils.utils import sample_smiles_nograd


class MetricsCalculator:
    """Calculates evaluation metrics for model performance."""

    def __init__(self, model, voc, rf_model, ad_model, train_smiles_set):
        self.model = model
        self.voc = voc
        self.rf_model = rf_model
        self.ad_model = ad_model
        self.train_smiles_set = train_smiles_set

    def calculate(
        self, processed_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """Calculate all evaluation metrics."""
        metrics = {}
        mask = np.array([d["valid"] for d in processed_data], dtype=bool)
        fps = [d["fp"] for d in processed_data]

        metrics["rf_probs"] = predict_activity_proba(
            fps, rf_model=self.rf_model
        )[mask]
        metrics["ad_dists"] = ad_domain_score(fps, ad_model=self.ad_model)[
            mask
        ]
        metrics["qed_scores"] = np.array(
            [d.get("qed", 0.0) for d in processed_data], dtype=np.float32
        )[mask]
        metrics["sa_scores"] = np.array(
            [d.get("sa", 10.0) for d in processed_data], dtype=np.float32
        )[mask]

        sampled_smiles, _ = sample_smiles_nograd(
            self.model, voc=self.voc, n_mols=100, block_size=200, top_k=10
        )

        metrics["validity"] = get_validity(sampled_smiles)
        metrics["uniqueness"] = get_uniqueness(sampled_smiles)
        metrics["diversity"] = get_diversity(sampled_smiles)
        metrics["novelty"] = get_novelty(sampled_smiles, self.train_smiles_set)

        img_np = self._generate_molecule_image(sampled_smiles)
        return metrics, img_np

    def _generate_molecule_image(
        self, smiles_list: List[str], n_samples: int = 8
    ) -> Optional[np.ndarray]:
        """Generate grid image of sampled molecules."""
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid_mols = [m for m in mols if m is not None]

        if not valid_mols:
            return None

        k = min(n_samples, len(valid_mols))
        idx = np.random.choice(len(valid_mols), size=k, replace=False)
        img = Draw.MolsToGridImage(
            [valid_mols[i] for i in idx], molsPerRow=4, subImgSize=(200, 200)
        )
        return np.array(img)


class TensorBoardLogger:
    """Logs evaluation metrics to TensorBoard."""

    def __init__(self, writer):
        self.writer = writer

    def log_metrics(
        self, step: int, metrics: Dict[str, Any], img_np: Optional[np.ndarray]
    ) -> None:
        """Write metrics and images to TensorBoard."""
        self._log_histograms(step, metrics)
        self._log_scalars(step, metrics)
        self._log_image(step, img_np)

    def _log_histograms(self, step: int, metrics: Dict[str, Any]) -> None:
        for key, data in [
            ("rf/prob_dist", metrics["rf_probs"]),
            ("ad/dist_dist", metrics["ad_dists"]),
            ("qed/score_dist", metrics["qed_scores"]),
            ("sa/score_dist", metrics["sa_scores"]),
        ]:
            if data.size > 0:
                self.writer.add_histogram(key, data, step)

    def _log_scalars(self, step: int, metrics: Dict[str, Any]) -> None:
        for key in ["validity", "uniqueness", "diversity", "novelty"]:
            self.writer.add_scalar(f"metrics/{key}", float(metrics[key]), step)

    def _log_image(self, step: int, img_np: Optional[np.ndarray]) -> None:
        if img_np is not None:
            self.writer.add_image(
                "samples/molecules", img_np, step, dataformats="HWC"
            )
