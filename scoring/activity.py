from typing import Any, List, Optional
import warnings
import joblib
import numpy as np
import numpy.typing as npt

rf_model = None

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names"
)


def load_rf_model(path: str) -> Any:
    """
    Load the Scikit-Learn Random Forest model from disk.

    Parameters
    ----------
    path : str
        The file path to the saved joblib model.

    Returns
    -------
    Any
        The loaded Scikit-Learn model object.
    """
    return joblib.load(path)


def predict_activity_proba(
    fps: List[Optional[npt.NDArray[np.float32]]],
    rf_model: Any,
    default_prob: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Predict the probability of activity using the Random Forest model.

    Parameters
    ----------
    fps : List[Optional[npt.NDArray[np.float32]]]
        A list of Morgan fingerprint numpy arrays. Invalid molecules
        should be represented as None.
    rf_model : Any
        The loaded Scikit-Learn Random Forest model.
    default_prob : float, optional
        The fallback probability to assign if a molecule is invalid
        (default is 0.0).

    Returns
    -------
    npt.NDArray[np.float32]
        A 1D numpy array of predicted probabilities (class 1) matching
        the length of the input list.
    """
    probs = np.full(len(fps), default_prob, dtype=np.float32)

    valid_idx = []
    valid_fps = []

    for i, fp in enumerate(fps):
        if fp is not None:
            valid_fps.append(fp)
            valid_idx.append(i)

    if not valid_fps:
        return probs

    fps_matrix = np.stack(valid_fps, axis=0)

    preds = rf_model.predict_proba(fps_matrix)[:, 1]

    for i, p in zip(valid_idx, preds):
        probs[i] = float(p)

    return probs
