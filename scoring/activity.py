import numpy as np
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem


rf_model = None


def init_rf_model(model_path):
    """
    Load the pre-trained Random Forest model into a global variable.
    
    :param model_path: Path to joblib RF model
    """
    global rf_model
    rf_model = load(model_path)


def morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=np.float32)


def predict_activity_proba(fps_list):
    """
    Predict activity probabilities for a list of FPS  using the loaded RF model.

    :param fps_list: List of FPS vectors
    :return: NumPy array of probabilities (0.0 for invalid FPS)
    """
    if rf_model is None:
            raise RuntimeError("RF model not loaded")

    proba_full = np.zeros(len(fps_list), dtype=np.float32)
        
    valid_idx = [i for i, fp in enumerate(fps_list) if fp is not None]

    if valid_idx:
        import pandas as pd
        df_features = pd.DataFrame([fps_list[i] for i in valid_idx])
        probabilities = rf_model.predict_proba(df_features)[:, 1]  # positive class
        for out_i, i in enumerate(valid_idx):
            proba_full[i] = float(probabilities[out_i])

    return proba_full
