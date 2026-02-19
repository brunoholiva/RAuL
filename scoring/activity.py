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


def predict_activity_proba(smiles_list):
    """
    Predict activity probabilities for a list of SMILES strings using the loaded RF model.

    :param smiles_list: List of SMILES strings
    :return: NumPy array of probabilities (0.0 for invalid SMILES)
    """
    if rf_model is None:
        raise RuntimeError("RF model not loaded")

    features = [morgan_fingerprint(smi) for smi in smiles_list]
    valid_idx = [i for i, fp in enumerate(features) if fp is not None]

    proba_full = np.zeros(len(smiles_list), dtype=np.float32)

    if valid_idx:
        # Only predict on valid SMILES
        import pandas as pd
        df_features = pd.DataFrame([features[i] for i in valid_idx])
        probabilities = rf_model.predict_proba(df_features)[:, 1]  # positive class
        for out_i, i in enumerate(valid_idx):
            proba_full[i] = float(probabilities[out_i])

    return proba_full
