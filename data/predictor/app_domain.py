import argparse
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def fp_to_array(mol, n_bits=2048, radius=2):
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    for i in range(n_bits):
        arr[i] = bv.GetBit(i)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--smiles_col", default="standardized_smiles")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    smiles = df[args.smiles_col].dropna().tolist()

    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            fps.append(fp_to_array(m, n_bits=args.n_bits, radius=args.radius))
    fps = np.array(fps, dtype=np.uint8)

    # sklearn NN with Jaccard distance on boolean vectors
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="jaccard", n_neighbors=5)
    nn.fit(fps.astype(bool))

    joblib.dump(
        {"nn": nn, "fps": fps, "radius": args.radius, "n_bits": args.n_bits},
        args.out_path
    )
    print(f"Saved kNN to {args.out_path} with {len(fps)} fingerprints")

if __name__ == "__main__":
    main()