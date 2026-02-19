import argparse
from vocabulary import SMILESTokenizer, create_vocabulary

def load_smiles(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to SMILES txt (one per line)")
    parser.add_argument("--out_path", type=str, required=True, help="output vocab file path")
    args = parser.parse_args()

    smiles = load_smiles(args.data_path)
    vocab = create_vocabulary(smiles, tokenizer=SMILESTokenizer())

    tokens = vocab.tokens()
    with open(args.out_path, "w") as f:
        for t in tokens:
            f.write(f"{t}\n")

    print(f"Saved {len(tokens)} tokens to {args.out_path}")

if __name__ == "__main__":
    main()