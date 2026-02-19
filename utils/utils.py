import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from tdc import Evaluator
from tqdm import tqdm
from pretraining.vocabulary import SMILESTokenizer

def randomize_smiles(smiles):
    # randomize SMILES for data augmentation
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    ans = list(range(mol.GetNumAtoms()))
    if not ans:
        return smiles
    np.random.shuffle(ans)
    new_mol = Chem.RenumberAtoms(mol, ans)
    return Chem.MolToSmiles(new_mol, canonical=False)

# @torch.no_grad()
def likelihood(model, seqs):
    nll_loss = nn.NLLLoss(reduction="none")
    seqs = seqs.cuda()
    logits, _, _ = model(seqs[:, :-1])
    log_probs = logits.log_softmax(dim=2)
    return nll_loss(log_probs.transpose(1, 2), seqs[:, 1:]).sum(dim=1)


@torch.no_grad()
def sample_SMILES(model, voc, n_mols=100, block_size=200, temperature=1.0, top_k=10):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    nll_loss = nn.NLLLoss(reduction="none")
    codes = torch.zeros((n_mols, 1), dtype=torch.long).to("cuda")
    codes[:] = voc["^"]
    nlls = torch.zeros(n_mols).to("cuda")

    model.eval()
    for k in range(block_size - 1):
        logits, _, _ = model(codes)  
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, k=top_k)
        # apply softmax to convert to probabilities
        probs = logits.softmax(dim=-1)
        log_probs = logits.log_softmax(dim=1)
        # sample from the distribution
        code_i = torch.multinomial(probs, num_samples=1)
        # print(probs)
        # append to the sequence and continue
        codes = torch.cat((codes, code_i), dim=1)

        nlls += nll_loss(log_probs, code_i.view(-1))
        if code_i.sum() == 0:
            break

    # codes = codes
    smiles = []
    Tokenizer = SMILESTokenizer()
    for i in range(n_mols):
        tokens_i = voc.decode(np.array(codes[i, :].cpu()))
        smiles_i = Tokenizer.untokenize(tokens_i)
        smiles.append(smiles_i)

    return smiles, codes, nlls


def calc_fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius = 2, nBits = 2048) for x in mols]
    smiles_canonicalized = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]
    return fps, smiles_canonicalized

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)



# testing metrics
from rdkit.DataStructs import FingerprintSimilarity

def model_uniqueness(smiles_list):
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    unique_smiles = set(valid_smiles)
    return len(unique_smiles) / max(1, len(valid_smiles))

def model_novelty(smiles_list, train_smiles_set):
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    unique_smiles = set(valid_smiles)
    novel_smiles = [s for s in unique_smiles if s not in train_smiles_set]
    return len(novel_smiles) / max(1, len(unique_smiles))

def model_diversity(smiles_list):
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    fps, _ = calc_fingerprints(valid_smiles)
    n = len(fps)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sims.append(FingerprintSimilarity(fps[i], fps[j]))
    avg_diversity = 1 - np.mean(sims)
    return avg_diversity


def model_validity(smiles_list):
    evaluator = Evaluator(name = 'Validity')
    return evaluator(smiles_list)
   
def reverse_sigmoid(value, low, high, k=0.25):
    """
    High values are BAD (0.0), Low values are GOOD (1.0).
    Example: Penalizing Molecular Weight.
    """
    if value < low: return 1.0
    if value > high: return 0.0
    return 1.0 / (1.0 + 10.0 ** (k * (value - (high + low) / 2) / (high - low))) # from reinvent


def sigmoid(value, low, high, k=0.25):
    """
    High values are GOOD (1.0), Low values are BAD (0.0).
    Example: QED or Activity Probability.
    """
    if value < low: return 0.0
    if value > high: return 1.0
    return 1.0 / (1.0 + 10.0 ** (k * ((high + low) / 2 - value) / (high - low)))


def step_function(value, min_val):
    """Hard cutoff. Below min_val is 0, above is 1."""
    return 1.0 if value >= min_val else 0.0
