# base code and pre training loop adapted from https://github.com/HXYfighter/ACARL

import os
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from vocabulary import SMILESTokenizer, read_vocabulary, create_vocabulary
from dataset import Dataset
from model import GPT, GPTConfig
from utils import model_validity, model_uniqueness, model_novelty, model_diversity, set_seed


def load_data(data_path):
    """Load SMILES data from text file.
    
    Args:
        data_path: Path to text file with one SMILES per line
    
    Returns:
        List of SMILES strings
    """
    with open(data_path, "r") as file:
        data = [line.rstrip() for line in file.readlines()]
    return data


def create_model(model_type, vocab_size, n_layer, n_head, n_embd, max_length, learning_rate, ckpt_load_path=None):
    """Create model and optimizer.
    
    Args:
        model_type: Type of model ('gpt' or 'rnn')
        vocab_size: Size of vocabulary
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        max_length: Maximum sequence length
        learning_rate: Learning rate for optimizer
        ckpt_load_path: Path to checkpoint to load (optional)
    
    Returns:
        Tuple of (model, optimizer)
    """
    if model_type == "gpt":
        model_config = GPTConfig(vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=max_length)
        model = GPT(model_config).to("cuda")
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if ckpt_load_path is not None:
        print(f"Loading checkpoint from {ckpt_load_path}")
        model.load_state_dict(torch.load(ckpt_load_path), strict=True)
    
    return model, optimizer


def get_lr(it, total_it):
    warmup_iters = args.warmup * total_it
    if it < warmup_iters: # linear warmup        
        lr_mult = it / warmup_iters
    else: # cosine learning rate decay        
        decay_ratio  = (it - warmup_iters) / (total_it - warmup_iters)
        lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay_ratio)))
    return args.learning_rate * lr_mult


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="gpt")
    parser.add_argument('--run_name', type=str, help="name for tensorboard run", required=True)
    parser.add_argument('--data_path', type=str, help="path to SMILES text file", required=True)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=200, help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate", required=False)
    parser.add_argument('--lr_decay', type=bool, default=True, help="whether learning rate decays", required=False) 
    parser.add_argument('--warmup', type=float, default=0.01, help="warmup iters", required=False) 
    parser.add_argument('--weight_decay', type=float, default=0.1, help="weight decay", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help="gradient normalization clip", required=False) 
    parser.add_argument('--aug_prob', type=float, default=0.5, help="probablity of augmentation", required=False)
    parser.add_argument('--max_length', type=int, default=200, help="max length of SMILES", required=False)
    parser.add_argument('--vocab_path', type=str, required=False)
    parser.add_argument('--train_split', type=float, default=0.9, help="train/val split ratio", required=False)
    parser.add_argument('--ckpt_load_path', type=str, default=None, required=False)
    parser.add_argument('--ckpt_save_path', type=str, default="ckpt/", required=False)
    # parser.add_argument('--local_rank', type=int, help="local gpu id", required=False)
    args = parser.parse_args()
    
    writer = SummaryWriter("runs/" + args.run_name)
    if not os.path.exists(args.ckpt_save_path + args.run_name):
        os.makedirs(args.ckpt_save_path + args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # Load dataset
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} SMILES from {args.data_path}")

    # Vocabulary
    if args.vocab_path != None:
        voc = read_vocabulary(args.vocab_path)
        print("Read vocabulary from: ", args.vocab_path)
    else:
        voc = create_vocabulary(data, tokenizer=SMILESTokenizer())
        print("Parse vocabulary from dataset")
        tokens = voc.tokens()
        f = open("new_vocab.txt", "w")
        for t in tokens:
            f.write(t + '\n')

    # Split train / val set
    split_idx = int(args.train_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    train_smiles_set = set(train_data)
    train_dataset = Dataset(smiles_list=train_data, vocabulary=voc, tokenizer=SMILESTokenizer(), aug_prob=args.aug_prob)
    val_dataset = Dataset(smiles_list=val_data, vocabulary=voc, tokenizer=SMILESTokenizer())
    print("Training size: ", train_dataset.__len__(), ", Validation size: ", val_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=Dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Model
    model, optimizer = create_model(
        model_type=args.model_type,
        vocab_size=voc.__len__(),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        ckpt_load_path=args.ckpt_load_path
    )

    scaler = torch.cuda.amp.GradScaler()
    model = torch.nn.DataParallel(model, device_ids=[0])

    num_batches = len(train_loader)
    for epoch in tqdm(range(args.max_epochs)):
        # training
        model.train()
        pbar = tqdm(enumerate(train_loader), total=num_batches, leave=False)
        for iter_num, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            lr = get_lr(iter_num + num_batches * epoch, num_batches * args.max_epochs) if args.lr_decay else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            writer.add_scalar('learning rate', lr, iter_num + num_batches * epoch)

            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):
                    logits, _, loss = model(x, y)
                    loss = loss.mean()
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            # optimizer.step()

            pbar.set_description(f"epoch {epoch + 1}, iter {iter_num}: train loss {loss.item():.5f}, lr {lr:e}")
            writer.add_scalar('training loss', loss, iter_num + num_batches * epoch)
            if (iter_num + num_batches * epoch) % 1000 == 0:
                validity = model_validity(model, vocab_path=args.vocab_path, block_size=args.max_length)
                writer.add_scalar('SMILES validity', validity, iter_num + num_batches * epoch)
                uniqueness = model_uniqueness(model, vocab_path=args.vocab_path, block_size=args.max_length)
                writer.add_scalar('SMILES uniqueness', uniqueness, iter_num + num_batches * epoch)
                novelty = model_novelty(model, vocab_path=args.vocab_path, train_smiles_set=train_smiles_set, block_size=args.max_length)
                writer.add_scalar('SMILES novelty', novelty, iter_num + num_batches * epoch)
                diversity = model_diversity(model, vocab_path=args.vocab_path, block_size=args.max_length)
                writer.add_scalar('SMILES diversity', diversity, iter_num + num_batches * epoch)
                model.train()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for iter_num, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                logits, _, loss = model(x, y)
                loss = loss.mean()
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        writer.add_scalar('validation loss', loss, epoch)

        # save checkpoint
        torch.save(model.module.state_dict(), args.ckpt_save_path + args.run_name + "/" + f"epoch{epoch}.pt")
