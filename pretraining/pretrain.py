import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from pretraining.dataset import Dataset
from pretraining.model import GPT, GPTConfig
from pretraining.vocabulary import SMILESTokenizer, create_vocabulary, read_vocabulary
from config import ModelConfig, TrainingConfig
from utils.rdkit_utils import get_diversity, get_novelty, get_uniqueness, get_validity
from utils.utils import set_seed, sample_smiles_nograd


def load_data(data_path: str) -> list[str]:
    """Load SMILES data from text file."""
    with open(data_path, "r") as file:
        return [line.rstrip() for line in file.readlines()]


def create_model(args, vocab_size: int, device: torch.device):
    """Encapsulates model and optimizer creation."""
    model_config = GPTConfig(
        vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.max_length,
    )
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.learning_rate, betas=(0.9, 0.95)
    )

    if args.ckpt_load_path is not None:
        print(f"Loading checkpoint from {args.ckpt_load_path}")
        model.load_state_dict(torch.load(args.ckpt_load_path, map_location=device), strict=True)

    return model, optimizer


class Pretrainer:
    """
    Handles the pretraining loop. 
    Encapsulates state to prevent passing loose variables between functions.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, voc, train_smiles_set, args, device, writer):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.voc = voc
        self.train_smiles_set = train_smiles_set
        self.args = args
        self.device = device
        self.writer = writer
        
        self.scaler = torch.amp.GradScaler(device="cuda")
        self.num_batches = len(train_loader)
        
        self.eval_model_cfg = ModelConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, max_length=args.max_length
        )
        self.eval_train_cfg = TrainingConfig(
            batch_size=256, max_steps=0, learning_rate=0.0, temperature=1.0, top_k=10, max_workers=1, save_every=0
        )

    def _get_lr(self, it: int, total_it: int) -> float:
        """Calculates learning rate with linear warmup and cosine decay."""
        warmup_iters = self.args.warmup * total_it
        if it < warmup_iters:
            lr_mult = it / max(1, warmup_iters)
        else:
            decay_ratio = (it - warmup_iters) / max(1, (total_it - warmup_iters))
            lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay_ratio)))
        return self.args.learning_rate * lr_mult

    def _evaluate_metrics(self, step: int):
        """Samples smiles and calculates chemical metrics."""
        self.model.eval()
        
        base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        smiles, _ = sample_smiles_nograd(
            base_model, self.voc, self.eval_model_cfg, self.eval_train_cfg
        )

        validity = get_validity(smiles)
        uniqueness = get_uniqueness(smiles)
        novelty = get_novelty(smiles, self.train_smiles_set)
        diversity = get_diversity(smiles)

        self.writer.add_scalar("SMILES validity", validity, step)
        self.writer.add_scalar("SMILES uniqueness", uniqueness, step)
        self.writer.add_scalar("SMILES novelty", novelty, step)
        self.writer.add_scalar("SMILES diversity", diversity, step)
        
        self.model.train()

    def _validate(self, epoch: int):
        """Runs the validation loop and logs loss."""
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, _, loss = self.model(x, y)
                val_losses.append(loss.mean().item())
                
        val_loss = float(np.mean(val_losses))
        self.writer.add_scalar("validation loss", val_loss, epoch)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.5f}")

    def _save_checkpoint(self, epoch: int):
        """Saves the model state dict."""
        save_dir = os.path.join(self.args.ckpt_save_path, self.args.run_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch{epoch}.pt")
        
        base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        torch.save(base_model.state_dict(), save_path)

    def train(self):
        """Main training loop."""
        total_iters = self.num_batches * self.args.max_epochs

        for epoch in range(self.args.max_epochs):
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=self.num_batches, leave=False)
            
            for iter_num, (x, y) in pbar:
                step = iter_num + self.num_batches * epoch
                x, y = x.to(self.device), y.to(self.device)

                lr = self._get_lr(step, total_iters) if self.args.lr_decay else self.args.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                self.writer.add_scalar("learning rate", lr, step)

                with torch.amp.autocast(device_type=self.device.type):
                    _, _, loss = self.model(x, y)
                    loss = loss.mean()
                    
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Logging & Evaluation
                pbar.set_description(f"epoch {epoch + 1}, iter {iter_num}: train loss {loss.item():.5f}, lr {lr:e}")
                self.writer.add_scalar("training loss", loss.item(), step)
                
                if step % 1000 == 0 and step > 0:
                    self._evaluate_metrics(step)

            self._validate(epoch)
            self._save_checkpoint(epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--warmup", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--vocab_path", type=str, required=False)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--ckpt_load_path", type=str, default=None)
    parser.add_argument("--ckpt_save_path", type=str, default="ckpt/")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    writer = SummaryWriter(os.path.join("runs/logging", args.run_name))

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    
    if args.vocab_path:
        voc = read_vocabulary(args.vocab_path)
        print("Read vocabulary from:", args.vocab_path)
    else:
        voc = create_vocabulary(data, tokenizer=SMILESTokenizer())
        print("Parsed vocabulary from dataset")
        with open("new_vocab.txt", "w") as f:
            for t in voc.tokens():
                f.write(t + "\n")

    split_idx = int(args.train_split * len(data))
    train_data, val_data = data[:split_idx], data[split_idx:]
    train_smiles_set = set(train_data)

    train_loader = torch.utils.data.DataLoader(
        Dataset(train_data, voc, SMILESTokenizer(), aug_prob=args.aug_prob),
        batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=Dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(val_data, voc, SMILESTokenizer()),
        batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=Dataset.collate_fn,
    )

    model, optimizer = create_model(args, len(voc), device)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    trainer = Pretrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        voc=voc,
        train_smiles_set=train_smiles_set,
        args=args,
        device=device,
        writer=writer
    )
    
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()