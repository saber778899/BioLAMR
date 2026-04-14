#!/usr/bin/env python3
"""
Training script for BioLAMR on RadioML2016.10a / RadioML2016.10b.

Usage:
    python train.py --dataset_path data/RML2016.10a_dict.pkl --num_classes 11
    python train.py --dataset_path data/RML2016.10b.dat     --num_classes 10

For multi-run statistical evaluation (4 seeds as in the paper):
    python train.py --dataset_path data/RML2016.10a_dict.pkl --num_classes 11 \
                    --seeds 42 128 256 512
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biolamr import BioLAMR, BioLAMRLoss
from dataset import RadioMLDataset, stratified_split


# ─────────────────────────── Trainer ───────────────────────────


class BioLAMRTrainer:
    """
    Training loop for BioLAMR with hierarchical learning rates,
    OneCycleLR scheduling, gradient clipping, and early stopping.
    """

    def __init__(self, model, config, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Loss function
        self.criterion = BioLAMRLoss(
            num_classes=model.num_classes,
            label_smoothing=config["label_smoothing"],
        )

        # Hierarchical learning-rate optimizer
        gpt2_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "gpt2" in n
        ]
        new_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "gpt2" not in n
        ]
        self.optimizer = optim.AdamW(
            [
                {"params": gpt2_params, "lr": config["learning_rate"] * 0.5},
                {"params": new_params,  "lr": config["learning_rate"]},
            ],
            weight_decay=config["weight_decay"],
        )

        self.scheduler = None  # initialized in train()
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
        }

    def _train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(loader, desc="Train")
        for signals, labels in pbar:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(signals)
            loss = self.criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config["grad_clip_norm"]
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        return total_loss / len(loader), 100.0 * correct / total

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for signals, labels in tqdm(loader, desc="Val  "):
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), 100.0 * correct / total

    def train(self, train_loader, val_loader, save_path="best_biolamr.pth"):
        cfg = self.config

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg["learning_rate"],
            epochs=cfg["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        print(f"\nTraining on {self.device} for up to {cfg['epochs']} epochs\n")

        for epoch in range(cfg["epochs"]):
            print(f"Epoch {epoch + 1}/{cfg['epochs']}")
            print("-" * 50)

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(
                f"  Train Loss: {train_loss:.6f}  Acc: {train_acc:.2f}%\n"
                f"  Val   Loss: {val_loss:.6f}  Acc: {val_acc:.2f}%"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                    },
                    save_path,
                )
                print(f"  >> Saved best model (val acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                print(
                    f"  Early-stop counter: "
                    f"{self.patience_counter}/{cfg['patience']}"
                )
                if self.patience_counter >= cfg["patience"]:
                    print("Early stopping triggered.")
                    break

        print(f"\nTraining complete. Best val acc: {self.best_val_acc:.2f}%")
        return self.history


# ─────────────────────── Plotting ───────────────────────


def plot_training_curves(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train", lw=2)
    axes[0].plot(history["val_loss"], label="Val", lw=2)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", lw=2)
    axes[1].plot(history["val_acc"], label="Val", lw=2)
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")


# ─────────────────────── Main ───────────────────────


def run_single_seed(args, seed):
    """Run one complete train/eval cycle with a given random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = RadioMLDataset(
        args.dataset_path, min_snr=args.min_snr, max_snr=args.max_snr
    )
    train_set, val_set, _ = stratified_split(
        dataset, random_state=seed
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = BioLAMR(
        num_classes=args.num_classes,
        seq_len=128,
        input_channels=2,
        d_model=768,
        gpt_layers=args.gpt_layers,
        res_layers=4,
        res_dim=64,
        use_dual_domain=True,
        dropout=args.dropout,
        gpt_type=args.gpt_type,
    )

    config = {
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "label_smoothing": args.label_smoothing,
        "grad_clip_norm": args.grad_clip_norm,
    }

    trainer = BioLAMRTrainer(model, config, device=str(device))
    save_name = f"best_biolamr_seed{seed}.pth"
    history = trainer.train(train_loader, val_loader, save_path=save_name)

    plot_training_curves(history, save_path=f"curves_seed{seed}.png")

    return trainer.best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train BioLAMR")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to RadioML dataset (.pkl or .dat)")
    parser.add_argument("--num_classes", type=int, default=11,
                        help="Number of modulation classes (11 for 10a, 10 for 10b)")
    parser.add_argument("--gpt_type", type=str, default="gpt2")
    parser.add_argument("--gpt_layers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_snr", type=int, default=-20)
    parser.add_argument("--max_snr", type=int, default=18)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds; use multiple for statistical runs")
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"  Seed: {seed}")
        print(f"{'=' * 60}")
        acc = run_single_seed(args, seed)
        results.append(acc)

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"Results over {len(results)} runs:")
        print(f"  Accuracies: {[f'{r:.2f}' for r in results]}")
        print(f"  Mean: {np.mean(results):.2f}%  Std: {np.std(results):.2f}%")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
