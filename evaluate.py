#!/usr/bin/env python3
"""
Evaluation script for BioLAMR.

Generates:
    - Overall accuracy, per-SNR accuracy, recall, and F1 score
    - Confusion matrix at a specified SNR
    - t-SNE visualization of learned features

Usage:
    python evaluate.py --dataset_path data/RML2016.10a_dict.pkl \
                       --checkpoint best_biolamr_seed42.pth \
                       --num_classes 11
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
)
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from biolamr import BioLAMR
from dataset import RadioMLDataset, stratified_split


# ─────────────────────── Helpers ───────────────────────


def load_model(checkpoint_path, num_classes, device, **model_kwargs):
    """Load a trained BioLAMR model from a checkpoint."""
    model = BioLAMR(num_classes=num_classes, **model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    if "val_acc" in ckpt:
        print(f"  Checkpoint val acc: {ckpt['val_acc']:.2f}%")
    return model


@torch.no_grad()
def predict(model, loader, device):
    """Run inference and return predictions, labels, SNRs, and features."""
    all_preds, all_labels, all_snrs, all_feats = [], [], [], []

    for signals, labels in tqdm(loader, desc="Evaluating"):
        signals = signals.to(device)
        logits = model(signals)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract penultimate-layer features (before classifier)."""
    feats_list, labels_list = [], []

    for signals, labels in tqdm(loader, desc="Extracting features"):
        signals = signals.to(device)
        # Reproduce forward pass up to pooling
        x = signals
        mean = torch.mean(x, dim=[1, 2], keepdim=True)
        std = torch.std(x, dim=[1, 2], keepdim=True) + 1e-8
        x = (x - mean) / std
        if model.use_dual_domain:
            x = model.dual_domain_processing(x)
        embedded = model.signal_embedding(x)
        gpt_input = model.dim_align(embedded)
        gpt_output = model.gpt2(inputs_embeds=gpt_input).last_hidden_state
        pooled = gpt_output.mean(dim=1)  # [B, gpt_dim]

        feats_list.append(pooled.cpu().numpy())
        labels_list.append(labels.numpy())

    return np.concatenate(feats_list), np.concatenate(labels_list)


# ────────────────── Per-SNR Evaluation ──────────────────


def evaluate_per_snr(model, dataset, test_indices, device, batch_size=256):
    """Compute accuracy for each SNR level."""
    snr_values = sorted(set(dataset.SNR[test_indices]))
    results = {}

    for snr in snr_values:
        mask = dataset.SNR[test_indices] == snr
        idx = test_indices[mask]
        if len(idx) == 0:
            continue

        from dataset import SubsetDataset

        subset = SubsetDataset(dataset, idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        preds, labels = predict(model, loader, device)
        acc = accuracy_score(labels, preds) * 100
        results[snr] = acc

    return results


# ──────────────── Confusion Matrix ──────────────────


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("BioLAMR Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")


# ──────────────── t-SNE Visualization ──────────────────


def plot_tsne(features, labels, class_names, save_path="tsne.png",
              perplexity=30, random_state=42, max_samples_per_class=200):
    """
    t-SNE visualization of learned feature representations.

    Args:
        features: (N, D) feature array.
        labels: (N,) integer label array.
        class_names: List of class name strings.
        save_path: Output image path.
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for t-SNE.
        max_samples_per_class: Subsample limit per class.
    """
    # Subsample for speed
    selected = []
    for c in range(len(class_names)):
        idx = np.where(labels == c)[0]
        if len(idx) > max_samples_per_class:
            idx = np.random.choice(idx, max_samples_per_class, replace=False)
        selected.append(idx)
    selected = np.concatenate(selected)

    feats_sub = features[selected]
    labels_sub = labels[selected]

    print(f"Running t-SNE on {len(feats_sub)} samples "
          f"(perplexity={perplexity}, seed={random_state}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    coords = tsne.fit_transform(feats_sub)

    plt.figure(figsize=(10, 8))
    for c in range(len(class_names)):
        mask = labels_sub == c
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=0.6, label=class_names[c],
        )
    plt.legend(markerscale=3, fontsize=8)
    plt.title("BioLAMR Feature Space (t-SNE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"t-SNE plot saved to {save_path}")


# ─────────────────────── Main ───────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate BioLAMR")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--gpt_type", type=str, default="gpt2")
    parser.add_argument("--gpt_layers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr_for_cm", type=int, default=10,
                        help="SNR level for confusion matrix")
    parser.add_argument("--tsne_snr", type=int, default=10,
                        help="SNR level for t-SNE visualization")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)

    # Load data
    dataset = RadioMLDataset(args.dataset_path)
    _, _, test_set = stratified_split(dataset, random_state=args.seed)
    test_indices = test_set.indices
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Load model
    model = load_model(
        args.checkpoint, args.num_classes, device,
        gpt_type=args.gpt_type, gpt_layers=args.gpt_layers,
    )

    # ── Overall metrics ──
    preds, labels = predict(model, test_loader, device)
    overall_acc = accuracy_score(labels, preds) * 100
    overall_f1 = f1_score(labels, preds, average="macro")
    overall_recall = recall_score(labels, preds, average="macro")

    print(f"\n{'=' * 50}")
    print(f"Overall Accuracy : {overall_acc:.2f}%")
    print(f"Macro Recall     : {overall_recall:.4f}")
    print(f"Macro F1 Score   : {overall_f1:.4f}")
    print(f"{'=' * 50}")

    # ── Per-SNR accuracy ──
    snr_results = evaluate_per_snr(
        model, dataset, test_indices, device, args.batch_size
    )
    print("\nPer-SNR Accuracy:")
    for snr in sorted(snr_results):
        print(f"  SNR {snr:+3d} dB : {snr_results[snr]:.2f}%")

    low_snrs = [s for s in snr_results if s <= -2]
    high_snrs = [s for s in snr_results if s >= 0]
    if low_snrs:
        low_acc = np.mean([snr_results[s] for s in low_snrs])
        print(f"\n  Low-SNR  (-20 to -2 dB) : {low_acc:.2f}%")
    if high_snrs:
        high_acc = np.mean([snr_results[s] for s in high_snrs])
        print(f"  High-SNR ( 0  to 18 dB) : {high_acc:.2f}%")

    # ── Confusion matrix at specified SNR ──
    cm_mask = dataset.SNR[test_indices] == args.snr_for_cm
    if cm_mask.any():
        from dataset import SubsetDataset

        cm_idx = test_indices[cm_mask]
        cm_set = SubsetDataset(dataset, cm_idx)
        cm_loader = DataLoader(cm_set, batch_size=args.batch_size, shuffle=False)
        cm_preds, cm_labels = predict(model, cm_loader, device)
        plot_confusion_matrix(
            cm_labels, cm_preds, dataset.classes,
            save_path=f"confusion_matrix_snr{args.snr_for_cm}.png",
        )

    # ── t-SNE visualization ──
    tsne_mask = dataset.SNR[test_indices] == args.tsne_snr
    if tsne_mask.any():
        from dataset import SubsetDataset

        tsne_idx = test_indices[tsne_mask]
        tsne_set = SubsetDataset(dataset, tsne_idx)
        tsne_loader = DataLoader(tsne_set, batch_size=args.batch_size, shuffle=False)
        features, feat_labels = extract_features(model, tsne_loader, device)
        plot_tsne(
            features, feat_labels, dataset.classes,
            save_path=f"tsne_snr{args.tsne_snr}.png",
            perplexity=args.tsne_perplexity,
            random_state=args.seed,
        )


if __name__ == "__main__":
    main()
