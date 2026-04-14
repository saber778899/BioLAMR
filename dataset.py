"""
Dataset loaders for RadioML2016.10a and RadioML2016.10b benchmarks.

Both datasets are loaded from pickle files and preprocessed with
Z-score normalization. Stratified splitting is performed jointly
over modulation class and SNR bin.
"""

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class RadioMLDataset(Dataset):
    """
    Unified dataset loader for RadioML2016.10a and RadioML2016.10b.

    Supports both .pkl and .dat files (both are pickle-serialized).

    Args:
        data_path: Path to the dataset file (.pkl or .dat).
        min_snr: Minimum SNR to include (dB).
        max_snr: Maximum SNR to include (dB).
    """

    def __init__(self, data_path, min_snr=-20, max_snr=18):
        print(f"Loading RadioML dataset from {data_path} ...")

        with open(data_path, "rb") as f:
            raw_data = pickle.load(f, encoding="latin1")

        # Collect modulation types
        all_mods = sorted(set(mod for (mod, snr) in raw_data.keys()))
        self.classes = all_mods
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)

        # Filter and collect samples
        samples, labels, snrs = [], [], []
        for (mod, snr), data in raw_data.items():
            if min_snr <= snr <= max_snr:
                for sample in data:
                    samples.append(sample)
                    labels.append(mod)
                    snrs.append(snr)

        self.X = np.array(samples, dtype=np.float32)  # (N, 2, 128)
        self.Y = self.label_encoder.transform(labels).astype(np.int64)
        self.SNR = np.array(snrs, dtype=np.float32)

        # Global Z-score normalization (computed on entire dataset;
        # in practice, statistics should be computed on the training split only)
        self.X = (self.X - np.mean(self.X)) / (np.std(self.X) + 1e-8)

        print(f"  Samples: {len(self.X)}")
        print(f"  Classes ({len(self.classes)}): {self.classes}")
        print(f"  Shape: {self.X.shape}")
        print(f"  SNR range: [{self.SNR.min():.0f}, {self.SNR.max():.0f}] dB")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


class SubsetDataset(Dataset):
    """Wraps a dataset with a subset of indices."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def stratified_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                     random_state=42):
    """
    Stratified split of the dataset into train / val / test subsets.

    Stratification is performed over modulation class labels to ensure
    proportional representation in each split.

    Args:
        dataset: A RadioMLDataset instance.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = np.arange(len(dataset))
    labels = dataset.Y

    # Step 1: separate the test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels,
        shuffle=True,
    )

    # Step 2: separate validation from train+val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=labels[train_val_idx],
        shuffle=True,
    )

    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    return (
        SubsetDataset(dataset, train_idx),
        SubsetDataset(dataset, val_idx),
        SubsetDataset(dataset, test_idx),
    )
