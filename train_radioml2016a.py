#!/usr/bin/env python3
"""
BioLAMR training script - RML2016.10a
Fixed version: resolves train accuracy lower than validation accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from biolamr import BioLAMR, BioLAMRLoss


class FixedBioLAMR(BioLAMR):
    """Fixed BioLAMR - balanced regularization strategy"""
    
    def __init__(self, gpt_type='gpt2', num_classes=11, seq_len=128, 
                 input_channels=2, use_dual_domain=True, dropout=0.15):
        super().__init__(
            gpt_type=gpt_type,
            num_classes=num_classes, 
            seq_len=seq_len,
            input_channels=input_channels,
            use_dual_domain=use_dual_domain,
            d_model=768,
            gpt_layers=6,
            res_layers=4,
            res_dim=64,
            patch_size=16,
            dropout=dropout,  # Use the passed dropout value
            gpu_id=0
        )
        
        self._configure_trainable_parameters()
        self._add_balanced_regularization(dropout)
    
    def _configure_trainable_parameters(self):
        """Configure trainable parameters"""
        print("Configuring parameter fine-tuning strategy...")
        
        for param in self.gpt2.parameters():
            param.requires_grad = False
        
        trainable_count = 0
        
        # Unfreeze LayerNorm and position encoding
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
                trainable_count += 1
        
        # Unfreeze attention in the last 2 layers
        num_layers = len(self.gpt2.h)
        for i in range(num_layers - 2, num_layers):
            for name, param in self.gpt2.h[i].named_parameters():
                if 'attn' in name:
                    param.requires_grad = True
                    trainable_count += 1
        
        # Unfreeze MLP in the last layer
        for param in self.gpt2.h[-1].mlp.parameters():
            param.requires_grad = True
            trainable_count += 1
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def _add_balanced_regularization(self, dropout):
        """Add balanced regularization - without overly affecting train accuracy"""
        if hasattr(self, 'classifier'):
            old_classifier = self.classifier
            # Add a moderate Dropout before the classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),  # Use the configured dropout value
                old_classifier
            )
        
        print(f"Added regularization layer: Dropout={dropout}")


class RML2016Dataset:
    """RML2016 dataset - simplified version"""
    
    def __init__(self, pkl_file, min_snr=-20, max_snr=18):
        print("Loading RML2016 data...")
        with open(pkl_file, 'rb') as f:
            self.raw_data = pickle.load(f, encoding='latin1')
        
        # Collect modulation types
        all_mods = sorted(list(set([mod for (mod, snr) in self.raw_data.keys()])))
        self.classes = all_mods
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        print(f"Modulation types: {self.classes}")
        
        # Process data
        samples, labels, snrs = [], [], []
        for (mod, snr), data in self.raw_data.items():
            if min_snr <= snr <= max_snr:
                for sample in data:
                    samples.append(sample)
                    labels.append(mod)
                    snrs.append(snr)
        
        self.X = np.array(samples, dtype=np.float32)
        self.Y = self.label_encoder.transform(labels).astype(np.int64)
        self.SNR = np.array(snrs, dtype=np.float32)
        
        # Normalization
        self.X = (self.X - np.mean(self.X)) / (np.std(self.X) + 1e-8)
        
        print(f"Data loaded: {len(self.X)} samples, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.Y[idx], dtype=torch.long)


class SubsetDataset:
    """Subset dataset"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def stratified_split(dataset, test_size=0.10, val_size=0.10, random_state=42):
    """Stratified split"""
    indices = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=dataset.Y, shuffle=True
    )
    
    val_ratio = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=random_state,
        stratify=dataset.Y[train_val_idx], shuffle=True
    )
    
    return (SubsetDataset(dataset, train_idx),
            SubsetDataset(dataset, val_idx),
            SubsetDataset(dataset, test_idx))


class FixedTrainer:
    """Fixed trainer with hierarchical learning rates"""
    
    def __init__(self, model, device='cuda', learning_rate=0.0005, 
                 weight_decay=0.01, epochs=50, patience=10, 
                 label_smoothing=0.05, grad_clip_norm=1.0):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        
        # Loss function with reduced label smoothing
        self.criterion = BioLAMRLoss(
            num_classes=model.num_classes, 
            label_smoothing=label_smoothing
        )
        
        print(f"Loss function: label_smoothing={label_smoothing}")
        
        # Optimizer
        self._setup_optimizer(weight_decay)
        
        # Learning rate scheduler
        self.scheduler = None  # Initialized in train()
        
        # Training history
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.best_val_acc = 0
        self.early_stop_counter = 0
        self.best_model_path = "best_biolamr_rml2016a.pth"
    
    def _setup_optimizer(self, weight_decay):
        """Setup optimizer with hierarchical learning rates"""
        gpt2_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'gpt2' in name:
                    gpt2_params.append(param)
                else:
                    new_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': gpt2_params, 'lr': self.learning_rate * 0.5},
            {'params': new_params, 'lr': self.learning_rate}
        ], weight_decay=weight_decay)
        
        print(f"Optimizer: LR={self.learning_rate}, Weight Decay={weight_decay}")
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc="Training")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """Validate"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"Starting training (device: {self.device})")
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.learning_rate,
            epochs=self.epochs, steps_per_epoch=len(train_loader),
            pct_start=0.3, anneal_strategy='cos'
        )
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 60)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            print(f"Train - Loss: {train_loss:.6f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%")
            print(f"Acc gap: {abs(val_acc - train_acc):.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                }, self.best_model_path)
                print(f"Saved best model (val acc: {val_acc:.2f}%)")
            else:
                self.early_stop_counter += 1
                print(f"Early stop counter: {self.early_stop_counter}/{self.patience}")
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered! Best val acc: {self.best_val_acc:.2f}%")
                    break
        
        print(f"\nTraining complete! Best val acc: {self.best_val_acc:.2f}%")
        return self.history


def plot_training_curves(history):
    """Plot training curves - loss and accuracy"""
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('biolamr_rml2016a_training.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function - fixed configuration"""
    config = {
        'batch_size': 128,
        'epochs': 30,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'patience': 10,
        'label_smoothing': 0.05,
        'dropout': 0.15,
        'grad_clip_norm': 1.0,
        'num_workers': 4,
        'min_snr': -20,
        'max_snr': 18,
        'gpt_type': 'gpt2',
        'use_dual_domain': True,
    }
    
    pkl_file = "/home/caict/code/LLM4RML/data/RML2016.10a/archive/RML2016.10a_dict.pkl"
    
    if not os.path.exists(pkl_file):
        print(f"Data file not found: {pkl_file}")
        return
    
    # Load data
    dataset = RML2016Dataset(pkl_file, min_snr=config['min_snr'], max_snr=config['max_snr'])
    train_dataset, val_dataset, test_dataset = stratified_split(dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    # Create model
    model = FixedBioLAMR(
        gpt_type=config['gpt_type'],
        num_classes=len(dataset.classes),
        seq_len=128,
        input_channels=2,
        use_dual_domain=config['use_dual_domain'],
        dropout=config['dropout']
    )
    
    # Train
    trainer = FixedTrainer(
        model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        epochs=config['epochs'],
        patience=config['patience'],
        label_smoothing=config['label_smoothing'],
        grad_clip_norm=config['grad_clip_norm']
    )
    
    history = trainer.train(train_loader, val_loader)

    print(f"\nTraining complete! Best val acc: {trainer.best_val_acc:.2f}%")
    print(f"Model saved: {trainer.best_model_path}")

    # Plot training curves
    plot_training_curves(history)


if __name__ == "__main__":
    main()
