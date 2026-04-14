#!/usr/bin/env python3
"""
BioLAMR training script - RML2016.10b
Modulation recognition training using the RML2016.10b.dat dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Model


class RML2016bDataset(Dataset):
    """RML2016.10b dataset loader - supports .dat format"""
    
    def __init__(self, dat_file, min_snr=-20, max_snr=18):
        """
        Initialize RML2016.10b dataset
        Args:
            dat_file: Path to RML2016.10b.dat file
            min_snr: Minimum SNR (dB)
            max_snr: Maximum SNR (dB)
        """
        print("Loading RML2016.10b data...")
        
        # RML2016.10b.dat is a pickle-formatted binary file
        with open(dat_file, 'rb') as f:
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
        
        self.X = np.array(samples, dtype=np.float32)  # (N, 2, 128)
        self.Y = self.label_encoder.transform(labels).astype(np.int64)
        self.SNR = np.array(snrs, dtype=np.float32)
        
        # Normalization
        self.X = (self.X - np.mean(self.X)) / (np.std(self.X) + 1e-8)
        
        print(f"Data loaded: {len(self.X)} samples, {len(self.classes)} classes")
        print(f"Data shape: {self.X.shape}")
        print(f"SNR range: {self.SNR.min():.1f} to {self.SNR.max():.1f} dB")
        
        # Show class distribution
        unique_labels, counts = np.unique(self.Y, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique_labels, counts):
            mod_name = self.classes[label]
            print(f"  {mod_name}: {count} samples ({count/len(self.Y)*100:.1f}%)")
    
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


def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Stratified split of dataset"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    indices = np.arange(len(dataset))
    labels = dataset.Y
    
    # Step 1: Separate test set
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=random_state,
        stratify=labels, shuffle=True
    )
    
    # Step 2: Separate validation set from train_val
    train_val_labels = labels[train_val_idx]
    val_size = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, random_state=random_state,
        stratify=train_val_labels, shuffle=True
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/len(dataset)*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/len(dataset)*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/len(dataset)*100:.1f}%)")
    
    return SubsetDataset(dataset, train_idx), SubsetDataset(dataset, val_idx), SubsetDataset(dataset, test_idx)


class FixedBioLAMR(nn.Module):
    """Fixed BioLAMR - unfreezes ~13% parameters (consistent with RML2016a version)"""

    def __init__(self, gpt_type='gpt2', num_classes=11, seq_len=128,
                 input_channels=2, use_dual_domain=True, dropout=0.15):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.use_dual_domain = use_dual_domain
        
        # Load pretrained GPT-2
        print(f"Loading pretrained {gpt_type} model...")
        self.gpt2 = GPT2Model.from_pretrained(gpt_type)
        self.hidden_size = self.gpt2.config.hidden_size
        
        # Signal embedding layer
        self.signal_embedding = nn.Linear(input_channels, self.hidden_size)
        
        # Dual-domain processing
        if use_dual_domain:
            self.freq_embedding = nn.Linear(input_channels, self.hidden_size)
            self.domain_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Configure trainable parameters
        self._configure_trainable_parameters()
    
    def _configure_trainable_parameters(self):
        """Configure trainable parameters - unfreeze ~13% (consistent with RML2016a)"""
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
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        # Time-domain embedding
        time_embed = self.signal_embedding(x)
        
        # Dual-domain processing
        if self.use_dual_domain:
            x_fft = torch.fft.rfft(x, dim=1)
            x_freq = torch.cat([x_fft.real, x_fft.imag], dim=-1)
            
            if x_freq.size(-1) != self.input_channels:
                x_freq = x_freq[..., :self.input_channels]
            
            freq_embed = self.freq_embedding(x_freq)
            
            if freq_embed.size(1) < time_embed.size(1):
                freq_embed = torch.nn.functional.pad(
                    freq_embed, (0, 0, 0, time_embed.size(1) - freq_embed.size(1))
                )
            elif freq_embed.size(1) > time_embed.size(1):
                freq_embed = freq_embed[:, :time_embed.size(1), :]
            
            combined = torch.cat([time_embed, freq_embed], dim=-1)
            embeddings = self.domain_fusion(combined)
        else:
            embeddings = time_embed
        
        # GPT-2 processing
        outputs = self.gpt2(inputs_embeds=embeddings)
        sequence_output = outputs.last_hidden_state
        
        # Classification
        pooled = sequence_output.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits


class Trainer:
    """Trainer"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'gpt2' in n], 
             'lr': config['learning_rate'] * 0.1},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'gpt2' not in n], 
             'lr': config['learning_rate']}
        ], weight_decay=config['weight_decay'])
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[config['learning_rate'] * 0.1, config['learning_rate']],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        self.best_val_acc = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_model_path = 'best_biolamr_rml2016b.pth'
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for signals, labels in pbar:
            signals, labels = signals.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels in tqdm(self.val_loader, desc='Validation'):
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader):
        print(f"\nStarting training (device: {self.device})...")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"  Train - Loss: {train_loss:.6f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  Saved best model (val acc: {val_acc:.2f}%)")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    print(f"\nEarly stopping triggered (patience={self.config['patience']})")
                    break
        
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
    plt.savefig('biolamr_rml2016b_training.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function - RML2016.10b version"""
    config = {
        'batch_size': 128,
        'epochs': 50,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'patience': 10,
        'num_workers': 4,
        'min_snr': -20,
        'max_snr': 18,
        'gpt_type': 'gpt2',
        'use_dual_domain': True,
        'dropout': 0.15,
        'label_smoothing': 0.05,
        'grad_clip_norm': 1.0
    }
    
    print("=" * 70)
    print("BioLAMR Training - RML2016.10b Dataset")
    print("=" * 70)
    
    # Data file path
    dat_file = "./data/RML2016.10b/archive/RML2016.10b.dat"
    
    if not os.path.exists(dat_file):
        print(f"Data file not found: {dat_file}")
        print("Please ensure the data file is in the correct path")
        return
    
    # Load data
    dataset = RML2016bDataset(dat_file, min_snr=config['min_snr'], max_snr=config['max_snr'])
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
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train(train_loader, val_loader)

    print(f"\nTraining complete! Best val acc: {trainer.best_val_acc:.2f}%")
    print(f"Model saved: {trainer.best_model_path}")

    # Plot training curves
    plot_training_curves(history)


if __name__ == "__main__":
    main()
