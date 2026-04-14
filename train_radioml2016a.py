#!/usr/bin/env python3
"""
BioLAMR 修复版训练脚本 - RML2016
解决训练准确率低于验证准确率的问题
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
    """修复版BioLAMR - 平衡的正则化策略"""
    
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
            dropout=dropout,  # 使用传入的dropout值
            gpu_id=0
        )
        
        self._configure_trainable_parameters()
        self._add_balanced_regularization(dropout)
    
    def _configure_trainable_parameters(self):
        """配置可训练参数"""
        print("配置参数微调策略...")
        
        for param in self.gpt2.parameters():
            param.requires_grad = False
        
        trainable_count = 0
        
        # 解冻LayerNorm和位置编码
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
                trainable_count += 1
        
        # 解冻最后2层的注意力
        num_layers = len(self.gpt2.h)
        for i in range(num_layers - 2, num_layers):
            for name, param in self.gpt2.h[i].named_parameters():
                if 'attn' in name:
                    param.requires_grad = True
                    trainable_count += 1
        
        # 解冻最后一层的MLP
        for param in self.gpt2.h[-1].mlp.parameters():
            param.requires_grad = True
            trainable_count += 1
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def _add_balanced_regularization(self, dropout):
        """添加平衡的正则化 - 不过度影响训练准确率"""
        if hasattr(self, 'classifier'):
            old_classifier = self.classifier
            # 只在分类头前添加一次适度的Dropout
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),  # 使用配置的dropout值
                old_classifier
            )
        
        print(f"添加正则化层: Dropout={dropout}")


class RML2016Dataset:
    """RML2016数据集 - 简化版"""
    
    def __init__(self, pkl_file, min_snr=-20, max_snr=18):
        print("加载 RML2016 数据...")
        with open(pkl_file, 'rb') as f:
            self.raw_data = pickle.load(f, encoding='latin1')
        
        # 收集调制类型
        all_mods = sorted(list(set([mod for (mod, snr) in self.raw_data.keys()])))
        self.classes = all_mods
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        print(f"调制类型: {self.classes}")
        
        # 处理数据
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
        
        # 标准化
        self.X = (self.X - np.mean(self.X)) / (np.std(self.X) + 1e-8)
        
        print(f"数据加载完成: {len(self.X)} 样本, {len(self.classes)} 类别")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.Y[idx], dtype=torch.long)


class SubsetDataset:
    """子数据集"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def stratified_split(dataset, test_size=0.10, val_size=0.10, random_state=42):
    """分层抽样划分"""
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
    """修复版训练器"""
    
    def __init__(self, model, device='cuda', learning_rate=0.0005, 
                 weight_decay=0.01, epochs=50, patience=10, 
                 label_smoothing=0.05, grad_clip_norm=1.0):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        
        # 修复后的损失函数 - 降低标签平滑
        self.criterion = BioLAMRLoss(
            num_classes=model.num_classes, 
            label_smoothing=label_smoothing
        )
        
        print(f"✅ 损失函数: 标签平滑={label_smoothing} (降低训练准确率惩罚)")
        
        # 优化器
        self._setup_optimizer(weight_decay)
        
        # 学习率调度器
        self.scheduler = None  # 将在train中初始化
        
        # 训练历史
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.best_val_acc = 0
        self.early_stop_counter = 0
        self.best_model_path = "best_biolamr_rml2016a.pth"
    
    def _setup_optimizer(self, weight_decay):
        """设置优化器"""
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
        
        print(f"✅ 优化器: LR={self.learning_rate}, Weight Decay={weight_decay}")
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc="Training")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 修复后的梯度裁剪
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
        """验证"""
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
        """完整训练"""
        print(f"开始训练 (设备: {self.device})")
        
        # 初始化调度器
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
            
            print(f"训练 - 损失: {train_loss:.6f}, 准确率: {train_acc:.2f}%")
            print(f"验证 - 损失: {val_loss:.6f}, 准确率: {val_acc:.2f}%")
            print(f"📊 准确率差距: {abs(val_acc - train_acc):.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                }, self.best_model_path)
                print(f"✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            else:
                self.early_stop_counter += 1
                print(f"⚠️  早停计数: {self.early_stop_counter}/{self.patience}")
                if self.early_stop_counter >= self.patience:
                    print(f"早停触发! 最佳验证准确率: {self.best_val_acc:.2f}%")
                    break
        
        print(f"\n✅ 训练完成! 最佳验证准确率: {self.best_val_acc:.2f}%")
        return self.history


def plot_training_curves(history):
    """绘制训练曲线 - 只显示损失和准确率"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
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
    """主函数 - 修复版配置"""
    config = {
        'batch_size': 128,
        'epochs': 30,
        'learning_rate': 0.0005,      # ✅ 提高学习率
        'weight_decay': 0.01,         # ✅ 降低权重衰减
        'patience': 10,               # ✅ 更宽容的早停
        'label_smoothing': 0.05,      # ✅ 关键: 降低标签平滑
        'dropout': 0.15,              # ✅ 关键: 降低Dropout
        'grad_clip_norm': 1.0,        # ✅ 放宽梯度裁剪
        'num_workers': 4,
        'min_snr': -20,
        'max_snr': 18,
        'gpt_type': 'gpt2',
        'use_dual_domain': True,
    }
    
    pkl_file = "/home/caict/code/LLM4RML/data/RML2016.10a/archive/RML2016.10a_dict.pkl"
    
    if not os.path.exists(pkl_file):
        print(f"❌ 数据文件不存在: {pkl_file}")
        return
    
    # 加载数据
    dataset = RML2016Dataset(pkl_file, min_snr=config['min_snr'], max_snr=config['max_snr'])
    train_dataset, val_dataset, test_dataset = stratified_split(dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    # 创建模型
    model = FixedBioLAMR(
        gpt_type=config['gpt_type'],
        num_classes=len(dataset.classes),
        seq_len=128,
        input_channels=2,
        use_dual_domain=config['use_dual_domain'],
        dropout=config['dropout']
    )
    
    # 训练
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

    print(f"\n✅ 训练完成! 最佳验证准确率: {trainer.best_val_acc:.2f}%")
    print(f"✅ 模型已保存: {trainer.best_model_path}")

    # 绘制训练曲线
    plot_training_curves(history)


if __name__ == "__main__":
    main()
