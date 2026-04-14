#!/usr/bin/env python3
"""
基于GPT-2的无线电调制识别模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model

class ChannelAttention(nn.Module):
    """通道注意力机制 - 从LLM4CP迁移"""
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 改为1D适配信号数据
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ResidualBlock1D(nn.Module):
    """1D残差块 - 适配信号处理"""
    def __init__(self, in_planes):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv1d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs


class LDDF(nn.Module):
    """
    轻量级双域注意力融合模块 (Lightweight Dual-Domain Fusion)

    设计要点：
    1. 保持输出维度与输入一致 [B, 2, 128]，完全兼容GPT-2
    2. 轻量级设计（~1000参数），不干扰GPT-2的预训练知识
    3. 使用通道注意力+空间注意力自适应融合时频域特征
    4. 残差连接保留原始时域信息
    """
    def __init__(self, channels=2, seq_len=128, reduction=2):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len

        # 通道注意力 - 学习时频域特征的通道重要性
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, 4, 128] -> [B, 4, 1]
            nn.Conv1d(channels * 2, channels * 2 // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 2 // reduction, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力 - 学习时频域特征的空间重要性
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 融合卷积 - 将拼接的时频域特征融合回原始维度
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

        # 残差连接的可学习权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, time_features, freq_features):
        """
        Args:
            time_features: [batch, 2, 128] - 时域特征
            freq_features: [batch, 2, 128] - 频域特征
        Returns:
            fused: [batch, 2, 128] - 融合后的特征（维度不变，兼容GPT-2）
        """
        # 拼接时频域特征
        concat = torch.cat([time_features, freq_features], dim=1)  # [B, 4, 128]

        # 通道注意力
        ca_weights = self.channel_attention(concat)  # [B, 4, 1]
        concat_ca = concat * ca_weights  # [B, 4, 128]

        # 空间注意力
        # 计算通道维度的平均和最大值
        avg_out = torch.mean(concat_ca, dim=1, keepdim=True)  # [B, 1, 128]
        max_out, _ = torch.max(concat_ca, dim=1, keepdim=True)  # [B, 1, 128]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, 128]
        sa_weights = self.spatial_attention(spatial_input)  # [B, 1, 128]
        concat_sa = concat_ca * sa_weights  # [B, 4, 128]

        # 融合到原始维度
        fused = self.fusion_conv(concat_sa)  # [B, 4, 128] -> [B, 2, 128]

        # 残差连接（保留原始时域信息）
        fused = fused + self.residual_weight * time_features  # [B, 2, 128]

        return fused  # [B, 2, 128] ✅ 维度保持不变


class SignalEmbedding(nn.Module):
    """信号嵌入模块 - 适配I/Q信号处理"""
    def __init__(self, input_dim, d_model, seq_len=1024):
        super(SignalEmbedding, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Token嵌入 - 将I/Q信号映射到高维空间
        self.value_embedding = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1, bias=False)
        
        # 位置嵌入 - 学习信号的时序位置信息
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [batch_size, 2, seq_len] -> [batch_size, d_model, seq_len]
        x = self.value_embedding(x).transpose(1, 2)  # [B, seq_len, d_model]
        
        # 添加位置嵌入
        seq_len = x.size(1)
        pos_emb = self.position_embedding[:, :seq_len, :]
        x = x + pos_emb
        
        return self.dropout(x)


class BioLAMR(nn.Module):
    """BioLAMR: 基于GPT-2的调制识别模型"""
    
    def __init__(self, 
                 gpt_type='gpt2',
                 d_model=768,
                 gpt_layers=6,
                 seq_len=1024,
                 input_channels=2,
                 num_classes=24,
                 res_layers=4,
                 res_dim=64,
                 patch_size=16,
                 use_dual_domain=True,
                 dropout=0.1,
                 gpu_id=0):
        super(BioLAMR, self).__init__()
        
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.use_dual_domain = use_dual_domain
        self.res_dim = res_dim
        self.res_layers = res_layers
        
        # 信号嵌入模块
        self.signal_embedding = SignalEmbedding(input_channels, d_model, seq_len)
        
        # GPT-2 骨干网络
        if gpt_type == 'gpt2-medium':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1024
        elif gpt_type == 'gpt2-large':
            self.gpt2 = GPT2Model.from_pretrained('gpt2-large', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1280
        else:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 768
        
        # 冻结GPT-2参数，只训练部分层
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # LayerNorm和位置编码
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 双域处理模块（可选）
        if use_dual_domain:
            # 时域处理分支
            self.time_branch = nn.Sequential(nn.Conv1d(input_channels, self.res_dim, 3, 1, 1))
            # 频域处理分支
            self.freq_branch = nn.Sequential(nn.Conv1d(input_channels, self.res_dim, 3, 1, 1))

            # 残差块
            for i in range(self.res_layers):
                self.time_branch.append(ResidualBlock1D(self.res_dim))
                self.freq_branch.append(ResidualBlock1D(self.res_dim))

            self.time_branch.append(nn.Conv1d(self.res_dim, input_channels, 3, 1, 1))
            self.freq_branch.append(nn.Conv1d(self.res_dim, input_channels, 3, 1, 1))

            # 轻量级双域注意力融合模块 (LDDF)
            self.lddf = LDDF(
                channels=input_channels,
                seq_len=seq_len,
                reduction=2
            )
        
        # 维度适配层
        self.dim_align = nn.Linear(d_model, self.gpt_dim) if d_model != self.gpt_dim else nn.Identity()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # 全局平均池化用于序列压缩
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def dual_domain_processing(self, x):
        """
        双域并行处理 - 使用轻量级注意力融合

        Args:
            x: [batch_size, 2, seq_len] - 输入I/Q信号
        Returns:
            fused_features: [batch_size, 2, seq_len] - 融合后的特征
        """
        if not self.use_dual_domain:
            return x

        # 时域处理
        time_features = self.time_branch(x)  # [B, 2, seq_len]

        # 频域处理
        # 将I/Q信号转换为复数进行FFT
        x_complex = torch.complex(x[:, 0, :], x[:, 1, :])  # [B, seq_len]
        x_fft = torch.fft.fft(x_complex, dim=1)  # FFT变换

        # 分离实部虚部重新组织
        x_freq = torch.stack([torch.real(x_fft), torch.imag(x_fft)], dim=1)  # [B, 2, seq_len]
        freq_features = self.freq_branch(x_freq)  # [B, 2, seq_len]

        # 轻量级注意力融合（替代简单相加）
        fused_features = self.lddf(time_features, freq_features)  # [B, 2, seq_len]

        return fused_features

    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, 2, seq_len] - I/Q信号数据
        Returns:
            logits: [batch_size, num_classes] - 分类logits
        """
        batch_size, channels, seq_len = x.shape
        
        # 数据标准化
        mean = torch.mean(x, dim=[1, 2], keepdim=True)
        std = torch.std(x, dim=[1, 2], keepdim=True) + 1e-8
        x = (x - mean) / std
        
        # 双域处理（可选）
        if self.use_dual_domain:
            x = self.dual_domain_processing(x)
        
        # 信号嵌入
        embedded = self.signal_embedding(x)  # [B, seq_len, d_model]
        
        # 维度适配到GPT-2
        gpt_input = self.dim_align(embedded)  # [B, seq_len, gpt_dim]
        
        # 通过GPT-2进行序列建模
        gpt_output = self.gpt2(inputs_embeds=gpt_input).last_hidden_state  # [B, seq_len, gpt_dim]
        
        # 序列压缩：全局平均池化
        pooled = gpt_output.mean(dim=1)  # [B, gpt_dim]
        
        # 分类预测
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits


class BioLAMRLoss(nn.Module):
    """BioLAMR损失函数"""
    
    def __init__(self, num_classes=24, label_smoothing=0.1):
        super(BioLAMRLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size] - 类别标签
        """
        return self.ce_loss(logits, targets)


def create_biolamr_model(num_classes=24, gpt_type='gpt2', **kwargs):
    """创建BioLAMR模型的工厂函数"""
    model = BioLAMR(
        gpt_type=gpt_type,
        num_classes=num_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_biolamr_model(num_classes=24, gpt_type='gpt2').to(device)
    
    # 测试输入 (模拟RML2018数据格式)
    batch_size = 4
    seq_len = 1024
    test_input = torch.randn(batch_size, 2, seq_len).to(device)  # [B, 2, 1024]
    
    # 前向传播测试
    with torch.no_grad():
        output = model(test_input)
        print(f"模型输出形状: {output.shape}")  # 应该是 [4, 24]
        print(f"输出示例: {output[0, :5]}")  # 显示第一个样本的前5个类别logits
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"参数冻结比例: {(total_params-trainable_params)/total_params*100:.1f}%")
