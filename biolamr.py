#!/usr/bin/env python3
"""
BioLAMR: GPT-2 based radio modulation recognition model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 1D pooling for signal data
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
    """1D residual block with channel attention for signal processing"""
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
    Lightweight Dual-Domain Fusion (LDDF) module

    Design highlights:
    1. Output dimension matches input [B, 2, 128], fully compatible with GPT-2
    2. Lightweight design (~1000 params), does not interfere with GPT-2 pretrained knowledge
    3. Channel attention + spatial attention for adaptive time-frequency feature fusion
    4. Residual connection preserves original time-domain information
    """
    def __init__(self, channels=2, seq_len=128, reduction=2):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len

        # Channel attention - learns channel importance of time-frequency features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, 4, 128] -> [B, 4, 1]
            nn.Conv1d(channels * 2, channels * 2 // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 2 // reduction, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention - learns spatial importance of time-frequency features
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # Fusion convolution - fuses concatenated features back to original dimension
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, time_features, freq_features):
        """
        Args:
            time_features: [batch, 2, 128] - Time-domain features
            freq_features: [batch, 2, 128] - Frequency-domain features
        Returns:
            fused: [batch, 2, 128] - Fused features (dimension unchanged, compatible with GPT-2)
        """
        # Concatenate time-frequency features
        concat = torch.cat([time_features, freq_features], dim=1)  # [B, 4, 128]

        # Channel attention
        ca_weights = self.channel_attention(concat)  # [B, 4, 1]
        concat_ca = concat * ca_weights  # [B, 4, 128]

        # Spatial attention
        # Compute channel-wise mean and max
        avg_out = torch.mean(concat_ca, dim=1, keepdim=True)  # [B, 1, 128]
        max_out, _ = torch.max(concat_ca, dim=1, keepdim=True)  # [B, 1, 128]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, 128]
        sa_weights = self.spatial_attention(spatial_input)  # [B, 1, 128]
        concat_sa = concat_ca * sa_weights  # [B, 4, 128]

        # Fuse to original dimension
        fused = self.fusion_conv(concat_sa)  # [B, 4, 128] -> [B, 2, 128]

        # Residual connection (preserve original time-domain information)
        fused = fused + self.residual_weight * time_features  # [B, 2, 128]

        return fused  # [B, 2, 128] - dimension preserved


class SignalEmbedding(nn.Module):
    """Signal embedding module for I/Q signal processing"""
    def __init__(self, input_dim, d_model, seq_len=1024):
        super(SignalEmbedding, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Value embedding - maps I/Q signals to high-dimensional space
        self.value_embedding = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1, bias=False)
        
        # Position embedding - learns temporal position information
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [batch_size, 2, seq_len] -> [batch_size, d_model, seq_len]
        x = self.value_embedding(x).transpose(1, 2)  # [B, seq_len, d_model]
        
        # Add position embedding
        seq_len = x.size(1)
        pos_emb = self.position_embedding[:, :seq_len, :]
        x = x + pos_emb
        
        return self.dropout(x)


class BioLAMR(nn.Module):
    """BioLAMR: GPT-2 based modulation recognition model"""
    
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
        
        # Signal embedding module
        self.signal_embedding = SignalEmbedding(input_channels, d_model, seq_len)
        
        # GPT-2 backbone
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
        
        # Freeze GPT-2 parameters, only train selected layers
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # LayerNorm and position encoding
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Dual-domain processing module (optional)
        if use_dual_domain:
            # Time-domain branch
            self.time_branch = nn.Sequential(nn.Conv1d(input_channels, self.res_dim, 3, 1, 1))
            # Frequency-domain branch
            self.freq_branch = nn.Sequential(nn.Conv1d(input_channels, self.res_dim, 3, 1, 1))

            # Residual blocks
            for i in range(self.res_layers):
                self.time_branch.append(ResidualBlock1D(self.res_dim))
                self.freq_branch.append(ResidualBlock1D(self.res_dim))

            self.time_branch.append(nn.Conv1d(self.res_dim, input_channels, 3, 1, 1))
            self.freq_branch.append(nn.Conv1d(self.res_dim, input_channels, 3, 1, 1))

            # Lightweight Dual-Domain Fusion (LDDF) module
            self.lddf = LDDF(
                channels=input_channels,
                seq_len=seq_len,
                reduction=2
            )
        
        # Dimension alignment layer
        self.dim_align = nn.Linear(d_model, self.gpt_dim) if d_model != self.gpt_dim else nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Global average pooling for sequence compression
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def dual_domain_processing(self, x):
        """
        Dual-domain parallel processing with lightweight attention fusion

        Args:
            x: [batch_size, 2, seq_len] - Input I/Q signal
        Returns:
            fused_features: [batch_size, 2, seq_len] - Fused features
        """
        if not self.use_dual_domain:
            return x

        # Time-domain processing
        time_features = self.time_branch(x)  # [B, 2, seq_len]

        # Frequency-domain processing
        # Convert I/Q signal to complex for FFT
        x_complex = torch.complex(x[:, 0, :], x[:, 1, :])  # [B, seq_len]
        x_fft = torch.fft.fft(x_complex, dim=1)  # FFT transform

        # Separate real and imaginary parts
        x_freq = torch.stack([torch.real(x_fft), torch.imag(x_fft)], dim=1)  # [B, 2, seq_len]
        freq_features = self.freq_branch(x_freq)  # [B, 2, seq_len]

        # Lightweight attention fusion (replaces simple addition)
        fused_features = self.lddf(time_features, freq_features)  # [B, 2, seq_len]

        return fused_features

    def forward(self, x):
        """
        Forward pass
        Args:
            x: [batch_size, 2, seq_len] - I/Q signal data
        Returns:
            logits: [batch_size, num_classes] - Classification logits
        """
        batch_size, channels, seq_len = x.shape
        
        # Per-sample normalization
        mean = torch.mean(x, dim=[1, 2], keepdim=True)
        std = torch.std(x, dim=[1, 2], keepdim=True) + 1e-8
        x = (x - mean) / std
        
        # Dual-domain processing (optional)
        if self.use_dual_domain:
            x = self.dual_domain_processing(x)
        
        # Signal embedding
        embedded = self.signal_embedding(x)  # [B, seq_len, d_model]
        
        # Dimension alignment to GPT-2
        gpt_input = self.dim_align(embedded)  # [B, seq_len, gpt_dim]
        
        # Sequence modeling via GPT-2
        gpt_output = self.gpt2(inputs_embeds=gpt_input).last_hidden_state  # [B, seq_len, gpt_dim]
        
        # Sequence compression: global average pooling
        pooled = gpt_output.mean(dim=1)  # [B, gpt_dim]
        
        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits


class BioLAMRLoss(nn.Module):
    """BioLAMR loss function with label smoothing"""
    
    def __init__(self, num_classes=24, label_smoothing=0.1):
        super(BioLAMRLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size] - Class labels
        """
        return self.ce_loss(logits, targets)


def create_biolamr_model(num_classes=24, gpt_type='gpt2', **kwargs):
    """Factory function to create a BioLAMR model"""
    model = BioLAMR(
        gpt_type=gpt_type,
        num_classes=num_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_biolamr_model(num_classes=24, gpt_type='gpt2').to(device)
    
    # Test input (simulating RML2018 data format)
    batch_size = 4
    seq_len = 1024
    test_input = torch.randn(batch_size, 2, seq_len).to(device)  # [B, 2, 1024]
    
    # Forward pass test
    with torch.no_grad():
        output = model(test_input)
        print(f"Output shape: {output.shape}")  # Expected: [4, 24]
        print(f"Output sample: {output[0, :5]}")  # First sample's top-5 class logits
    
    # Model parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen ratio: {(total_params-trainable_params)/total_params*100:.1f}%")
