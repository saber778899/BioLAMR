"""
BioLAMR: A Biomimetically Inspired Large Language Model Adaptation Framework
for Automatic Modulation Recognition.

This module defines the core model architecture of BioLAMR, including:
    - ChannelAttention: Channel attention mechanism for residual blocks
    - ResidualBlock1D: 1D residual block with channel attention
    - LDDF: Lightweight Dual-Domain Fusion module
    - SignalEmbedding: Convolutional signal embedding module
    - BioLAMR: Main model integrating all components with GPT-2 backbone
    - BioLAMRLoss: Label-smoothing cross-entropy loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism (CA).

    Generates per-channel importance weights via global average pooling
    and max pooling, followed by a shared MLP with a bottleneck.

    Args:
        channels: Number of input channels.
        reduction: Bottleneck reduction ratio.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class ResidualBlock1D(nn.Module):
    """
    1D residual block with channel attention for signal processing.

    Consists of two Conv1d layers and a channel attention gate,
    with an identity skip connection.

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv1d(channels, channels, 3, 1, 1)
        self.ca = ChannelAttention(channels, reduction=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out) * out
        return x + out


class LDDF(nn.Module):
    """
    Lightweight Dual-Domain Fusion (LDDF) module.

    Fuses time-domain and frequency-domain features through sequential
    channel attention and spatial attention, inspired by selective
    attention mechanisms in cognitive neuroscience (Posner, 1980).

    Five-stage pipeline:
        1. Feature concatenation
        2. Channel attention (feature-based, "what")
        3. Spatial attention (position-based, "where")
        4. Channel projection and normalization
        5. Learnable residual connection

    Args:
        channels: Number of input channels per domain (C=2 for I/Q).
        seq_len: Temporal sequence length (L).
        reduction: Reduction ratio for channel attention.
    """

    def __init__(self, channels=2, seq_len=128, reduction=2):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels * 2, channels * 2 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 2 // reduction, channels * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # Fusion convolution (2C -> C)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # Learnable residual weight (lambda)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, time_features, freq_features):
        """
        Args:
            time_features: [B, C, L] time-domain features X_t.
            freq_features: [B, C, L] frequency-domain features X_f.

        Returns:
            Y: [B, C, L] fused output features.
        """
        # Stage 1: Feature concatenation  X_c = [X_t; X_f]
        X_c = torch.cat([time_features, freq_features], dim=1)  # [B, 2C, L]

        # Stage 2: Channel attention  M_c
        M_c = self.channel_attention(X_c)
        X_c = X_c * M_c

        # Stage 3: Spatial attention  M_s
        avg_out = torch.mean(X_c, dim=1, keepdim=True)
        max_out, _ = torch.max(X_c, dim=1, keepdim=True)
        S = torch.cat([avg_out, max_out], dim=1)
        M_s = self.spatial_attention(S)
        X_c = X_c * M_s

        # Stage 4: Channel projection and normalization
        X_fused = self.fusion_conv(X_c)

        # Stage 5: Learnable residual connection
        Y = X_fused + self.residual_weight * time_features

        return Y


class SignalEmbedding(nn.Module):
    """
    Convolutional signal embedding module.

    Converts continuous I/Q signals into GPT-2 compatible sequence
    representations via three stages:
        1. Value embedding (Conv1d with kernel size 3)
        2. Learnable positional encoding
        3. Dropout regularization

    Args:
        input_channels: Number of input channels (C=2 for I/Q).
        d_model: Embedding dimension.
        seq_len: Maximum sequence length.
    """

    def __init__(self, input_channels, d_model, seq_len=1024):
        super().__init__()
        self.value_embedding = nn.Conv1d(
            input_channels, d_model, kernel_size=3, padding=1, bias=False
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: [B, C, L] input signal.

        Returns:
            E: [B, L, d_model] embedded sequence.
        """
        x = self.value_embedding(x).transpose(1, 2)  # [B, L, d_model]
        seq_len = x.size(1)
        x = x + self.position_embedding[:, :seq_len, :]
        return self.dropout(x)


class BioLAMR(nn.Module):
    """
    BioLAMR: Biomimetically Inspired Large Language Model Adaptation
    Framework for Automatic Modulation Recognition.

    Architecture overview:
        Input I/Q signal  ->  Per-sample normalization
                          ->  Dual-domain feature extraction
                              (time branch + frequency branch)
                          ->  LDDF fusion
                          ->  Convolutional signal embedding
                          ->  Feature distribution alignment
                          ->  GPT-2 backbone (hierarchical fine-tuning)
                          ->  Global average pooling
                          ->  Classification head
                          ->  Modulation prediction

    Args:
        num_classes: Number of modulation classes
                     (11 for RadioML2016.10a, 10 for RadioML2016.10b).
        seq_len: Input sequence length (default: 128).
        input_channels: Number of input channels (default: 2, I/Q).
        d_model: Signal embedding dimension (default: 768).
        gpt_layers: Number of GPT-2 transformer blocks (default: 12).
        res_layers: Number of residual blocks per branch (default: 4).
        res_dim: Intermediate dimension in residual branches (default: 64).
        use_dual_domain: Whether to enable dual-domain processing.
        dropout: Dropout rate for the classification head.
        gpt_type: Pretrained GPT-2 variant ('gpt2', 'gpt2-medium', etc.).
    """

    def __init__(
        self,
        num_classes=11,
        seq_len=128,
        input_channels=2,
        d_model=768,
        gpt_layers=12,
        res_layers=4,
        res_dim=64,
        use_dual_domain=True,
        dropout=0.15,
        gpt_type="gpt2",
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_dual_domain = use_dual_domain

        # ── Dual-Domain Feature Extraction ──
        if use_dual_domain:
            # Time-domain branch: F_time
            self.time_branch = nn.Sequential(
                nn.Conv1d(input_channels, res_dim, 3, 1, 1)
            )
            # Frequency-domain branch: F_freq
            self.freq_branch = nn.Sequential(
                nn.Conv1d(input_channels, res_dim, 3, 1, 1)
            )
            for _ in range(res_layers):
                self.time_branch.append(ResidualBlock1D(res_dim))
                self.freq_branch.append(ResidualBlock1D(res_dim))
            # Dimension recovery
            self.time_branch.append(nn.Conv1d(res_dim, input_channels, 3, 1, 1))
            self.freq_branch.append(nn.Conv1d(res_dim, input_channels, 3, 1, 1))

            # Lightweight Dual-Domain Fusion (LDDF) module
            self.lddf = LDDF(
                channels=input_channels, seq_len=seq_len, reduction=2
            )

        # ── Convolutional Signal Embedding ──
        self.signal_embedding = SignalEmbedding(input_channels, d_model, seq_len)

        # ── GPT-2 Backbone ──
        self.gpt2 = GPT2Model.from_pretrained(
            gpt_type, output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.gpt_dim = self.gpt2.config.hidden_size

        # Feature distribution alignment  (Pi_align)
        self.dim_align = (
            nn.Linear(d_model, self.gpt_dim)
            if d_model != self.gpt_dim
            else nn.Identity()
        )

        # ── Classification Head ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.gpt_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # ── Hierarchical Parameter Fine-tuning ──
        self._configure_hierarchical_finetuning()

    def _configure_hierarchical_finetuning(self):
        """
        Hierarchical parameter fine-tuning strategy inspired by
        auditory cortical hierarchical processing (Rauschecker & Scott, 2009).

        Unfreezing scheme for GPT-2 Small (12 blocks):
            - LayerNorm (gamma, beta) in all layers   : trainable
            - Position embedding (wpe)                 : trainable
            - Self-attention in last 2 blocks (10-11)  : trainable
            - FFN / MLP in the last block (11)         : trainable
            - All other GPT-2 parameters               : frozen

        This results in ~8.9% trainable parameters (11.1M / 125.2M).
        """
        # Freeze all GPT-2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm and position embeddings
        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name:
                param.requires_grad = True

        # Unfreeze self-attention in the last 2 transformer blocks
        num_layers = len(self.gpt2.h)
        for i in range(num_layers - 2, num_layers):
            for name, param in self.gpt2.h[i].named_parameters():
                if "attn" in name:
                    param.requires_grad = True

        # Unfreeze MLP in the last transformer block
        for param in self.gpt2.h[-1].mlp.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable parameters: {trainable_params:,} "
            f"({trainable_params / total_params * 100:.1f}%)"
        )

    def dual_domain_processing(self, x):
        """
        Dual-domain parallel processing with LDDF fusion.

        Args:
            x: [B, C, L] normalized I/Q signal.

        Returns:
            Y: [B, C, L] fused time-frequency features.
        """
        # Time-domain branch
        X_t = self.time_branch(x)

        # Frequency-domain branch
        # Complexification: z = I + jQ
        z = torch.complex(x[:, 0, :], x[:, 1, :])
        # Discrete Fourier Transform
        Z = torch.fft.fft(z, dim=1)
        # Real-imaginary decoupling
        x_freq = torch.stack([Z.real, Z.imag], dim=1)
        X_f = self.freq_branch(x_freq)

        # LDDF fusion
        Y = self.lddf(X_t, X_f)
        return Y

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [B, C, L] raw I/Q signal (C=2, L=seq_len).

        Returns:
            logits: [B, num_classes] classification logits.
        """
        # Per-sample amplitude normalization
        mean = torch.mean(x, dim=[1, 2], keepdim=True)
        std = torch.std(x, dim=[1, 2], keepdim=True) + 1e-8
        x = (x - mean) / std

        # Dual-domain feature extraction and fusion
        if self.use_dual_domain:
            x = self.dual_domain_processing(x)

        # Convolutional signal embedding
        embedded = self.signal_embedding(x)  # [B, L, d_model]

        # Feature distribution alignment
        gpt_input = self.dim_align(embedded)  # [B, L, gpt_dim]

        # GPT-2 backbone: hierarchical sequence modeling
        gpt_output = self.gpt2(inputs_embeds=gpt_input).last_hidden_state

        # Global average pooling
        pooled = gpt_output.mean(dim=1)  # [B, gpt_dim]

        # Classification head
        logits = self.classifier(pooled)  # [B, num_classes]

        return logits


class BioLAMRLoss(nn.Module):
    """
    Loss function for BioLAMR with label smoothing.

    Args:
        num_classes: Number of target classes.
        label_smoothing: Label smoothing factor (default: 0.05).
    """

    def __init__(self, num_classes=11, label_smoothing=0.05):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets)


def create_biolamr_model(num_classes=11, gpt_type="gpt2", **kwargs):
    """Factory function to create a BioLAMR model instance."""
    return BioLAMR(num_classes=num_classes, gpt_type=gpt_type, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_biolamr_model(num_classes=11, gpt_type="gpt2").to(device)

    # Test forward pass (RadioML2016.10a format: 2 x 128)
    test_input = torch.randn(4, 2, 128).to(device)
    with torch.no_grad():
        output = model(test_input)
        print(f"Output shape: {output.shape}")  # Expected: [4, 11]

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen ratio: {(total - trainable) / total * 100:.1f}%")
