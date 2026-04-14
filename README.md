# BioLAMR

**BioLAMR: A Biomimetically Inspired Large Language Model Adaptation Framework for Automatic Modulation Recognition**

This repository contains the official implementation of BioLAMR, published in *Biomimetics* (2026).

## Overview

BioLAMR transfers the sequence-modeling capability of a pretrained GPT-2 Small backbone to the task of automatic modulation recognition (AMR) through three bio-inspired components:

1. **Lightweight Dual-Domain Fusion (LDDF)** — Parallel time-domain and frequency-domain feature extraction with channel & spatial attention, inspired by auditory time–frequency processing.
2. **Convolutional Signal Embedding** — A Conv1d-based projection that maps continuous I/Q signals into GPT-2-compatible sequences, bypassing discrete tokenization.
3. **Hierarchical Parameter Fine-Tuning** — Selective unfreezing of GPT-2 layers (LayerNorm, position embeddings, top-layer attention & MLP), keeping ~8.9 % of parameters trainable. Inspired by hierarchical processing in the auditory cortex.

## Architecture

```
I/Q Signal (2 × 128)
     │
     ▼
Per-sample Normalization
     │
     ├──► Time Branch (Conv1d + ResBlocks + CA)  ──┐
     │                                              ├──► LDDF Fusion
     └──► FFT → Freq Branch (Conv1d + ResBlocks + CA) ──┘
                                                    │
                                                    ▼
                                   Convolutional Signal Embedding
                                                    │
                                                    ▼
                                   Feature Distribution Alignment
                                                    │
                                                    ▼
                                      GPT-2 Small (12 blocks)
                                      [Hierarchical Fine-Tuning]
                                                    │
                                                    ▼
                                     Global Average Pooling → MLP
                                                    │
                                                    ▼
                                          Modulation Prediction
```

## Results

| Dataset | Overall Acc. | Low-SNR Acc. (-20 to -2 dB) | High-SNR Acc. (0 to 18 dB) |
|---|---|---|---|
| RadioML2016.10a | **64.99 ± 0.31 %** | **36.78 %** | **93.21 %** |
| RadioML2016.10b | **67.43 ± 0.27 %** | **38.14 %** | **96.72 %** |

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Transformers (Hugging Face)
- scikit-learn, numpy, matplotlib, seaborn, tqdm

```bash
pip install -r requirements.txt
```

## Dataset

Download the RadioML benchmark datasets:

- [RadioML2016.10a](https://www.deepsig.ai/datasets/) — 11 modulation classes, 220,000 samples
- [RadioML2016.10b](https://www.deepsig.ai/datasets/) — 10 modulation classes, 1,200,000 samples

Place the dataset files (`.pkl` or `.dat`) under a `data/` directory.

## Usage

### Training

```bash
# RadioML2016.10a (11 classes)
python train_radioml2016a.py

# RadioML2016.10b (10 classes)
python train_radioml2016b.py
```

Note: Please modify the dataset path in each training script before running.

## File Structure

```
BioLAMR/
├── biolamr.py              # Model definition (BioLAMR, LDDF, SignalEmbedding, etc.)
├── train_radioml2016a.py   # Training script for RadioML2016.10a
├── train_radioml2016b.py   # Training script for RadioML2016.10b
├── requirements.txt
├── .gitignore
└── README.md
```

## License

This project is released for academic research purposes.


