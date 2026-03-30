# ViP-Bloc
ViP-Bloc is the first end-to-end video-based photoplethysmography (ViPPG) framework that jointly encodes cardiac periodicity as a structural graph topology prior and provides non-repudiable blockchain provenance for every physiological prediction.
# ViP-Bloc: Periodicity-Aware Relational Graph Networks with Blockchain Provenance for Privacy-Preserving Contactless Physiological Monitoring

<p align="center">
  <img src="assets/vipbloc_overview.png" width="820" alt="ViP-Bloc Architecture Overview"/>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/TBME.2026.XXXXXXX">
    <img src="https://img.shields.io/badge/IEEE%20TBME-Under%20Review-blue?style=flat-square&logo=ieee"/>
  </a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-Preprint-red?style=flat-square&logo=arxiv"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/CUDA-11.8-76b900?style=flat-square&logo=nvidia"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

> **ViP-Bloc** is the first end-to-end video-based photoplethysmography (ViPPG) framework that jointly encodes cardiac periodicity as a structural graph topology prior and provides non-repudiable blockchain provenance for every physiological prediction — achieving 52% MAE reduction over the strongest published baseline (LSTS) on PURE while operating at 4.5× lower FLOPs than a dense Transformer.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Blockchain Layer](#blockchain-layer)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

---

## Overview

Vision-based photoplethysmography (ViPPG) recovers heart rate (HR), heart rate variability (HRV), and respiratory rate from subtle periodic chrominance changes in standard RGB facial video — enabling fully contactless physiological monitoring for telemedicine, driver-safety systems, and continuous health tracking.

Despite a decade of progress, two fundamental gaps remain:

1. **Quasi-periodicity underexploitation.** Existing CNN, Transformer, and graph-based rPPG models capture local BVP waveform morphology but cannot structurally enforce cardiac cycle-to-cycle consistency. Dense attention over cross-cycle pairs costs O(T²) and is prohibitive for sequences longer than 64 frames.

2. **Privacy and auditability.** No prior ViPPG work provides verifiable, tamper-evident records of AI-generated physiological measurements — a hard requirement in regulated telemedicine and forensic contexts.

**ViP-Bloc** addresses both gaps in a single unified framework through four contributions:

| Contribution | What it does |
|---|---|
| **C1 — Temporal-Periodic Graph** | Partitions edges into *intra-period* (local BVP morphology, 9-frame window) and *inter-period* (cross-cycle cardiac consistency, Δ ∈ [9,45] frames) relation types — the first structural periodicity prior in rPPG |
| **C2 — R-GCN + Graph Transformer** | Two-stage propagation: relation-specific R-GCN aggregation + C-head Graph Transformer attention over the sparse periodic graph, at 4.5× fewer FLOPs than a dense Transformer |
| **C3 — TriAug** | Three-component augmentation (Temporal CutMix, Normalized Difference Frames, Multi-Scale MPOS) with a Fourier-domain proof linking temporal masking to Dirichlet kernel spectral regularisation |
| **C4 — Blockchain Provenance** | SHA-256 permissioned ledger commits hashed predictions + metadata, providing 100% tamper detection with only 10.5% inference overhead and a formal security proof |

---

## Key Results

### HR Estimation (Mean Absolute Error, lower is better)

| Dataset | MAE (bpm) | RMSE (bpm) | Pearson r | vs. LSTS baseline |
|---|---|---|---|---|
| **PURE** | **0.72** | — | **1.00** | −52% MAE |
| **UBFC-rPPG** | **0.15** | — | — | −71% MAE |
| **MMPD** | **4.74** | **10.16** | **0.72** | −0.06 bpm (p=0.03) |

All comparisons are statistically significant by two-tailed Wilcoxon signed-rank test (p < 0.01).

### Inter-Period Graph Ablation (ΔMAE over intra-only baseline)

| Dataset | ΔMAE improvement |
|---|---|
| PURE | −0.66 bpm |
| UBFC-rPPG | −0.37 bpm |
| MMPD | −0.70 bpm |

### Module Ablation (UBFC-rPPG)

| GT | R-GCN | MAE (bpm) |
|---|---|---|
| ✗ | ✓ | 0.37 |
| ✓ | ✗ | 0.59 |
| ✓ | ✓ | **0.15** |

### Blockchain Trust Layer (500-run evaluation)

| Metric | Value |
|---|---|
| Tamper detection rate | **100%** |
| End-to-end latency | < 50 ms |
| Inference overhead | **10.5%** |

---

## Architecture

```
Input: RGB facial video (T=128 frames, 128×128 px)
          │
          ▼
    ┌─────────────────┐
    │   TriAug        │  TCM + NDF + MPOS
    │ Preprocessing   │  (Fourier-domain guarantee)
    └────────┬────────┘
             │ Z₀ ∈ ℝ^{T×H×W×3}
             ▼
    ┌─────────────────┐
    │  Swin Transformer│  4 stages, L₁=12 layers, 8 heads
    │  Spatial Encoder │  Pool3D → h_t ∈ ℝ^{d_g}
    └────────┬────────┘
             │ {h₁,...,h_T}  node features
             ▼
    ┌──────────────────────────────────────┐
    │   Temporal-Periodic Graph  G=(V,E,R) │
    │                                      │
    │  R_intra: window [i-P, i+F]          │  P=F=1 (PURE/UBFC)
    │  R_inter: Δ ∈ [Δ_min, Δ_max]        │  P=F=5 (MMPD)
    │                                      │
    │  Stage 1 — Relational GCN (Eq. 4)   │
    │  g_i = σ(Σ_r Σ_j 1/|N_r| W_r h_j   │
    │          + W₀ h_i)                   │
    │                                      │
    │  Stage 2 — Graph Transformer (Eq. 5) │
    │  o_i = ‖_{c=1}^C (W₁ᶜg_i +          │  C=8 heads, d_g=128
    │          Σ_j α_ij^c W₂ᶜ g_j)        │
    └────────┬─────────────┬───────────────┘
             │ Z_RGCN      │ Z_GT
             ▼             ▼
    ┌──────────────────────┐
    │  Cross-Attention     │  Learned sigmoid gate
    │  Fusion Gate         │  α_gate ∈ (0.3, 0.7) per node
    └──────────┬───────────┘
               │ Z̃ ∈ ℝ^{T×d_g}
               ▼
    ┌──────────────────────┐
    │  Conv1D Decoder      │  k_c=7 frames (~233 ms)
    │  + Periodicity Head  │  L_total = L_time + λ₁L_freq
    └──────────┬───────────┘           + λ₂L_HR + λ₃L_period
               │ ŝ ∈ ℝ^T, ŷ_t
               ▼
    ┌──────────────────────┐
    │  Blockchain Trust    │  SHA-256 hash commitment
    │  Layer               │  Permissioned ledger
    └──────────────────────┘
               │ HR estimate + audit record
               ▼
           Output
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0.0
- CUDA 11.8
- NVIDIA GPU with ≥ 16 GB VRAM (tested on A100 80 GB)

### Setup

```bash
# Clone repository
git clone https://github.com/akortheanchor/ViP-Bloc.git
cd ViP-Bloc

# Create and activate conda environment
conda create -n vipbloc python=3.10 -y
conda activate vipbloc

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
torch==2.0.0
torchvision==0.15.0
torch-geometric==2.3.1
timm==0.9.2
numpy>=1.24.0
scipy>=1.10.0
opencv-python>=4.7.0
einops>=0.6.1
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.12.0
hashlib
```

---

## Datasets

ViP-Bloc is evaluated on three public benchmarks:

| Dataset | Subjects | Videos | Scenarios | Resolution |
|---|---|---|---|---|
| **PURE** | 10 | 60 | Stationary, talking, slow/fast translation, small/large rotation | 640×480, 30 fps |
| **UBFC-rPPG** | 42 | 42 | Indoor stationary | 640×480, 30 fps |
| **MMPD** | 33 | 660 | Indoor/outdoor, Fitzpatrick I–VI skin tones, walking, talking, gym | 320×240, 30 fps |

### Download and Prepare

```bash
# 1. Download datasets from official sources:
#    PURE:      https://www.tu-ilmenau.de/neurob/data-sets-code/pulse-rate-dataset-pure/
#    UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg
#    MMPD:      https://github.com/McJackTang/MMPD_rPPG_dataset

# 2. Organise under data/raw/
mkdir -p data/raw/{PURE,UBFC-rPPG,MMPD}

# 3. Run preprocessing (face detection via RetinaFace + crop to 128×128)
python scripts/preprocess.py --dataset PURE     --input data/raw/PURE     --output data/processed/PURE
python scripts/preprocess.py --dataset UBFC     --input data/raw/UBFC-rPPG --output data/processed/UBFC
python scripts/preprocess.py --dataset MMPD     --input data/raw/MMPD     --output data/processed/MMPD
```

---

## Training

### Single Dataset

```bash
# PURE (window [P,F]=[1,1], T=128, 30 epochs)
python train.py \
  --config configs/pure.yaml \
  --dataset PURE \
  --gpu 0

# UBFC-rPPG
python train.py \
  --config configs/ubfc.yaml \
  --dataset UBFC \
  --gpu 0

# MMPD (window [P,F]=[5,5], 10 epochs)
python train.py \
  --config configs/mmpd.yaml \
  --dataset MMPD \
  --gpu 0
```

### Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `T` | 128 | Temporal sequence length (frames) |
| `d_g` | 128 | Graph embedding dimension |
| `C` | 8 | Graph Transformer attention heads |
| `[P, F]` | [1,1] / [5,5] | R-GCN intra-period window (PURE/UBFC / MMPD) |
| `Δ_min`, `Δ_max` | 15, 25 (train) / 9, 45 (infer) | Inter-period lag range (frames) |
| `λ₁`, `λ₂`, `λ₃` | 0.5, 1.0, 0.2 | Loss weights (freq, HR, period) |
| Optimizer | Adam, lr=1e-4, wd=0.01 | With cosine-annealed OneCycle scheduler |
| Batch size | 4 | Per GPU |
| Swin layers | L₁=12 | 8-head Swin Transformer backbone |

### Configuration File Structure

```yaml
# configs/pure.yaml
model:
  backbone: swin_base
  swin_layers: 12
  swin_heads: 8
  d_g: 128
  gt_heads: 8
  gt_layers: 1
  kernel_size: 7

graph:
  T: 128
  P: 1
  F: 1
  delta_min: 15      # training range
  delta_max: 25
  delta_min_infer: 9  # inference range
  delta_max_infer: 45

loss:
  lambda1: 0.5
  lambda2: 1.0
  lambda3: 0.2

training:
  epochs: 30
  batch_size: 4
  lr: 1.0e-4
  weight_decay: 0.01

blockchain:
  enabled: true
  ledger_path: ledger/pure_ledger.json
```

---

## Evaluation

```bash
# Evaluate on PURE with pretrained checkpoint
python evaluate.py \
  --config configs/pure.yaml \
  --checkpoint checkpoints/vipbloc_pure.pth \
  --dataset PURE \
  --metrics MAE RMSE r HRV

# Cross-dataset evaluation (train on PURE, test on UBFC)
python evaluate.py \
  --config configs/ubfc.yaml \
  --checkpoint checkpoints/vipbloc_pure.pth \
  --dataset UBFC \
  --cross_dataset

# Full ablation (intra-only vs full model)
python ablation.py \
  --config configs/pure.yaml \
  --checkpoint checkpoints/vipbloc_pure.pth \
  --ablation inter_period

# Generate attention visualisations (Figures 9 & 10)
python visualise_attention.py \
  --checkpoint checkpoints/vipbloc_pure.pth \
  --subject A B C D \
  --output figures/
```

---

## Pretrained Models

| Dataset | MAE (bpm) | r | Download |
|---|---|---|---|
| PURE | 0.72 | 1.00 | [vipbloc_pure.pth](#) |
| UBFC-rPPG | 0.15 | — | [vipbloc_ubfc.pth](#) |
| MMPD | 4.74 | 0.72 | [vipbloc_mmpd.pth](#) |

> Pretrained weights will be released upon paper acceptance.

---

## Blockchain Layer

ViP-Bloc includes a lightweight permissioned blockchain for tamper-evident audit trails of physiological predictions.

### How It Works

Each inference produces a commitment tuple:

```
D_k = {ŷ, t, M}
```

where `ŷ` is the predicted HR/HRV signal, `t` is the UTC timestamp, and `M` is device + session metadata. This tuple is SHA-256 hashed and appended to the ledger as:

```
B_k = SHA256(D_k ‖ hash(B_{k-1}))
```

The chain structure guarantees that tampering with any prior record invalidates all subsequent hashes — providing 100% tamper detection across 500 benchmark runs.

### Usage

```python
from vipbloc.blockchain import BlockchainLedger

ledger = BlockchainLedger(ledger_path="ledger/session.json")

# After each inference
hr_prediction = model.predict(video_frames)
ledger.commit(prediction=hr_prediction, metadata=session_meta)

# Verify integrity at any time
is_valid = ledger.verify()
print(f"Ledger integrity: {'OK' if is_valid else 'TAMPERED'}")
```

### Security Properties

| Property | Guarantee |
|---|---|
| Tamper detection | 100% (500-run evaluation) |
| Latency | < 50 ms per commit |
| Inference overhead | 10.5% |
| Asymptotic complexity | O(n·s) append, O(n) verify |
| Collision model | SHA-256 collision resistance (2⁻¹²⁸ probability) |

---

## Project Structure

```
ViP-Bloc/
├── configs/                   # YAML training configs per dataset
│   ├── pure.yaml
│   ├── ubfc.yaml
│   └── mmpd.yaml
├── vipbloc/
│   ├── models/
│   │   ├── swin_encoder.py    # Swin Transformer backbone + Pool3D gate
│   │   ├── rgcn.py            # Relational GCN (Eq. 4)
│   │   ├── graph_transformer.py  # Graph Transformer (Eq. 5)
│   │   ├── fusion_gate.py     # Cross-attention sigmoid gate
│   │   ├── decoder.py         # Conv1D temporal decoder
│   │   └── vipbloc.py         # Full end-to-end model
│   ├── graph/
│   │   ├── temporal_graph.py  # G=(V,E,R) construction
│   │   ├── intra_edges.py     # R_intra: local window edges
│   │   └── inter_edges.py     # R_inter: cross-cycle edges
│   ├── augmentation/
│   │   ├── triaug.py          # TriAug pipeline
│   │   ├── tcm.py             # Temporal CutMix (Fourier proof)
│   │   ├── ndf.py             # Normalized Difference Frames
│   │   └── mpos.py            # Multi-Scale MPOS projections
│   ├── loss/
│   │   └── composite_loss.py  # L_time + λ₁L_freq + λ₂L_HR + λ₃L_period
│   └── blockchain/
│       ├── ledger.py          # SHA-256 chained ledger
│       └── verifier.py        # Tamper detection
├── scripts/
│   ├── preprocess.py          # Face detection + crop pipeline
│   └── dataset_utils.py       # PURE / UBFC / MMPD loaders
├── train.py                   # Main training script
├── evaluate.py                # Evaluation + metrics
├── ablation.py                # Module ablation runner
├── visualise_attention.py     # Swin attention heatmaps
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use ViP-Bloc in your research, please cite:

```bibtex
@article{akoramurthy2026vipbloc,
  author    = {Akoramurthy, B. and Surendiran, B.},
  title     = {{ViP-Bloc}: Periodicity-Aware Relational Graph Networks
               with Blockchain Provenance for Privacy-Preserving
               Contactless Physiological Monitoring},
  journal   = {Signal, Image and Video Processing},
  year      = {2026},
  note      = {Under review},
  institution = {National Institute of Technology Puducherry, India;
                 }
}
```

---

## Authors

| Author | Affiliation | Email |
|---|---|---|
| **Akoramurthy B** | Dept. of CSE, NIT Puducherry, India | cs22d1005@nitpy.ac.in |
| **Surendiran B** | Dept. of CSE, NIT Puducherry, India | surendiran@nitpy.ac.in |

This work is part of the **Visvesvaraya PhD- Phase-II** research programme.

---

## Acknowledgements



We thank the creators of the PURE, UBFC-rPPG, and MMPD datasets for making their data publicly available. The Swin Transformer backbone is based on the implementation by [Liu et al. (2021)](https://github.com/microsoft/Swin-Transformer). The R-GCN implementation builds on [PyTorch Geometric](https://pyg.org/).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>NIT Puducherry  · 2026</b><br/>
  <i>For questions or collaboration, open an issue or contact cs22d1005@nitpy.ac.in</i>
</p>
