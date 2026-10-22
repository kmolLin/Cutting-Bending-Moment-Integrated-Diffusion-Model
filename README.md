# Physics-Guided Diffusion Model for Cutting Bending Moment Synthesis

## Overview
In smart manufacturing, accurate tool wear prediction is often limited by data deficiency and cross-condition adaptation. Conventional data-driven generative models (e.g., GANs) synthesize high-frequency dynamic signals without physical guidance, which may cause frequency shifts and negative transfer in downstream tasks.

To address this issue, we propose a **Physics-Guided Diffusion Model** that integrates a mechanistic cutting bending moment model with a Denoising Diffusion Probabilistic Model (DDPM).

### ✨ Key Contributions
- **Mechanics-Integrated Architecture**: Uses ideal cutting bending moments as a deterministic structural prior at the input layer ($[B_x, B_y, R, \theta]$).
- **Dynamic Feature Modulation (FiLM)**: Injects the tool wear state ($VB$) and diffusion timestep ($t$) seamlessly into the 1D-CNN U-Net backbone.
- **Superior Cross-Condition Generalization**: Generates highly realistic, wear-related signal variations that outperform traditional augmentation methods in downstream prediction tasks.

---

## 📦 Released Features (Code & Data)
This repository provides:
- [x] Complete PyTorch implementation of the 1D-CNN U-Net Diffusion Model (`models/unet1d.py`).
- [x] Training scripts with the custom DDPM noise schedule (`diffusion/ddpm.py`, `train.py`).
- [x] Inference scripts for synthesizing physics-guided bending moments (`generate.py`).
- [x] Sample dataset from the milling cutting experiments (`data/sample_dataset/`, `data/Tool_Wear.csv`) and analytical mechanistic simulator (`data/mechanistic_model.py`).
- [x] Pre-trained weights for quick reproduction (`checkpoints/pretrained_diffusion.pth`).

---

## 📂 Repository Structure
```text
├── data/                  # Sample datasets and signal pre-processing scripts
│   ├── sample_dataset/    # Sample experimental & simulated cutting pass CSVs (NO1, NO10)
│   ├── Tool_Wear.csv      # Ground-truth tool flank wear (VB) measurements
│   ├── dataset.py         # PyTorch Dataset & polar feature transformation (Bx, By, R, theta)
│   └── mechanistic_model.py # Analytical cutting bending moment simulator
├── models/                # 1D-CNN U-Net and FiLM conditioning implementations
│   └── unet1d.py          # UNet1D, ResnetBlock, FiLM, and SinusoidalPosEmb
├── diffusion/             # DDPM forward/reverse process formulations
│   └── ddpm.py            # DiffusionManager (q_sample, p_sample) & DiffusionPredictor
├── checkpoints/           # Pre-trained weights for quick reproduction
│   └── pretrained_diffusion.pth
├── train.py               # Script for training the diffusion model
├── generate.py            # Script for inference and signal synthesis
└── requirements.txt       # Environment dependencies
```

---

## 🛠️ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Signal Synthesis (Quick Reproduction with Pre-trained Weights)
Run `generate.py` to synthesize realistic cutting bending moments for an unseen cutting condition (`NO10`) using the provided pre-trained checkpoint and sample dataset:
```bash
# Synthesize Condition NO10, Cut #1 (Fresh tool: VB = 0.025 mm)
python generate.py --condition NO10 --cut_idx 1

# Synthesize Condition NO10, Cut #20 (Worn tool: VB = 0.219 mm)
python generate.py --condition NO10 --cut_idx 20
```
Synthesized numerical signals (`.csv`) and time-domain comparison plots (`.png`) are saved to `./outputs/`.

### 3. Training the Diffusion Model
To train the Physics-Guided Diffusion Model on your dataset (or test training on the included sample dataset):
```bash
# Quick training verification on the included sample dataset
python train.py --data_dir ./data/sample_dataset --toolwear_csv ./data/Tool_Wear.csv \
                --train_cond NO1 --test_cond NO10 --epochs 10

# Full training on all experimental conditions
python train.py --data_dir ./processed_csv_individual_10sec --toolwear_csv ./Tool_Wear.csv \
                --train_cond NO1 NO2 NO3 NO4 NO5 NO6 NO7 NO8 --test_cond NO9 NO10 --epochs 500
```

### 4. Mechanistic Bending Moment Simulation
You can also run the analytical cutting bending moment simulator in `data/mechanistic_model.py` to generate ideal physical priors ($B_{x,\text{sim}}, B_{y,\text{sim}}$) from spindle speed, feed rate, axial/radial depth of cut, and tool geometry.
