# Bending Moment Synthesis by Integrating Cutting Mechanics with Diffusion Model for Tool Wear Prediction

[![Paper Status: Under Review](https://img.shields.io/badge/Paper-Under_Review-yellow.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

> **📢 Important Notice (Update: July 2026)**
>
> This repository contains the official PyTorch implementation for the paper **"Bending Moment Synthesis by Integrating Cutting Mechanics with Diffusion Model for Tool Wear Prediction"**.
> 
> ⚠️ **The manuscript is currently under peer review.** To comply with journal policies and ensure the integrity of the review process, the complete source code, pre-trained weights, and sample datasets are temporarily withheld. **All materials will be made open-source and publicly available immediately upon the acceptance of the paper.** ---

## 📖 Overview

In smart manufacturing, accurate tool wear prediction is often limited by data deficiency and cross-condition adaptation. Conventional data-driven generative models (e.g., GANs) synthesize high-frequency dynamic signals without physical guidance, which may cause frequency shifts and negative transfer in downstream tasks. 

To address this issue, we propose a **Physics-Guided Diffusion Model ** that integrates a mechanistic cutting bending moment model with a Denoising Diffusion Probabilistic Model (DDPM).

### ✨ Key Contributions:
- **Mechanics-Integrated Architecture**: Uses ideal cutting bending moments as a deterministic structural prior at the input layer.
- **Dynamic Feature Modulation (FiLM)**: Injects the tool wear state and diffusion timestep seamlessly into the 1D-CNN U-Net backbone.
- **Superior Cross-Condition Generalization**: Generates highly realistic, wear-related signal variations that outperform traditional augmentation methods in downstream prediction tasks.

## 🚀 Upcoming Release (Code & Data)

Once the paper is accepted, this repository will be updated with:
- [ ] Complete PyTorch implementation of the 1D-CNN U-Net Diffusion Model.
- [ ] Training scripts with the custom DDPM noise schedule.
- [ ] Inference scripts for synthesizing physical-guided bending moments.
- [ ] Sample dataset from the cutting experiments.
- [ ] Pre-trained weights for quick reproduction.

## 📂 Expected Repository Structure
```text
├── data/                  # Sample datasets and signal pre-processing scripts
├── models/                # 1D-CNN U-Net and FiLM conditioning implementations
├── diffusion/             # DDPM forward/reverse process formulations
├── train.py               # Script for training the diffusion model
├── generate.py            # Script for inference and signal synthesis
└── requirements.txt       # Environment dependencies
