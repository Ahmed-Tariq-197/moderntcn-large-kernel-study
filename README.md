
![3D Work Division Diagram](work_division_3d.png)

> **Status:** Work in Progress. Data audit and model architectures (`tcn.py`, `modern_tcn.py`) are implemented.

<div align="center">

# 🧠 ModernTCN: Large Kernels vs. Dilations
**A From-Scratch PyTorch Reimplementation for Long-Sequence Time Series Forecasting**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-success.svg)]()
[![Course](https://img.shields.io/badge/Lab_4-Deep_Learning-purple.svg)]()

*Evaluating the trade-offs between Effective Receptive Fields (ERF), parameter efficiency, and forecasting accuracy.*

</div>

---

## 📌 Abstract
This repository presents a rigorous empirical study on the architectural shift from standard dilated convolutions (TCN) to large-kernel depthwise convolutions (ModernTCN), as introduced in the ICLR 2024 paper *"ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"*. 

We reimplement the ModernTCN architecture entirely from scratch in PyTorch, avoiding high-level abstractions. We benchmark its performance against a standard Lab 4 TCN baseline and a DLinear baseline on the **ETTh1** and **ETTm1** datasets, using an input window of \(L=336\) and prediction horizons of \(H \in \{96, 336\}\). Finally, we conduct an ablation study on kernel sizes and visualize the Effective Receptive Field (ERF) using input-sensitivity autograd gradients.

## 📊 Project Management & Work Division
<div align="center">
  <img src="work_division_3d.png" alt="3D Work Division Diagram" width="100%">
</div>

---

## 🏗️ Model Architectures

<details>
<summary><b>1. ModernTCN (Ours)</b></summary>
<br>
Instead of increasing dilation factors (which increases parameter count and requires deep networks to achieve a large receptive field), ModernTCN utilizes large-kernel Depthwise Convolutions combined with two distinct Feed-Forward Networks (ConvFFN1 and ConvFFN2).

- **Patch Embedding:** Splits the input sequence into overlapping patches.
- **Modern Block:**
  - Large-kernel DWConv (e.g., \(k=51\)) to capture long-range dependencies.
  - ConvFFN1: Pointwise convolution operating *within* each variable.
  - ConvFFN2: Pointwise convolution operating *across* variables.
  - Residual connections and Batch Normalization.
</details>

<details>
<summary><b>2. Lab 4 TCN Baseline</b></summary>
<br>
A Temporal Convolutional Network using causal dilated convolutions. The receptive field grows exponentially with depth (\(RF = 1 + 2(k-1)(2^L - 1)\)), contrasting directly with ModernTCN's large-kernel approach.
</details>

<details>
<summary><b>3. DLinear Baseline</b></summary>
<br>
A simple linear model that decomposes the time series into trend and seasonal components, serving as a sanity-check baseline to ensure deep models are actually learning meaningful representations.
</details>

---

## 📂 Repository Structure
```text
├── models/       # PyTorch implementations (ModernTCN, TCN, DLinear)
├── scripts/      # Data audit and parameter counting scripts
├── logs/         # JSON artifacts (data audit results, ablation logs)
├── notebooks/    # End-to-end Colab notebook (Reproducibility)
├── results/      # MSE/MAE tables, ablation, ERF plots
└── docs/         # Discussion notes and design docs
