<div align="center">

# HILP Sim-to-Real Experimentation Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.3-A50026?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.7-00897B?style=flat-square)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
<!-- [![ICLR](https://img.shields.io/badge/ICLR-2026-red?style=flat-square)](https://iclr.cc/) -->

<p align="center">
  <!-- <a href="Camera_Latent_Adaptation_of_Foundation_Policies_for_Sim_to_Real_Transfer.pdf"><img src="https://img.shields.io/badge/Paper-PDF-orange?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper"></a> -->
  &nbsp;&nbsp;
  <a href="https://seohong.me/projects/hilp/"><img src="https://img.shields.io/badge/HILP-Project_Page-green?style=for-the-badge&logo=github&logoColor=white" alt="Project Page"></a>
  &nbsp;&nbsp;
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-lightgrey?style=for-the-badge&logo=latex&logoColor=white" alt="Citation"></a>
</p>

[**Overview**](#overview) · [**What's Included**](#whats-included) · [**Quick Start**](#quick-start) · [**Experiments**](#experiments) · [**Citation**](#citation)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=DaRL-GenAI.hilp-sim2real&style=flat)

</div>

A unified experimentation framework for **Hilbert space foundation policies (HILP)** with built-in support for sim-to-real transfer via physics perturbations (gravity, friction). Run HILP training and evaluation across goal-conditioned and zero-shot RL settings, with tools to sweep dynamics, batch-evaluate checkpoints, and analyze transfer performance.

Built on top of [HILP (Park et al., ICML 2024)](https://seohong.me/projects/hilp/) and extended with the **Found-adapt** latent adaptation method for sim-to-real research ([Da et al., ICLR 2026](Camera_Latent_Adaptation_of_Foundation_Policies_for_Sim_to_Real_Transfer.pdf)).

---

### 📰 News

> **[2026.1]** Found-adapt (built on this codebase) accepted at **ICLR 2026**!

### 📦 Releases

> **[2026.1]** Initial release — GCRL and ZSRL implementations with full physics override support.

---

## Overview

This repo lets you:

- **Train HILP foundation policies** from offline data using intrinsic Hilbert-space rewards
- **Evaluate under perturbed dynamics** — swap gravity and friction at train or eval time with a single flag
- **Batch-sweep** checkpoints × physics combinations and aggregate results automatically
- **Run latent adaptation (Found-adapt)** to close the sim-to-real gap without retraining the policy

The framework covers two experiment tracks:

| Track | Environments | Entry Point |
|-------|-------------|-------------|
| **Goal-Conditioned RL (GCRL)** | antmaze-large, antmaze-ultra, kitchen | `hilp_gcrl/` |
| **Zero-Shot RL (ZSRL)** | walker, cheetah, quadruped, jaco | `hilp_zsrl/` |

---

## What's Included

```
HILP/
├── hilp_gcrl/                    # Goal-conditioned RL track
│   ├── main.py                   # Train HILP from offline D4RL data
│   ├── eval.py                   # Evaluate a checkpoint (single or sweep mode)
│   ├── batch_eval.py             # Grid sweep: checkpoints × physics settings
│   ├── src/
│   │   ├── agents/hilp.py        # HILP agent (encoder + latent-conditioned policy)
│   │   ├── d4rl_utils.py         # Dataset loading
│   │   └── dataset_utils.py      # GCDataset wrapper
│   ├── jaxrl_m/                  # JAX RL model utilities
│   ├── d4rl_ext/                 # antmaze-ultra environment extension
│   └── README.md                 # Full GCRL usage guide
│
├── hilp_zsrl/                    # Zero-shot RL track
│   ├── url_benchmark/
│   │   ├── train_gravity_offline.py   # Retrain under gravity sweep
│   │   ├── train_friction_offline.py  # Retrain under friction sweep
│   │   └── test_offline.py            # Direct-transfer eval (frozen policy)
│   ├── convert.py                # ExORL dataset → replay.pt
│   └── README.md                 # Full ZSRL usage guide
│
├── hilp_gcrl_env.yml             # Conda environment spec (GCRL)
└── README.md                     # This file
```

---

## Quick Start

### Goal-Conditioned RL

Full setup instructions: [hilp_gcrl/README.md](hilp_gcrl/README.md)

```bash
conda create -n hilp_gcrl python=3.8 -y && conda activate hilp_gcrl
pip install mujoco==2.3.7 dm-control "gymnasium[mujoco]"
pip install -r hilp_gcrl/requirements.txt

# Train a foundation policy on antmaze-large
python hilp_gcrl/main.py \
    --run_group EXP \
    --env_name antmaze-large-diverse-v2 \
    --seed 0

# Evaluate at default physics
python hilp_gcrl/eval.py \
    --restore_path path/to/params_1000000.pkl \
    --env_name antmaze-large-diverse-v2

# Evaluate under perturbed dynamics (sim-to-real)
python hilp_gcrl/eval.py \
    --restore_path path/to/params_1000000.pkl \
    --env_name antmaze-large-diverse-v2 \
    --gravity_z -34.0 \
    --friction "6,0.6,0.6"

# Batch sweep across checkpoints × physics
python hilp_gcrl/batch_eval.py \
    --env_name antmaze-large-diverse-v2 \
    --model_paths path/to/model1.pkl path/to/model2.pkl \
    --gravity_values -9.81 -15.0 -24.0 -34.0 -44.0 \
    --friction "1,0.5,0.5"
```

### Zero-Shot RL

Full setup instructions: [hilp_zsrl/README.md](hilp_zsrl/README.md)

```bash
conda create -n hilp_zsrl python=3.8 -y && conda activate hilp_zsrl
pip install -r hilp_zsrl/requirements.txt

# Step 1 — Convert ExORL dataset to replay buffer
python hilp_zsrl/url_benchmark/convert.py \
    --save_path path/to/save/ \
    --env walker --task runs --method rnd --num_episodes 5000

# Step 2a — Retrain under gravity sweep (Found-adapt)
PYTHONPATH=hilp_zsrl python hilp_zsrl/url_benchmark/train_gravity_offline.py \
    --domain walker \
    --replay_buffer path/to/walker/replay.pt \
    --gravities -9.81 -15.0 -24.0 -34.0 -44.0

# Step 2b — Direct-transfer eval (frozen policy, no retraining)
PYTHONPATH=hilp_zsrl python hilp_zsrl/url_benchmark/test_offline.py \
    task=walker_run \
    load_model=path/to/latest.pt \
    load_replay_buffer=path/to/replay.pt \
    gravities="[-9.81,-15.0,-24.0,-34.0,-44.0]"
```

---

## Experiments

### Physics Perturbation Settings

| Setting | Parameter | Values |
|---------|-----------|--------|
| G1–G4 | Gravity Z (m/s²) | −15, −24, −34, −44 (default: −9.81) |
| F1–F6 | Friction (sliding, torsional, rolling) | [4,0.4,0.4] → [18,1.8,1.8] |

### Supported Environments

| Environment | Track | Tasks |
|-------------|-------|-------|
| antmaze-large-diverse-v2 | GCRL | Navigation |
| antmaze-ultra-diverse-v0 | GCRL | Long-horizon navigation |
| kitchen-partial-v0 | GCRL | Multi-stage manipulation |
| walker | ZSRL | Stand, Walk, Flip, Run |
| cheetah | ZSRL | Run |
| quadruped | ZSRL | Run |
| jaco | ZSRL | Reach |

### Adaptation Methods

| Method | Retrains Policy? | Task-Agnostic? | Description |
|--------|-----------------|----------------|-------------|
| Direct-Transfer | No | Yes | Deploy frozen policy as-is |
| GAT / UGAT | Partial | No | Learn grounding module per task |
| PAD | Yes | No | Task-specific deployment adaptation |
| **Found-adapt** | **No** | **Yes** | Latent-only adaptation via MetaDynamic + adapter |

---
