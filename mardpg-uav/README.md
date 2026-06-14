# MARDPG‑UAV: Multi‑Agent Recurrent DPG for UAV Path Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository implements a **decentralised multi‑agent path planning system** for quadcopters operating in a partially observable 3D environment with static obstacles. The framework uses **Multi‑Agent Recurrent Deterministic Policy Gradient (MARDPG)** under the Centralised Training with Decentralised Execution (CTDE) paradigm, featuring:

- Per‑agent LSTM belief approximation
- Recurrent FC->LSTM centralized critic
- Uncorrelated Gaussian noise
- Per‑agent episode termination (no global collision truncation)
- Sin/cos attitude encoding (wrap‑free)
- Action rate limiting for smooth trajectories

---

## Technical Details

- **Observation space**: 49-dim
- **Critic architecture**: Recurrent FC $\to$ LSTM centralized critic
- **Exploration**: Gaussian noise
- **BPTT**: length 80

**Note on Stage-7 Difficulty**: 
Please note that generating an evaluation scene harder than stage-7 (dense static obstacles) is currently bounded by the maximum possible 16 grid spawns. Requesting static_obs > 16 prints a warning and is clamped by the env. Genuine Out-Of-Distribution evaluation comes from increasing goal distance and denser/faster dynamic threats.

---

## Installation

**Requirements**: Python 3.9+, PyTorch 2.0+, NumPy, Matplotlib, pandas, W&B

```bash
pip install -r requirements.txt
```

## Training Hyperparameters
*(Confirm)*
- Actor Learning Rate: 1e-4
- Critic Learning Rate: 1e-3
- Batch Size: 64
- BPTT Length: 80
- Burn-in Length: 10
- Discount Factor (Gamma): 0.99
- Soft Update (Tau): 0.005
- Noise Sigma: 0.1

