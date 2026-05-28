# MARDPG‑UAV: Multi‑Agent Recurrent DPG for UAV Path Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository implements a **decentralised multi‑agent path planning system** for fixed‑wing‑style UAVs operating in a partially observable 3D environment with static obstacles. The framework uses **Multi‑Agent Recurrent Deterministic Policy Gradient (MARDPG)** under the Centralised Training with Decentralised Execution (CTDE) paradigm, featuring:

- Per‑agent LSTM belief approximation
- Centralised attention critic with multi‑head self‑attention
- Ornstein‑Uhlenbeck exploration with exponential annealing
- Per‑agent episode termination (no global collision truncation)
- Sin/cos attitude encoding (wrap‑free)
- Action rate limiting for smooth trajectories

Full mathematical derivations, stability analysis, and hyperparameter justifications are provided in the accompanying PDF.

---

## Table of Contents

- [Mathematical Formulation](#mathematical-formulation)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Hyperparameters](#hyperparameters)
- [Results (Expected)](#results-expected)
- [Citation](#citation)
- [License](#license)

---

## Mathematical Formulation

All equations, assumptions, and derivations are documented in [`MARDPG_UAV_Mathematical_Formulation_Revised.pdf`](MARDPG_UAV_Mathematical_Formulation_Revised.pdf).  
Key sections:

- **UAV kinematic model** (semi‑implicit Euler, constant airspeed `v=3.0 m/s`)
- **Observation space** (36‑dim: sin/cos attitude, previous action, 25‑beam lidar, goal vector)
- **Reward function** (progress + collision penalty + separation penalty + free‑space bonus)
- **Centralised attention critic** (permutation‑invariant Q‑function)
- **Per‑agent termination & validity masking** (for BPTT)

---

## Key Features

| Feature | Implementation |
|---------|----------------|
| Multi‑agent RL | MARDPG (deterministic policy gradient, CTDE) |
| Partial observability | LSTM (hidden size 128) as belief state |
| Coordination | Centralised critic with scaled dot‑product attention |
| Exploration | Ornstein‑Uhlenbeck process (`κ=0.15`, annealed `σ`) |
| Termination | Per‑agent flags, episode ends only when all agents are done |
| Observation encoding | Sin/cos for yaw/pitch, previous action feedback |
| Rate limiting | `δ_max = 0.175 rad/step` for smooth commands |
| Replay buffer | Episode‑based, FIFO (2000 episodes), BPTT length 50 |

---

## Installation

**Requirements**: Python 3.9+, PyTorch 2.0+, NumPy, Gymnasium (or custom environment)

```bash
git clone https://github.com/your-org/mardpg-uav.git
cd mardpg-uav
pip install -r requirements.txt
