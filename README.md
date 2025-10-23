# ddpg-marl-boxpushing-attn
Multi-agent box-pushing using DDPG with attention-based critic and actor architectures.

### Multi-Agent Reinforcement Learning with Attention-based Centralized Critic

![Box Pushing Animation](Docs/Animation_Attention_network.gif)


## 🚀 Overview
This project implements a **Deep Deterministic Policy Gradient (DDPG)** framework for a **multi-agent unicycle system** that learns to cooperatively push a box toward a goal in a continuous 2D environment.

The system uses:
- **Runge-Kutta 4th Order (RK4)** dynamics
- **Friction & inertia modeling**
- **Curriculum learning** (shrinking search radius)
- **Attention-based centralized critic** for cooperative learning

---

## 📈 Results

| Framework | Convergence (Episodes) | Smoothness | Robustness |
|------------|------------------------|-------------|-------------|
| Baseline Critic | ~3700 | ⚪⚪⚪⚫⚫ | ⚪⚪⚪⚫⚫ |
| Attention-Based Critic | ~1200 | ⚪⚪⚪⚪⚪ | ⚪⚪⚪⚪⚫ |
| Extended Critic | ~6700 | ⚪⚪⚪⚪⚪ | ⚪⚪⚫⚫⚫ |

---
