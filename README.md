# ddpg-marl-boxpushing-attn
Multi-agent box-pushing using DDPG with attention-based critic and actor architectures.

### Multi-Agent Reinforcement Learning with Attention-based Centralized Critic

![Box Pushing Animation](Docs/Animation_Attention_network.gif)


## 🚀 Overview
This project implements a **Deep Deterministic Policy Gradient (DDPG)** framework for a **multi-agent unicycle system** that learns to cooperatively push a box toward a goal in a continuous 2D environment.

The system uses:
- **Runge-Kutta 4th Order (RK4)** dynamics
- **Friction & inertia modeling**
- **Attention-based centralized critic** for cooperative learning
---
## ⚙️ Environment Dynamics

| Entity | Variables | Description |
|--------|------------|--------------|
| Agent | (x, y, θ, F, ω) | Position, orientation, and action |
| Box | (x_b, y_b, θ_b) | Center position and rotation |
| Goal | (X_G, Y_G ) | Position of the Goal |

- Both agents and box motion are integrated using **RK4**.
- Once an agent reaches the box, it “sticks” and starts pushing.
- Rewards are **staged**:
  - **Phase 1:** Reach the box (terminal reward)
  - **Phase 2:** Push box toward goal (instantaneous reward)

---
## 📈 Results

| Framework | Convergence (Episodes) | Smoothness | Robustness |
|------------|------------------------|-------------|-------------|
| Baseline Critic | ~3700 | ⚪⚪⚪⚫⚫ | ⚪⚪⚪⚫⚫ |
| Extended Baseline Critic | ~6700 | ⚪⚪⚪⚪⚪ | ⚪⚪⚫⚫⚫ |
| Attention-Based Critic | ~1200 | ⚪⚪⚪⚪⚪ | ⚪⚪⚪⚪⚫ |

---
