# ddpg-marl-boxpushing-attn
Multi-agent box-pushing using DDPG with attention-based critic and actor architectures.

### Multi-Agent Reinforcement Learning with Attention-based Centralized Critic

![Box Pushing Animation](Docs/Animation_Attention_network.gif)


## 📈 Results

| Framework | Convergence (Episodes) | Smoothness | Robustness |
|------------|------------------------|-------------|-------------|
| Baseline Critic | ~3700 | ⚪⚪⚪⚫⚫ | ⚪⚪⚪⚫⚫ |
| Attention-Based Critic | ~1200 | ⚪⚪⚪⚪⚪ | ⚪⚪⚪⚪⚫ |
| Extended Critic | ~6700 | ⚪⚪⚪⚪⚪ | ⚪⚪⚫⚫⚫ |

---
