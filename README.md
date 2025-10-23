# ddpg-marl-boxpushing-attn
Multi-agent box-pushing using DDPG with attention-based critic and actor architectures.

### Multi-Agent Reinforcement Learning with Attention-based Centralized Critic

![Box Pushing Animation](Docs/Animation_Attention_network.gif)


## 🚀 Overview
In this work, we investigate the role of attention within a centralized critic in a cooperative continuous-space task involving both coordination and physical interaction.
A detailed 2D two-phase box-pushing environment is developed using second-order unicycle agents, torque-based dynamics, and a two-phase cooperative objective.
Within the Deep Deterministic Policy Learning framework, we compare three actor–critic systems: a standard baseline critic, an extended critic with additional environmental information, and an attention critic.
All setups employ the same actor trained to handle both task phases.
Results show that only the attention-based critic achieves reliable convergence and efficient phase transitions, demonstrating that attention is structurally required when a single critic must represent multiple task contexts.
The system uses:
- **Runge-Kutta 4th Order (RK4)** dynamics
- **Friction & inertia modeling**
- **Attention-based centralized critic** for cooperative learning
---
## ⚙️ Environment Dynamics
Each agent is modeled as a unicycle robot with two continuous control outputs: a forward force and a steering angle. These controls respect non-holonomic constraints, requiring agents to rotate gradually to reorient. Figure below is an illustration of the game indicating the agents $r_i, i=1,2$ moving in the direction of their velocities $v_i$, taking the input force  $f_i\$ with the steering angle $alpha_i$ to push the box $b$ toward the goal $G$. 


| Entity | Variables | Description |
|--------|------------|--------------|
| Agent | ($x, y, \alpha, F, ω$) | Position, orientation, and action |
| Box | ($x_b, y_b, \alpha_b$) or ($p_b,\alpha_b$)| Center position and rotation |
| Goal | ($x_G, Y_G$) | Position of the Goal |

![Box game_image](Docs/box_game.jpg)

The equation of motion for the agents before attaching to the box is:

$$
\begin{aligned}
\dot{x}_i &= v_i \cos(\beta_i) \\
\dot{y}_i &= v_i \sin(\beta_i) \\
\dot{\beta}_i &= \alpha_i \\
\dot{v}_i &= \frac{f_i - \mu_a v_i}{m}
\end{aligned}
$$

and the equation of motion after agents are attached to the box is:

$$
\begin{aligned}
\dot{\vec{v}}_b &= \frac{\vec{f_b} - \mu_b \vec{v}_b}{M} \\
\dot{\vec{p}}_b &= \vec{v}_b \\
\dot{\omega}_b &= \frac{\tau - \mu_r \omega_b}{I} \\
\dot{\beta}_b &= \omega_b
\end{aligned}
$$

- Both agents and box motion are integrated using **RK4**.
- Once an agent reaches the box, it “sticks” and starts pushing.
- Rewards are **staged**:
  - **Phase 1:** Reach the box (terminal reward)
$$
\begin{align}
r_{\text{t}} &= 
\begin{cases}
+15, & \text{if the agent reaches the box} \\
-5, & \text{if the agent hits the left wall} \\
-3, & \text{if the agent hits any other wall}
\end{cases} 
\nonumber 
\end{align}
$$

$$
\begin{align}
r_v &= \min(0, 0.1 \cdot v_i) \text{ penalizing backward movement}
\nonumber 
\end{align} 
$$

  - **Phase 2:** Push box toward goal (instantaneous reward)

$$
\begin{align}
r_p &= -\ 0.01 \cdot \omega_b 
\ +\ \frac{1}{1 + d}\nonumber  \quad \text{d is the euclidian distance between the box and the goal}
\end{align}
$$

$$
\begin{align}
r_h &= -5 , & \text{if the box hits a wall} \nonumber 
\end{align}
$$

So the general form of rewward is:

$$
\begin{align}
r = 
\begin{cases}
r_t + r_v, & \text{if in Phase 1 (Reaching)} \\
r_p + r_h, & \text{if in Phase 2 (Pushing)}
\end{cases}
\end{align}
$$


---
## 📈 Results

| Framework | Convergence (Episodes) | Smoothness | Robustness |
|------------|------------------------|-------------|-------------|
| Baseline Critic | ~3700 | ⚪⚪⚪⚫⚫ | ⚪⚪⚪⚫⚫ |
| Extended Baseline Critic | ~6700 | ⚪⚪⚪⚪⚪ | ⚪⚪⚫⚫⚫ |
| Attention-Based Critic | ~1200 | ⚪⚪⚪⚪⚪ | ⚪⚪⚪⚪⚫ |

---
