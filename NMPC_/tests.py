import numpy as np
from costfunction import _rk4_step,phase1_cost,_compute_theta_ref
from Environment import UnicyclePushBoxEnv



env = UnicyclePushBoxEnv()
dt = 0.1
params = {
    "m": 1,
    "mu_a": 4.5,
    "v_max": 5,
    "env_size": 40.0
}
weights = {
    "w_p": 10.0,
    "w_theta": 5.0,
    "w_v": 0.5,

    "w_f": 0.1,
    "w_omega": 0.1,

    "w_df": 0.1,
    "w_domega": 0.1,

    "w_p_f": 20.0,
    "w_theta_f": 10.0,
    "w_v_f": 1.0
}

x0 = np.array([7.0, 7.0, 0.0, 0.0])
box_pos = np.array([10.0, 10.0])
goal_pos = np.array([15.0, 15.0])
ell= 2.5
N = 15

theta_ref = _compute_theta_ref(box_pos, goal_pos)

U_good = []
for _ in range(N):
    U_good.append([3.0, 0.0])   # push forward

U_good = np.array(U_good).flatten()
U_zero = np.zeros((N, 2)).flatten()
u_prev = np.array([0.0, 0.0])
U_rand = np.random.uniform(
    low=[0, -np.pi/10],
    high=[8, np.pi/10],
    size=(N,2)
).flatten()
J_zero = phase1_cost(U_zero, x0, box_pos, goal_pos, u_prev, N, dt, ell, weights, params)
J_good = phase1_cost(U_good, x0, box_pos, goal_pos, u_prev, N, dt, ell, weights, params)
J_rand = phase1_cost(U_rand, x0, box_pos, goal_pos, u_prev, N, dt, ell, weights, params)

print(J_zero)
print(J_good)
print(J_rand)

i = 0

# Force attachment manually
env.attached[i] = 1
env.box_reached[i] = True

# Define a clean attachment point (e.g., right side of box)
env.agent_attachment_dis[i] = np.array([2.0, 0.0])  # half-width

# Place agent correctly
env.agent_pos[i] = env.box_pos + env.box_rotation_matrix @ env.agent_attachment_dis[i]

# Set agent heading forward
env.agent_theta[i] = np.pi/2

for _ in range(50):
    action = np.array([5.0, 0.01])  # forward force, no rotation
    env.step(action)

    print("box pos:", env.box_pos, "theta:", env.box_theta )
    
    