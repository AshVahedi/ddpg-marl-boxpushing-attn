import numpy as np
from costfunction import phase1_cost, _compute_p_ref

from optimizerfunction import solve_phase1_nmpc
from Environment import UnicyclePushBoxEnv

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def animate_from_state(X, goal, box_size, env_size, episode=0, interval=30):

    """
    X: np.ndarray of shape (10, L)
       state trajectory

    goal: [xg, yg]
    box_size: [width, height]
    env_size: scalar
    """

    save_dir = "results_animation"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"animation_ep{episode}.gif")

    # --- Extract trajectories ---
    agent_traj = X[0:2, :].T        # (L,2)
    box_traj = X[4:6, :].T          # (L,2)
    box_theta = X[6, :]             # (L,)

    L = agent_traj.shape[0]

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env_size)
    ax.set_ylim(0, env_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Episode {episode}")
    ax.grid(True)

    # Goal
    ax.plot(goal[0], goal[1], 'bx', markersize=10, label='Goal')

    # Agent
    agent_dot, = ax.plot([], [], 'ro', label='Agent')

    # Trajectory (optional dashed line)
    ax.plot(agent_traj[:,0], agent_traj[:,1], '--', alpha=0.3)

    # Box patch
    box_patch = plt.Polygon([[0, 0]], closed=True, fc='gray', ec='k', alpha=0.5)
    ax.add_patch(box_patch)

    ax.legend()

    # --- Box geometry ---
    w, h = box_size[0] / 2.0, box_size[1] / 2.0

    local_corners = np.array([
        [-w, -h],
        [ w, -h],
        [ w,  h],
        [-w,  h]
    ])

    # --- Update function ---
    def update(frame):

        # Agent
        agent_dot.set_data(
            [agent_traj[frame, 0]],
            [agent_traj[frame, 1]]
        )

        # Box rotation
        theta = box_theta[frame]
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        rotated = (R @ local_corners.T).T + box_traj[frame]
        box_patch.set_xy(rotated)

        return agent_dot, box_patch

    # --- Animation ---
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=L,
        interval=interval,
        blit=True,
        repeat=False
    )

    ani.save(save_path, writer='pillow', fps=1000 // interval)
    plt.close(fig)


env = UnicyclePushBoxEnv()

params = {
    "m": 1.0,
    "mu_a": 4.5,
    "v_max": 5.0,
    "env_size": 40.0,
    "f_max": 8.0,
    "omega_max": np.pi / 10,
    "dt" :0.1
}
max_step = 500 
N=15
ell = 2.5

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


env.reset()
states = env.get_full_state()
goal_pos = env.goal
u_prev = np.array([0,0],dtype=np.float32)
done = False
box_reached = False
step = 0
s=[states.copy()]
while  not box_reached and step <=max_step:
    x0 =states[:4]
    box_pos =states[4:6]
    u ,res =  solve_phase1_nmpc(x0,
                                box_pos,
                                goal_pos,
                                u_prev,
                                N,
                                params["dt"],
                                ell,
                                weights,
                                params)
    p_ref = _compute_p_ref(env.box_pos,env.goal,ell)
    u_apply = u[0]
    states, done, goal_reached, box_reached = env.step(u_apply)
    
    s.append(states.copy()) 
    u_prev = u_apply
    step +=1



X = np.array(s).T   # shape (10, L)

animate_from_state(
    X,
    goal=goal_pos,
    box_size=[4.0, 2.0],
    env_size=40,
    episode=0
)