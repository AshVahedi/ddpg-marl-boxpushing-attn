import numpy as np
from costfunction import phase1_cost

from optimizerfunction import solve_phase1_nmpc
from Environment import UnicyclePushBoxEnv



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
while not box_reached:
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
    u_apply = u[0]
    states, done, goal_reached, box_reached = env.step(u_apply)
    
    u_prev = u_apply