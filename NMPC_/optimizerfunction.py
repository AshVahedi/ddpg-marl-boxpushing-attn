from scipy.optimize import minimize
import numpy as np
from costfunction import phase1_cost


def solve_phase1_nmpc(x0, box_pos, goal_pos, u_prev, N, dt, ell, weights, params):
    U0 = np.zeros((N, 2))
    U0[:, 0] = 2.0
    U0[:, 1] = 0.0
    U0 = U0.flatten()

    bounds = []
    for _ in range(N):
        bounds.append((0.0, params["f_max"]))
        bounds.append((-params["omega_max"], params["omega_max"]))

    res = minimize(
        phase1_cost,
        U0,
        args=(x0, box_pos, goal_pos, u_prev, N, dt, ell, weights, params),
        method="SLSQP",
        bounds=bounds
    )

    U_opt = res.x.reshape(N, 2)
    return U_opt, res