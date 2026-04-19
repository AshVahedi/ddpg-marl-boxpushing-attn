from scipy.optimize import minimize
import numpy as np
from costfunction import phase1_cost,phase2_cost


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


def solve_phase2_nmpc(x0, goal, u_prev, N, dt, weights, params, r_local):
    """
    Solve Phase 2 NMPC.

    x0      : [x_b, y_b, theta_b, vbx, vby, omega_b, theta_a]
    goal    : [xg, yg]
    u_prev  : previously applied control [f, omega]
    N       : horizon
    dt      : timestep
    weights : phase 2 weights dictionary
    params  : model/constraint parameters
    r_local : fixed attachment vector in box frame
    """

    # Initial guess
    U0 = np.zeros((N, 2))
    U0[:, 0] = 2.0   # moderate push
    U0[:, 1] = 0.0   # no turning initially
    U0 = U0.flatten()

    # Bounds
    bounds = []
    for _ in range(N):
        bounds.append((0.0, float(params["f_max"])))
        bounds.append((-float(params["omega_max"]), float(params["omega_max"])))

    res = minimize(
        phase2_cost,
        U0,
        args=(x0, goal, u_prev, N, dt, weights, params, r_local),
        method="SLSQP",
        bounds=bounds
    )

    U_opt = res.x.reshape(N, 2)
    return U_opt, res