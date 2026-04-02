
import numpy as np


def _wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def _dynamics(x, u, params):
    m = params["m"]
    mu_a = params["mu_a"]

    X, Y, theta, v = x
    f, omega = u

    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = omega
    dv = (f - mu_a * v) / m

    return np.array([dx, dy, dtheta, dv])


def _rk4_step(x, u, dt, params):
    k1 = _dynamics(x, u, params)
    k2 = _dynamics(x + 0.5 * dt * k1, u, params)
    k3 = _dynamics(x + 0.5 * dt * k2, u, params)
    k4 = _dynamics(x + dt * k3, u, params)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    x_next[0:2] = np.clip(x_next[0:2], 0.0, params["env_size"])
    x_next[2] = _wrap_to_pi(x_next[2])
    x_next[3] = max(0.0, x_next[3])

    return x_next

def _compute_p_ref(box_pos, goal_pos, ell):

    e_bg = goal_pos-box_pos
    norm = np.linalg.norm(e_bg)
    if norm < 1e-6:
        return box_pos.copy()
    unit_e_bg = e_bg/norm
    
    return box_pos - unit_e_bg*ell

def _compute_theta_ref(box_pos, goal_pos):

    e_bg = goal_pos -box_pos
    return np.arctan2(e_bg[1], e_bg[0])
    
def phase1_cost(U_flat, x0, box_pos, goal_pos, u_prev, N, dt, ell, weights, params):
    
    J_pos = 0.0
    J_theta = 0.0
    J_v = 0.0
    J_u = 0.0
    J_du = 0.0
    U = U_flat.reshape(N, 2)

    w_p = weights["w_p"]
    w_theta = weights["w_theta"]
    w_v = weights["w_v"]
    w_f = weights["w_f"]
    w_omega = weights["w_omega"]
    w_df = weights["w_df"]
    w_domega = weights["w_domega"]
    w_p_f = weights["w_p_f"]
    w_theta_f = weights["w_theta_f"]
    w_v_f = weights["w_v_f"]

    x = x0.copy()
    J = 0.0

    p_ref = _compute_p_ref(box_pos, goal_pos, ell)
    theta_ref = _compute_theta_ref(box_pos, goal_pos)

    for i in range(N):
        u = U[i]
        f = u[0]
        omega = u[1]

        p = x[:2]
        theta = x[2]
        v = x[3]

        e_p = p - p_ref
        e_theta = _wrap_to_pi(theta - theta_ref)

        if i == 0:
            du = u - u_prev
        else:
            du = u - U[i - 1]

        J += (
            w_p * np.dot(e_p, e_p)
            + w_theta * e_theta**2
            + w_v * v**2
            + w_f * f**2
            + w_omega * omega**2
            + w_df * du[0]**2
            + w_domega * du[1]**2
        )

        x = _rk4_step(x, u, dt, params)
        pos_term = w_p * np.dot(e_p, e_p)
        theta_term = w_theta * e_theta**2
        v_term = w_v * v**2
        u_term = w_f * f**2 + w_omega * omega**2
        du_term = w_df * du[0]**2 + w_domega * du[1]**2

        J_pos += pos_term
        J_theta += theta_term
        J_v += v_term
        J_u += u_term
        J_du += du_term

        J += pos_term + theta_term + v_term + u_term + du_term

    p = x[:2]
    theta = x[2]
    v = x[3]
    e_p = p - p_ref
    e_theta = _wrap_to_pi(theta - theta_ref)

    J += (
        w_p_f * np.dot(e_p, e_p)
        + w_theta_f * e_theta**2
        + w_v_f * v**2
    )

    return J

def phase2_cost(U_flat, x0, goal, u_prev, N, dt, weights, params, r_local):
    """
    Phase 2 NMPC cost (pushing)

    x0: [x_b, y_b, theta_b, vbx, vby, omega_b, theta_a]
    U_flat: flattened control sequence (2N,)
    """

    U = U_flat.reshape(N, 2)

    x = x0.copy()
    J = 0.0

    for k in range(N):

        u = U[k]

        # --- box position ---
        box_pos = x[0:2]

        # --- tracking cost ---
        e = box_pos - goal
        J += weights["w_pos"] * np.dot(e, e)

        # --- control effort ---
        J += weights["w_f"] * (u[0]**2)
        J += weights["w_omega"] * (u[1]**2)

        # --- smoothness ---
        du = u - u_prev
        J += weights["w_df"] * (du[0]**2)
        J += weights["w_domega"] * (du[1]**2)

        # --- rollout ---
        x = rk4_step_phase2(x, u, dt, params, r_local)

        u_prev = u

    # --- terminal cost ---
    e_terminal = x[0:2] - goal
    J += weights["w_pos_f"] * np.dot(e_terminal, e_terminal)

    return float(J)