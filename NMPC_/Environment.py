import numpy as np
import matplotlib.pyplot as plt

class UnicyclePushBoxEnv:
    """
        Continuous-time 2D environment for a single unicycle agent interacting with a rigid box.

        The system operates in two implicit phases:
        1) Free-motion phase: the agent moves under second-order unicycle dynamics.
        2) Pushing phase: once contact is established, the agent is rigidly attached to the box,
        and applies force/torque that drives the box dynamics.

        Dynamics:
        - Agent: second-order system with force input and angular rate control.
        - Box: rigid body with translational and rotational dynamics, including friction.
        - Integration: Runge-Kutta 4 (RK4) for agent dynamics; box dynamics approximated via RK4 on velocities.

        Notes:
        - Internal state includes velocities (agent and box), which are required for prediction (e.g., NMPC).
        - Observation returned to external users may omit velocities (partially observable setting).
        - Contact is approximated using discrete boundary points (corners and midpoints).
    """

    def __init__(self, agent_radius=1, dt=0.1, num_agents =1,max_steps=500, box_size=(4.0, 2.0), mode = "NMPC"):

        """
        Initialize the environment parameters and internal state.

        Defines:
        - Agent parameters: mass, friction, force limits, angular velocity limits.
        - Box parameters: mass, inertia, translational and rotational friction.
        - Simulation parameters: time step, environment size.
        - Geometry: box dimensions and agent interaction radius.
        - Goal position for the box.

        Internal state variables (initialized later in reset()):
        - Agent: position, heading, velocity, attachment status.
        - Box: position, orientation, linear velocity, angular velocity.

        The environment supports multiple agents, but is typically used with a single agent
        for NMPC-based control design.
        """

        self.mode = mode
        self.dt = dt
        self.max_steps = max_steps
        self.agent_radius = agent_radius
        self.box_size = np.array(box_size)
        self.env_size = 40.0
        self.goal = np.array([35.0, 15.0])

        self.agent_mass = 1.0
        self.agent_friction = 4.5
        self.agent_force_max =np.array([8.0],dtype=np.float32)
        self.agent_omega_max = np.array([np.pi / 10],dtype=np.float32)

        self.box_friction_linear = 6.0
        self.box_friction_rotary = 9.8
        self.box_mass = 20
        self.box_inertia = (1/12) * self.box_mass * (self.box_size[0]**2 + self.box_size[1]**2)

        self.num_agents = num_agents
        self.terminal_reward = 10

        self.contact_margin = 0.2 
        self.contact_threshold = self.agent_radius + self.contact_margin
        self.reset()

    def get_full_state(self):
        """
    Return the full dynamic state of the system.

    State vector:
        s = [
            x_a, y_a, theta_a, v_a,
            x_b, y_b, theta_b,
            v_bx, v_by, omega_b
        ]

    where:
    - (x_a, y_a, theta_a, v_a): agent position, heading, and velocity
    - (x_b, y_b, theta_b): box position and orientation
    - (v_bx, v_by): box linear velocity
    - omega_b: box angular velocity

    Returns:
        np.ndarray: full state vector (dimension = 10)

    Notes:
    - This state is required for model-based prediction (e.g., NMPC).
    - Unlike observations, this includes all dynamic variables.
    - After attachment, agent velocity is typically zero but still included for consistency.
    """
        return np.array([
            self.agent_pos[0][0],
            self.agent_pos[0][1],
            self.agent_theta[0],
            self.agent_vel[0],

            self.box_pos[0],
            self.box_pos[1],
            self.box_theta,

            self.box_vel[0],
            self.box_vel[1],
            self.box_omega
        ])
    
    def get_observation(self):
        """
        Return a reduced observation of the system state.

        Observation vector:
            o = [x_a, y_a, theta_a, x_b, y_b, theta_b]

        where:
        - Agent position and heading are included.
        - Box position and orientation are included.
        - Velocities are omitted.

        Returns:
            np.ndarray: observation vector (dimension = 6)

        Notes:
        - This represents a partially observable state.
        - Not suitable for model-based control (e.g., NMPC) due to missing dynamics.
        - Useful for visualization, logging, or learning-based policies.
        """
        state = []
        for i in range(self.num_agents):
            state += [self.agent_pos[i][0], self.agent_pos[i][1], self.agent_theta[i]]
        state += [self.box_pos[0], self.box_pos[1], self.box_theta]
        return np.array(state, dtype=np.float32)

    def reset(self):
        """
    Reset the environment to an initial state.

    Initializes:
    - Agent state: position, heading angle, velocity, and attachment flag.
    - Box state: position, orientation, linear velocity, angular velocity.
    - Attachment geometry: cleared for all agents.

    Returns:
        np.ndarray: observation vector (typically partial state) for external use.

    Notes:
    - Agent and box initial positions may be fixed or randomized depending on implementation.
    - Velocities are reset to zero.
    - This method prepares the system for a new episode.
        """
        self.agent_pos = [np.random.uniform(2.0, 10.0, size=2) for _ in range(self.num_agents)]
        self.agent_theta = [np.random.uniform(0, 8*np.pi/10) for _ in range(self.num_agents)]
        self.agent_vel = np.zeros(self.num_agents)
        self.box_pos = np.array([20.0, 20.0])
        self.box_theta = np.random.uniform(-np.pi, np.pi)
        self.box_vel = np.zeros(2)
        self.box_omega = 0.0

        self.agent_attachment_dis = [np.zeros(2) for _ in range(self.num_agents)]
        self.agent_start_pos = self.agent_pos.copy()
        self.box_start_pos, _ = self.box_corners_and_midpoints

        self.dist = 50
        self.step_count = 0
        self.done = False
        self.attached = [0 for _ in range(self.num_agents)]
        self.goal_reached = False
        self.box_reached =  [0 for _ in range(self.num_agents)]

    def _unicycle_dynamics(self, state, action):
        mass = self.agent_mass
        friction_coeff = self.agent_friction

        def f(state, action):
            x, y, theta, v = state
            force, omega = action
            force = force.item()
            omega = omega.item()
            acc = (force - friction_coeff * v) / mass 
            dx = v * np.cos(theta)
            dy = v * np.sin(theta)
            dtheta = omega
            dv = acc
            return np.array([dx, dy, dtheta, dv])

        k1 = f(state, action)
        k2 = f(state + 0.5 * self.dt * k1, action)
        k3 = f(state + 0.5 * self.dt * k2, action)
        k4 = f(state + self.dt * k3, action)

        next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state [3] =  max(0,next_state[3])
        return next_state

    def _box_dynamics(self, x, u, r_local):
        """
        x = [x_b, y_b, theta_b, vbx, vby, omega_b, theta_a]
        u = [f, omega_a]
        """
        
        xb, yb, theta_b, vbx, vby, omega_b, theta_a = x
        f, omega_a = u


        # --- Force direction ---
        Fx = f * np.cos(theta_a)
        Fy = f * np.sin(theta_a)

        # --- Rotate attachment vector ---
        R = self.box_rotation_matrix

        r = R @ r_local

        # --- Torque ---
        torque = r[0] * Fy - r[1] * Fx

        # --- Dynamics ---
        dx_b = vbx
        dy_b = vby
        dtheta_b = omega_b

        dvbx = (Fx - self.box_friction_linear * vbx) / self.box_mass
        dvby = (Fy - self.box_friction_linear * vby) / self.box_mass
        
        domega_b = (torque - self.box_friction_rotary * omega_b) / self.box_inertia

        dtheta_a = omega_a

        return np.array([
            dx_b, dy_b, dtheta_b,
            dvbx.item(), dvby.item(), domega_b.item(),
            dtheta_a.item()
        ])
    
    def _rk4_step_phase2(self, x, u, r_local):
        """
        RK4 discretization for Phase 2
        """

        dt = self.dt

        k1 = self._box_dynamics(x, u, r_local)
        k2 = self._box_dynamics(x + 0.5 * dt * k1, u, r_local)
        k3 = self._box_dynamics(x + 0.5 * dt * k2, u, r_local)
        k4 = self._box_dynamics(x + dt * k3, u, r_local)

        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # --- wrap angles ---
        x_next[2] = self._normalize_angle(x_next[2])   # box angle
        x_next[6] = self._normalize_angle(x_next[6])   # agent heading

        # --- velocity clamp ---
        v_max = 5.0  
        x_next[3] = np.clip(x_next[3], -v_max, v_max)
        x_next[4] = np.clip(x_next[4], -v_max, v_max)

        # --- angular clamp (optional) ---
        x_next[5] = np.clip(x_next[5], -10.0, 10.0)

        # --- workspace clamp ---
        x_next[0:2] = np.clip(x_next[0:2], 0.0, self.env_size)

        return x_next
        
    def _distance_to_box(self):
        ''' finds the min distance between agent and 
        the known points of the box ( mid points and corners)'''
        corners, midpoints = self.box_corners_and_midpoints
        all_points = np.vstack((corners, midpoints))
        dists = [np.linalg.norm(self.agent_pos - pt) for pt in all_points]
        return min(dists)

    def step(self, actions):  # actions is a flat array [f1, w1, f2, w2, ...]
        for i in range(self.num_agents):
            if self.mode == "RL":
                force = actions[2*i] * self.agent_force_max + self.agent_force_offset
            else:  # NMPC
                force = np.clip(actions[2*i], 0, self.agent_force_max)

            if self.mode == "RL":
                omega = actions[2*i+1] * self.agent_omega_max 
            else:  # NMPC
                force = np.clip(actions[2*i+1], -self.agent_omega_max, self.agent_omega_max)
        self.step_count += 1
        reward = 0.0
        done = False

        # Store all force vectors and torque contributions
        total_force = np.zeros(2)
        total_torque = 0.0

        R_box = self.box_rotation_matrix

        for i in range(self.num_agents):
            if self.mode == "RL":
                force = actions[2*i] * self.agent_force_max + self.agent_force_offset
                omega = actions[2*i+1] * self.agent_omega_max
            else:  # NMPC
                force = np.clip(actions[2*i], 0, self.agent_force_max)
                omega =  np.clip(actions[2*i+1],-self.agent_omega_max,  self.agent_omega_max)

            if not self.attached[i]:
                current_state_agent = np.array([
                    *self.agent_pos[i],
                    self.agent_theta[i],
                    self.agent_vel[i]  # use the agent's actual velocity
                ])
                next_state = self._unicycle_dynamics(current_state_agent, [force, omega])
                self.agent_pos[i] = np.clip(next_state[:2], 0, self.env_size) 
                self.agent_theta[i] = self._normalize_angle(next_state[2])
                self.agent_vel[i] = next_state[3]

                # Check attachment
                corners, midpoints = self.box_corners_and_midpoints
                all_points = np.vstack((corners, midpoints))
                dists = [np.linalg.norm(self.agent_pos[i] - pt) for pt in all_points]
                min_dist = min(dists)
                if  not self.box_reached[i] and min_dist < self.contact_threshold:
                    self.attached[i] = 1
                    self.box_reached[i] = True
                    self.agent_attachment_dis[i] = self.agent_pos[i] - self.box_pos
                    # --- Attachment vector (fixed) ---
                    r_local = self.agent_attachment_dis[i]
                    
            else:
                # --- Freeze agent velocity ---
                self.agent_vel[i] = 0.0

                # --- Build Phase 2 state ---
                x = np.array([
                    self.box_pos[0],
                    self.box_pos[1],
                    self.box_theta,
                    self.box_vel[0],
                    self.box_vel[1],
                    self.box_omega,
                    self.agent_theta[i]
                ])

                # --- Control ---
                u = np.array([force, omega])
                
                # --- Attachment vector (fixed) ---
                r_local = self.agent_attachment_dis[i]

                # --- RK4 update ---
                x_next = self._rk4_step_phase2(x, u, r_local)

                # --- Write back box state ---
                self.box_pos = x_next[0:2]
                self.box_theta = self._normalize_angle(x_next[2])
                self.box_vel = x_next[3:5]
                self.box_omega = x_next[5]

                # --- Update agent heading ---
                self.agent_theta[i] = self._normalize_angle(x_next[6])

                # --- Attach agent position to box ---
                R = self.box_rotation_matrix

                self.agent_pos[i] =self.box_pos + R @ r_local
        
        if self.step_count >= self.max_steps:
            done = True

        return self.get_full_state(), done, self.goal_reached, self.box_reached[0]

    def plot_current(self,trajectory=0):
        plt.figure(figsize=(6,6))

        corners, _ = self.box_corners_and_midpoints
        if trajectory:
             for traj in trajectory:
                traj = np.array(traj)

                plt.plot(traj[:, 0], traj[:, 1], 'r-', lw=1.5, label='Agent Trajectory')

        # Draw box
        plt.plot(*np.append(corners, [corners[0]], axis=0).T, 'k-', lw=2, label='Box')
        # Draw agent current pos
        plt.plot(self.agent_pos[0], self.agent_pos[1], 'ro', label='Agent')
        # Draw agent start pos
        plt.plot(self.agent_start_pos[0], self.agent_start_pos[1], 'go', label='Agent Start')
        # Draw goal
        plt.plot(self.goal[0], self.goal[1], 'bx', label='Goal', markersize=10)

        plt.plot(*np.append(self.box_start_pos, [self.box_start_pos[0]], axis=0).T, 'b-', lw=2, label='Box_initial')        

        plt.xlim(0, self.env_size)
        plt.ylim(0, self.env_size)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Current Environment State')
        plt.grid(True)
        plt.legend()
        plt.show()


    def _normalize_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    @property
    def state(self):
        flat_agent_state = []
        for i in range(self.num_agents):
            flat_agent_state += [*self.agent_pos[i], self.agent_theta[i], self.agent_vel[i]]
        return np.array([
            *flat_agent_state,
            *self.box_pos,
            self.box_theta
        ], dtype=np.float32)

    @property
    def box_center(self):
        return self.box_pos.copy()

    @property
    def box_rotation_matrix(self):
        return np.array([
            [np.cos(self.box_theta), -np.sin(self.box_theta)],
            [np.sin(self.box_theta),  np.cos(self.box_theta)]
        ])

    @property
    def box_corners_and_midpoints(self):
        w, h = self.box_size / 2.0
        R = np.array([
            [np.cos(self.box_theta), -np.sin(self.box_theta)],
            [np.sin(self.box_theta),  np.cos(self.box_theta)]
        ])

        local = np.array([
            [-w, -h],
            [ w, -h],
            [ w,  h],
            [-w,  h]
        ])
        global_corners = self.box_pos + (R @ local.T).T
        midpoints = [(global_corners[i] + global_corners[(i+1)%4]) / 2 for i in range(4)]
        return global_corners, np.array(midpoints)
