import numpy as np
import matplotlib.pyplot as plt

class UnicyclePushBoxEnv:
    def __init__(self, agent_radius=1, dt=0.1, num_agents =2,max_steps=500, box_size=(4.0, 2.0)):
        '''
        initializing the environment for two cooperative agents
        '''
        self.dt = dt
        self.max_steps = max_steps
        self.agent_radius = agent_radius
        self.box_size = np.array(box_size)
        self.env_size = 40.0
        self.goal = np.array([35.0, 35.0])

        self.agent_mass = 1.0
        self.agent_friction = 4.5
        self.agent_force_max = 8.0
        self.agent_force_offset = 3.0
        self.agent_omega_max = np.pi / 10

        self.box_friction_linear = 6.0
        self.box_friction_rotary = 9.8
        self.box_mass = 20
        self.box_inertia = (1/12) * self.box_mass * (self.box_size[0]**2 + self.box_size[1]**2)

        self.num_agents = num_agents
        self.terminal_reward = 10

        self.reset(agent_radius)

    def reset(self, agent_radius):
        self.agent_radius = agent_radius
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

        return self._get_state()

    def reset_evaluation(self):
        self.agent_pos = [np.array([4.0, y]) for y in np.linspace(4.0, 30.0, self.num_agents)]
        self.agent_theta = [0.0 for _ in range(self.num_agents)]
        self.agent_attachment_dis = [np.zeros(2) for _ in range(self.num_agents)]
        self.agent_start_pos = self.agent_pos.copy()

        self.box_pos = np.array([17.0, 17.0])
        self.box_theta = 0.0
        self.box_vel = np.zeros(2)
        self.box_omega = 0.0
        self.box_start_pos, _ = self.box_corners_and_midpoints

        self.step_count = 0
        self.attached = [0 for _ in range(self.num_agents)]
        self.goal_reached = False
        self.box_reached = [0 for _ in range(self.num_agents)]

        return self._get_state()

    def _get_state(self):
        state = []
        for i in range(self.num_agents):
            state += [self.agent_pos[i][0], self.agent_pos[i][1], self.agent_theta[i]]
        state += [self.box_pos[0], self.box_pos[1], self.box_theta]
        return np.array(state, dtype=np.float32)

    def _unicycle_dynamics(self, state, action):
        mass = self.agent_mass
        friction_coeff = self.agent_friction

        def f(state, action):
            x, y, theta, v = state
            force, omega = action
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
        return next_state

    def _box_dynamics(self, pos, vel, theta, omega, force_vec, torque):
        '''
            pos: box's CoM position
            vel: box's CoM velocity
            theta:box's orientation
            omega: box's angular velocity
            force_vec: [force_x , force_y] applied to the box
            agent_pos: agent's position 
        '''
       
        m = self.box_mass   
        mu = self.box_friction_linear
        mu_rot = self.box_friction_rotary
        I = self.box_inertia

        def linear_acc(v, F):
            return (F - mu * v) / m

        def angular_acc(w, tau):
            return (tau - mu_rot * w) / I

        # RK4 for linear motion
        k1_v = self.dt * linear_acc(vel, force_vec)
        k2_v = self.dt * linear_acc(vel + 0.5 * k1_v, force_vec)
        k3_v = self.dt * linear_acc(vel + 0.5 * k2_v, force_vec)
        k4_v = self.dt * linear_acc(vel + k3_v, force_vec)
        new_vel = vel + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        new_pos = pos + self.dt * new_vel

        # RK4 for angular motion
        k1_w = self.dt * angular_acc(omega, torque)
        k2_w = self.dt * angular_acc(omega + 0.5 * k1_w, torque)
        k3_w = self.dt * angular_acc(omega + 0.5 * k2_w, torque)
        k4_w = self.dt * angular_acc(omega + k3_w, torque)
        new_omega = omega + (k1_w + 2 * k2_w + 2 * k3_w + k4_w) / 6
        new_theta = self._normalize_angle(theta + self.dt * new_omega)

        return new_pos, new_vel, new_theta, new_omega


    def _distance_to_box(self):
        ''' finds the min distance between agent and 
        the known points of the box ( mid points and corners)'''
        corners, midpoints = self.box_corners_and_midpoints
        all_points = np.vstack((corners, midpoints))
        dists = [np.linalg.norm(self.agent_pos - pt) for pt in all_points]
        return min(dists)

    def step(self, actions):  # actions is a flat array [f1, w1, f2, w2, ...]
        '''
        actions: list of actions, one per agent. Each action is [force, omega] scaled in [-1,1].
        '''
        self.step_count += 1
        reward = 0.0
        done = False

        # Store all force vectors and torque contributions
        total_force = np.zeros(2)
        total_torque = 0.0

        R_box = self.box_rotation_matrix

        for i in range(self.num_agents):
            force = actions[2 * i] * self.agent_force_max + self.agent_force_offset
            omega = actions[2 * i + 1] * self.agent_omega_max

            if not self.attached[i]:
                current_state_agent = np.array([
                    *self.agent_pos[i],
                    self.agent_theta[i],
                    self.agent_vel[i]  # use the agent's actual velocity
                ])
                next_state = self._unicycle_dynamics(current_state_agent, [force, omega])
                self.agent_pos[i] = next_state[:2]
                self.agent_theta[i] = self._normalize_angle(next_state[2])
                self.agent_vel[i] = next_state[3]
                if self.agent_vel[i]<-0.5:
                    reward+=-0.2
                # Check attachment
                corners, midpoints = self.box_corners_and_midpoints
                all_points = np.vstack((corners, midpoints))
                dists = [np.linalg.norm(self.agent_pos[i] - pt) for pt in all_points]
                min_dist = min(dists)
                if min_dist < self.agent_radius:
                    self.attached[i] = 1
                    self.box_reached[i] = 1
                    self.agent_attachment_dis[i] = self.agent_pos[i] - self.box_pos
                    reward += self.terminal_reward 

            else:
                # Attached agent applies force
                direction = np.array([np.cos(self.agent_theta[i]), np.sin(self.agent_theta[i])])
                force_vec = force * direction
                attach_world = self.box_pos + R_box @ self.agent_attachment_dis[i]
                torque_arm = attach_world - self.box_pos
                torque = torque_arm[0] * force_vec[1] - torque_arm[1] * force_vec[0]

                total_force += force_vec
                total_torque += torque
                self.agent_vel[i]= 0 # freeze the agent
        # Update box dynamics if all agents are attached

        if sum(self.attached) == self.num_agents:
            self.box_pos, self.box_vel, self.box_theta, self.box_omega = self._box_dynamics(
                self.box_pos, self.box_vel, self.box_theta, self.box_omega,
                total_force,total_torque  # dummy pos for compatibility
            )

            # Update attached agent positions
            R_box = self.box_rotation_matrix
            for i in range(self.num_agents):
                if self.attached[i]:
                    self.agent_pos[i] = self.box_pos + R_box @ self.agent_attachment_dis[i]

            dist_to_goal = np.linalg.norm(self.box_pos - self.goal)
            reward += 1.0 / (1.0 + dist_to_goal)
            reward += -0.01 * abs(self.box_omega)

            if dist_to_goal < 1.0:
                done = True
                self.goal_reached = True

            corners, _ = self.box_corners_and_midpoints
            for i in corners:
                if i[0] < 0 or i[0] > self.env_size or i[1] < 0 or i[1] > self.env_size:
                    reward = -self.terminal_reward / 10
                    done = True

            
        for i in range(self.num_agents):
            if (
                self.agent_pos[i][0] < 0 or
                self.agent_pos[i][1] < 0 
            ):
                reward = -self.terminal_reward / 2
                done = True
                    
                    
            if (
                self.agent_pos[i][0] > self.env_size or
                self.agent_pos[i][1] > self.env_size
            ):
                    reward = -self.terminal_reward / 5
                    done = True

        if self.step_count >= self.max_steps:
            done = True

        return self._get_state(), reward, done, self.goal_reached, self.box_reached

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
