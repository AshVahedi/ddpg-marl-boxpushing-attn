import torch
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Optional fallback

import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PolicyEvaluator:
    def __init__(self, env_class, actor, device, action_bounds):
        """
        Parameters:
        - env_class: class reference (not instance), e.g., UnicycleGoalEnv
        - actor: trained actor network
        - device: torch device (cpu or cuda)
        - action_bounds: np.array([max_force, max_omega])
        """
        self.env_class = env_class
        self.actor = actor
        self.device = device
        self.action_bounds = action_bounds
        self.env_size =env_class().env_size

    def evaluate(self, max_steps,printer = 0):
        self.actor.eval()
        env = self.env_class()

        num_agents = env.num_agents

        trajectories = [[] for _ in range(num_agents)]
        box_positions =[]
        box_theta =[]

        state = env.reset_evaluation()

        corners_initial ,_= self.box_corners_and_midpoints(env.box_theta,env.box_size,env.box_pos)
        for i in range(num_agents):
            x = state[3*i]
            y = state[3*i+1]
            trajectories[i].append([x, y])        
            
        box_positions.append(env.box_pos.copy())
        box_theta.append(env.box_theta)

        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor(state_tensor).cpu().numpy().flatten()

            state, reward, done,goal_reached,box_reached = env.step(action)
            if printer:
                print(f"Step {step:3d} | Network Output: {action}")
            
            
            for i in range(num_agents):
                x = state[3*i]
                y = state[3*i+1]
                trajectories[i].append([x, y])
            box_positions.append(env.box_pos.copy())
            box_theta.append(env.box_theta)

            if done:
                
                corners_end,_ =self.box_corners_and_midpoints(env.box_theta,env.box_size,env.box_pos)
                break

        self.actor.train()
        return [np.array(traj) for traj in trajectories], [corners_initial, corners_end], box_positions, box_theta, goal_reached, box_reached


    def save(self, trajectories, box_positions, goal, episode, save_dir="results"):
        """Terminates learning and saves actor and final trajectory plot.

        Parameters:
        - trajectories: List of arrays of trajectory points
        - goal: Final goal point
        - save_dir: Directory to store outputs
        - actor_name: Filename for saving actor weights
        - plot_name: Filename for saving trajectory plot
        """
        env_size = self.env_size
        os.makedirs(save_dir, exist_ok=True)
        actor_name =f"actor_{episode}.pth"
        plot_name = f"trajectory_{episode}.png"
        # Save actor
        actor_path = os.path.join(save_dir, actor_name)
        torch.save(self.actor.state_dict(), actor_path)

        # Save trajectory plot
        fig, ax = plt.subplots(figsize=(6, 6))
        corners_initial = box_positions[0]
        corners_end= box_positions[1]
        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], '-o', markersize=2)
            ax.plot(traj[0, 0], traj[0, 1], 'go', label='Start')
            ax.plot(traj[-1, 0], traj[-1, 1], 'ro', label='End')
        
        ax.plot(*np.append(corners_end, [corners_end[0]], axis=0).T, 'b-', lw=2, label='Box')
        ax.plot(*np.append(corners_initial, [corners_initial[0]], axis=0).T, 'k-', lw=2, label='Box')
        ax.plot(goal[0], goal[1], 'bx', markersize=10, label='Goal')
        ax.set_xlim(0, env_size)
        ax.set_ylim(0, env_size)
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"episode = {episode}")

        plot_path = os.path.join(save_dir, plot_name)
        plt.savefig(plot_path)
        plt.close(fig)
        
    def animate(self, agent_traj, box_traj, box_theta,p_2_a, goal, box_info,env_size, episode, interval=30):
        #p_2_a= params_to_animate as [tau,gamma,sigma,sigma_decay_rate,num_agents,emb]
        save_dir = "results_animation"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"animation_ep{episode}.gif")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, env_size)
        ax.set_ylim(0, env_size)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"t:{p_2_a[0]},g:{p_2_a[1]}, s:{p_2_a[2]},sdr:{p_2_a[3]}, NA:{p_2_a[4]},emb:{p_2_a[5]},Epi:{episode}")
        ax.grid(True)

        ax.plot(goal[0], goal[1], 'bx', markersize=10, label='Goal')

        agent_dots = [ax.plot([], [], 'ro', label=f'Agent {i}')[0] for i in range(len(agent_traj))]
        for traj in agent_traj:
            ax.plot([p[0] for p in traj], [p[1] for p in traj], linestyle='--', alpha=0.3)

        box_patch = plt.Polygon([[0, 0]], closed=True, fc='gray', ec='k', alpha=0.5)
        ax.add_patch(box_patch)
        ax.legend()

        # Get box size from env instance
        w, h = box_info[0]/2,box_info[1]/2
        local_corners = np.array([
            [-w, -h],
            [ w, -h],
            [ w,  h],
            [-w,  h]
        ])


        def update(frame):
            for i, dot in enumerate(agent_dots):
                dot.set_data(agent_traj[i][frame][0], agent_traj[i][frame][1])

            R = np.array([
                [np.cos(box_theta[frame]), -np.sin(box_theta[frame])],
                [np.sin(box_theta[frame]),  np.cos(box_theta[frame])]
            ])
            rotated = (R @ local_corners.T).T + box_traj[frame]
            box_patch.set_xy(rotated)

            return (*agent_dots, box_patch)
        


        ani = animation.FuncAnimation(
            fig, update, frames=len(box_traj), blit=True,
            interval=interval, repeat=False
        )

        ani.save(save_path, writer='pillow', fps=1000 // interval)
        plt.close(fig)

    def box_corners_and_midpoints(self,box_theta,box_size,box_pos):
        w, h =box_size / 2.0
        R = np.array([
            [np.cos(box_theta), -np.sin(box_theta)],
            [np.sin(box_theta),  np.cos(box_theta)]
        ])

        local = np.array([
            [-w, -h],
            [ w, -h],
            [ w,  h],
            [-w,  h]
        ])
        global_corners = box_pos + (R @ local.T).T
        midpoints = [(global_corners[i] + global_corners[(i+1)%4]) / 2 for i in range(4)]
        return global_corners, np.array(midpoints)
