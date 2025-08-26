import torch
import numpy as np
import random

from replaybuffer import ReplayBuffer
from network import Actor, Critic, AttentionCritic
from noisegenerator import OUNoise
from evaluator import PolicyEvaluator
from environment import UnicyclePushBoxEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np
import torch


actor_path = "results/actor_0.pth"
num_agents = 2
obs_dim = 9
action_dim = 4 
agent_radius =1
max_steps = 500

env = UnicyclePushBoxEnv(agent_radius,num_agents = num_agents)

# Step 1: Re-create the model architecture (must match original!)
loaded_actor = Actor(obs_dim, action_dim).to(device)

# Step 2: Load saved weights
loaded_actor.load_state_dict(torch.load(actor_path, map_location=device))

# Step 3: Set to evaluation mode (important for dropout/batchnorm)
action_bounds = np.array([env.agent_force_max, env.agent_omega_max])

evaluator = PolicyEvaluator(UnicyclePushBoxEnv, loaded_actor, device, action_bounds)

trajectories, box_start_and_end, box_positions, box_theta, goal_reached ,box_reached= evaluator.evaluate(max_steps,printer =1)