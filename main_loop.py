import torch
import numpy as np
import random

from replaybuffer import ReplayBuffer
from network import Actor, Critic, AttentionCritic
from noisegenerator import OUNoise
from evaluator import PolicyEvaluator
from environment import UnicyclePushBoxEnv
from ddpg_loop import training_loop
from AttentionStructure import AttentionBlock, Embedding, AttentionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np
import torch
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For deterministic operations (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(44)  # or any fixed integer

# Hyperparameters

gamma = 0.99
tau = 0.008
lr=0.0002
buffer_capacity = 10000
sigma = 0.8; # noise generator sample


agent_radius = 1 # initiate agent s radius "IT MUST BE 1"
num_agents = 2
agent_state_dim = 3
box_dim =3
agent_action_dim = 2
goal_dim =2


# Env & network setup
env = UnicyclePushBoxEnv(agent_radius,num_agents = num_agents)
action_dim = agent_action_dim*num_agents # two different actions [force, phi(aka: steering angle) for agent 1, 2]
action_bounds = np.array([env.agent_force_max, env.agent_omega_max])
obs_dim = agent_state_dim * num_agents + box_dim # network input size [x_a_1, y_a_1, theta_a_1, x_a_2, y_a_2, theta_a_2,  x_b,y_b,theta_b]

# Networks

# Buffer and noise
buffer = ReplayBuffer(buffer_capacity)
noise = OUNoise(action_dim)
# evaluator initialization
decay_rate =0.9999  
embeded_length=8
# Training

    # Optimizers
                                        
actor = Actor(obs_dim, action_dim).to(device)

critic = AttentionModel(
                 num_agents,
                 agent_state_dim,   # e.g. 3  (x, y, theta)
                 agent_action_dim,  # e.g. 2  (force, omega)
                 box_dim,           # e.g. 3  (x_b, y_b, theta_b)
                 goal_dim,          # e.g. 2  (x_G, y_G)
                 embed_dim=8,
                 attn_dim=8,
                 critic_hidden_dim=16)

# critic = AttentionCritic(num_agents,embed_dim =embeded_length).to(device) attention-based critic using nn.multiheadtransformer

target_actor = Actor(obs_dim, action_dim).to(device)
target_critic = AttentionModel(
                 num_agents,
                 agent_state_dim,   # e.g. 3  (x, y, theta)
                 agent_action_dim,  # e.g. 2  (force, omega)
                 box_dim,           # e.g. 3  (x_b, y_b, theta_b)
                 goal_dim,          # e.g. 2  (x_G, y_G)
                 embed_dim=8,
                 attn_dim=8,
                 critic_hidden_dim=16)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

evaluator = PolicyEvaluator(UnicyclePushBoxEnv, actor, critic, device, action_bounds)

actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

w =training_loop(env,buffer,noise,evaluator,
            actor,critic,target_actor,target_critic,actor_opt,critic_opt,
            tau,gamma,sigma,
            num_agents,decay_rate,embeded_length)

np.save("results/weights.npy", w)
