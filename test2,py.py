
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
from torchviz import make_dot
import hiddenlayer as hl




actor_path = "results/actor_577.pth"
critic_path = "results/critic_577.pth"
num_agents = 2
obs_dim = 9
action_dim = 4 
agent_radius =1
max_steps = 500
embeded_length = 16


env = UnicyclePushBoxEnv(agent_radius,num_agents = num_agents)

# Step 1: Re-create the model architecture (must match original!)
loaded_actor = Actor(obs_dim, action_dim).to(device)
loaded_critic = AttentionCritic(num_agents,embed_dim =16).to(device)
# Step 2: Load saved weights
loaded_actor.load_state_dict(torch.load(actor_path, map_location=device))
loaded_critic.load_state_dict(torch.load(critic_path, map_location=device))
loaded_critic.eval()
# Step 3: Set to evaluation mode (important for dropout/batchnorm)
action_bounds = np.array([env.agent_force_max, env.agent_omega_max])

evaluator = PolicyEvaluator(UnicyclePushBoxEnv, loaded_actor,loaded_critic, device, action_bounds)


state = torch.randn(1, 9, requires_grad=True)
action = torch.randn(1, 4, requires_grad=True)
value = loaded_critic(state, action)

torch.onnx.export(
    loaded_critic,
    (state, action),
    "attention_critic.onnx",
    export_params=True,
    opset_version=14,  # ← this is the fix
    input_names=["state", "action"],
    output_names=["Q_value"],
    dynamic_axes={
        "state": {0: "batch_size"},
        "action": {0: "batch_size"},
        "Q_value": {0: "batch_size"}
    }
)

make_dot(value, params=dict(list(loaded_critic.named_parameters()))).render("critic_graph", format="png")
