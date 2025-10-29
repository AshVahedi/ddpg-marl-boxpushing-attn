import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None  # store for later use

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # identical to nn.TransformerEncoderLayer forward, but we store attn weights
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn_weights = attn_weights.detach()  # shape [batch, nhead, tokens, tokens]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=24):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state,printer =0):
        a1 = self.relu(self.fc1(state))  # First hidden layer activations
        a2 = self.relu(self.fc2(a1))     # Second hidden layer activations
        out = self.tanh(self.out(a2))    # Output (actions)
        if printer: 
            return out, a1, a2
        else:
            return out

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))



class AttentionCritic(nn.Module):
    def __init__(self, num_agents, input_dim_per_agent=3, embed_dim=16, nhead=1):
        super().__init__()
        self.num_agents = num_agents
        self.total_tokens = num_agents + 2  # agents + box_cm + goal
        self.embed_dim = embed_dim

        # Encoders for inputs
        self.token_encoder = nn.Linear(input_dim_per_agent + 2, embed_dim)  # agent state + action
        self.box_encoder = nn.Linear(3, embed_dim)
        self.goal_encoder = nn.Linear(2, embed_dim)

        # Attention block
        self.attn = nn.TransformerEncoder(
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True),
            num_layers=1
        )
          # A single-layer transformer encoder (multi-head attention + feedforward + residuals
          # Input is a sequence of tokens (agents + box + goal), each of shape

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * self.total_tokens, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ) # predicts Q_value

    def forward(self, state, action, evaluator = 0):
        # state: [B, 3N + 3 + 2], action: [B, 2N]
        B = state.size(0)
        agent_state = state[:, :3 * self.num_agents].view(B, self.num_agents, 3)
        agent_action = action.view(B, self.num_agents, 2)
        agent_input = torch.cat([agent_state, agent_action], dim=-1)  # [B, N, 5]
        agent_embed = self.token_encoder(agent_input)  # [B, N, D]

        box_state = state[:, 3 * self.num_agents : 3 * self.num_agents + 3]  # [B, 3]
        goal_state = torch.FloatTensor([35,35]).unsqueeze(0).repeat(B, 1) # [B, 2]

        box_embed = self.box_encoder(box_state).unsqueeze(1)  # [B, 1, D]
        goal_embed = self.goal_encoder(goal_state).unsqueeze(1)  # [B, 1, D]



        tokens = torch.cat([agent_embed, box_embed, goal_embed], dim=1)  # [B, N+2, D]
        attn_out = self.attn(tokens)  # [B, N+2, D]
        
        attn_weights = self.attn.layers[0].last_attn_weights  # [B, nhead, N+2, N+2]

        flat = attn_out.view(B, -1)
        value = self.value_head(flat)

        if evaluator:
            return tokens, attn_out, attn_weights, value
        else:
            return value
