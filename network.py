import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Assumes actions are in [-1, 1]
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True),
            num_layers=1
        ) # A single-layer transformer encoder (multi-head attention + feedforward + residuals
          # Input is a sequence of tokens (agents + box + goal), each of shape

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * self.total_tokens, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ) # predicts Q_value

    def forward(self, state, action):
        # state: [B, 3N + 3 + 2], action: [B, 2N]
        B = state.size(0)
        agent_state = state[:, :3 * self.num_agents].view(B, self.num_agents, 3)
        agent_action = action.view(B, self.num_agents, 2)
        agent_input = torch.cat([agent_state, agent_action], dim=-1)  # [B, N, 5]
        agent_embed = self.token_encoder(agent_input)  # [B, N, D]

        box_state = state[:, 3 * self.num_agents : 3 * self.num_agents + 3]  # [B, 3]
        goal_state = state[:, -2:]  # [B, 2]

        box_embed = self.box_encoder(box_state).unsqueeze(1)  # [B, 1, D]
        goal_embed = self.goal_encoder(goal_state).unsqueeze(1)  # [B, 1, D]

        tokens = torch.cat([agent_embed, box_embed, goal_embed], dim=1)  # [B, N+2, D]
        attn_out = self.attn(tokens)  # [B, N+2, D]

        flat = attn_out.view(B, -1)  # [B, (N+2)*D]
        value = self.value_head(flat)  # [B, 1]
        return value
