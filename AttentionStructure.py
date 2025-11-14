import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Simple embedding MLP for entities (agents / box / goal) ----------
class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: (B, L, input_dim)  or  (B, input_dim)
        out = self.lin(x)
        return F.relu(out)

class Critic(nn.Module):
    def __init__(self, embed_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self,x):
        return self.net(x)


# ---------- Single-head self-attention with explicit Q, K, V ----------
class AttentionBlock(nn.Module):
    def __init__(self, dim, attn_dim):
        """
        dim      = embedding dimension of tokens
        attn_dim = internal attention dimension (d_k, d_v)
        """
        super().__init__()

        # explicit Q, K, V projections
        self.W_Q = nn.Linear(dim, attn_dim, bias=False)
        self.W_K = nn.Linear(dim, attn_dim, bias=False)
        self.W_V = nn.Linear(dim, attn_dim, bias=False)

        self.proj = nn.Linear(attn_dim, dim)   # back to token dim
        self.norm = nn.LayerNorm(dim)

        self.scale = attn_dim ** 0.5           # sqrt(d_k)

    def forward(self, x):
        """
        x: (B, L, D)
        returns:
          out  : (B, L, D)
          attn : (B, L, L)
        """
        Q = self.W_Q(x)   # (B, L, H)
        K = self.W_K(x)   # (B, L, H)
        V = self.W_V(x)   # (B, L, H)

        # scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / self.scale   # (B, L, L)
        attn   = F.softmax(scores, dim=-1)              # row-wise over keys

        context = attn @ V                              # (B, L, H)

        out = self.proj(context)                        # (B, L, D)
        out = self.norm(out + x)                        # residual + norm

        return out, attn


# ---------- Centralized Attention Critic ----------
class AttentionModel(nn.Module):
    """
    Args:
        num_agents        : number of agents in the system
        agent_state_dim   : dimension of each agent's state vector
        agent_action_dim  : dimension of each agent's action vector
        box_dim           : dimension of the box state vector
        goal_dim          : dimension of the goal position vector
        embed_dim         : embedding dimension for all entity tokens
        attn_dim          : internal attention dimension (d_k and d_v)
    """

    def __init__(self,
                 num_agents,
                 agent_state_dim,   # e.g. 3  (x, y, theta)
                 agent_action_dim,  # e.g. 2  (force, omega)
                 box_dim,           # e.g. 3  (x_b, y_b, theta_b)
                 goal_dim,          # e.g. 2  (x_G, y_G)
                 embed_dim=8,
                 attn_dim=8,
                 critic_hidden_dim = 16):


        super().__init__()

        self.num_agents       = num_agents
        self.agent_state_dim  = agent_state_dim
        self.agent_action_dim = agent_action_dim
        self.box_dim          = box_dim
        self.goal_dim         = goal_dim
        self.embed_dim        = embed_dim
        self.attn_dim         = attn_dim

        # ---- entity embeddings (your Embedding class) ----
        self.agent_embedding = Embedding(agent_state_dim + agent_action_dim, embed_dim)
        self.box_embedding   = Embedding(box_dim, embed_dim)
        self.goal_embedding  = Embedding(goal_dim, embed_dim)

        # ---- positional embedding for tokens: [agent_1, ..., agent_N, box, goal] ----
        self.num_tokens   = num_agents + 2
        self.pos_embedding = nn.Embedding(self.num_tokens, embed_dim)

        # ---- self-attention over all tokens ----
        self.attention = AttentionBlock(embed_dim, attn_dim)

        # ---- global query for pooling (like CLS token, but as a query) ----
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ---- final head to scalar Q ----
        self.output = Critic(embed_dim*self.num_tokens, critic_hidden_dim) 

    def forward(self, state, action, goal_pos):
        """
        state:  (B, S)
            layout assumed:
              [ agent_1_state (agent_state_dim),
                agent_2_state (agent_state_dim),
                ...,
                agent_N_state (agent_state_dim),
                box_state     (box_dim) ]

        action: (B, num_agents * agent_action_dim)
            flattened joint action

        goal_pos: (B, goal_dim)  OR  (goal_dim,) shared for batch

        returns:
            Q    : (B, 1)
            attn : (B, L, L) self-attention weights over tokens
        """
        B = state.size(0)
        N = self.num_agents

        # ---- split state into agents and box ----
        total_agent_state_dim = N * self.agent_state_dim
        agent_state_flat = state[:, :total_agent_state_dim]                 # (B, N * agent_state_dim)
        box_state        = state[:, total_agent_state_dim:
                                     total_agent_state_dim + self.box_dim]  # (B, box_dim)

        agent_state = agent_state_flat.view(B, N, self.agent_state_dim)     # (B, N, agent_state_dim)

        # ---- reshape actions per agent ----
        agent_action = action.view(B, N, self.agent_action_dim)             # (B, N, agent_action_dim)

        # ---- agent tokens ----
        agent_input = torch.cat([agent_state, agent_action], dim=-1)        # (B, N, state+action)
        agent_tokens = self.agent_embedding(agent_input)                    # (B, N, D)

        # ---- box token ----
        box_tokens = self.box_embedding(box_state).unsqueeze(1)             # (B, 1, D)

        # ---- goal token ----
        if goal_pos.dim() == 1:
            # broadcast single goal for batch
            goal_pos = goal_pos.unsqueeze(0).expand(B, -1)                  # (B, goal_dim)


        goal_tokens = self.goal_embedding(goal_pos).unsqueeze(1)            # (B, 1, D)

        # ---- concatenate all entity tokens ----
        tokens = torch.cat([agent_tokens, box_tokens, goal_tokens], dim=1)  # (B, L, D), L = N + 2

        B, L, D = tokens.shape

        # ---- add positional embeddings ----
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)  # (B, L)
        pos_emb = self.pos_embedding(pos_ids)                                       # (B, L, D)
        tokens = tokens + pos_emb                                                   # (B, L, D)

        # ---- self-attention over all tokens ----
        context, attn = self.attention(tokens)                                      # (B, L, D), (B, L, L)

        #flattening the output of attention 
        flat = context.view(B, self.num_tokens * self.embed_dim)  # (B, L*D)

        # ---- final Q value (Critic) ----
        Q = self.output(flat)                                                     # (B, 1)

        # you now have:
        #   - attn: self-attention over tokens (who attends to whom)
        #   - Q: how global query weights each token for Q
        return Q, attn

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        """
        dim       : token embedding dimension (D)
        num_heads : number of attention heads (H), must divide dim
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Separate Q, K, V projections (your style)
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)

        # Output projection + residual norm
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.scale = self.head_dim ** 0.5  # sqrt(d_k per head)

    def forward(self, x):
        """
        x: (B, L, D)
        returns:
            out  : (B, L, D)
            attn : (B, H, L, L) attention weights per head
        """
        B, L, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        # ---- project to Q, K, V ----
        Q = self.W_Q(x)  # (B, L, D)
        K = self.W_K(x)  # (B, L, D)
        V = self.W_V(x)  # (B, L, D)

        # ---- reshape to multi-head: (B, H, L, Hd) ----
        Q = Q.view(B, L, H, Hd).transpose(1, 2)  # (B, H, L, Hd)
        K = K.view(B, L, H, Hd).transpose(1, 2)  # (B, H, L, Hd)
        V = V.view(B, L, H, Hd).transpose(1, 2)  # (B, H, L, Hd)

        # ---- scaled dot-product attention per head ----
        # scores: (B, H, L, L)
        scores = Q @ K.transpose(-2, -1) / self.scale
        attn = F.softmax(scores, dim=-1)  # softmax over keys

        # ---- apply attention to V ----
        # context: (B, H, L, Hd)
        context = attn @ V

        # ---- merge heads back: (B, L, D) ----
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        out = self.out_proj(context)  # (B, L, D)
        out = self.norm(out + x)      # residual + norm

        return out, attn

