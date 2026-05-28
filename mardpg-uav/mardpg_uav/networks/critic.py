"""
Centralized critic network.
Input: all agents' LSTM hidden states + all agents' actions.
Reference: Section 6.2 and 5.1 of blueprint.
"""
import torch
import torch.nn as nn

class AttentionCritic(nn.Module):
    def __init__(self, n_agents=5, obs_dim=34, action_dim=2, d_model=128, d_ff=256):
        super().__init__()
        self.encoder = nn.Linear(obs_dim + action_dim, d_model)  # 38 → 128
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.readout = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1))
        self.scale = d_model ** 0.5

    def forward(self, obs_all, act_all, agent_idx):
        # obs_all: (B, N, 36), act_all: (B, N, 2)
        x = torch.relu(self.encoder(torch.cat([obs_all, act_all], dim=-1)))  # (B,N,128)
        Q = self.W_Q(x); K = self.W_K(x); V = self.W_V(x)
        A = torch.softmax(Q @ K.transpose(-1,-2) / self.scale, dim=-1)
        x = self.ln1(x + A @ V)
        x = self.ln2(x + self.ffn(x))
        return self.readout(x[:, agent_idx, :]).squeeze(-1)   # (B,)
