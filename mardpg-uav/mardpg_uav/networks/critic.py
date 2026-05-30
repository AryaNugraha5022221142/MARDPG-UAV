"""
Centralized critic network.
Input: all agents' LSTM hidden states + all agents' actions.
Reference: Section 6.2 and 5.1 of blueprint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCritic(nn.Module):
    def __init__(self, n_agents=5, lstm_hidden=128, action_dim=2, d_model=128, d_ff=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.encoder = nn.Linear(lstm_hidden + action_dim, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.readout = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), 
            nn.Linear(64, 1))

    def forward(self, hidden_all, act_all, agent_idx):
        # hidden_all: (B, N, lstm_hidden), act_all: (B, N, 2)
        x = torch.relu(self.encoder(torch.cat([hidden_all, act_all], dim=-1)))
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return self.readout(x[:, agent_idx, :]).squeeze(-1)
