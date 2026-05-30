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
        self.d_model = d_model
        
        # Shared Encoder & MHA
        self.encoder = nn.Linear(lstm_hidden + action_dim, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.ln1 = nn.LayerNorm(d_model)
        
        # Twin Q-networks (TD3 style to prevent overestimation)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_ff, d_model))
        self.ln2_1 = nn.LayerNorm(d_model)
        self.readout1 = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_ff, d_model))
        self.ln2_2 = nn.LayerNorm(d_model)
        self.readout2 = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, hidden_all, act_all, agent_idx):
        x = torch.relu(self.encoder(torch.cat([hidden_all, act_all], dim=-1)))
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        
        # Q1
        x1 = self.ln2_1(x + self.ffn1(x))
        q1 = self.readout1(x1[:, agent_idx, :]).squeeze(-1)
        
        # Q2
        x2 = self.ln2_2(x + self.ffn2(x))
        q2 = self.readout2(x2[:, agent_idx, :]).squeeze(-1)
        
        return q1, q2

    def Q1(self, hidden_all, act_all, agent_idx):
        """Used for actor policy gradient"""
        x = torch.relu(self.encoder(torch.cat([hidden_all, act_all], dim=-1)))
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        x1 = self.ln2_1(x + self.ffn1(x))
        return self.readout1(x1[:, agent_idx, :]).squeeze(-1)
