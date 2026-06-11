"""
Centralised recurrent critic with late action injection (N3-3).
Action enters at the second FC layer so the policy gradient dQ/da is not
diluted by the high-dimensional joint observation at the input.
"""
import torch
import torch.nn as nn


class RecurrentCritic(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 fc_hidden: int = 128, lstm_hidden: int = 128):
        super().__init__()
        self.obs_input = n_agents * obs_dim
        self.act_input = n_agents * action_dim
        self.fc1  = nn.Linear(self.obs_input, fc_hidden)
        self.fc2  = nn.Linear(fc_hidden + self.act_input, fc_hidden)
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, batch_first=True)
        self.fc_q = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_all, act_all, hidden=None, seq_len=None):
        BS_total = obs_all.shape[0]
        o = obs_all.reshape(BS_total, -1)
        a = act_all.reshape(BS_total, -1)
        x = torch.relu(self.fc1(o))
        x = torch.cat([x, a], dim=-1)
        x = torch.relu(self.fc2(x))
        if seq_len is None:
            seq_len    = BS_total
            batch_size = 1
        else:
            batch_size = BS_total // seq_len
        x = x.view(batch_size, seq_len, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        q = self.fc_q(lstm_out).squeeze(-1)
        return q.view(BS_total), hidden

