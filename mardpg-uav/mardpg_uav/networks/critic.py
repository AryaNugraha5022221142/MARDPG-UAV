"""
Critic with late action injection, supporting all four corners of the ablation:

  n_agents  = N  -> CENTRALIZED  (sees every agent's obs+action)  [MARDPG/MADDPG]
  n_agents  = 1  -> INDEPENDENT  (sees only its own agent)        [IDDPG]
  recurrent = True/False -> LSTM temporal core / per-timestep MLP

The OWN-vs-ALL slicing is done by the agent (MARDPGAgent._critic_inputs) BEFORE
calling forward(); this class only needs its input dimensions sized correctly,
which is what `n_agents` controls here.
"""
import torch
import torch.nn as nn


class _FFCore(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden_size = hidden
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, x, hidden=None):
        return self.net(x), None


class RecurrentCritic(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 fc_hidden: int = 128, lstm_hidden: int = 128,
                 recurrent: bool = True):
        super().__init__()
        self.obs_input = n_agents * obs_dim
        self.act_input = n_agents * action_dim
        self.recurrent = recurrent
        self.fc1 = nn.Linear(self.obs_input, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden + self.act_input, fc_hidden)
        if recurrent:
            self.lstm = nn.LSTM(fc_hidden, lstm_hidden, batch_first=True)
        else:
            self.lstm = _FFCore(fc_hidden, lstm_hidden)
        self.fc_q = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_all, act_all, hidden=None, seq_len=None):
        BS_total = obs_all.shape[0]
        o = obs_all.reshape(BS_total, -1)
        a = act_all.reshape(BS_total, -1)
        x = torch.relu(self.fc1(o))
        x = torch.cat([x, a], dim=-1)
        x = torch.relu(self.fc2(x))
        if seq_len is None:
            seq_len = BS_total
            batch_size = 1
        else:
            batch_size = BS_total // seq_len
        x = x.view(batch_size, seq_len, -1)
        core_out, hidden = self.lstm(x, hidden)
        q = self.fc_q(core_out).squeeze(-1)
        return q.view(BS_total), hidden
