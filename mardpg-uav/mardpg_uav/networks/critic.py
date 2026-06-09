"""
Centralised recurrent critic — faithful to Section V.B and Fig. 4.
Input : concatenation of all agents' observations and actions.
        [o_1, ..., o_N, a_1, ..., a_N]  at each time step
Output: single Q value for agent i.
"""
import torch, torch.nn as nn

class RecurrentCritic(nn.Module):
    def __init__(self, n_agents: int, feature_dim: int, action_dim: int,
                 fc_hidden: int = 128, lstm_hidden: int = 128):
        super().__init__()
        # Total input = N * (feature_dim + action_dim)
        input_dim = n_agents * (feature_dim + action_dim)

        # FC layers before LSTM (paper: "using the fully-connected layer
        #   of all agents as input to the hidden LSTM layer")
        self.fc1  = nn.Linear(input_dim, fc_hidden)
        self.fc2  = nn.Linear(fc_hidden, fc_hidden)
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, batch_first=True)
        self.fc_q = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_all, act_all, hidden=None, seq_len=None):
        """
        obs_all : (batch*seq, n_agents, obs_dim)
        act_all : (batch*seq, n_agents, action_dim)
        seq_len : required to reshape for LSTM correctly. If not provided, assumes batch=1 and seq=batch*seq
        returns : q (batch*seq,), hidden
        """
        BS_total = obs_all.shape[0]
        x  = torch.cat([obs_all.view(BS_total, -1),
                         act_all.view(BS_total, -1)], dim=-1)
        x  = torch.relu(self.fc1(x))
        x  = torch.relu(self.fc2(x))          # (BS_total, fc_hidden)
        
        if seq_len is None:
            # Assume single sequence if not provided
            seq_len = BS_total
            batch_size = 1
        else:
            batch_size = BS_total // seq_len
            
        x = x.view(batch_size, seq_len, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        
        q = self.fc_q(lstm_out).squeeze(-1)   # (batch_size, seq_len)
        return q.view(BS_total), hidden
