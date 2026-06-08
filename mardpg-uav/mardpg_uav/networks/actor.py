"""
Actor network with shared feature extractor and LSTM.
Decentralized execution: each agent uses only its own observation history.
Reference: Section 6.1 and 5.3 of blueprint.
"""
import torch, torch.nn as nn
from .shared import SharedFeatureExtractor

class Actor(nn.Module):
    def __init__(self, shared_extractor: SharedFeatureExtractor,
                 lstm_hidden: int = 128,
                 max_delta_angle: float = 0.5236):   # pi/6
        super().__init__()
        self.shared      = shared_extractor
        self.action_bound = max_delta_angle

        self.lstm   = nn.LSTM(self.shared.feature_dim, lstm_hidden, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden, 2)    # 2D: [rho, tau]
        self.tanh   = nn.Tanh()

    def forward(self, obs_sequence, hidden=None):
        """
        obs_sequence: (batch, seq_len, 30)
        returns: action (batch, 2), hidden
        """
        B, T, D = obs_sequence.shape
        feat    = self.shared(obs_sequence.view(B * T, D)).view(B, T, -1)
        lstm_out, hidden = self.lstm(feat, hidden)
        action  = self.tanh(self.fc_out(lstm_out[:, -1, :])) * self.action_bound
        return action, hidden
