"""
Actor network with shared feature extractor and LSTM.
Decentralized execution: each agent uses only its own observation history.
Reference: Section 6.1 and 5.3 of blueprint.
"""
import torch
import torch.nn as nn
import numpy as np
from .shared import SharedFeatureExtractor


class Actor(nn.Module):
    def __init__(self, shared_extractor: SharedFeatureExtractor,
                 lstm_hidden: int = 128,
                 action_bound: float = 3.0):
        super().__init__()
        self.shared = shared_extractor
        self.action_bound = action_bound
        
        # LSTM processes feature sequences (Section 6.1)
        self.lstm = nn.LSTM(self.shared.feature_dim, lstm_hidden, batch_first=True)
        
        # Agent-specific head (Section 10.1)
        self.fc_out = nn.Linear(lstm_hidden, 3)
        self.tanh = nn.Tanh()
        
    def forward(self, obs_sequence: torch.Tensor,
                hidden: tuple = None) -> tuple:
        """
        Args:
            obs_sequence: (batch, seq_len, 34)
            hidden: (h_0, c_0) for LSTM
        Returns:
            action: (batch, 3) scaled to [-v_max, v_max]
            hidden: updated LSTM state
        """
        batch_size, seq_len, obs_dim = obs_sequence.shape
        
        # Extract features for all timesteps
        flat_obs = obs_sequence.view(batch_size * seq_len, obs_dim)
        features = self.shared(flat_obs)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM (Section 6.1)
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Output action at last timestep
        action = self.tanh(self.fc_out(lstm_out[:, -1, :]))
        action = action * self.action_bound  # Scale to [-v_max, v_max]
        
        return action, hidden
