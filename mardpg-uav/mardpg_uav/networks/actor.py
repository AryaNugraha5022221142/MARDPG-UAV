import torch, torch.nn as nn
from .shared import SharedFeatureExtractor
from mardpg_uav.algorithm.nn_blocks import _FFCore

class Actor(nn.Module):

    def __init__(self, shared_extractor: SharedFeatureExtractor, lstm_hidden: int=128, max_delta_angle: float=0.5236, action_dim: int=2, recurrent: bool=True):
        super().__init__()
        self.shared = shared_extractor
        self.action_bound = max_delta_angle
        self.recurrent = recurrent
        in_dim = self.shared.feature_dim + action_dim
        if recurrent:
            self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True)
        else:
            self.lstm = _FFCore(in_dim, lstm_hidden)
        self.fc_out = nn.Linear(lstm_hidden, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, obs_sequence, prev_action, hidden=None):
        (B, T, D) = obs_sequence.shape
        feat = self.shared(obs_sequence.view(B * T, D)).view(B, T, -1)
        x = torch.cat([feat, prev_action], dim=-1)
        (core_out, hidden) = self.lstm(x, hidden)
        action = self.tanh(self.fc_out(core_out)) * self.action_bound
        return (action, hidden)
