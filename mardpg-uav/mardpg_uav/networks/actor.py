"""
Actor network with shared feature extractor and an optional temporal core.
  recurrent=True  -> LSTM            (MARDPG, the full method)
  recurrent=False -> per-timestep MLP (MADDPG / IDDPG ablations)

The temporal core keeps the attribute name `.lstm` and the call signature
core(x, hidden) -> (out, hidden) so the rest of the codebase (target-action
unroll, soft updates, actor_private_params, hidden-state plumbing) is unchanged.
"""
import torch, torch.nn as nn
from .shared import SharedFeatureExtractor


class _FFCore(nn.Module):
    """Feed-forward drop-in for nn.LSTM: applied independently per timestep,
    ignores/returns hidden=None. I/O shapes match nn.LSTM(batch_first=True)."""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden_size = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, x, hidden=None):
        return self.net(x), None


class Actor(nn.Module):
    def __init__(self, shared_extractor: SharedFeatureExtractor,
                 lstm_hidden: int = 128,
                 max_delta_angle: float = 0.5236,
                 action_dim: int = 2,
                 recurrent: bool = True):
        super().__init__()
        self.shared       = shared_extractor
        self.action_bound = max_delta_angle
        self.recurrent    = recurrent
        in_dim = self.shared.feature_dim + action_dim
        if recurrent:
            self.lstm = nn.LSTM(in_dim, lstm_hidden, batch_first=True)
        else:
            self.lstm = _FFCore(in_dim, lstm_hidden)
        self.fc_out = nn.Linear(lstm_hidden, 2)   # [rho, tau]
        self.tanh   = nn.Tanh()

    def forward(self, obs_sequence, prev_action, hidden=None):
        B, T, D = obs_sequence.shape
        feat = self.shared(obs_sequence.view(B * T, D)).view(B, T, -1)
        x = torch.cat([feat, prev_action], dim=-1)
        core_out, hidden = self.lstm(x, hidden)
        action = self.tanh(self.fc_out(core_out)) * self.action_bound
        return action, hidden
