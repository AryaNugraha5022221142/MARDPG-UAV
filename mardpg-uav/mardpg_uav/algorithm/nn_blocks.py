import torch.nn as nn

class _FFCore(nn.Module):
    """Feed-forward drop-in for nn.LSTM: applied independently per timestep,
    ignores/returns hidden=None. I/O shapes match nn.LSTM(batch_first=True)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden_size = hidden
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, x, hidden=None):
        return (self.net(x), None)