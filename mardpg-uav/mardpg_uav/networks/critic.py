"""
Centralised recurrent critic — faithful to Section V.B and Fig. 4.

Two classes:
  RecurrentCritic      – original per-agent critic (kept for reference).
  SharedCentralCritic  – shared trunk + per-agent Q heads (Group 2 fix).
                         Lower FC + LSTM are shared across agents; each agent
                         has its own linear head. Matches paper §V.C: "sharing
                         parameters in the lower layer of the network."
"""
import torch
import torch.nn as nn


class RecurrentCritic(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 fc_hidden: int = 128, lstm_hidden: int = 128):
        super().__init__()
        input_dim = n_agents * (obs_dim + action_dim)
        self.fc1  = nn.Linear(input_dim, fc_hidden)
        self.fc2  = nn.Linear(fc_hidden, fc_hidden)
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, batch_first=True)
        self.fc_q = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_all, act_all, hidden=None, seq_len=None):
        BS_total = obs_all.shape[0]
        x = torch.cat([obs_all.reshape(BS_total, -1),
                       act_all.reshape(BS_total, -1)], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if seq_len is None:
            seq_len   = BS_total
            batch_size = 1
        else:
            batch_size = BS_total // seq_len
        x = x.view(batch_size, seq_len, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        q = self.fc_q(lstm_out).squeeze(-1)
        return q.view(BS_total), hidden


class SharedCentralCritic(nn.Module):
    """
    One critic shared across ALL agents.

    Architecture
    ------------
    Shared trunk  : Linear(N*(obs+act) -> fc_hidden)
                    Linear(fc_hidden -> fc_hidden)
                    LSTM(fc_hidden -> lstm_hidden)
    Per-agent heads: N × Linear(lstm_hidden -> 1)

    The trunk runs ONCE per update step.  Each head converts the shared
    LSTM state into a per-agent Q value.  Because rewards differ per agent
    the heads diverge during training, while the shared trunk learns a
    common representation of the joint state-action space.
    """
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 fc_hidden: int = 128, lstm_hidden: int = 128):
        super().__init__()
        self.n_agents   = n_agents
        self.lstm_hidden = lstm_hidden
        input_dim = n_agents * (obs_dim + action_dim)

        # Shared lower layers
        self.fc1  = nn.Linear(input_dim, fc_hidden)
        self.fc2  = nn.Linear(fc_hidden, fc_hidden)
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, batch_first=True)

        # Per-agent Q heads
        self.heads = nn.ModuleList(
            [nn.Linear(lstm_hidden, 1) for _ in range(n_agents)])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def trunk(self, obs_all: torch.Tensor, act_all: torch.Tensor,
              seq_len: int) -> torch.Tensor:
        """
        Run the shared trunk.

        Parameters
        ----------
        obs_all : (batch*seq, n_agents, obs_dim)
        act_all : (batch*seq, n_agents, action_dim)
        seq_len : int

        Returns
        -------
        h : (batch, seq, lstm_hidden)  — LSTM output, all time steps
        """
        BS      = obs_all.shape[0]
        batch   = BS // seq_len
        x = torch.cat([obs_all.reshape(BS, -1),
                       act_all.reshape(BS, -1)], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))           # (B*T, fc_hidden)
        x = x.view(batch, seq_len, -1)
        h, _   = self.lstm(x)                 # (B, T, lstm_hidden)
        return h

    def q_from_trunk(self, h: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Apply agent-specific head to pre-computed trunk output.

        Parameters
        ----------
        h        : (batch, seq, lstm_hidden)
        agent_id : int

        Returns
        -------
        q : (batch*seq,)  — flat, same layout as RecurrentCritic
        """
        B, T, _ = h.shape
        return self.heads[agent_id](h).squeeze(-1).reshape(B * T)

    # ------------------------------------------------------------------
    # Convenience forward (used by compute_actor_loss)
    # ------------------------------------------------------------------
    def forward(self, obs_all: torch.Tensor, act_all: torch.Tensor,
                agent_id: int, seq_len: int) -> torch.Tensor:
        """Returns flat Q values (batch*seq,) for one agent."""
        return self.q_from_trunk(self.trunk(obs_all, act_all, seq_len), agent_id)
