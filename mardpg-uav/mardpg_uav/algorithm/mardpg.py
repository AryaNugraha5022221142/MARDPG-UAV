"""
MARDPG Agent with CTDE and parameter sharing.
Reference: Sections 4, 5, and 10 of blueprint.

Changes in v2
-------------
- Per-agent critic removed; agent now holds a *reference* to the shared
  SharedCentralCritic created in train.py (set via agent.critic = ...).
- update_critic() removed; the combined critic update lives in train.py
  so the trunk runs only once per update step.
- compute_actor_loss() no longer freezes/unfreezes the critic internally;
  train.py wraps the whole actor backward in a single freeze/unfreeze.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..networks.shared import SharedFeatureExtractor
from ..networks.actor  import Actor
from ..networks.critic import RecurrentCritic


class MARDPGAgent:
    def __init__(self, agent_id: int, n_agents: int,
                 obs_dim: int = 32, action_dim: int = 2,
                 action_bound: float = 0.5236,
                 lstm_hidden: int = 128, fc_hidden: int = 128,
                 lr_actor: float = 0.01, lr_critic: float = 0.01,
                 tau: float = 0.01, gamma: float = 0.99,
                 gradient_clip: float = 1.0, burn_in: int = 10,
                 device: str = 'cpu'):

        self.agent_id      = agent_id
        self.n_agents      = n_agents
        self.gamma         = gamma
        self.tau           = tau
        self.gradient_clip = gradient_clip
        self.burn_in       = burn_in
        self.device        = torch.device(device)
        self.action_bound  = action_bound

        # Actor (shared encoder + private LSTM + head)
        self.shared_extractor = SharedFeatureExtractor().to(self.device)
        self.actor            = Actor(self.shared_extractor, lstm_hidden,
                                      max_delta_angle=action_bound).to(self.device)
        self.actor_target     = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.eval()

        # Critic: N independent instances
        self.critic        = RecurrentCritic(n_agents, obs_dim, action_dim, fc_hidden, lstm_hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()

        # Optimizers
        self.actor_private_params = (list(self.actor.lstm.parameters()) +
                                     list(self.actor.fc_out.parameters()))
        self.actor_optimizer = optim.Adam(self.actor_private_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # Hidden states
        self.actor_hidden      = None
        self.eval_actor_hidden = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, prev_action: np.ndarray,
                      evaluate: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            pa_t   = torch.FloatTensor(prev_action).unsqueeze(0).unsqueeze(0).to(self.device)
            if evaluate:
                if self.eval_actor_hidden is None:
                    self.eval_actor_hidden = self._init_hidden(1)
                act, self.eval_actor_hidden = self.actor(obs_t, pa_t, self.eval_actor_hidden)
            else:
                if self.actor_hidden is None:
                    self.actor_hidden = self._init_hidden(1)
                act, self.actor_hidden = self.actor(obs_t, pa_t, self.actor_hidden)
            return act[:, -1, :][0].cpu().numpy()

    def reset_hidden(self, batch_size: int = 1, eval_mode: bool = False):
        h = self._init_hidden(batch_size)
        if eval_mode:
            self.eval_actor_hidden = h
        else:
            self.actor_hidden = h

    def _init_hidden(self, batch_size: int):
        h = torch.zeros(1, batch_size, self.actor.lstm.hidden_size).to(self.device)
        return (h, h.clone())

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------
    def compute_critic_loss(self, obs_all_seq: torch.Tensor,
                            act_all_seq: torch.Tensor,
                            next_obs_all_seq: torch.Tensor,
                            next_act_all_seq: torch.Tensor,
                            rewards: torch.Tensor,
                            dones: torch.Tensor,
                            mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        DPG critic loss (paper Eq. 8).
        """
        batch_size, total_seq_len = mask.shape
        learn_start = self.burn_in

        # Current Q
        q, _ = self.critic(obs_all_seq, act_all_seq, hidden=None, seq_len=total_seq_len)
        q = q.view(batch_size, total_seq_len)

        # Target Q
        with torch.no_grad():
            q_next, _ = self.critic_target(next_obs_all_seq, next_act_all_seq, hidden=None, seq_len=total_seq_len)
            q_next = q_next.view(batch_size, total_seq_len)
            
            y = rewards + self.gamma * q_next * (~dones)

        q_learn = q[:, learn_start:]
        y_learn = y[:, learn_start:].detach()
        mask_learn = mask[:, learn_start:]

        valid_steps = mask_learn.sum()
        if valid_steps < 1e-6:
            return (((q_learn - y_learn) ** 2) * mask_learn).sum() * 0.0, q_learn, valid_steps.item()

        critic_loss = (((q_learn - y_learn) ** 2) * mask_learn).sum() / valid_steps
        
        return critic_loss, q_learn, valid_steps.item()

    # ------------------------------------------------------------------
    # Actor loss  (critic freeze handled externally in train.py)
    # ------------------------------------------------------------------
    def compute_actor_loss(self, obs_all_seq: torch.Tensor,
                           act_all_seq: torch.Tensor,
                           prev_act_all_seq: torch.Tensor,
                           mask: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        DPG actor loss (paper Eq. 11).
        """
        batch_size, total_seq_len = mask.shape

        # Joint actions: detach all, then re-attach THIS agent's actions
        joint_actions = act_all_seq.clone().detach()

        my_obs  = obs_all_seq[:, self.agent_id, :].view(batch_size, total_seq_len, -1)
        my_pa   = prev_act_all_seq[:, self.agent_id, :].view(batch_size, total_seq_len, -1)

        my_acts, _ = self.actor(my_obs, my_pa, hidden=None)
        joint_actions[:, self.agent_id, :] = my_acts.reshape(-1, my_acts.shape[-1])

        # Query independent critic
        q_flat, _ = self.critic(obs_all_seq, joint_actions, hidden=None, seq_len=total_seq_len)
        q_full = q_flat.view(batch_size, total_seq_len)

        learn_start = self.burn_in
        q_learn    = q_full[:, learn_start:]
        mask_learn = mask[:, learn_start:]

        valid_steps = mask_learn.sum()
        if valid_steps < 1e-6:
            # Return 0 loss but preserve computational graph mathematically
            return (q_learn * mask_learn).sum() * 0.0, valid_steps.item()

        loss = -(q_learn * mask_learn).sum() / valid_steps
        return loss, valid_steps.item()

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------
    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def share_parameters(self, other: 'MARDPGAgent'):
        """Share feature extractor (lower layers) with agent 0."""
        self.shared_extractor      = other.shared_extractor
        self.actor.shared          = self.shared_extractor
        self.actor_target.shared   = other.actor_target.shared
