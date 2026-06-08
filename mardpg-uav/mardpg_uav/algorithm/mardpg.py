"""
MARDPG Agent with CTDE and parameter sharing.
Reference: Sections 4, 5, and 10 of blueprint.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from ..networks.shared import SharedFeatureExtractor
from ..networks.actor import Actor
from ..networks.critic import RecurrentCritic

class MARDPGAgent:
    def __init__(self, agent_id: int, n_agents: int,
                 obs_dim: int = 30, action_dim: int = 2,
                 action_bound: float = 0.5236,   # pi/6
                 lstm_hidden: int = 128, fc_hidden: int = 128,
                 lr_actor: float = 0.01, lr_critic: float = 0.01,
                 tau: float = 0.01, gamma: float = 0.99,
                 gradient_clip: float = 1.0, burn_in: int = 10,
                 device: str = 'cpu'):

        self.agent_id     = agent_id
        self.n_agents     = n_agents
        self.gamma        = gamma
        self.tau          = tau
        self.gradient_clip = gradient_clip
        self.burn_in      = burn_in
        self.device       = torch.device(device)
        self.action_bound = action_bound

        import copy
        self.shared_extractor = SharedFeatureExtractor().to(self.device)
        self.actor            = Actor(self.shared_extractor, lstm_hidden,
                                      max_delta_angle=action_bound).to(self.device)
        self.actor_target     = copy.deepcopy(self.actor).to(self.device)

        # Single Q critic (paper does not use twin Q)
        self.critic         = RecurrentCritic(n_agents, obs_dim, action_dim,
                                               fc_hidden, lstm_hidden).to(self.device)
        self.critic_target  = copy.deepcopy(self.critic).to(self.device)

        self.actor_private_params = (list(self.actor.lstm.parameters()) +
                                     list(self.actor.fc_out.parameters()))
        self.actor_optimizer  = optim.Adam(self.actor_private_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        self.critic_target.eval()
        self.actor_target.eval()

        # Hidden states
        self.actor_hidden      = None
        self.eval_actor_hidden = None

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            if evaluate:
                if self.eval_actor_hidden is None:
                    self.eval_actor_hidden = self._init_hidden(1)
                action, self.eval_actor_hidden = self.actor(obs_tensor, self.eval_actor_hidden)
            else:
                if self.actor_hidden is None:
                    self.actor_hidden = self._init_hidden(1)
                action, self.actor_hidden = self.actor(obs_tensor, self.actor_hidden)
            return action[0].cpu().numpy()

    def reset_hidden(self, batch_size: int = 1, eval_mode: bool = False):
        h = self._init_hidden(batch_size)
        if eval_mode:
            self.eval_actor_hidden = h
        else:
            self.actor_hidden = h

    def _init_hidden(self, batch_size: int):
        return (torch.zeros(1, batch_size, self.actor.lstm.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.actor.lstm.hidden_size).to(self.device))

    def update_critic(self, obs_all_seq: torch.Tensor, act_all_seq: torch.Tensor,
                      next_obs_all_seq: torch.Tensor, next_act_all_seq: torch.Tensor,
                      rewards: torch.Tensor, dones: torch.Tensor, seq_len: int, mask: torch.Tensor):
        """
        obs_all_seq : (batch*seq, n_agents, obs_dim)
        act_all_seq : (batch*seq, n_agents, action_dim)
        """
        batch_size = rewards.shape[0]

        # Current Q
        self.critic.train()
        q_flat, _  = self.critic(obs_all_seq, act_all_seq, seq_len=seq_len)     # (batch*seq,)
        q_cur   = q_flat.view(batch_size, seq_len)

        with torch.no_grad():
            self.critic_target.eval()
            q_next_flat, _ = self.critic_target(next_obs_all_seq, next_act_all_seq, seq_len=seq_len)
            q_next      = q_next_flat.view(batch_size, seq_len)

            r     = rewards[:, :, self.agent_id]
            d     = dones[:, :, self.agent_id]
            y     = r + self.gamma * q_next * (~d)

        eps         = 1e-8
        critic_loss = (((q_cur - y.detach()) ** 2) * mask).sum() / (mask.sum() + eps)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        return critic_loss.item(), q_cur.detach().mean().item(), 0.0

    def compute_actor_loss(self, obs_all_seq: torch.Tensor, actor_act_all_seq: torch.Tensor, mask: torch.Tensor):
        """
        Uses obs_all_seq (not hidden states) for critic input.
        """
        batch_size, seq_len = mask.shape
        self.critic.eval()
        q_flat, _ = self.critic(obs_all_seq, actor_act_all_seq, seq_len=seq_len)  # (batch*seq,)
        q      = q_flat.view(batch_size, seq_len)
        eps    = 1e-8
        return -(q * mask).sum() / (mask.sum() + eps)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def share_parameters(self, other_agent: 'MARDPGAgent'):
        """Share feature extractor — §14.2."""
        self.shared_extractor = other_agent.shared_extractor
        self.actor.shared = self.shared_extractor
        self.actor_target.shared = other_agent.actor_target.shared
        if hasattr(self, 'target_shared_extractor'):
            delattr(self, 'target_shared_extractor')
