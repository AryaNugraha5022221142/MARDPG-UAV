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
from ..networks.critic import AttentionCritic


class MARDPGAgent:
    def __init__(self, agent_id: int, n_agents: int, obs_dim: int = 36,
                 action_dim: int = 2, hidden_dim: int = 128,
                 lr_actor: float = 0.0001, lr_critic: float = 0.0003,
                 tau: float = 0.01, gamma: float = 0.99,
                 burn_in: int = 12,
                 gradient_clip: float = 1.0, device: str = 'cpu'):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.burn_in = burn_in
        self.gradient_clip = gradient_clip
        self.device = torch.device(device)
        
        # Shared feature extractor (Section 10.1)
        self.shared_extractor = SharedFeatureExtractor().to(self.device)
        
        # Actor network (decentralized execution)
        self.actor = Actor(self.shared_extractor, hidden_dim).to(self.device)
        self.actor_target = Actor(self.shared_extractor, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic network (centralized training)
        self.critic = AttentionCritic(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.critic_target = AttentionCritic(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize targets
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        # Hidden states (reset per episode)
        self.actor_hidden = None
        self.eval_actor_hidden = None  # Separate hidden for evaluation
        
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Args:
            obs: (obs_dim,) single observation
        Returns:
            action: (action_dim,)
        """
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
            
            return action.squeeze(0).cpu().numpy()
    
    def reset_hidden(self, batch_size: int = 1, eval_mode: bool = False):
        h = self._init_hidden(batch_size)
        if eval_mode:
            self.eval_actor_hidden = h
        else:
            self.actor_hidden = h
    
    def _init_hidden(self, batch_size: int):
        return (torch.zeros(1, batch_size, self.actor.lstm.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.actor.lstm.hidden_size).to(self.device))
    
    def update(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor,
               batch_rewards: torch.Tensor, batch_dones: torch.Tensor,
               all_agents: list) -> Tuple[float, float]:
        """
        Update actor and critic using BPTT.
        Args:
            batch_obs: (batch, seq_len, n_agents, obs_dim)
            batch_actions: (batch, seq_len, n_agents, action_dim)
            batch_rewards: (batch, seq_len, n_agents)
            batch_dones: (batch, seq_len, n_agents)
            all_agents: list of MARDPGAgent instances
        """
        batch_size, seq_len, n_agents, obs_dim = batch_obs.shape
        T_burn = self.burn_in
        
        # Move to device
        obs = batch_obs.to(self.device)
        actions = batch_actions.to(self.device)
        rewards = batch_rewards.to(self.device)
        dones = batch_dones.to(self.device)
        
        # ==========================================
        # 1. Extract hidden states from all actors
        # ==========================================
        agent_hiddens = []  # List of (batch, seq_len, hidden_dim)
        
        for i, agent in enumerate(all_agents):
            agent_obs = obs[:, :, i, :]
            flat_obs = agent_obs.reshape(batch_size * seq_len, obs_dim)
            features = agent.actor.shared(flat_obs)
            features = features.view(batch_size, seq_len, -1)
            
            lstm_out, _ = agent.actor.lstm(features, None)
            agent_hiddens.append(lstm_out)
            
        # ==========================================
        # 2. Critic Update (Eq 12-13 with masking)
        # ==========================================
        # Flatten time and batch dimensions for AttentionCritic
        obs_all = obs.reshape(batch_size * seq_len, n_agents, obs_dim)
        act_all = actions.reshape(batch_size * seq_len, n_agents, -1)
        
        Q_current_flat = self.critic(obs_all, act_all, self.agent_id)
        Q_current = Q_current_flat.view(batch_size, seq_len)
        
        with torch.no_grad():
            target_actions = []
            for i, agent in enumerate(all_agents):
                # Shift for next state: duplicate last element or handle terminal
                next_obs_i = torch.cat([obs[:, 1:, i, :], 
                                      obs[:, -1:, i, :]], dim=1) 
                next_flat = next_obs_i.reshape(batch_size * seq_len, obs_dim)
                next_features = agent.actor_target.shared(next_flat).view(batch_size, seq_len, -1)
                next_lstm_out, _ = agent.actor_target.lstm(next_features, None)
                next_act = agent.actor_target.tanh(agent.actor_target.fc_out(next_lstm_out)) * agent.actor_target.action_bound
                target_actions.append(next_act.view(batch_size * seq_len, -1))
            
            next_act_all = torch.stack(target_actions, dim=1)  # (B*L, N, 2)
            next_obs_all = torch.cat([obs[:, 1:, :, :], obs[:, -1:, :, :]], dim=1)
            next_obs_all = next_obs_all.reshape(batch_size * seq_len, n_agents, obs_dim)
            
            target_q_flat = self.critic_target(next_obs_all, next_act_all, self.agent_id)
            Q_target_next = target_q_flat.view(batch_size, seq_len)
            
            r = rewards[:, :, self.agent_id]
            batch_dones_agent = dones[:, :, self.agent_id]
            
            # TD target — Eq.91: bootstrap zeroed by per-agent flag
            y = r + self.gamma * Q_target_next * (~batch_dones_agent)

        # Compute validity mask — Eq.87
        burn_mask  = torch.arange(seq_len).unsqueeze(0) >= T_burn
        agent_done_prev = torch.cat([torch.zeros_like(dones[:, :1, self.agent_id]),
                                      dones[:, :-1, self.agent_id]], dim=1)
        mask = burn_mask.to(self.device) & ~agent_done_prev  # (B, L)
        
        # Masked Bellman loss — Eq.93
        eps = 1e-8
        critic_loss = ((Q_current - y.detach())**2 * mask).sum() / (mask.sum() + eps)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        # ==========================================
        # 3. Actor Update (Eq 11)
        # ==========================================
        for p in self.critic.parameters():
            p.requires_grad = False
            
        actor_actions = []
        for i, agent in enumerate(all_agents):
            if i == self.agent_id:
                # We need actions from current policy that keep gradient
                # using the already computed agent_hiddens
                h_i = agent_hiddens[i]  # (B, L, hidden_dim)
                act_i = agent.actor.tanh(agent.actor.fc_out(h_i)) * agent.actor.action_bound
                actor_actions.append(act_i.view(batch_size * seq_len, -1))
            else:
                actor_actions.append(actions[:, :, i, :].detach().reshape(batch_size * seq_len, -1))
                
        actor_act_all = torch.stack(actor_actions, dim=1)
        
        self.critic.eval()
        q_value_flat = self.critic(obs_all, actor_act_all, self.agent_id)
        self.critic.train()
        
        q_value = q_value_flat.view(batch_size, seq_len)
        
        # Policy gradient: maximize Masked Q
        actor_loss = -(q_value * mask).sum() / (mask.sum() + eps)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        
        for p in self.critic.parameters():
            p.requires_grad = True
            
        # ==========================================
        # 4. Soft Update Target Networks
        # ==========================================
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())
    
    def share_parameters(self, other_agent: 'MARDPGAgent'):
        """Share feature extractor — §14.2. Rebuilds optimiser to track correct params."""
        self.shared_extractor = other_agent.shared_extractor
        self.actor.shared = self.shared_extractor
        self.actor_target.shared = self.shared_extractor
        # Rebuild optimiser so it tracks the now-shared extractor parameters
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=self.actor_optimizer.defaults['lr'])
