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
    def __init__(self, agent_id: int, n_agents: int, obs_dim: int = 34,
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
        import copy
        self.shared_extractor = SharedFeatureExtractor().to(self.device)
        self.target_shared_extractor = copy.deepcopy(self.shared_extractor).to(self.device)
        
        # Actor network (decentralized execution)
        self.actor = Actor(self.shared_extractor, hidden_dim).to(self.device)
        self.actor_target = Actor(self.target_shared_extractor, hidden_dim).to(self.device)
        self.private_actor_params = list(self.actor.lstm.parameters()) + list(self.actor.fc_out.parameters())
        self.actor_optimizer = optim.Adam(self.private_actor_params, lr=lr_actor)
        
        # Critic network (centralized training)
        self.critic = AttentionCritic(n_agents=n_agents, lstm_hidden=hidden_dim, action_dim=action_dim).to(self.device)
        self.critic_target = AttentionCritic(n_agents=n_agents, lstm_hidden=hidden_dim, action_dim=action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize targets
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        self.critic_target.eval()   # freeze dropout / batchnorm for stable targets
        
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
            
            return action[0, 0].cpu().numpy()
    
    def reset_hidden(self, batch_size: int = 1, eval_mode: bool = False):
        h = self._init_hidden(batch_size)
        if eval_mode:
            self.eval_actor_hidden = h
        else:
            self.actor_hidden = h
    
    def _init_hidden(self, batch_size: int):
        return (torch.zeros(1, batch_size, self.actor.lstm.hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.actor.lstm.hidden_size).to(self.device))
    
    def update_critic(self, detached_hidden_all, act_all, next_hidden_all, next_act_all, 
                      rewards, dones, seq_len, mask):
        """Phase 1: Critic update with detached hiddens (Fixes Bugs 1 & 2)"""
        batch_size = rewards.shape[0]
        
        # Current Q-values (Online Critic in Train Mode)
        self.critic.train()
        q1_flat, q2_flat = self.critic(detached_hidden_all, act_all, self.agent_id)
        q1_current = q1_flat.view(batch_size, seq_len)
        q2_current = q2_flat.view(batch_size, seq_len)
        
        with torch.no_grad():
            self.critic_target.eval() # FIX (Bug 8): Disable dropout for target calculation
            
            # Target Policy Smoothing (Fix Bug 7)
            noise = (torch.randn_like(next_act_all) * 0.1).clamp(-0.1, 0.1)
            next_act_all_noisy = (next_act_all + noise).clamp(-self.actor.action_bound, self.actor.action_bound)
            
            # Twin-Q Minimum
            target_q1_flat, target_q2_flat = self.critic_target(next_hidden_all, next_act_all_noisy, self.agent_id)
            target_q_next = torch.min(target_q1_flat, target_q2_flat).view(batch_size, seq_len)
            
            r = rewards[:, :, self.agent_id]
            batch_dones_agent = dones[:, :, self.agent_id]
            y = r + self.gamma * target_q_next * (~batch_dones_agent)

        eps = 1e-8
        critic_loss_1 = (((q1_current - y.detach())**2) * mask).sum() / (mask.sum() + eps)
        critic_loss_2 = (((q2_current - y.detach())**2) * mask).sum() / (mask.sum() + eps)
        critic_loss = critic_loss_1 + critic_loss_2
        
        # Update Critic safely
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        return critic_loss.item()

    def compute_actor_loss(self, hidden_all_with_grad, actor_act_all, mask):
        """Phase 2: Actor loss generation (Fixes Bugs 1 & 3)"""
        batch_size, seq_len = mask.shape
        
        self.critic.eval() # FIX (Bug 8): Disable dropout when evaluating Q for policy gradient
        q_value_flat = self.critic.Q1(hidden_all_with_grad, actor_act_all, self.agent_id)
        
        q_value = q_value_flat.view(batch_size, seq_len)
        eps = 1e-8
        
        # Deterministic Policy Gradient
        actor_loss = -(q_value * mask).sum() / (mask.sum() + eps)
        return actor_loss
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())
    
    def share_parameters(self, other_agent: 'MARDPGAgent'):
        """Share feature extractor — §14.2."""
        import copy
        self.shared_extractor = other_agent.shared_extractor
        self.actor.shared = self.shared_extractor
        self.actor_target.shared = copy.deepcopy(other_agent.shared_extractor)
        if hasattr(self, 'target_shared_extractor'):
            delattr(self, 'target_shared_extractor')
        # Note: shared_extractor parameters will be optimized externally.
