"""
MARDPG Agent with CTDE and parameter sharing.
Reference: Sections 4, 5, and 10 of blueprint.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..networks.shared import SharedFeatureExtractor
from ..networks.actor import Actor
from ..networks.critic import Critic


class MARDPGAgent:
    def __init__(self, agent_id: int, n_agents: int, obs_dim: int = 30,
                 action_dim: int = 2, hidden_dim: int = 128,
                 lr_actor: float = 0.001, lr_critic: float = 0.001,
                 tau: float = 0.01, gamma: float = 0.99,
                 gradient_clip: float = 1.0, device: str = 'cpu'):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.gradient_clip = gradient_clip
        self.device = torch.device(device)
        
        # Shared feature extractor (Section 10.1)
        self.shared_extractor = SharedFeatureExtractor().to(self.device)
        
        # Actor network (decentralized execution)
        self.actor = Actor(self.shared_extractor, hidden_dim).to(self.device)
        self.actor_target = Actor(self.shared_extractor, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic network (centralized training)
        self.critic = Critic(n_agents, hidden_dim).to(self.device)
        self.critic_target = Critic(n_agents, hidden_dim).to(self.device)
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
            batch_dones: (batch, seq_len)
            all_agents: list of MARDPGAgent instances
        """
        batch_size, seq_len, n_agents, obs_dim = batch_obs.shape
        
        # Move to device
        obs = batch_obs.to(self.device)
        actions = batch_actions.to(self.device)
        rewards = batch_rewards.to(self.device)
        dones = batch_dones.to(self.device)
        
        # ==========================================
        # 1. Extract hidden states from all actors
        # ==========================================
        agent_hiddens = []  # List of (batch, seq_len, hidden_dim)
        next_agent_hiddens = []
        
        for i, agent in enumerate(all_agents):
            # Process observations through shared extractor + LSTM
            # obs[:, :, i, :] -> (batch, seq_len, obs_dim)
            agent_obs = obs[:, :, i, :]
            
            # Flatten for batch processing
            flat_obs = agent_obs.reshape(batch_size * seq_len, obs_dim)
            features = agent.actor.shared(flat_obs)
            features = features.view(batch_size, seq_len, -1)
            
            # Run through LSTM
            lstm_out, _ = agent.actor.lstm(features, None)
            agent_hiddens.append(lstm_out)
            
            # For target: process next observations
            next_obs = torch.cat([obs[:, 1:, i, :], 
                                  obs[:, -1:, i, :]], dim=1)  # Shift for next state
            next_flat = next_obs.reshape(batch_size * seq_len, obs_dim)
            next_features = agent.actor_target.shared(next_flat).view(batch_size, seq_len, -1)
            next_lstm_out, _ = agent.actor_target.lstm(next_features, None)
            next_agent_hiddens.append(next_lstm_out)
        
        # ==========================================
        # 2. Critic Update (Eq 12-13)
        # ==========================================
        # Current Q
        # Use last timestep hidden states and actions
        current_hiddens = [h[:, -1, :] for h in agent_hiddens]
        current_actions = [actions[:, -1, i, :] for i in range(n_agents)]
        current_q, _ = self.critic(current_hiddens, current_actions)
        current_q = current_q.squeeze(-1)  # (batch,)
        
        # Target Q
        with torch.no_grad():
            # Target actions from all agents
            next_hiddens = [h[:, -1, :] for h in next_agent_hiddens]
            target_actions = []
            for i, agent in enumerate(all_agents):
                # Get next actions from target actors
                next_act, _ = agent.actor_target(obs[:, -1, i, :].unsqueeze(1), None)
                target_actions.append(next_act.squeeze(1))
            
            target_q, _ = self.critic_target(next_hiddens, target_actions)
            target_q = target_q.squeeze(-1)
            
            # Compute target value (Eq 13)
            # y = r + gamma * Q' * (1 - done)
            rewards_agent = rewards[:, -1, self.agent_id]
            dones_mask = (~dones[:, -1]).float()
            target_value = rewards_agent + self.gamma * target_q * dones_mask
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        
        # ==========================================
        # 3. Actor Update (Eq 11)
        # ==========================================
        # Freeze critic
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # Get actions from current policy for this agent
        # For other agents, use their current actions from batch (stop gradient)
        actor_actions = []
        for i, agent in enumerate(all_agents):
            if i == self.agent_id:
                act, _ = agent.actor(obs[:, -1, i, :].unsqueeze(1), None)
                actor_actions.append(act.squeeze(1))
            else:
                actor_actions.append(actions[:, -1, i, :].detach())
        
        # Compute Q for actor update
        actor_hiddens = [h[:, -1, :].detach() if i != self.agent_id else h[:, -1, :]
                        for i, h in enumerate(agent_hiddens)]
        q_value, _ = self.critic(actor_hiddens, actor_actions)
        
        # Policy gradient: maximize Q
        actor_loss = -q_value.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        
        # Unfreeze critic
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
        """Share lower-layer parameters (Section 10.1)."""
        self.shared_extractor = other_agent.shared_extractor
        self.actor.shared = self.shared_extractor
        self.actor_target.shared = self.shared_extractor
