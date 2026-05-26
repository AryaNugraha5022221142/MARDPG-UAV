"""
Episode-based replay buffer for BPTT.
Reference: Section 7.1 and 7.2 of blueprint.
"""
import random
import numpy as np
import torch
from collections import deque
from typing import Dict, List, Tuple


class Episode:
    def __init__(self):
        self.observations = []  # List of [n_agents, obs_dim]
        self.actions = []       # List of [n_agents, action_dim]
        self.rewards = []       # List of [n_agents]
        self.dones = []         # List of bool
    
    def append(self, obs, actions, rewards, done):
        self.observations.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(done)
    
    def length(self):
        return len(self.dones)


class EpisodeReplayBuffer:
    def __init__(self, capacity: int = 1000, seq_len: int = 80):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)
        
    def add_episode(self, episode: Episode):
        if episode.length() > 0:
            self.buffer.append(episode)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch of episode segments for BPTT.
        Returns tensors of shape (batch, seq_len, n_agents, dim)
        """
        episodes = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for ep in episodes:
            T = ep.length()
            if T < self.seq_len:
                continue
            
            # Random starting point (Section 7.2)
            start = random.randint(0, T - self.seq_len)
            end = start + self.seq_len
            
            # Stack episode data
            obs_seg = np.stack(ep.observations[start:end])      # (seq, n, obs)
            act_seg = np.stack(ep.actions[start:end])           # (seq, n, act)
            rew_seg = np.stack(ep.rewards[start:end])           # (seq, n)
            done_seg = np.array(ep.dones[start:end], dtype=bool) # (seq,)
            
            batch_obs.append(obs_seg)
            batch_actions.append(act_seg)
            batch_rewards.append(rew_seg)
            batch_dones.append(done_seg)
        
        if len(batch_obs) == 0:
            return None
        
        # Convert to tensors
        obs_tensor = torch.tensor(np.stack(batch_obs), dtype=torch.float32)
        act_tensor = torch.tensor(np.stack(batch_actions), dtype=torch.float32)
        rew_tensor = torch.tensor(np.stack(batch_rewards), dtype=torch.float32)
        done_tensor = torch.tensor(np.stack(batch_dones), dtype=torch.bool)
        
        return obs_tensor, act_tensor, rew_tensor, done_tensor
    
    def __len__(self):
        return len(self.buffer)
