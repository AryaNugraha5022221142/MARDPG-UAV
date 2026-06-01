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
        self.observations = []
        self.actions = []
        self.rewards = []
        self.agent_dones = []   # List of np.ndarray shape [n_agents] — was bool

    def append(self, obs, actions, rewards, agent_dones):
        self.observations.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.agent_dones.append(agent_dones.copy())  # [n_agents] bool array

    def length(self):
        return len(self.agent_dones)


class EpisodeReplayBuffer:
    def __init__(self, capacity: int = 1000, seq_len: int = 80, max_steps: int = 250):
        self.capacity = capacity
        self.seq_len = seq_len
        self.max_ep_len = max(max_steps, seq_len) + 1
        
        self.obs_buffer = None
        self.act_buffer = None
        self.rew_buffer = None
        self.done_buffer = None
        self.ep_lens = np.zeros(capacity, dtype=np.int32)
        
        self.ptr = 0
        self.size = 0
        
    def add_episode(self, episode: Episode):
        T = episode.length()
        if T == 0:
            return
            
        if self.obs_buffer is None:
            n_agents = len(episode.agent_dones[0])
            obs_shape = episode.observations[0].shape
            act_shape = episode.actions[0].shape
            
            self.obs_buffer = np.zeros((self.capacity, self.max_ep_len) + obs_shape, dtype=np.float32)
            self.act_buffer = np.zeros((self.capacity, self.max_ep_len) + act_shape, dtype=np.float32)
            self.rew_buffer = np.zeros((self.capacity, self.max_ep_len, n_agents), dtype=np.float32)
            self.done_buffer = np.ones((self.capacity, self.max_ep_len, n_agents), dtype=bool)
            
        idx = self.ptr
        
        obs = np.stack(episode.observations)
        act = np.stack(episode.actions)
        rew = np.stack(episode.rewards)
        dones = np.stack(episode.agent_dones)
        
        pad_len = max(0, self.seq_len + 1 - T)
        
        self.ep_lens[idx] = T + pad_len
        
        self.obs_buffer[idx, :T] = obs
        self.act_buffer[idx, :T] = act
        self.rew_buffer[idx, :T] = rew
        self.done_buffer[idx, :T] = dones
        
        if pad_len > 0:
            # fill neutral obs
            neutral_obs = np.zeros_like(obs[0])
            if neutral_obs.ndim == 2:
                neutral_obs[:, 6:31] = 1.0
            
            self.obs_buffer[idx, T:T+pad_len] = neutral_obs
            self.act_buffer[idx, T:T+pad_len] = 0.0
            self.rew_buffer[idx, T:T+pad_len] = 0.0
            self.done_buffer[idx, T:T+pad_len] = True
            
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if self.size == 0:
            return None
            
        batch_idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch_obs = np.zeros((batch_size, self.seq_len) + self.obs_buffer.shape[2:], dtype=np.float32)
        batch_obs_next = np.zeros((batch_size, self.seq_len) + self.obs_buffer.shape[2:], dtype=np.float32)
        batch_actions = np.zeros((batch_size, self.seq_len) + self.act_buffer.shape[2:], dtype=np.float32)
        batch_rewards = np.zeros((batch_size, self.seq_len) + self.rew_buffer.shape[2:], dtype=np.float32)
        batch_dones = np.zeros((batch_size, self.seq_len) + self.done_buffer.shape[2:], dtype=bool)
        
        for i, idx in enumerate(batch_idxs):
            T = self.ep_lens[idx]
            if T <= self.seq_len:
                start = 0
            else:
                start = np.random.randint(0, T - self.seq_len)
            end = start + self.seq_len
            
            batch_obs[i] = self.obs_buffer[idx, start:end]
            batch_obs_next[i] = self.obs_buffer[idx, start+1:end+1]
            batch_actions[i] = self.act_buffer[idx, start:end]
            batch_rewards[i] = self.rew_buffer[idx, start:end]
            batch_dones[i] = self.done_buffer[idx, start:end]
            
        return (
            torch.tensor(batch_obs),
            torch.tensor(batch_obs_next),
            torch.tensor(batch_actions),
            torch.tensor(batch_rewards),
            torch.tensor(batch_dones)
        )
    
    def __len__(self):
        return self.size
