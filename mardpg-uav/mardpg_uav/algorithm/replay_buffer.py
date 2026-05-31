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
        batch_obs = []
        batch_obs_next = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        attempts = 0
        max_attempts = batch_size * 20
        
        while len(batch_obs) < batch_size and attempts < max_attempts:
            ep = random.choice(self.buffer)
            attempts += 1
            
            T = ep.length()
            if T < self.seq_len + 1:
                # Zero padding for short episodes
                pad_len = self.seq_len + 1 - T
                n_agents = len(ep.agent_dones[0])
                obs_shape = ep.observations[0].shape
                act_shape = ep.actions[0].shape
                
                def _make_neutral_obs(shape):
                    obs = np.zeros(shape, dtype=np.float32)
                    if len(shape) == 2:
                        obs[:, 6:31] = 1.0  # lidar at max range
                        obs[:, 0] = 0.0; obs[:, 1] = 1.0
                        obs[:, 2] = 0.0; obs[:, 3] = 1.0
                    return obs
                
                obs_pad = [_make_neutral_obs(obs_shape) for _ in range(pad_len)]
                act_pad = [np.zeros(act_shape, dtype=np.float32) for _ in range(pad_len)]
                rew_pad = [np.zeros(n_agents, dtype=np.float32) for _ in range(pad_len)]
                done_pad = [np.ones(n_agents, dtype=bool) for _ in range(pad_len)] # Padding states act as absorbing terminal states
                
                ep_obs = ep.observations + obs_pad
                ep_act = ep.actions + act_pad
                ep_rew = ep.rewards + rew_pad
                ep_dones = ep.agent_dones + done_pad
                
                start = 0
                end = self.seq_len
                
                batch_obs.append(np.stack(ep_obs[start:end]))
                batch_obs_next.append(np.stack(ep_obs[start+1:end+1]))
                batch_actions.append(np.stack(ep_act[start:end]))
                batch_rewards.append(np.stack(ep_rew[start:end]))
                batch_dones.append(np.array(ep_dones[start:end], dtype=bool))
            else:
                # Random starting point
                start = random.randint(0, T - self.seq_len - 1)
                end = start + self.seq_len
                
                # Stack episode data
                batch_obs.append(np.stack(ep.observations[start:end]))
                batch_obs_next.append(np.stack(ep.observations[start+1:end+1]))
                batch_actions.append(np.stack(ep.actions[start:end]))
                batch_rewards.append(np.stack(ep.rewards[start:end]))
                batch_dones.append(np.array(ep.agent_dones[start:end], dtype=bool))
            
        if len(batch_obs) < batch_size:
            return None
        
        # Convert to tensors
        obs_tensor = torch.tensor(np.stack(batch_obs), dtype=torch.float32)
        obs_next_tensor = torch.tensor(np.stack(batch_obs_next), dtype=torch.float32)
        act_tensor = torch.tensor(np.stack(batch_actions), dtype=torch.float32)
        rew_tensor = torch.tensor(np.stack(batch_rewards), dtype=torch.float32)
        done_tensor = torch.tensor(np.stack(batch_dones),
                                   dtype=torch.bool)  # shape (B, seq, n_agents)
        
        return obs_tensor, obs_next_tensor, act_tensor, rew_tensor, done_tensor
    
    def __len__(self):
        return len(self.buffer)
