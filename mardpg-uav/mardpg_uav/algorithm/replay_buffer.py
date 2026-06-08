"""
Replay buffer faithful to Algorithm 1 of Xue & Chen (2024).
Stores full episodes; samples random length-T windows for BPTT.
Buffer capacity expressed in transitions (10^5).
"""
import numpy as np, torch

class SequenceReplayBuffer:
    """
    Stores complete episodes. Samples contiguous windows of length seq_len
    for BPTT, constructing histories h_{i,t} as in Algorithm 1.
    """
    def __init__(self, capacity: int = 100_000, seq_len: int = 80,
                 n_agents: int = 5, obs_dim: int = 30, action_dim: int = 2):
        self.capacity   = capacity
        self.seq_len    = seq_len
        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Circular buffer over transitions
        self.obs     = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents),             dtype=np.float32)
        self.dones   = np.ones ((capacity, n_agents),             dtype=bool)
        self.ep_ids  = np.zeros( capacity,                        dtype=np.int32)

        self.ptr       = 0
        self.size      = 0
        self._ep_count = 0

    def add_transition(self, obs, actions, rewards, dones):
        idx = self.ptr
        self.obs[idx]     = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx]   = dones
        self.ep_ids[idx]  = self._ep_count
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def end_episode(self):
        self._ep_count += 1

    def sample(self, batch_size: int):
        """
        Sample batch_size windows of length seq_len from the same episode.
        Returns arrays shaped (batch_size, seq_len, ...).
        """
        valid   = []
        ep_ids  = self.ep_ids[:self.size]
        indices = np.arange(self.size)

        # Collect all valid start indices (window stays within one episode)
        for start in range(self.size - self.seq_len):
            end = start + self.seq_len
            if np.all(ep_ids[start:end] == ep_ids[start]):
                valid.append(start)

        if len(valid) < batch_size:
            return None

        starts = np.random.choice(valid, size=batch_size, replace=False)
        idx_t  = np.stack([np.arange(s, s + self.seq_len) for s in starts])   # (B, T)

        def gather(buf):
            return buf[idx_t]

        batch_obs     = gather(self.obs)       # (B, T, N, obs_dim)
        batch_act     = gather(self.actions)   # (B, T, N, act_dim)
        batch_rew     = gather(self.rewards)   # (B, T, N)
        batch_done    = gather(self.dones)     # (B, T, N)

        # Next obs: clamp to episode boundary
        next_idx = np.minimum(idx_t + 1, self.size - 1)
        batch_obs_next = self.obs[next_idx]

        to_t = lambda a: torch.FloatTensor(a)
        to_b = lambda a: torch.BoolTensor(a)

        return (to_t(batch_obs),      to_t(batch_obs_next),
                to_t(batch_act),      to_t(batch_rew),
                to_b(batch_done))

    def __len__(self):
        return self.size
