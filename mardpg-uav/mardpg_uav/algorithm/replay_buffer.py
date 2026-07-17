import numpy as np
import torch

class SequenceReplayBuffer:

    def __init__(self, capacity: int=100000, seq_len: int=90, n_agents: int=5, obs_dim: int=49, action_dim: int=2, tail_pad: int=None):
        self.capacity = capacity
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.tail_pad = tail_pad if tail_pad is not None else seq_len - 1
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.prev_actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.dones = np.ones((capacity, n_agents), dtype=bool)
        self.pads = np.zeros(capacity, dtype=bool)
        self.ep_ids = np.full(capacity, -1, dtype=np.int64)
        self.ptr = 0
        self.size = 0
        self._ep_count = 0
        self._current_ep_len = 0
        self.valid_mask = np.zeros(capacity, dtype=bool)

    def add_transition(self, obs, prev_actions, actions, rewards, next_obs, dones, pad: bool=False):
        """Store one environment transition (or one inert pad step)."""
        idx = self.ptr
        
        # Store the transition data
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.prev_actions[idx] = prev_actions
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.pads[idx] = pad
        self.ep_ids[idx] = self._ep_count
        self._current_ep_len += 1
        
        # Only mark as valid if we have a complete window ending at this position
        # A window is valid if:
        # 1. We have at least seq_len transitions in this episode
        # 2. The window [idx-seq_len+1, idx] all belong to the same episode
        if self._current_ep_len >= self.seq_len:
            start_idx = idx - self.seq_len + 1
            # Handle wraparound case
            if start_idx < 0:
                # Window wraps around - check if all positions in the window are from current ep
                # Positions [0, idx] and [capacity+start_idx, capacity-1] should be current ep
                wrap_start = self.capacity + start_idx
                all_current_ep = (
                    np.all(self.ep_ids[wrap_start:] == self._ep_count) and
                    np.all(self.ep_ids[:idx + 1] == self._ep_count)
                )
                if all_current_ep:
                    self.valid_mask[idx] = True
            else:
                # No wraparound - just check if start position is from current episode
                if self.ep_ids[start_idx] == self._ep_count:
                    self.valid_mask[idx] = True
                    
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def end_episode(self):
        """Close the current episode and append tail-pad steps (inert copies of the
        last transition, pad=True) so every real transition — including the terminal
        step that ends the episode — appears across the full window-position range,
        not just once. Short episodes are still guaranteed >= 1 window."""
        ep_len = self._current_ep_len
        if ep_len > 0 and self.size > 0:
            n_pad = max(self.tail_pad, self.seq_len - ep_len)
            if n_pad > 0:
                last = (self.ptr - 1) % self.capacity
                obs_pad = self.obs[last].copy()
                nobs_pad = self.next_obs[last].copy()
                pact_pad = self.actions[last].copy()
                zero_a = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
                zero_r = np.zeros(self.n_agents, dtype=np.float32)
                all_done = np.ones(self.n_agents, dtype=bool)
                for _ in range(n_pad):
                    self.add_transition(obs_pad, pact_pad, zero_a, zero_r, nobs_pad, all_done, pad=True)
        self._ep_count += 1
        self._current_ep_len = 0

    def sample(self, batch_size: int):
        valid_end_indices = np.nonzero(self.valid_mask)[0]
        if len(valid_end_indices) < batch_size:
            return None

        # valid_mask marks the END of a valid [idx-seq_len+1, idx] window.
        # Convert to start indices so idx_t is correctly aligned.
        start_candidates = valid_end_indices - (self.seq_len - 1)

        # Guard: drop any that would go negative (buffer just started)
        safe_mask    = start_candidates >= 0
        start_pool   = start_candidates[safe_mask]
        if len(start_pool) < batch_size:
            return None

        chosen_idx = np.random.choice(len(start_pool), size=batch_size, replace=False)
        starts = start_pool[chosen_idx]
        idx_t = starts[:, None] + np.arange(self.seq_len)[None, :]
        batch_obs = self.obs[idx_t]
        batch_next_obs = self.next_obs[idx_t]
        batch_prev_act = self.prev_actions[idx_t]
        batch_act = self.actions[idx_t]
        batch_rew = self.rewards[idx_t]
        batch_done = self.dones[idx_t]
        batch_pad = self.pads[idx_t]
        to_t = lambda a: torch.FloatTensor(a)
        to_b = lambda a: torch.BoolTensor(a)
        return (to_t(batch_obs), to_t(batch_next_obs), to_t(batch_prev_act), to_t(batch_act), to_t(batch_rew), to_b(batch_done), to_b(batch_pad))

    def save(self, path: str):
        import os, tempfile
        d = os.path.dirname(os.path.abspath(path)) or '.'
        os.makedirs(d, exist_ok=True)
        (fd, tmp) = tempfile.mkstemp(suffix='.tmp', dir=d)
        try:
            with os.fdopen(fd, 'wb') as f:
                np.savez_compressed(f, obs=self.obs[:self.size], next_obs=self.next_obs[:self.size], prev_actions=self.prev_actions[:self.size], actions=self.actions[:self.size], rewards=self.rewards[:self.size], dones=self.dones[:self.size], pads=self.pads[:self.size], ep_ids=self.ep_ids[:self.size], valid_mask=self.valid_mask[:self.size], ptr=self.ptr, size=self.size, ep_count=self._ep_count, current_ep_len=self._current_ep_len)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def load(self, path: str):
        d = np.load(path)
        n = int(d['size'])
        if n > self.capacity:
            raise ValueError(f'Saved buffer ({n}) larger than capacity ({self.capacity}).')
        self.obs[:n] = d['obs']
        self.next_obs[:n] = d['next_obs']
        self.prev_actions[:n] = d['prev_actions']
        self.actions[:n] = d['actions']
        self.rewards[:n] = d['rewards']
        self.dones[:n] = d['dones']
        self.pads[:n] = d['pads']
        self.ep_ids[:n] = d['ep_ids']
        self.valid_mask[:] = False
        self.valid_mask[:n] = d['valid_mask']
        self.ptr = int(d['ptr'])
        self.size = n
        self._ep_count = int(d['ep_count'])
        self._current_ep_len = int(d['current_ep_len'])

    def __len__(self):
        return self.size

    def invalidate_oldest(self, frac: float) -> int:
        """Mark the OLDEST `frac` of currently-valid windows unsampleable.
        Used on curriculum promotion to de-emphasise stale previous-stage data
        (the slots are reclaimed normally as new transitions overwrite them).
        Returns the number of windows invalidated."""
        if frac <= 0.0:
            return 0
        valid = np.nonzero(self.valid_mask)[0]
        if len(valid) == 0:
            return 0
        k = int(len(valid) * float(frac))
        if k <= 0:
            return 0
        
        # Calculate age correctly for circular buffer.
        # Age = how many steps ago was this index written?
        # For a circular buffer, we need to handle wrap-around:
        age = (self.ptr - 1 - valid) % self.capacity
        
        oldest = valid[np.argsort(-age)[:k]]
        self.valid_mask[oldest] = False
        return int(k)
