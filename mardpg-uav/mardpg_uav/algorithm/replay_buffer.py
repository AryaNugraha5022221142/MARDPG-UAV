"""
Sequence replay buffer (v3) — audit fixes C1, C2, I2.

Changes vs v2
-------------
1. [C2] next_obs is stored EXPLICITLY per transition instead of being derived
   from obs[idx + 1] at sample time. This removes two silent corruption modes:
     (a) windows whose end+1 slot had been overwritten by a newer episode
         (the old invalidation range was off by one for that case), and
     (b) the full-buffer case, where `np.minimum(idx+1, size-1)` never
         clamps once size == capacity, so the final step of the newest
         window bootstrapped from the OLDEST data at the write pointer.

2. [C1] The artificial "episode-end padding transition" previously appended
   by train.py (zero action, zero reward, dones=True) is no longer needed
   and must NOT be added. That transition was being sampled and trained on
   as a genuine terminal, regressing Q(s_T, 0) -> 0 for every timed-out
   agent. With explicit next_obs, the last real transition of a timed-out
   episode keeps done=False and bootstraps correctly (time-limit truncation
   is no longer treated as a true terminal).

3. [I2] Episodes shorter than seq_len previously produced ZERO valid
   windows — quick-crash episodes (the highest-signal negative examples,
   especially right after a curriculum promotion) were invisible to
   learning. Short episodes are now right-padded with inert copies of
   their last transition (pad=True) so each yields exactly one valid
   window. Pad steps carry dones=True / r=0 but are excluded from every
   loss via the per-step `pad` mask returned by sample(); they only fill
   the tensor shape.

Sampling contract
-----------------
sample(batch_size) returns
    (obs, next_obs, prev_actions, actions, rewards, dones, pads)
with shapes
    obs/next_obs : (B, T, N, obs_dim)
    prev/actions : (B, T, N, act_dim)
    rewards      : (B, T, N)
    dones        : (B, T, N)   bool, status AFTER the step
    pads         : (B, T)      bool, True = inert padding step
Windows are always contiguous and contained in a single episode.
"""
import numpy as np
import torch


class SequenceReplayBuffer:
    def __init__(self, capacity: int = 100_000, seq_len: int = 90,
                 n_agents: int = 5, obs_dim: int = 43, action_dim: int = 2):
        self.capacity   = capacity
        self.seq_len    = seq_len
        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Circular buffer over transitions
        self.obs          = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.next_obs     = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.prev_actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.actions      = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards      = np.zeros((capacity, n_agents),             dtype=np.float32)
        self.dones        = np.ones ((capacity, n_agents),             dtype=bool)
        self.pads         = np.zeros( capacity,                        dtype=bool)
        self.ep_ids       = np.full ( capacity, -1,                    dtype=np.int64)

        self.ptr             = 0
        self.size            = 0
        self._ep_count       = 0
        self._current_ep_len = 0

        self.valid_mask = np.zeros(capacity, dtype=bool)

    # ------------------------------------------------------------------
    def add_transition(self, obs, prev_actions, actions, rewards,
                       next_obs, dones, pad: bool = False):
        """Store one environment transition (or one inert pad step)."""
        idx = self.ptr

        # Overwriting slot idx invalidates every window that CONTAINS idx,
        # i.e. windows starting in [idx - seq_len + 1, idx]. With next_obs
        # stored explicitly, no window reads outside [start, start+seq_len),
        # so this range is now exactly sufficient (old code additionally
        # leaked through the end+1 next-obs read).
        start_invalid = max(0, idx - self.seq_len + 1)
        self.valid_mask[start_invalid: idx + 1] = False

        self.obs[idx]          = obs
        self.next_obs[idx]     = next_obs
        self.prev_actions[idx] = prev_actions
        self.actions[idx]      = actions
        self.rewards[idx]      = rewards
        self.dones[idx]        = dones
        self.pads[idx]         = pad
        self.ep_ids[idx]       = self._ep_count

        self._current_ep_len += 1

        if self._current_ep_len >= self.seq_len:
            start_idx = idx - self.seq_len + 1
            # start_idx >= 0: windows are not allowed to wrap the array edge.
            # ep_id check is a cheap belt-and-braces guard against any future
            # change that writes an episode non-contiguously.
            if start_idx >= 0 and self.ep_ids[start_idx] == self._ep_count:
                self.valid_mask[start_idx] = True

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    def end_episode(self):
        """Close the current episode. Short episodes are padded to seq_len
        with inert copies of their final transition so they produce exactly
        one valid window (pad steps are masked out of all losses)."""
        ep_len = self._current_ep_len
        if 0 < ep_len < self.seq_len and self.size > 0:
            last     = (self.ptr - 1) % self.capacity
            obs_pad  = self.obs[last].copy()
            nobs_pad = self.next_obs[last].copy()
            pact_pad = self.actions[last].copy()          # masked anyway
            zero_a   = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
            zero_r   = np.zeros(self.n_agents, dtype=np.float32)
            all_done = np.ones(self.n_agents, dtype=bool)
            for _ in range(self.seq_len - ep_len):
                self.add_transition(obs_pad, pact_pad, zero_a, zero_r,
                                    nobs_pad, all_done, pad=True)
        self._ep_count       += 1
        self._current_ep_len  = 0

    # ------------------------------------------------------------------
    def sample(self, batch_size: int):
        valid_indices = np.nonzero(self.valid_mask)[0]
        if len(valid_indices) < batch_size:
            return None

        starts = np.random.choice(valid_indices, size=batch_size, replace=False)
        idx_t  = starts[:, None] + np.arange(self.seq_len)[None, :]      # (B, T)

        batch_obs      = self.obs[idx_t]            # (B, T, N, obs_dim)
        batch_next_obs = self.next_obs[idx_t]       # (B, T, N, obs_dim)
        batch_prev_act = self.prev_actions[idx_t]   # (B, T, N, act_dim)
        batch_act      = self.actions[idx_t]        # (B, T, N, act_dim)
        batch_rew      = self.rewards[idx_t]        # (B, T, N)
        batch_done     = self.dones[idx_t]          # (B, T, N)
        batch_pad      = self.pads[idx_t]           # (B, T)

        to_t = lambda a: torch.FloatTensor(a)
        to_b = lambda a: torch.BoolTensor(a)

        return (to_t(batch_obs), to_t(batch_next_obs),
                to_t(batch_prev_act), to_t(batch_act),
                to_t(batch_rew), to_b(batch_done), to_b(batch_pad))

    # ------------------------------------------------------------------
    # Persistence (audit fix C3 — needed for full resume of long runs)
    # ------------------------------------------------------------------
    def save(self, path: str):
        np.savez_compressed(
            path,
            obs=self.obs[:self.size], next_obs=self.next_obs[:self.size],
            prev_actions=self.prev_actions[:self.size],
            actions=self.actions[:self.size], rewards=self.rewards[:self.size],
            dones=self.dones[:self.size], pads=self.pads[:self.size],
            ep_ids=self.ep_ids[:self.size], valid_mask=self.valid_mask[:self.size],
            ptr=self.ptr, size=self.size,
            ep_count=self._ep_count, current_ep_len=self._current_ep_len)

    def load(self, path: str):
        d = np.load(path)
        n = int(d['size'])
        if n > self.capacity:
            raise ValueError(f"Saved buffer ({n}) larger than capacity ({self.capacity}).")
        self.obs[:n]          = d['obs'];          self.next_obs[:n] = d['next_obs']
        self.prev_actions[:n] = d['prev_actions']; self.actions[:n]  = d['actions']
        self.rewards[:n]      = d['rewards'];      self.dones[:n]    = d['dones']
        self.pads[:n]         = d['pads'];         self.ep_ids[:n]   = d['ep_ids']
        self.valid_mask[:]    = False
        self.valid_mask[:n]   = d['valid_mask']
        self.ptr              = int(d['ptr'])
        self.size             = n
        self._ep_count        = int(d['ep_count'])
        self._current_ep_len  = int(d['current_ep_len'])

    def __len__(self):
        return self.size

    def invalidate_oldest(self, frac: float) -> int:
        """[N3-4] Mark the OLDEST `frac` of currently-valid windows unsampleable.
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
        # Age = steps back from the write pointer in circular order.
        age = (self.ptr - 1 - valid) % self.capacity
        oldest = valid[np.argsort(-age)[:k]]
        self.valid_mask[oldest] = False
        return int(k)
