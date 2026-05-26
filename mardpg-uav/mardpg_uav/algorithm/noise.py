"""
Gaussian exploration noise with linear decay.
Reference: Section 11.1 of blueprint.
"""
import numpy as np


class GaussianNoise:
    def __init__(self, dim: int, std_start: float = 0.2,
                 std_end: float = 0.05, decay_episodes: int = 10000):
        self.dim = dim
        self.std_start = std_start
        self.std_end = std_end
        self.decay_episodes = decay_episodes
        self.episode = 0
        
    def sample(self) -> np.ndarray:
        std = max(self.std_end,
                  self.std_start - (self.std_start - self.std_end) *
                  (self.episode / self.decay_episodes))
        return np.random.normal(0, std, size=self.dim)
    
    def step(self):
        self.episode += 1
