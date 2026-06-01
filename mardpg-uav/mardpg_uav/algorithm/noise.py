"""
Gaussian exploration noise with sigmoid annealing.
Reference: Section 11.1 of blueprint.
"""
import numpy as np

class GaussianNoise:
    """Gaussian exploration noise representing physical sensor/actuator uncertainty with sigmoid decay."""
    def __init__(self, n_agents, action_dim=3, kappa=0.15,
                 sigma0=0.25, sigma_inf=0.05, anneal_steps=3_000_000):
        self.n = n_agents
        self.dim = action_dim
        self.sigma0 = sigma0
        self.sigma_inf = sigma_inf
        self.anneal_steps = anneal_steps
        self.total_steps = 0
        self.current_sigma = sigma0

    def reset(self):
        pass # Pure Gaussian requires no episode-boundary resets
        
    def get_sigma(self):
        return self.current_sigma

    def sample(self, dt=0.1):
        decay = 1.0 / (1.0 + np.exp((self.total_steps - self.anneal_steps / 2) / (self.anneal_steps / 10.0)))
        sigma_t = self.sigma_inf + (self.sigma0 - self.sigma_inf) * decay
        self.current_sigma = sigma_t
        self.total_steps += 1
        
        return np.random.normal(0, sigma_t, size=(self.n, self.dim)).astype(np.float32)
