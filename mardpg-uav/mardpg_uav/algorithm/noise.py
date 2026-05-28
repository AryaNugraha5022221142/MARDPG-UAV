"""
Gaussian exploration noise with linear decay.
Reference: Section 11.1 of blueprint.
"""
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process with exponential annealing — §4.3 Eqs.17-19."""
    def __init__(self, n_agents, action_dim=2, kappa=0.15,
                 sigma0=0.25, sigma_inf=0.05, anneal_steps=3_000_000):
        self.n = n_agents
        self.dim = action_dim
        self.kappa = kappa
        self.sigma0 = sigma0
        self.sigma_inf = sigma_inf
        self.anneal_steps = anneal_steps
        self.epsilon = np.zeros((n_agents, action_dim), dtype=np.float32)
        self.total_steps = 0
        self.current_sigma = sigma0

    def reset(self):
        self.epsilon[:] = 0.0   # reset OU state per episode
        
    def get_sigma(self):
        return self.current_sigma

    def sample(self, dt=0.1):
        """Update and return noise for all agents — OU discrete update with Sigmoid decay."""
        # Sigmoid decay: scale = 1 / (1 + exp((total_steps - anneal_steps/2) / (anneal_steps/10)))
        decay = 1.0 / (1.0 + np.exp((self.total_steps - self.anneal_steps / 2) / (self.anneal_steps / 10.0)))
        sigma_t = self.sigma_inf + (self.sigma0 - self.sigma_inf) * decay
        self.current_sigma = sigma_t
        xi = np.random.randn(self.n, self.dim).astype(np.float32)
        self.epsilon = (self.epsilon * (1 - self.kappa * dt)
                        + sigma_t * np.sqrt(dt) * xi)                   # OU
        self.total_steps += 1
        return self.epsilon.copy()
