import numpy as np

class GaussianNoise:
    """Uncorrelated Gaussian exploration noise with annealing."""

    def __init__(self, n_agents: int, action_dim: int=2, sigma: float=0.3, sigma_min: float=0.05, decay: float=0.9995):
        self.n = n_agents
        self.dim = action_dim
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay = decay

    def decay_sigma(self):
        self.sigma = max(self.sigma_min, self.sigma * self.decay)

    def sample(self) -> np.ndarray:
        return np.random.normal(0, self.sigma, size=(self.n, self.dim)).astype(np.float32)