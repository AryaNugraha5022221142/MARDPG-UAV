import numpy as np

class GaussianNoise:
    """Uncorrelated Gaussian exploration noise."""
    def __init__(self, n_agents: int, action_dim: int = 2, sigma: float = 0.1):
        self.n = n_agents
        self.dim = action_dim
        self.sigma = sigma

    def reset(self):
        pass

    def sample(self) -> np.ndarray:
        return np.random.normal(0, self.sigma, size=(self.n, self.dim)).astype(np.float32)

