import numpy as np
from visualize_eval import plot_trajectory_3d, plot_trajectory_top_down, animate

class DummyEnv:
    def __init__(self):
        self.obstacles = []
        self.dynamic_obstacles = []

env = DummyEnv()
env_cfg = {'n_agents': 2, 'env_size': [50.0, 50.0, 60.0]}
render = {
    'path': np.zeros((10, 2, 3)),
    'dyn_path': np.zeros((10, 0)),
    'dyn_r': np.array([]),
    'reached': np.array([False, False]),
    'collided': np.array([False, False]),
    'goals': np.zeros((2, 3))
}

plot_trajectory_3d(env, env_cfg, render, "Title", "test_3d.png")
plot_trajectory_top_down(env, env_cfg, render, "Title", "test_2d.png")
animate(env, env_cfg, render['path'], render['dyn_path'], render['dyn_r'], render['goals'], "Title", "test_vid.mp4")
print("Success!")
