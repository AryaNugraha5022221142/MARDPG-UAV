import os
import yaml
import numpy as np
from mardpg_uav.environment.uav_env import MultiUAVEnv

with open(os.path.join(os.path.dirname(__file__), '../config/default.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

def test_obs_shape():
    env = MultiUAVEnv(cfg)
    obs = env.reset()
    assert obs.shape == (env.n_agents, env.obs_dim), f"Got {obs.shape}"
    assert env.obs_dim == 46
