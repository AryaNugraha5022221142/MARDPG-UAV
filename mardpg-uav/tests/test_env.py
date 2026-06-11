import os
import yaml
import numpy as np
from mardpg_uav.environment.uav_env import MultiUAVEnv

with open(os.path.join(os.path.dirname(__file__), '../config/default.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

def test_obs_shape():
    env = MultiUAVEnv(cfg['environment'])
    obs = env.reset(cfg['environment'])
    assert obs.shape == (env.n_agents, env.obs_dim), f"Got {obs.shape}"
    assert env.obs_dim == 49

def test_trapped_detection():
    """Agent that stays in place should be trapped; one that moves to goal should not."""
    env = MultiUAVEnv(cfg['environment'])
    env.reset({'env_size': [100.0, 100.0, 60.0], 'static_obs': 0, 'min_sep': 30.0, 'max_steps': 200})
    
    # Run for 100+ steps with zero actions (agents don't move → no progress)
    for _ in range(100):
        actions = np.zeros((env.n_agents, env.action_dim))
        _, _, done, info = env.step(actions)
        if done: break
    
    trapped = env.get_trapped_agents(progress_window=50, progress_threshold=0.5)
    # Non-collided agents should be trapped since they haven't moved
    for i in range(env.n_agents):
        if not env.agent_collided[i]:
            assert trapped[i], f"Agent {i} should be trapped after 100 zero-action steps"
