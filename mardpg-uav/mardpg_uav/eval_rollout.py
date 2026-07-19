import numpy as np
import torch
import os
from mardpg_uav.utils.metrics import MetricsTracker
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from scripts.train import load_config
_VARIANT_FLAGS = {'mardpg': (True, True), 'maddpg': (False, True), 'ind_rdpg': (True, False), 'iddpg': (False, False)}

def load_agents(checkpoint_dir, config_path, device='cpu', variant='mardpg', recurrent=None, centralized=None):
    from scripts.evaluate_multiagent import load_agents_strict
    agents, cfg = load_agents_strict(checkpoint_dir, config_path, device, variant)
    return agents, cfg

