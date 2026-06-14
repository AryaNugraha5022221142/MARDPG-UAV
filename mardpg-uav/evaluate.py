"""
Evaluation script for MARDPG-NAV.
Reference: Section 12 of blueprint.
"""
import yaml
import torch
import numpy as np
from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.eval_rollout import run_eval, make_learned_act_fn, load_agents

def evaluate(checkpoint_dir: str, config_path: str = "config/default.yaml",
             n_eval_episodes: int = 250, device: str = 'cpu', static_obs: int = 16):
    import os
    if not os.path.exists(config_path):
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        if os.path.exists(fallback):
            config_path = fallback
            
    agents, cfg = load_agents(checkpoint_dir, config_path, device)
    env_cfg = cfg['environment']
    env = MultiUAVEnv(env_cfg)
    
    if static_obs == 16:
        # Default to stage 7 (Dynamic Threats) if using the final density
        final_stage_cfg = {
            'env_size': [100.0, 100.0, 60.0], 'static_obs': static_obs,
            'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
            'min_sep': 40.0, 'max_steps': 1500
        }
    else:
        final_stage_cfg = {
            'env_size': [100.0, 100.0, 60.0], 'static_obs': static_obs,
            'min_sep': 40.0, 'max_steps': 1500
        }
    
    act_fn, on_start = make_learned_act_fn(agents, env)
    stats, _ = run_eval(env, final_stage_cfg, act_fn, n_episodes=n_eval_episodes, base_seed=42, on_episode_start=on_start)
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success Rate:  {stats['success_rate']:.2%}")
    print(f"Collision Rate: {stats['collision_rate']:.2%}")
    print(f"Trapped Rate:   {stats['trapped_rate']:.2%}")
    if 'path_efficiency' in stats:
        print(f"Path Efficiency: {stats['path_efficiency']:.3f}")
    print("=" * 50)
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint directory')
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--episodes', type=int, default=250)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--static-obs', type=int, default=16,
                        help='16 = final training stage; 20 = out-of-distribution '
                             'generalization test (report separately).')
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.config, args.episodes, args.device, args.static_obs)
