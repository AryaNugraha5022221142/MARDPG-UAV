import sys
import os
import time
import math
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.eval_rollout import load_agents

class LearnedPolicy:
    def __init__(self, agents, name):
        self.agents = agents
        self.name = name

    def reset(self, env):
        self._prev = [np.zeros(env.action_dim, np.float32) for _ in range(env.n_agents)]
        for ag in self.agents:
            ag.actor.eval()
            ag.reset_hidden(batch_size=1, eval_mode=True)

    def act(self, env, obs):
        acts = []
        for (i, ag) in enumerate(self.agents):
            if env.agent_done[i]:
                acts.append(np.zeros(env.action_dim, np.float32))
            else:
                a = ag.select_action(obs[i], self._prev[i], evaluate=True)
                assert a.shape == (env.action_dim,), f"Agent {i} returned {a.shape}, expected {(env.action_dim,)}"
                acts.append(np.clip(a, -ag.actor.action_bound, ag.actor.action_bound))
        acts = np.array(acts, np.float32)
        assert acts.shape == (env.n_agents, env.action_dim), f"actions shape = {acts.shape}"
        self._prev = acts.copy()
        return acts

def build_base_scenarios(env_cfg):
    s_cfg = dict(env_size=[100.0, 100.0, 60.0], min_start_sep=12.0, static_obs=16, max_h=50.0, min_sep=40.0, max_steps=1500)
    return {
        'S1_Static_Dynamic': dict(s_cfg, static_obs=20, dynamic_obs=2),
        'S3_Fast_Dynamic': dict(s_cfg, static_obs=25, dynamic_obs=3)
    }

def run_episode(env, policy, stage_cfg, env_cfg, seed):
    # Call generalization's run_episode
    import scripts.evaluate_generalization as eg
    (ep, ag, rnd) = eg.run_episode(env, policy, stage_cfg, env_cfg, seed)
    ep.update(rnd)
    return ep, ag

def build_experiment_configs():
    return {'sensor_noise': [{'lidar_noise': 0.0, 'exp_val': 'sigma=0.0'}, {'lidar_noise': 0.1, 'exp_val': 'sigma=0.1'}, {'lidar_noise': 0.2, 'exp_val': 'sigma=0.2'}, {'lidar_noise': 0.3, 'exp_val': 'sigma=0.3'}, {'lidar_noise': 0.4, 'exp_val': 'sigma=0.4'}, {'lidar_noise': 0.5, 'exp_val': 'sigma=0.5'}], 'variable_speed': [{'variable_speed': False, 'exp_val': 'Constant'}, {'variable_speed': True, 'exp_val': 'Dynamic'}]}

def run_evaluations(args, wlogger=None):
    cfg = yaml.safe_load(open(args.config))
    env_cfg = cfg['environment']
    (agents, _) = load_agents(args.checkpoint, args.config, args.device, variant=args.variant)
    policy = LearnedPolicy(agents, name='MARDPG')
    
    experiments = build_experiment_configs()
    base_scenarios = build_base_scenarios(env_cfg)
    for (exp_name, exp_sweeps) in experiments.items():
        exp_dir = os.path.join(args.outdir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        ep_records = []
        ag_records = []
        for sweep in exp_sweeps:
            exp_val = sweep['exp_val']
            for (scenario_name, scenario_cfg) in base_scenarios.items():
                t0 = time.time()
                stage_cfg = dict(scenario_cfg)
                stage_cfg.update({k: v for (k, v) in sweep.items() if k != 'exp_val'})
                import copy
                env = MultiUAVEnv(copy.deepcopy(env_cfg))
                
                sub = []
                for ep in range(args.episodes):
                    seed = args.base_seed + ep
                    if scenario_name == 'S1_Static_Dynamic':
                        rng_tmp = np.random.RandomState(seed)
                        stage_cfg['dynamic_obs'] = rng_tmp.randint(1, 3)
                    elif scenario_name == 'S3_Fast_Dynamic':
                        rng_tmp = np.random.RandomState(seed)
                        stage_cfg['dynamic_obs'] = rng_tmp.randint(2, 4)
                    (ep_stats, ag_stats) = run_episode(env, policy, stage_cfg, env_cfg, seed)
                    ep_stats.update(experiment=exp_name, condition=exp_val, scenario=scenario_name)
                    ep_records.append(ep_stats)
                    ag_records.extend(ag_stats)
                    sub.append(ep_stats)
                    
                cr = np.mean([r['collision_rate'] for r in sub])
                best_ep = max(sub, key=lambda x: (x['mission_success'], x['success_rate'], x['team_reward'], -x['steps']))
                if not args.no_render:
                    try:
                        from mardpg_uav.rendering import RenderConfig
                        from mardpg_uav.rendering.media import generate_episode_media
                        title = f"Robustness | {exp_name}={exp_val} | {scenario_name} (seed {best_ep['seed']}) | SR {best_ep['success_rate']:.0%}"
                        tag = f"best_{scenario_name}_{exp_val}"
                        rcfg = RenderConfig(enable_render=True, record_video=True,
                                            save_png=True, output_directory=exp_dir)
                        produced = generate_episode_media(env, env_cfg, best_ep, rcfg,
                                                          tag, title, exp_dir)
                        if args.wandb and wlogger:
                            wlogger.log_media(
                                produced,
                                prefix=f"video/{exp_name}/{scenario_name}/{exp_val}")
                    except Exception as e:
                        pass
                
                for ep in sub:
                    ep.pop('path', None)
                    ep.pop('dyn_path', None)
                    ep.pop('dyn_r', None)
                    ep.pop('reached', None)
                    ep.pop('collided', None)
                    ep.pop('goals', None)
                    
        df_ep = pd.DataFrame(ep_records)
        df_ag = pd.DataFrame(ag_records)
        df_ep.to_csv(os.path.join(exp_dir, 'eval_episodes.csv'), index=False)
        df_ag.to_csv(os.path.join(exp_dir, 'eval_agents.csv'), index=False)
        
        def agg(g):
            n = len(g)
            out = dict(n_episodes=n)
            for col in ['success_rate', 'collision_rate', 'dyn_collision_rate', 'path_eff_paper', 'path_eff_reached', 'flight_time_s']:
                v = g[col].astype(float)
                out[f'{col}_mean'] = np.nanmean(v)
            return pd.Series(out)
        
        df_sum = df_ep.groupby(['scenario', 'condition']).apply(agg).reset_index()
        df_sum.to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)
        
        if args.wandb and wlogger:
            for (_, row) in df_sum.iterrows():
                wlogger._wandb.log({f"{exp_name}/{row['scenario']}/{row['condition']}/success_rate": row['success_rate_mean'], f"{exp_name}/{row['scenario']}/{row['condition']}/collision_rate": row['collision_rate_mean'], f"{exp_name}/{row['scenario']}/{row['condition']}/flight_time_s": row['flight_time_s_mean']})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint to evaluate')
    p.add_argument('--config', default='config/default.yaml', help='Base config file')
    p.add_argument('--variant', default='mardpg', choices=['mardpg', 'maddpg', 'iddpg'], help='Variant architecture')
    p.add_argument('--episodes', type=int, default=50, help='Episodes per condition/scenario')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='robustness_results')
    p.add_argument('--base-seed', type=int, default=20000)
    p.add_argument('--no-render', action='store_true', help='Disable generating trajectory plots and animations')
    p.add_argument('--wandb', action='store_true', help='Log results to W&B')
    p.add_argument('--wandb-project', default='mardpg-uav-eval')
    p.add_argument('--wandb-name', default=None)
    p.add_argument('--suite', default='quick', choices=['quick', 'full'], help='For compatibility, currently ignored as we run fixed scenarios')
    args = p.parse_args()
    
    wlogger = None
    if args.wandb:
        from mardpg_uav.wandb_logger import WandbLogger
        wlogger = WandbLogger(True, args.wandb_project, args.config, args.wandb_name or f'Robustness_{os.path.basename(os.path.dirname(args.checkpoint))}')
        if wlogger.use_wandb:
            wlogger._wandb.config.update(vars(args))
    
    run_evaluations(args, wlogger)
    
    if wlogger and wlogger.use_wandb:
        wlogger._wandb.finish()

if __name__ == '__main__':
    main()
