import sys
import os
import time
import math
import argparse
import yaml
import numpy as np
import pandas as pd
import torch


from mardpg_uav.environment.uav_env import MultiUAVEnv
from scripts.evaluate_multiagent import load_agents_strict as load_agents

class LearnedPolicy:
    def __init__(self, agents, name):
        self.agents = agents
        self.name = name
        self.current_agents = agents

    def reset(self, env):
        self._prev = [np.zeros(env.action_dim, np.float32) for _ in range(env.n_agents)]
        
        if env.n_agents > len(self.agents):
            import copy
            new_agents = []
            for i in range(env.n_agents):
                if i < len(self.agents):
                    new_agents.append(self.agents[i])
                else:
                    ag_copy = copy.deepcopy(self.agents[i % len(self.agents)])
                    ag_copy.agent_id = i
                    new_agents.append(ag_copy)
            self.current_agents = new_agents
        else:
            self.current_agents = self.agents[:env.n_agents]

        for ag in self.current_agents:
            ag.actor.eval()
            ag.reset_hidden(batch_size=1, eval_mode=True)

    def act(self, env, obs):
        acts = []
        for (i, ag) in enumerate(self.current_agents):
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
    s_cfg = dict(env_size=[100.0, 100.0, 60.0], min_start_sep=12.0, static_obs=16, max_h=50.0, min_sep=40.0, max_steps=1500, dynamic_radius=2.0, dynamic_speed=(1.0, 2.0), conflict_frac=1.0, ring_frac=0.35)
    return {
        'S1_Static_Dynamic': dict(s_cfg, static_obs=20, dynamic_obs=2),
        'S3_Fast_Dynamic': dict(s_cfg, static_obs=25, dynamic_obs=3)
    }

def run_episode(env, policy, stage_cfg, env_cfg, seed, live=None):
    # Call generalization's run_episode
    import scripts.evaluate_generalization as eg
    (ep, ag, rnd) = eg.run_episode(env, policy, stage_cfg, env_cfg, seed, live=live)
    ep.update(rnd)
    return ep, ag

def build_experiment_configs():
    return {'sensor_noise': [{'lidar_noise': 0.0, 'exp_val': 'sigma=0.0'}, {'lidar_noise': 0.1, 'exp_val': 'sigma=0.1'}, {'lidar_noise': 0.2, 'exp_val': 'sigma=0.2'}, {'lidar_noise': 0.3, 'exp_val': 'sigma=0.3'}, {'lidar_noise': 0.4, 'exp_val': 'sigma=0.4'}, {'lidar_noise': 0.5, 'exp_val': 'sigma=0.5'}]}

def run_evaluations(args, wlogger=None):
    cfg = yaml.safe_load(open(args.config))
    env_cfg = cfg['environment']
    (agents, _) = load_agents(args.checkpoint, args.config, args.device, variant=args.variant)
    policy = LearnedPolicy(agents, name='MARDPG')
    
    try:
        from mardpg_uav.rendering import select_backend, LiveRenderer
        _HAVE_RENDER = True
    except Exception:
        _HAVE_RENDER = False
        
    if not args.no_render and _HAVE_RENDER:
        select_backend('auto', want_interactive=args.realtime)
        
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
                
                live = None
                if not args.no_render and args.realtime and _HAVE_RENDER:
                    live = LiveRenderer(env, env_cfg)
                
                sub = []
                for ep in range(args.episodes):
                    seed = args.base_seed + ep
                    if scenario_name == 'S1_Static_Dynamic':
                        rng_tmp = np.random.RandomState(seed)
                        stage_cfg['dynamic_obs'] = rng_tmp.randint(1, 3)
                    elif scenario_name == 'S3_Fast_Dynamic':
                        rng_tmp = np.random.RandomState(seed)
                        stage_cfg['dynamic_obs'] = rng_tmp.randint(2, 4)
                    (ep_stats, ag_stats) = run_episode(env, policy, stage_cfg, env_cfg, seed, live=live)
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
                                produced)
                    except Exception:
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
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    out[f'{col}_mean'] = np.nanmean(v)
            return pd.Series(out)
        
        df_sum = df_ep.groupby(['scenario', 'condition']).apply(agg).reset_index()
        df_sum.to_csv(os.path.join(exp_dir, 'summary.csv'), index=False)
        
        if args.wandb and wlogger:
            for (_, row) in df_sum.iterrows():
                wlogger.log({f"{exp_name}/{row['scenario']}/{row['condition']}/success_rate": row['success_rate_mean'], f"{exp_name}/{row['scenario']}/{row['condition']}/collision_rate": row['collision_rate_mean'], f"{exp_name}/{row['scenario']}/{row['condition']}/flight_time_s": row['flight_time_s_mean']})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint to evaluate')
    p.add_argument('--config', default='config/default.yaml', help='Base config file')
    p.add_argument('--variant', default='mardpg', choices=['mardpg', 'maddpg', 'iddpg', 'ind_rdpg'], help='Variant architecture')
    p.add_argument('--episodes', type=int, default=50, help='Episodes per condition/scenario')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='robustness_results')
    p.add_argument('--base-seed', type=int, default=20000)
    p.add_argument('--no-render', action='store_true', help='Disable generating trajectory plots and animations')
    p.add_argument('--wandb', action='store_true', help='Log results to W&B')
    p.add_argument('--wandb-project', default='mardpg-uav-eval')
    p.add_argument('--wandb-name', default=None)
    p.add_argument('--suite', default='quick', choices=['quick', 'full'], help='For compatibility, currently ignored as we run fixed scenarios')
    p.add_argument('--realtime', action='store_true', help='Enable live rendering')
    args = p.parse_args()
    
    wlogger = None
    if args.wandb:
        from mardpg_uav.wandb_logger import WandbLogger
        wlogger = WandbLogger(True, args.wandb_project, vars(args), args.wandb_name or f'Robustness_{os.path.basename(os.path.dirname(args.checkpoint))}')
        if wlogger.use_wandb:
            wlogger._wandb.config.update(vars(args))
    
    run_evaluations(args, wlogger)
    
    if wlogger and wlogger.use_wandb:
        wlogger.finish()

if __name__ == '__main__':
    main()
