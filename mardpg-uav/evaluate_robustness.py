"""
evaluate_robustness.py
======================
Dedicated script for robustness and generalization testing.
Evaluates a single trained model under various unseen conditions:
- Sensor Noise
- Sensor Range
- Goal Distribution (Goal distance)
- Variable Speed (Changing dynamics)
"""
import os
import time
import math
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.eval_rollout import load_agents

# ===========================================================================
# Policy Provider (Learned policy only)
# ===========================================================================
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
        for i, ag in enumerate(self.agents):
            if env.agent_done[i]:
                acts.append(np.zeros(env.action_dim, np.float32))
            else:
                a = ag.select_action(obs[i], self._prev[i], evaluate=True)
                acts.append(np.clip(a, -ag.actor.action_bound, ag.actor.action_bound))
        acts = np.array(acts, np.float32)
        self._prev = acts.copy()
        return acts

# ===========================================================================
# Single episode rollout, policy-agnostic, fully instrumented.
# ===========================================================================
def run_episode(env, policy, stage_cfg, env_cfg, seed):
    n_agents = env_cfg['n_agents']
    dt = env_cfg.get('dt', 0.1)
    iu_min = env_cfg.get('inter_uav_min_dist', 1.0)

    # Reproducible scene for THIS episode
    env.scene_gen.rng.seed(seed)
    env.rangefinder.rng.seed(seed)
    np.random.seed(seed)
    
    obs = env.reset(stage_cfg)
    policy.reset(env)

    start_pos = env.agents_state[:, :3].copy()
    path = [start_pos.copy()]
    dyn_path = []
    dp_start = []
    if hasattr(env, 'dynamic_obstacles'):
        for o in env.dynamic_obstacles:
            dp_start.append(o.position.copy())
    dyn_path.append(dp_start)
    
    cum_reward = np.zeros(n_agents)
    time_to_goal = np.full(n_agents, np.nan)
    coll_type = [None] * n_agents
    infer_time_total = 0.0
    n_decisions = 0
    info = {}

    for t in range(stage_cfg.get('max_steps', 1500)):
        # Dynamic variable speed injection for Variable-Speed robustness experiment
        if stage_cfg.get('variable_speed', False):
            env.agent_v = np.random.uniform(0.5, 5.0, env.n_agents)

        live_before = ~env.agent_done.copy()
        dyn_before = (env.agent_dyn_collided.copy() if hasattr(env, 'agent_dyn_collided') else np.zeros(n_agents, bool))

        t0 = time.perf_counter()
        acts = policy.act(env, obs)
        infer_time_total += time.perf_counter() - t0
        n_decisions += int(live_before.sum())

        obs, rewards, done, info = env.step(acts)
        cum_reward += np.asarray(rewards)
        path.append(env.agents_state[:, :3].copy())
        
        dp = []
        for o in env.dynamic_obstacles:
            dp.append(o.position.copy())
        dyn_path.append(dp)

        for i in range(n_agents):
            if info['step_reached'][i] and np.isnan(time_to_goal[i]):
                time_to_goal[i] = (t + 1) * dt

        pos_now = env.agents_state[:, :3]
        for i in range(n_agents):
            if info['step_collisions'][i] and coll_type[i] is None:
                if env.agent_dyn_collided[i] and not dyn_before[i]:
                    coll_type[i] = 'dynamic'
                else:
                    dmin = min((np.linalg.norm(pos_now[i] - pos_now[j])
                                for j in range(n_agents)
                                if j != i and live_before[j]), default=np.inf)
                    coll_type[i] = 'inter_uav' if dmin < iu_min else 'static'
        if done:
            break

    path = np.array(path)
    T = len(path)
    reached = env.agent_reached.copy()
    collided = env.agent_collided.copy()
    dyn_collided = (env.agent_dyn_collided.copy() if hasattr(env, 'agent_dyn_collided') else np.zeros(n_agents, bool))
    goals = env.goals.copy()

    flight_dist = np.zeros(n_agents)
    for i in range(n_agents):
        flight_dist[i] = float(np.sum(np.linalg.norm(np.diff(path[:, i, :], axis=0), axis=1)))

    straight = np.array([np.linalg.norm(goals[i] - start_pos[i]) for i in range(n_agents)])

    tot_actual = float(flight_dist.sum())
    path_eff_paper = float(straight.sum() / tot_actual) if tot_actual > 1e-8 else np.nan
    reached_eff = []
    for i in range(n_agents):
        if reached[i] and flight_dist[i] > 1e-8:
            reached_eff.append(min(straight[i] / flight_dist[i], 1.0))
    path_eff_reached = float(np.mean(reached_eff)) if reached_eff else np.nan

    neither = (~reached) & (~collided)
    trapped_rate_paper = float(neither.mean())
    trapped_rate_progress = float(np.mean(info.get('trapped', np.zeros(n_agents, bool))))
    
    dyn_path = np.array(dyn_path) if dyn_path else np.zeros((len(path), 0, 3))
    dyn_r = np.array([o.size[0] for o in env.dynamic_obstacles]) if hasattr(env, 'dynamic_obstacles') else np.array([])

    agent_rows = []
    for i in range(n_agents):
        peff = (min(straight[i] / flight_dist[i], 1.0)
                if (reached[i] and flight_dist[i] > 1e-8) else np.nan)
        agent_rows.append(dict(
            agent=i, reached=bool(reached[i]), collided=bool(collided[i]),
            dyn_collided=bool(dyn_collided[i]),
            collision_type=(coll_type[i] if collided[i] else ''),
            flight_distance_m=flight_dist[i],
            straight_distance_m=float(straight[i]),
            time_to_goal_s=float(time_to_goal[i]),
            path_efficiency=float(peff),
            cumulative_reward=float(cum_reward[i])))

    ep = dict(
        seed=seed, steps=T - 1, flight_time_s=(T - 1) * dt,
        success_rate=float(reached.mean()),
        mission_success=bool(reached.all()),
        collision_rate=float(collided.mean()),
        dyn_collision_rate=float(dyn_collided.mean()),
        trapped_rate_paper=trapped_rate_paper,
        trapped_rate_progress=trapped_rate_progress,
        team_reward=float(cum_reward.sum()),
        mean_agent_reward=float(cum_reward.mean()),
        path_eff_paper=path_eff_paper,
        path_eff_reached=path_eff_reached,
        mean_flight_distance_m=float(flight_dist.mean()),
        total_flight_distance_m=float(flight_dist.sum()),
        mean_inference_ms_per_decision=1e3 * infer_time_total / max(1, n_decisions),
        n_static_collisions=sum(1 for c in coll_type if c == 'static'),
        n_inter_uav_collisions=sum(1 for c in coll_type if c == 'inter_uav'),
        n_dynamic_collisions=sum(1 for c in coll_type if c == 'dynamic'),
        path=path,
        dyn_path=dyn_path,
        dyn_r=dyn_r,
        reached=reached,
        collided=collided,
        goals=goals,
    )
    return ep, agent_rows

# ===========================================================================
# Statistics helpers
# ===========================================================================
def _se(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    return (v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

def _wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)

# ===========================================================================
# Experiment Definitions
# ===========================================================================
def build_base_scenarios(env_cfg):
    env_size = env_cfg.get('env_size', [100.0, 100.0, 60.0])
    return {
        'S1_Static_Dynamic': {
            'env_size': env_size,
            'static_obs': 16,
            'dynamic_obs': 0, # Injected in loop
            'dynamic_radius': 2.0,
            'dynamic_speed': [1.0, 2.0],
            'min_sep': 40.0,
            'max_steps': 1500,
        },
        'S2_Longer_Distance': {
            'env_size': env_size,
            'static_obs': 16,
            'dynamic_obs': 0,
            'min_sep': 60.0,
            'max_steps': 1500,
        },
        'S3_Fast_Dynamic': {
            'env_size': env_size,
            'static_obs': 16,
            'dynamic_obs': 0, # Injected in loop 
            'dynamic_radius': 2.5,
            'dynamic_speed': [2.5, 3.5],
            'min_sep': 40.0,
            'max_steps': 1500,
        }
    }

def build_experiment_configs():
    # Returns dictionary of experiment groups and their parameter sweeps
    return {
        'sensor_noise': [
            {'lidar_noise': 0.0, 'exp_val': 'sigma=0.0'},
            {'lidar_noise': 0.1, 'exp_val': 'sigma=0.1'},
            {'lidar_noise': 0.2, 'exp_val': 'sigma=0.2'},
            {'lidar_noise': 0.3, 'exp_val': 'sigma=0.3'},
            {'lidar_noise': 0.4, 'exp_val': 'sigma=0.4'},
            {'lidar_noise': 0.5, 'exp_val': 'sigma=0.5'},
        ],
        'sensor_range': [
            {'sensor_range': 15.0, 'exp_val': '15m'},
            {'sensor_range': 20.0, 'exp_val': '20m'},
            {'sensor_range': 25.0, 'exp_val': '25m'},
        ],
        'goal_distribution': [
            {'min_sep': 50.0, 'exp_val': '50m'},
            {'min_sep': 60.0, 'exp_val': '60m'},
            {'min_sep': 70.0, 'exp_val': '70m'},
        ],
        'variable_speed': [
            {'variable_speed': False, 'exp_val': 'Constant'},
            {'variable_speed': True, 'exp_val': 'Dynamic'},
        ]
    }

# ===========================================================================
# Driver
# ===========================================================================
def run_evaluations(args):
    cfg = yaml.safe_load(open(args.config))
    env_cfg = cfg['environment']
    
    # Load Policy
    print(f"Loading agents from {args.checkpoint}...")
    agents, _ = load_agents(args.checkpoint, args.config, args.device, variant=args.variant)
    policy = LearnedPolicy(agents, name="MARDPG")
    
    env = MultiUAVEnv(env_cfg)
    
    experiments = build_experiment_configs()
    base_scenarios = build_base_scenarios(env_cfg)
    
    for exp_name, exp_sweeps in experiments.items():
        exp_dir = os.path.join(args.outdir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n=====================================")
        print(f"Running Experiment: {exp_name}")
        print(f"=====================================")
        
        ep_records = []
        ag_records = []
        
        for sweep in exp_sweeps:
            exp_val = sweep['exp_val']
            print(f"\n---> Sweep condition: {exp_val}")
            
            for scenario_name, scenario_cfg in base_scenarios.items():
                print(f"  [Scenario: {scenario_name}]")
                t0 = time.time()
                
                # Merge scenario config with sweep overrides
                stage_cfg = dict(scenario_cfg)
                stage_cfg.update({k: v for k, v in sweep.items() if k != 'exp_val'})
                
                # Re-initialize environment to prevent state pollution across sweeps
                import copy
                env = MultiUAVEnv(copy.deepcopy(env_cfg))

                for ep in range(args.episodes):
                    seed = args.base_seed + ep
                    # Inject dynamic obstacles number deterministically based on scenario and seed
                    if scenario_name == 'S1_Static_Dynamic':
                        # Randomize 1-2 based on seed to match paper/generalization script behavior
                        rng_tmp = np.random.RandomState(seed)
                        stage_cfg['dynamic_obs'] = rng_tmp.randint(1, 3)
                    elif scenario_name == 'S3_Fast_Dynamic':
                        rng_tmp = np.random.RandomState(seed)
                        stage_cfg['dynamic_obs'] = rng_tmp.randint(2, 4)

                    ep_stats, ag_stats = run_episode(env, policy, stage_cfg, env_cfg, seed)
                    
                    ep_stats.update(experiment=exp_name, condition=exp_val, scenario=scenario_name)
                    ep_records.append(ep_stats)
                    for r in ag_stats:
                        r.update(experiment=exp_name, condition=exp_val, scenario=scenario_name, seed=seed)
                        ag_records.append(r)
                        
                dur = time.time() - t0
                # Quick summary
                sub = [r for r in ep_records if r['condition'] == exp_val and r['scenario'] == scenario_name]
                sr = np.mean([r['success_rate'] for r in sub])
                cr = np.mean([r['collision_rate'] for r in sub])
                print(f"     SR: {sr:.1%} | Coll: {cr:.1%} | Time: {dur:.1f}s")
                
                # Plot/Render the BEST episode for this condition/scenario
                best_ep = max(sub, key=lambda x: (x['mission_success'], x['success_rate'], x['team_reward'], -x['steps']))
                try:
                    from visualize_eval import plot_trajectory_3d, plot_trajectory_top_down, animate
                    title = f"Robustness | {exp_name}={exp_val} | {scenario_name} (seed {best_ep['seed']}) | SR {best_ep['success_rate']:.0%}"
                    out_png_3d = os.path.join(exp_dir, f'best_3d_{scenario_name}_{exp_val}.png')
                    plot_trajectory_3d(env, env_cfg, best_ep, title, out_png_3d)
                    
                    out_png_2d = os.path.join(exp_dir, f'best_topdown_{scenario_name}_{exp_val}.png')
                    plot_trajectory_top_down(env, env_cfg, best_ep, title, out_png_2d)
                    
                    out_vid = os.path.join(exp_dir, f'best_vid_{scenario_name}_{exp_val}.mp4')
                    animate(env, env_cfg, best_ep['path'], best_ep['dyn_path'], best_ep['dyn_r'], best_ep['goals'], title, out_vid)

                    if args.wandb:
                        import wandb
                        log_dict = {
                            f"video/{exp_name}/{scenario_name}/{exp_val}/3d": wandb.Image(out_png_3d),
                            f"video/{exp_name}/{scenario_name}/{exp_val}/topdown": wandb.Image(out_png_2d)
                        }
                        if os.path.exists(out_vid):
                            log_dict[f"video/{exp_name}/{scenario_name}/{exp_val}/animation"] = wandb.Video(out_vid, format="mp4")
                        elif os.path.exists(out_vid.replace('.mp4', '.gif')):
                            log_dict[f"video/{exp_name}/{scenario_name}/{exp_val}/animation"] = wandb.Video(out_vid.replace('.mp4', '.gif'), format="gif")
                        wandb.log(log_dict)
                except Exception as ex:
                    print(f"[WARN] plot/render failed: {ex}")
                
                # Delete path payload from memory to avoid OOM
                for ep in sub:
                    ep.pop('path', None)
                    ep.pop('dyn_path', None)
                    ep.pop('dyn_r', None)
                    ep.pop('reached', None)
                    ep.pop('collided', None)
                    ep.pop('goals', None)
                
        # Save results for this experiment
        df_ep = pd.DataFrame(ep_records)
        df_ag = pd.DataFrame(ag_records)
        df_ep.to_csv(os.path.join(exp_dir, 'eval_episodes.csv'), index=False)
        df_ag.to_csv(os.path.join(exp_dir, 'eval_agents.csv'), index=False)
        
        # Summary grouping
        def agg(g):
            n = len(g)
            out = dict(n_episodes=n)
            for col in ['success_rate', 'collision_rate', 'dyn_collision_rate', 
                        'path_eff_paper', 'path_eff_reached', 'flight_time_s']:
                v = g[col].astype(float)
                out[f'{col}_mean'] = np.nanmean(v)
                out[f'{col}_se'] = _se(v)
            msr, lo, hi = _wilson(int(g['mission_success'].sum()), n)
            out['mission_success_rate'] = msr
            out['mission_success_ci_lo'] = lo
            out['mission_success_ci_hi'] = hi
            return pd.Series(out)

        df_sum = df_ep.groupby(['condition', 'scenario'], sort=False).apply(agg).reset_index()
        df_sum.to_csv(os.path.join(exp_dir, 'eval_summary.csv'), index=False)
        print(f"Saved results for {exp_name} to {exp_dir}/")
        
        if args.wandb:
            import wandb
            for _, row in df_sum.iterrows():
                wandb.log({
                    f"{exp_name}/{row['scenario']}/{row['condition']}/success_rate": row['success_rate_mean'],
                    f"{exp_name}/{row['scenario']}/{row['condition']}/collision_rate": row['collision_rate_mean'],
                    f"{exp_name}/{row['scenario']}/{row['condition']}/flight_time_s": row['flight_time_s_mean'],
                })

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint to evaluate')
    p.add_argument('--config', default='config/default.yaml', help='Base config file')
    p.add_argument('--variant', default='mardpg', choices=['mardpg', 'maddpg', 'iddpg'], help='Variant architecture')
    p.add_argument('--episodes', type=int, default=50, help='Episodes per condition/scenario')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='robustness_results')
    p.add_argument('--base-seed', type=int, default=20000)
    p.add_argument('--wandb', action='store_true', help='Log results to W&B')
    p.add_argument('--wandb-project', default='mardpg-uav-eval')
    p.add_argument('--wandb-name', default=None)
    p.add_argument('--suite', default='quick', choices=['quick', 'full'], help='For compatibility, currently ignored as we run fixed scenarios')
    args = p.parse_args()
    
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name or f"Robustness_{os.path.basename(os.path.dirname(args.checkpoint))}")
        wandb.config.update(vars(args))

    run_evaluations(args)
    
    if args.wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
