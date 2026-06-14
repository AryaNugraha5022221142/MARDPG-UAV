"""
evaluate_generalization.py — generalization & detailed evaluation for MARDPG-UAV.

Goes beyond evaluate.py in three ways the curriculum-only eval cannot:

  1. GENERALIZATION SWEEP. Evaluates the policy on a suite of configs tagged
     'in_dist' (matching curriculum stages) or 'ood' (UNSEEN difficulty). The
     overfitting test is the SHAPE of SR vs difficulty: graceful degradation =
     genuine navigation skill; a cliff just past the trained range = scene
     overfitting.

     IMPORTANT CAVEAT (honest): the scene generator caps static obstacles at 16
     (a 4x4 grid; obstacles.py: n_static = min(n_static, 16)). So you CANNOT get
     a harder *static* scene than stage 6/7 without changing the grid. Genuine
     OOD here comes from longer goal distance (min_sep, trained <= 40) and
     denser/faster DYNAMIC threats (beyond stage 7). Requesting static_obs > 16
     prints a warning and is clamped by the env.

  2. DETAILED PER-EPISODE / PER-AGENT METRICS: flight distance, flight time,
     time-to-goal, per-decision inference time (real-time feasibility vs the
     dt=0.1 s / 10 Hz control budget), cumulative reward, path efficiency, and a
     heuristic collision-type label (static / inter-UAV / dynamic).

  3. CSV EXPORT of every episode and every agent, plus an aggregated summary,
     plus a 3D render of the best episode in a chosen config.

Reproducibility: each episode is seeded (base_seed + index) on BOTH the scene
RNG and the lidar-noise RNG, so any episode (e.g. the best one) can be
regenerated exactly for rendering. Eval action selection is deterministic
(no exploration noise), so seed fully determines the rollout.

Reuses the corrected rendering from visualize_eval.py — keep both files in the
same directory.

Usage:
    python evaluate_generalization.py --checkpoint checkpoints/final --episodes 50
    python evaluate_generalization.py --checkpoint checkpoints/final --suite quick --episodes 30
"""
import os
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.eval_rollout import load_agents

# Reuse corrected renderer from the improved visualizer.
from visualize_eval import plot_trajectory_3d, plot_trajectory_top_down, AGENT_COLORS, HAZARD_COLOR


# ---------------------------------------------------------------------------
# Evaluation suite (in-distribution anchors + genuine OOD)
# ---------------------------------------------------------------------------
def build_suite(quick: bool = False):
    """Return a list of (name, regime, stage_cfg). regime in {'in_dist','ood'}."""
    base = dict(env_size=[100.0, 100.0, 60.0], min_start_sep=12.0)

    def cfg(**kw):
        c = dict(base); c.update(kw); return c

    suite = [
        # ---- In-distribution anchors (match curriculum) ----
        ('stage4_train', 'in_dist',
         cfg(static_obs=7, max_h=20.0, min_sep=40.0, max_steps=1000)),
        ('stage6_train', 'in_dist',
         cfg(static_obs=16, max_h=50.0, min_sep=40.0, max_steps=1500)),
        ('stage7_train', 'in_dist',
         cfg(static_obs=16, max_h=50.0, min_sep=40.0, max_steps=1500,
             dynamic_obs=(1, 2), dynamic_radius=2.0, dynamic_speed=(1.0, 2.0))),

        # ---- OOD: longer goal distance (horizon never trained) ----
        ('ood_dist_50', 'ood',
         cfg(static_obs=16, max_h=50.0, min_sep=50.0, max_steps=1500)),
        ('ood_dist_60', 'ood',
         cfg(static_obs=16, max_h=50.0, min_sep=60.0, max_steps=1500)),

        # ---- OOD: denser / faster dynamic threats ----
        ('ood_dyn_dense', 'ood',
         cfg(static_obs=16, max_h=50.0, min_sep=40.0, max_steps=1500,
             dynamic_obs=(3, 4), dynamic_radius=2.0, dynamic_speed=(1.0, 2.0))),
        ('ood_dyn_fast', 'ood',
         cfg(static_obs=16, max_h=50.0, min_sep=40.0, max_steps=1500,
             dynamic_obs=(2, 3), dynamic_radius=2.5, dynamic_speed=(2.5, 3.5))),

        # ---- OOD: combined stress ----
        ('ood_combined', 'ood',
         cfg(static_obs=16, max_h=50.0, min_sep=60.0, max_steps=1500,
             dynamic_obs=(2, 3), dynamic_radius=2.5, dynamic_speed=(2.0, 3.0))),

        # ---- Easy sanity floor (should be ~trivial; failure => bug, not policy) ----
        ('sanity_freespace', 'in_dist',
         cfg(static_obs=0, min_sep=30.0, max_steps=600)),
    ]
    if quick:
        keep = {'stage7_train', 'ood_dist_60', 'ood_dyn_fast', 'sanity_freespace'}
        suite = [s for s in suite if s[0] in keep]
    return suite


# ---------------------------------------------------------------------------
# Single episode rollout with full instrumentation
# ---------------------------------------------------------------------------
def run_episode(env, agents, stage_cfg, env_cfg, seed):
    n_agents = env_cfg['n_agents']
    dt = env_cfg.get('dt', 0.1)
    iu_min = env_cfg.get('inter_uav_min_dist', 1.0)

    # Reproducible scene + lidar noise for THIS episode.
    env.scene_gen.rng.seed(seed)
    env.rangefinder.rng.seed(seed)
    obs = env.reset(stage_cfg)

    for ag in agents:
        ag.actor.eval()
        ag.reset_hidden(batch_size=1, eval_mode=True)

    prev = [np.zeros(env.action_dim, dtype=np.float32) for _ in range(n_agents)]
    start_pos = env.agents_state[:, :3].copy()
    path = [start_pos.copy()]
    dyn = getattr(env, 'dynamic_obstacles', [])
    dyn_path = [[d.position.copy() for d in dyn]] if dyn else []
    dyn_r = [float(d.size[0]) for d in dyn] if dyn else []

    cum_reward = np.zeros(n_agents)
    time_to_goal = np.full(n_agents, np.nan)
    coll_type = [None] * n_agents
    infer_time_total = 0.0
    n_decisions = 0
    info = {}

    for t in range(stage_cfg['max_steps']):
        live_before = ~env.agent_done.copy()
        dyn_before = env.agent_dyn_collided.copy() if hasattr(env, 'agent_dyn_collided') \
            else np.zeros(n_agents, bool)

        t0 = time.perf_counter()
        acts = []
        for i, ag in enumerate(agents):
            if env.agent_done[i]:
                acts.append(np.zeros(env.action_dim, dtype=np.float32))
            else:
                a = ag.select_action(obs[i], prev[i], evaluate=True)
                acts.append(np.clip(a, -ag.actor.action_bound, ag.actor.action_bound))
                n_decisions += 1
        infer_time_total += time.perf_counter() - t0
        acts = np.array(acts, dtype=np.float32)

        obs, rewards, done, info = env.step(acts)
        prev = acts.copy()
        cum_reward += np.asarray(rewards)
        path.append(env.agents_state[:, :3].copy())
        if dyn:
            dyn_path.append([d.position.copy() for d in dyn])

        # time-to-goal at first reach
        for i in range(n_agents):
            if info['step_reached'][i] and np.isnan(time_to_goal[i]):
                time_to_goal[i] = (t + 1) * dt

        # collision-type label (heuristic) at the step an agent first collides
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

    path = np.array(path)                       # (T, N, 3)
    dyn_path = np.array(dyn_path) if dyn else None
    T = len(path)
    reached = env.agent_reached.copy()
    collided = env.agent_collided.copy()
    dyn_collided = (env.agent_dyn_collided.copy()
                    if hasattr(env, 'agent_dyn_collided') else np.zeros(n_agents, bool))
    goals = env.goals.copy()

    # Per-agent flight distance (done agents are frozen => zero increments after).
    flight_dist = np.zeros(n_agents)
    for i in range(n_agents):
        flight_dist[i] = float(np.sum(np.linalg.norm(np.diff(path[:, i, :], axis=0), axis=1)))

    # Per-agent path efficiency (only meaningful for reached agents).
    path_eff = np.full(n_agents, np.nan)
    for i in range(n_agents):
        if reached[i] and flight_dist[i] > 1e-8:
            straight = np.linalg.norm(goals[i] - start_pos[i])
            path_eff[i] = min(straight / flight_dist[i], 1.0)

    agent_rows = []
    for i in range(n_agents):
        agent_rows.append(dict(
            agent=i, reached=bool(reached[i]), collided=bool(collided[i]),
            dyn_collided=bool(dyn_collided[i]),
            collision_type=(coll_type[i] if collided[i] else ''),
            flight_distance_m=flight_dist[i],
            time_to_goal_s=float(time_to_goal[i]),
            path_efficiency=float(path_eff[i]),
            cumulative_reward=float(cum_reward[i])))

    ep = dict(
        seed=seed, steps=T - 1, flight_time_s=(T - 1) * dt,
        success_rate=float(reached.mean()),
        mission_success=bool(reached.all()),
        collision_rate=float(collided.mean()),
        dyn_collision_rate=float(dyn_collided.mean()),
        team_reward=float(cum_reward.sum()),
        mean_flight_distance_m=float(flight_dist.mean()),
        total_flight_distance_m=float(flight_dist.sum()),
        mean_path_efficiency=float(np.nanmean(path_eff)) if reached.any() else np.nan,
        mean_inference_ms_per_decision=1e3 * infer_time_total / max(1, n_decisions),
        safe_inter_uav_ratio=float(info.get('safe_inter_uav_ratio', 1.0)),
        n_static_collisions=sum(1 for c in coll_type if c == 'static'),
        n_inter_uav_collisions=sum(1 for c in coll_type if c == 'inter_uav'),
        n_dynamic_collisions=sum(1 for c in coll_type if c == 'dynamic'),
    )
    render = dict(path=path, dyn_path=dyn_path, dyn_r=dyn_r,
                  reached=reached, collided=collided, goals=goals)
    return ep, agent_rows, render


# ---------------------------------------------------------------------------
def _episode_score(ep):
    """Ranking for 'best episode': all-reached first, then reward, then steps."""
    return (1 if ep['mission_success'] else 0,
            ep['success_rate'], ep['team_reward'], -ep['steps'])


def evaluate_suite(checkpoint, config, episodes, device, outdir,
                   render, base_seed, quick):
    if not os.path.exists(config):
        fb = os.path.join(os.path.dirname(os.path.abspath(__file__)), config)
        if os.path.exists(fb):
            config = fb
    cfg = yaml.safe_load(open(config))
    env_cfg, net_cfg, algo_cfg = cfg['environment'], cfg['network'], cfg['algorithm']
    n_agents = env_cfg['n_agents']

    os.makedirs(outdir, exist_ok=True)
    env = MultiUAVEnv(env_cfg)
    agents, _ = load_agents(checkpoint, config, device)

    # warmup so the first timed inference isn't penalised by lazy init
    _ = agents[0].select_action(np.zeros(env.obs_dim, np.float32),
                                np.zeros(env.action_dim, np.float32), evaluate=True)
    agents[0].reset_hidden(batch_size=1, eval_mode=True)

    suite = build_suite(quick=quick)
    ep_records, agent_records = [], []
    best = {}   # config_name -> (score, seed)

    for name, regime, stage_cfg in suite:
        if stage_cfg.get('static_obs', 0) > 16:
            print(f"[WARN] {name}: static_obs={stage_cfg['static_obs']} will be "
                  f"clamped to 16 by the scene grid (not a harder scene).")
        print(f"\n=== {name} ({regime}) | {episodes} episodes ===")
        t_cfg = time.time()
        for e in range(episodes):
            seed = base_seed + e
            ep, arows, render_data = run_episode(env, agents, stage_cfg, env_cfg, seed)
            ep.update(config_name=name, regime=regime, episode=e)
            ep_records.append(ep)
            for r in arows:
                r.update(config_name=name, episode=e, seed=seed)
                agent_records.append(r)
            sc = _episode_score(ep)
            if name not in best or sc > best[name][0]:
                best[name] = (sc, seed)
        dur = time.time() - t_cfg
        sub = [r for r in ep_records if r['config_name'] == name]
        sr = np.mean([r['success_rate'] for r in sub])
        cr = np.mean([r['collision_rate'] for r in sub])
        print(f"  SR {sr:.1%} | collision {cr:.1%} | {dur:.0f}s "
              f"({dur / max(1, episodes):.1f}s/ep)")

    # mark is_best per config
    best_seeds = {n: s for n, (_, s) in best.items()}
    for r in ep_records:
        r['is_best'] = (r['seed'] == best_seeds.get(r['config_name']))

    df_ep = pd.DataFrame(ep_records)
    df_ag = pd.DataFrame(agent_records)

    # ---- aggregated summary with SE ----
    def agg(g):
        n = len(g)
        out = dict(n_episodes=n, regime=g['regime'].iloc[0])
        for col in ['success_rate', 'collision_rate', 'dyn_collision_rate',
                    'mission_success', 'team_reward', 'mean_flight_distance_m',
                    'flight_time_s', 'mean_path_efficiency',
                    'mean_inference_ms_per_decision', 'safe_inter_uav_ratio']:
            v = g[col].astype(float)
            out[f'{col}_mean'] = v.mean()
            out[f'{col}_se'] = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        out['static_coll_total'] = int(g['n_static_collisions'].sum())
        out['inter_uav_coll_total'] = int(g['n_inter_uav_collisions'].sum())
        out['dynamic_coll_total'] = int(g['n_dynamic_collisions'].sum())
        return pd.Series(out)

    df_sum = df_ep.groupby('config_name', sort=False).apply(agg).reset_index()

    df_ep.to_csv(os.path.join(outdir, 'eval_episodes.csv'), index=False)
    df_ag.to_csv(os.path.join(outdir, 'eval_agents.csv'), index=False)
    df_sum.to_csv(os.path.join(outdir, 'eval_summary.csv'), index=False)
    print(f"\nWrote eval_episodes.csv, eval_agents.csv, eval_summary.csv to {outdir}/")

    # ---- generalization summary plot ----
    _plot_generalization(df_sum, os.path.join(outdir, 'generalization_summary.png'))

    # ---- render best episode of EVERY config (3D trajectory only) ----
    # Side panels (top-down, separation, distance-to-goal) are intentionally
    # dropped: all of that is reconstructable from eval_episodes.csv /
    # eval_agents.csv. One focused 3D plot per config instead.
    if render:
        suite_cfg = {n: s for n, r, s in suite}
        for name in df_ep['config_name'].unique():
            seed = best_seeds[name]
            stage_cfg = dict(suite_cfg[name])
            print(f"Rendering best episode of '{name}' (seed {seed}) ...")
            _, _, rnd = run_episode(env, agents, stage_cfg, env_cfg, seed)
            plot_trajectory_3d(
                env, env_cfg, rnd,
                f"BEST | {name} (seed {seed}) | "
                f"reached {int(rnd['reached'].sum())}/{n_agents}, "
                f"collided {int(rnd['collided'].sum())}/{n_agents}",
                os.path.join(outdir, f'best_3d_{name}.png'))
            plot_trajectory_top_down(
                env, env_cfg, rnd,
                f"BEST TOP-DOWN | {name} (seed {seed}) | "
                f"reached {int(rnd['reached'].sum())}/{n_agents}, "
                f"collided {int(rnd['collided'].sum())}/{n_agents}",
                os.path.join(outdir, f'best_top_down_{name}.png'))

    return df_ep, df_ag, df_sum


def _plot_generalization(df_sum, out_path):
    df = df_sum.copy()
    order = df['config_name'].tolist()
    x = np.arange(len(order))
    colors = ['#0072B2' if r == 'in_dist' else '#D55E00' for r in df['regime']]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(order)), 5))
    ax.bar(x, df['success_rate_mean'], yerr=df['success_rate_se'], capsize=4,
           color=colors, alpha=0.85)
    ax.plot(x, df['collision_rate_mean'], 'k^--', ms=7, label='collision rate')
    ax.axhline(0.80, color='green', ls=':', lw=1.2, label='0.80 target')
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel('rate'); ax.set_ylim(0, 1.0)
    ax.set_title('Generalization: blue=in-distribution, orange=OOD '
                 '(graceful drop = generalizes; cliff = overfit)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--episodes', type=int, default=50,
                   help='episodes per config (SE ~ 0.5/sqrt(n) at p=0.5; 50 -> ~7pp)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='eval_results')
    p.add_argument('--no-render', action='store_true',
                   help='skip the per-config 3D trajectory renders')
    p.add_argument('--base-seed', type=int, default=10_000)
    p.add_argument('--suite', choices=['full', 'quick'], default='full')
    p.add_argument('--wandb', action='store_true', help='Use wandb to log evaluation results')
    p.add_argument('--wandb-project', default='mardpg-uav-eval', help='Wandb project name')
    p.add_argument('--wandb-name', default=None, help='Wandb run name')
    a = p.parse_args()
    
    if a.wandb:
        import wandb
        wandb.init(project=a.wandb_project, name=a.wandb_name, config=vars(a))
        
    df_ep, df_ag, df_sum = evaluate_suite(a.checkpoint, a.config, a.episodes, a.device, a.outdir,
                   not a.no_render, a.base_seed, a.suite == 'quick')
                   
    if a.wandb:
        # Log aggregated summary
        for _, row in df_sum.iterrows():
            cfg_name = row['config_name']
            log_dict = {f"eval/{cfg_name}/{k}": v for k, v in row.items() if isinstance(v, (int, float, np.number))}
            wandb.log(log_dict)
            
        import wandb
        # Log generalization summary plot
        img_path = os.path.join(a.outdir, 'generalization_summary.png')
        if os.path.exists(img_path):
            wandb.log({"eval/generalization_summary": wandb.Image(img_path)})
            
        # Log renders
        if not a.no_render:
            for cfg_name in df_ep['config_name'].unique():
                img_path_3d = os.path.join(a.outdir, f'best_3d_{cfg_name}.png')
                img_path_td = os.path.join(a.outdir, f'best_top_down_{cfg_name}.png')
                if os.path.exists(img_path_3d):
                    wandb.log({f"eval/renders/best_3d_{cfg_name}": wandb.Image(img_path_3d)})
                if os.path.exists(img_path_td):
                    wandb.log({f"eval/renders/best_top_down_{cfg_name}": wandb.Image(img_path_td)})
                    
        wandb.finish()


if __name__ == "__main__":
    main()
