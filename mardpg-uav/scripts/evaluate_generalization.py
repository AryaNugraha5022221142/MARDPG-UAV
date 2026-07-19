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
try:
    from mardpg_uav.rendering import plot_trajectory_3d, plot_trajectory_top_down, select_backend, LiveRenderer
    _HAVE_RENDER = True
except Exception as _e:
    _HAVE_RENDER = False
    _RENDER_IMPORT_ERR = _e

def build_suite(quick: bool=False):
    base = dict(env_size=[100.0, 100.0, 60.0], min_start_sep=12.0)

    def cfg(**kw):
        c = dict(base)
        c.update(kw)
        return c
    s6_kwargs = dict(static_obs=16, max_h=50.0, min_sep=40.0, max_steps=1500, conflict_frac=1.0, ring_frac=0.35)

    def ood_dense(**kw):
        c = cfg(**s6_kwargs)
        c.update(kw)
        return c
    suite = [('ood_dense_20', 'ood', ood_dense(static_obs=20)), ('ood_dense_25', 'ood', ood_dense(static_obs=25)), ('ood_dense_max', 'ood', ood_dense(static_obs=30))]
    if quick:
        keep = {'ood_dense_20', 'ood_dense_25', 'ood_dense_max'}
        suite = [s for s in suite if s[0] in keep]
    return suite

class LearnedPolicy:
    kind = 'learned'

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

def run_episode(env, policy, stage_cfg, env_cfg, seed, live=None):
    n_agents = stage_cfg.get('n_agents', env_cfg['n_agents'])
    dt = env_cfg.get('dt', 0.1)
    iu_min = env_cfg.get('inter_uav_min_dist', 1.0)
    env.scene_gen.rng.seed(seed)
    env.rangefinder.rng.seed(seed)
    obs = env.reset(stage_cfg)
    policy.reset(env)
    
    if live is not None:
        live.reset(env)
        
    start_pos = env.agents_state[:, :3].copy()
    path = [start_pos.copy()]
    dyn = getattr(env, 'dynamic_obstacles', [])
    dyn_path = [[d.position.copy() for d in dyn]] if dyn else []
    dyn_r = [float(np.max(d.size)) for d in dyn] if dyn else []
    cum_reward = np.zeros(n_agents)
    time_to_goal = np.full(n_agents, np.nan)
    coll_type = [None] * n_agents
    infer_time_total = 0.0
    n_decisions = 0
    info = {}
    for t in range(stage_cfg['max_steps']):
        live_before = ~env.agent_done.copy()
        dyn_before = env.agent_dyn_collided.copy() if hasattr(env, 'agent_dyn_collided') else np.zeros(n_agents, bool)
        t0 = time.perf_counter()
        acts = policy.act(env, obs)
        infer_time_total += time.perf_counter() - t0
        n_decisions += int(live_before.sum())
        (obs, rewards, done, info) = env.step(acts)
        if live is not None:
            live.step(env)
        cum_reward += np.asarray(rewards)
        path.append(env.agents_state[:, :3].copy())
        if dyn:
            dyn_path.append([d.position.copy() for d in dyn])
        for i in range(n_agents):
            if info['step_reached'][i] and np.isnan(time_to_goal[i]):
                time_to_goal[i] = (t + 1) * dt
        pos_now = env.agents_state[:, :3]
        for i in range(n_agents):
            if info['step_collisions'][i] and coll_type[i] is None:
                if env.agent_dyn_collided[i] and (not dyn_before[i]):
                    coll_type[i] = 'dynamic'
                else:
                    dmin = min((np.linalg.norm(pos_now[i] - pos_now[j]) for j in range(n_agents) if j != i and live_before[j]), default=np.inf)
                    coll_type[i] = 'inter_uav' if dmin < iu_min else 'static'
        if done:
            break
    path = np.array(path)
    T = len(path)
    reached = env.agent_reached.copy()
    collided = env.agent_collided.copy()
    dyn_collided = env.agent_dyn_collided.copy() if hasattr(env, 'agent_dyn_collided') else np.zeros(n_agents, bool)
    goals = env.goals.copy()
    flight_dist = np.zeros(n_agents)
    for i in range(n_agents):
        flight_dist[i] = float(np.sum(np.linalg.norm(np.diff(path[:, i, :], axis=0), axis=1)))
    straight = np.array([np.linalg.norm(goals[i] - start_pos[i]) for i in range(n_agents)])
    tot_actual = float(flight_dist.sum())
    path_eff_paper = float(straight.sum() / tot_actual) if tot_actual > 1e-08 else np.nan
    reached_eff = []
    for i in range(n_agents):
        if reached[i] and flight_dist[i] > 1e-08:
            reached_eff.append(min(straight[i] / flight_dist[i], 1.0))
    path_eff_reached = float(np.mean(reached_eff)) if reached_eff else np.nan
    neither = ~reached & ~collided
    trapped_rate_paper = float(neither.mean())
    trapped_rate_progress = float(np.mean(info.get('trapped', np.zeros(n_agents, bool))))
    agent_rows = []
    for i in range(n_agents):
        peff = min(straight[i] / flight_dist[i], 1.0) if reached[i] and flight_dist[i] > 1e-08 else np.nan
        agent_rows.append(dict(agent=i, reached=bool(reached[i]), collided=bool(collided[i]), dyn_collided=bool(dyn_collided[i]), collision_type=coll_type[i] if collided[i] else '', flight_distance_m=flight_dist[i], straight_distance_m=float(straight[i]), time_to_goal_s=float(time_to_goal[i]), path_efficiency=float(peff), cumulative_reward=float(cum_reward[i])))
    ep = dict(seed=seed, steps=T - 1, flight_time_s=(T - 1) * dt, success_rate=float(reached.mean()), mission_success=bool(reached.all()), collision_rate=float(collided.mean()), dyn_collision_rate=float(dyn_collided.mean()), trapped_rate_paper=trapped_rate_paper, trapped_rate_progress=trapped_rate_progress, team_reward=float(cum_reward.sum()), mean_agent_reward=float(cum_reward.mean()), path_eff_paper=path_eff_paper, path_eff_reached=path_eff_reached, mean_flight_distance_m=float(flight_dist.mean()), total_flight_distance_m=float(flight_dist.sum()), mean_inference_ms_per_decision=1000.0 * infer_time_total / max(1, n_decisions), safe_inter_uav_ratio=float(info.get('safe_inter_uav_ratio', 1.0)), n_static_collisions=sum((1 for c in coll_type if c == 'static')), n_inter_uav_collisions=sum((1 for c in coll_type if c == 'inter_uav')), n_dynamic_collisions=sum((1 for c in coll_type if c == 'dynamic')))
    render = dict(path=path, dyn_path=np.array(dyn_path) if dyn else None, dyn_r=dyn_r, reached=reached, collided=collided, goals=goals)
    return (ep, agent_rows, render)

def _episode_score(ep):
    return (1 if ep['mission_success'] else 0, ep['success_rate'], ep['team_reward'], -ep['steps'])

def _se(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    return v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

def _wilson(k, n, z=1.96):
    """Wilson score interval for a Bernoulli proportion (mission success)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))

def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _paired_delta(a, b):
    """Approximate two-sided test on the difference of per-episode means a-b.
    Episodes are paired by seed; we use the paired differences' SE."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    n = len(d)
    if n < 2:
        return dict(delta=float(np.mean(d)) if n else np.nan, se=np.nan, z=np.nan, p=np.nan, n=n)
    delta = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    z = delta / se if se > 0 else np.nan
    p = 2.0 * (1.0 - _normal_cdf(abs(z))) if se > 0 else np.nan
    return dict(delta=delta, se=se, z=z, p=p, n=n)

def evaluate_suite(methods, config, episodes, device, outdir, render, base_seed, quick, render_method, realtime=False):
    if not os.path.exists(config):
        fb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config)
        if os.path.exists(fb):
            config = fb
    cfg = yaml.safe_load(open(config))
    env_cfg = cfg['environment']
    n_agents = env_cfg['n_agents']
    os.makedirs(outdir, exist_ok=True)
    
    if render and _HAVE_RENDER:
        select_backend('auto', want_interactive=realtime)
        
    env = MultiUAVEnv(env_cfg)
    live = None
    if render and realtime and _HAVE_RENDER:
        live = LiveRenderer(env, env_cfg)
        
    providers = {}
    for (name, kind, payload) in methods:
        (variant, ckpt) = payload
        (agents, _) = load_agents(ckpt, config, device, variant=variant)
        for ag in agents:
            _ = ag.select_action(np.zeros(env.obs_dim, np.float32), np.zeros(env.action_dim, np.float32), evaluate=True)
            ag.reset_hidden(batch_size=1, eval_mode=True)
        providers[name] = LearnedPolicy(agents, name=name)
    suite = build_suite(quick=quick)
    (ep_records, agent_records) = ([], [])
    best = {}
    for (name, prov) in providers.items():
        for (cname, regime, stage_cfg) in suite:
            t_cfg = time.time()
            for e in range(episodes):
                seed = base_seed + e
                (ep, arows, _) = run_episode(env, prov, stage_cfg, env_cfg, seed, live=live)
                ep.update(method=name, config_name=cname, regime=regime, episode=e)
                ep_records.append(ep)
                for r in arows:
                    r.update(method=name, config_name=cname, episode=e, seed=seed)
                    agent_records.append(r)
                sc = _episode_score(ep)
                key = (name, cname)
                if key not in best or sc > best[key][0]:
                    best[key] = (sc, seed)
            sub = [r for r in ep_records if r['method'] == name and r['config_name'] == cname]
            sr = np.mean([r['success_rate'] for r in sub])
            sr_se = _se([r['success_rate'] for r in sub])
            msr = np.mean([r['mission_success'] for r in sub])
            cr = np.mean([r['collision_rate'] for r in sub])
            dur = time.time() - t_cfg
    for r in ep_records:
        r['is_best'] = r['seed'] == best.get((r['method'], r['config_name']), (None, None))[1]
    df_ep = pd.DataFrame(ep_records)
    df_ag = pd.DataFrame(agent_records)

    def agg(g):
        n = len(g)
        out = dict(n_episodes=n, regime=g['regime'].iloc[0])
        rate_cols = ['success_rate', 'collision_rate', 'dyn_collision_rate', 'trapped_rate_paper', 'trapped_rate_progress', 'mean_agent_reward', 'team_reward', 'path_eff_paper', 'path_eff_reached', 'mean_flight_distance_m', 'flight_time_s', 'mean_inference_ms_per_decision', 'safe_inter_uav_ratio']
        for col in rate_cols:
            v = g[col].astype(float)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[f'{col}_mean'] = np.nanmean(v)
            out[f'{col}_se'] = _se(v)
        k = int(g['mission_success'].sum())
        (msr, lo, hi) = _wilson(k, n)
        out['mission_success_rate'] = msr
        out['mission_success_ci_lo'] = lo
        out['mission_success_ci_hi'] = hi
        out['static_coll_total'] = int(g['n_static_collisions'].sum())
        out['inter_uav_coll_total'] = int(g['n_inter_uav_collisions'].sum())
        out['dynamic_coll_total'] = int(g['n_dynamic_collisions'].sum())
        return pd.Series(out)
    df_sum = df_ep.groupby(['method', 'config_name'], sort=False).apply(agg).reset_index()
    primary = methods[0][0]
    cmp_rows = []
    if len(providers) > 1:
        for (cname, regime, _) in suite:
            base_g = df_ep[(df_ep.method == primary) & (df_ep.config_name == cname)]
            for name in providers:
                if name == primary:
                    continue
                other_g = df_ep[(df_ep.method == name) & (df_ep.config_name == cname)]
                merged = base_g.merge(other_g, on='episode', suffixes=('_p', '_o'))
                for metric in ['success_rate', 'collision_rate', 'mission_success']:
                    d = _paired_delta(merged[f'{metric}_p'], merged[f'{metric}_o'])
                    cmp_rows.append(dict(config_name=cname, regime=regime, primary=primary, baseline=name, metric=metric, **d))
    df_cmp = pd.DataFrame(cmp_rows)
    df_ep.to_csv(os.path.join(outdir, 'eval_episodes.csv'), index=False)
    df_ag.to_csv(os.path.join(outdir, 'eval_agents.csv'), index=False)
    df_sum.to_csv(os.path.join(outdir, 'eval_summary.csv'), index=False)
    if not df_cmp.empty:
        df_cmp.to_csv(os.path.join(outdir, 'eval_method_comparison.csv'), index=False)
    _plot_generalization(df_sum, list(providers.keys()), os.path.join(outdir, 'generalization_summary.png'))
    if render and _HAVE_RENDER:
        rm = render_method if render_method in providers else primary
        suite_cfg = {n: s for (n, r, s) in suite}
        for cname in df_ep['config_name'].unique():
            seed = best[rm, cname][1]
            stage_cfg = dict(suite_cfg[cname])
            (_, _, rnd) = run_episode(env, providers[rm], stage_cfg, env_cfg, seed)
            title = f"BEST [{rm}] | {cname} (seed {seed}) | reached {int(rnd['reached'].sum())}/{n_agents}, collided {int(rnd['collided'].sum())}/{n_agents}"
            plot_trajectory_3d(env, env_cfg, rnd, title, os.path.join(outdir, f'best_3d_{rm}_{cname}.png'))
            plot_trajectory_top_down(env, env_cfg, rnd, title.replace('BEST', 'BEST TOP-DOWN'), os.path.join(outdir, f'best_top_down_{rm}_{cname}.png'))
    elif render and (not _HAVE_RENDER):
        pass
    return (df_ep, df_ag, df_sum, df_cmp)

def _plot_generalization(df_sum, method_order, out_path):
    configs = list(dict.fromkeys(df_sum['config_name'].tolist()))
    nM = len(method_order)
    x = np.arange(len(configs))
    width = 0.8 / max(1, nM)
    cmap = plt.get_cmap('tab10')
    (fig, (ax1, ax2)) = plt.subplots(2, 1, figsize=(max(9, 1.3 * len(configs)), 8), sharex=True)
    for (mi, m) in enumerate(method_order):
        sub = df_sum[df_sum.method == m].set_index('config_name').reindex(configs)
        off = (mi - (nM - 1) / 2) * width
        ax1.bar(x + off, sub['success_rate_mean'], width, yerr=sub['success_rate_se'], capsize=3, label=m, color=cmap(mi % 10), alpha=0.9)
        ax2.bar(x + off, sub['collision_rate_mean'], width, yerr=sub['collision_rate_se'], capsize=3, color=cmap(mi % 10), alpha=0.9)
    ax1.axhline(0.8, color='green', ls=':', lw=1.2, label='0.80 ref')
    ax1.set_ylabel('Success rate')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Per-agent success (top) and collision (bottom). Error bars = SE. Compare in-dist vs OOD spacing to SE.')
    ax1.legend(fontsize=8, ncol=min(nM + 1, 4))
    ax1.grid(axis='y', alpha=0.3)
    ax2.set_ylabel('Collision rate')
    ax2.set_ylim(0, max(0.3, df_sum['collision_rate_mean'].max() * 1.3))
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=30, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='primary method checkpoint dir')
    p.add_argument('--name', default='MARDPG', help='display name for the primary method')
    p.add_argument('--variant', default='mardpg', choices=['mardpg', 'maddpg', 'iddpg', 'ind_rdpg'], help='architecture of the primary checkpoint')
    p.add_argument('--baseline', action='append', nargs=3, default=[], metavar=('NAME', 'VARIANT', 'CKPT'), help='add a learned baseline; repeatable')
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--episodes', type=int, default=100, help='episodes per (method,config). Use 250 to match the paper; SE ~ 0.5/sqrt(n) at p=0.5 (250 -> ~3pp).')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='eval_results')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--render-method', default=None, help='which method to render best episodes for (default: primary)')
    p.add_argument('--base-seed', type=int, default=10000)
    p.add_argument('--suite', choices=['full', 'quick'], default='full')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', default='mardpg-uav-eval')
    p.add_argument('--wandb-name', default=None)
    p.add_argument('--realtime', action='store_true', help='Enable live rendering')
    a = p.parse_args()
    methods = [(a.name, 'learned', (a.variant, a.checkpoint))]
    for (nm, var, ck) in a.baseline:
        methods.append((nm, 'learned', (var, ck)))
    if a.wandb:
        import wandb
        try:
            wandb.init(project=a.wandb_project, name=a.wandb_name, config=vars(a), settings=wandb.Settings(init_timeout=15))
        except Exception as e:
            print(f"Warning: wandb init failed ({e}). Proceeding without wandb.")
            a.wandb = False
    (df_ep, df_ag, df_sum, df_cmp) = evaluate_suite(methods, a.config, a.episodes, a.device, a.outdir, not a.no_render, a.base_seed, a.suite == 'quick', a.render_method or a.name, a.realtime)
    if a.wandb:
        import wandb
        wandb.log({'eval/summary': wandb.Table(dataframe=df_sum)})
        if not df_cmp.empty:
            wandb.log({'eval/method_comparison': wandb.Table(dataframe=df_cmp)})
        img = os.path.join(a.outdir, 'generalization_summary.png')
        if os.path.exists(img):
            wandb.log({'eval/generalization_summary': wandb.Image(img)})
        wandb.finish()
if __name__ == '__main__':
    main()
