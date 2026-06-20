"""
evaluate_multiagent.py  (v2) — multi-SEED-correct evaluation with explicit
variance decomposition.

What changed vs v1 and WHY
==========================
v1 varied SCENE seeds per episode for a SINGLE checkpoint and reported ±SE over
those scenes. That SE is *within-seed scene-sampling noise* — it is NOT the
variance of your METHOD. A method-level claim ("MARDPG beats X") must account
for TRAINING-seed variance, which is almost always larger and is the only
dispersion a reviewer cares about.

v2 makes the TRAINING SEED the unit of statistical analysis:

  1. Each method takes a LIST of checkpoints (one per training seed).
  2. All methods/seeds see the SAME fixed set of scene seeds  -> paired over
     scenes; differences are not scene-luck.
  3. Aggregation is two-level:
        scene  -> seed     : mean over scenes = that seed's point estimate
        seed   -> method   : mean/std/SE over SEEDS  (the honest dispersion)
     Both `seed_se` (honest) and `scene_se_within_seed` (what v1 reported) are
     printed together so the difference is visible.
  4. rliable-style robust aggregate: IQM over the (seed x config) score matrix
     with a STRATIFIED BOOTSTRAP 95% CI that resamples SEEDS. Self-contained
     (numpy only); no rliable/scipy dependency. With 3 seeds the CI is wide —
     that is correct, not a defect.
  5. Checkpoint loads are VERIFIED: a fingerprint of the actor weights before
     and after load must change, else we RAISE. This closes the v1 hole where a
     dim mismatch was swallowed (`[WARN] agent i:`) and produced believable
     random-policy numbers.

Everything good in v1 is kept: instrumented paired rollouts, APF baseline,
Wilson CIs for Bernoulli mission-success, encounter-conditioned
conflict-resolution, the exact training-stage configs as the eval MDP.

Honesty notes to keep in the thesis (unchanged from v1)
------------------------------------------------------
  * APF is a potential-field reactive controller, NOT ORCA-3D.
  * Trained reward/hyperparameters deviate from the paper (terminal anchors,
    sigma, lr, buffer) -> absolute numbers are not digit-comparable.
  * conflict_resolution_rate / uav_collision_rate are often saturated in this
    weakly-coupled env; report them but do not over-claim coordination from a
    ceiling'd metric.

Usage
-----
  python evaluate_multiagent.py \
    --method MARDPG  mardpg   ckpts/cl_mardpg_seed0/final,ckpts/cl_mardpg_seed1/final,ckpts/cl_mardpg_seed2/final \
    --method IndRDPG ind_rdpg ckpts/cl_ind_rdpg_seed0/final,ckpts/cl_ind_rdpg_seed1/final,ckpts/cl_ind_rdpg_seed2/final \
    --method MADDPG  maddpg   ckpts/cl_maddpg_seed0/final,ckpts/cl_maddpg_seed1/final,ckpts/cl_maddpg_seed2/final \
    --method IDDPG   iddpg    ckpts/cl_iddpg_seed0/final,ckpts/cl_iddpg_seed1/final,ckpts/cl_iddpg_seed2/final \
    --apf --episodes 100 --device cpu --suite quick

A checkpoint entry may be a comma list (one dir per seed) or a glob
(e.g. 'ckpts/cl_mardpg_seed*/final'). Outputs:
    eval_episodes.csv   : one row per (method, seed, config, scene)  [raw]
    eval_per_seed.csv   : one row per (method, seed, config)         [seed point estimates]
    eval_summary.csv    : one row per (method, config)               [across-seed stats]
    eval_method_iqm.csv : one row per method                         [robust aggregate + bootstrap CI]
analyze_factorial.py consumes eval_per_seed.csv for the 2x2 decomposition.
"""
import os
import re
import glob
import time
import math
import hashlib
import argparse

import numpy as np
import pandas as pd
import torch

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.train import CURRICULUM, load_config
from mardpg_uav.algorithm.mardpg import MARDPGAgent

ENCOUNTER_DIST = 6.0  # metres; matches utils/metrics.ENCOUNTER_DIST

# variant -> (recurrent, centralized); mirrors eval_rollout._VARIANT_FLAGS
_VARIANT_FLAGS = {
    'mardpg':   (True,  True),
    'maddpg':   (False, True),
    'ind_rdpg': (True,  False),
    'iddpg':    (False, False),
}


# ===========================================================================
# Verified checkpoint loading  (audit: silent load failure -> random policy)
# ===========================================================================
def _module_fingerprint(module) -> str:
    h = hashlib.sha1()
    for p in module.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_agents_strict(checkpoint_dir, config_path, device, variant):
    """Build agents for `variant` and load the checkpoint, RAISING if the load
    did not actually change the actor weights (the silent-failure mode that v1
    swallowed). Returns (agents, cfg)."""
    recurrent, centralized = _VARIANT_FLAGS[variant]
    cfg = load_config(config_path)
    env_cfg, net_cfg = cfg['environment'], cfg['network']
    n_agents = env_cfg['n_agents']

    agents = []
    for i in range(n_agents):
        ag = MARDPGAgent(
            agent_id=i, n_agents=n_agents,
            obs_dim=env_cfg.get('obs_dim', 35), action_dim=2,
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            recurrent=recurrent, centralized=centralized, device=device)

        init_fp = _module_fingerprint(ag.actor)

        apath = os.path.join(checkpoint_dir, f"agent_{i}.pt")
        if not os.path.exists(apath):
            print(f"[LOAD WARN] missing {apath}, falling back to agent_0.pt")
            apath = os.path.join(checkpoint_dir, "agent_0.pt")
            if not os.path.exists(apath):
                raise FileNotFoundError(f"[LOAD FAIL] missing {apath}")
        ckpt = torch.load(apath, map_location=device)

        if 'actor_private' in ckpt:
            if i == 0:
                spath = os.path.join(checkpoint_dir, "shared_actor.pt")
                if not os.path.exists(spath):
                    raise FileNotFoundError(f"[LOAD FAIL] missing {spath}")
                sc = torch.load(spath, map_location=device)
                ag.shared_extractor.load_state_dict(sc['shared_actor'])  # strict
            ag.actor.load_state_dict(ckpt['actor_private'], strict=False)
        elif 'actor' in ckpt:
            ag.actor.load_state_dict(ckpt['actor'])
        else:
            raise KeyError(f"[LOAD FAIL] {apath} has neither 'actor_private' "
                           f"nor 'actor' (keys: {list(ckpt.keys())})")

        post_fp = _module_fingerprint(ag.actor)
        # Agent 0 always changes (shared + private). Agents >0 change via the
        # private lstm/fc_out load; share_parameters runs afterwards.
        if post_fp == init_fp:
            raise RuntimeError(
                f"[LOAD FAIL] {checkpoint_dir} agent {i}: actor weights are "
                f"IDENTICAL to init after load -> nothing loaded (key/shape "
                f"mismatch). Numbers from this checkpoint would be a RANDOM "
                f"policy. Fix the checkpoint/variant before evaluating.")
        agents.append(ag)

    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    return agents, cfg


# ===========================================================================
# Evaluation suite — exact training stages as the eval MDP (conflict_frac > 0)
# ===========================================================================
def _stage_cfg(idx_0based):
    c = dict(CURRICULUM[idx_0based])
    c.pop('criteria', None)
    c.pop('name', None)
    return c


def build_suite(quick=False):
    s4 = _stage_cfg(3)
    s6 = _stage_cfg(5)
    s7 = _stage_cfg(6)

    def ood_static(**kw):
        c = dict(s6); c.update(kw); return c

    def ood_dyn(**kw):
        c = dict(s7); c.update(kw); return c

    suite = [
        ('stage4_train',     'in_dist', s4),
        ('stage6_train',     'in_dist', s6),
        ('stage7_train',     'in_dist', s7),
        ('ood_dist_60',      'ood',     ood_static(min_sep=60.0)),
        ('ood_dyn_fast',     'ood',     ood_dyn(dynamic_obs=(2, 3),
                                                dynamic_radius=2.5,
                                                dynamic_speed=(2.5, 3.5))),
        ('sanity_crossings', 'in_dist', dict(env_size=[100.0, 100.0, 60.0],
                                             static_obs=0, max_steps=600,
                                             min_sep=30.0, min_start_sep=12.0,
                                             conflict_frac=1.0, ring_frac=0.35)),
    ]
    if quick:
        keep = {'stage7_train', 'ood_dist_60', 'ood_dyn_fast', 'sanity_crossings'}
        suite = [s for s in suite if s[0] in keep]
    return suite


# ===========================================================================
# Policy providers
# ===========================================================================
class LearnedPolicy:
    kind = 'learned'

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


class APFPolicy:
    kind = 'classical'

    def __init__(self, env, name='APF'):
        from mardpg_uav.apf import APFController
        self.ctrl = APFController(env)
        self.name = name

    def reset(self, env):
        pass

    def act(self, env, obs):
        return self.ctrl.act()


# ===========================================================================
# Single episode rollout — policy-agnostic, fully instrumented (kept from v1)
# ===========================================================================
def run_episode(env, policy, stage_cfg, env_cfg, seed, capture_render=False):
    n_agents = env_cfg['n_agents']
    dt = env_cfg.get('dt', 0.1)
    iu_min = env_cfg.get('inter_uav_min_dist', 1.0)

    env.scene_gen.rng.seed(seed)
    env.rangefinder.rng.seed(seed)
    obs = env.reset(stage_cfg)
    policy.reset(env)

    start_pos = env.agents_state[:, :3].copy()
    path = [start_pos.copy()]
    
    dyn = getattr(env, 'dynamic_obstacles', []) if capture_render else []
    dyn_path = [[d.position.copy() for d in dyn]] if dyn else []
    dyn_r = [d.size[0] for d in dyn] if dyn else []
    
    cum_reward = np.zeros(n_agents)
    time_to_goal = np.full(n_agents, np.nan)
    steps_taken = np.zeros(n_agents, dtype=int)
    coll_type = [None] * n_agents
    info = {}

    for t in range(stage_cfg['max_steps']):
        live_before = ~env.agent_done.copy()
        for i in range(n_agents):
            if live_before[i]:
                steps_taken[i] += 1
                
        dyn_before = (env.agent_dyn_collided.copy()
                      if hasattr(env, 'agent_dyn_collided')
                      else np.zeros(n_agents, bool))
        acts = policy.act(env, obs)
        obs, rewards, done, info = env.step(acts)
        cum_reward += np.asarray(rewards)
        path.append(env.agents_state[:, :3].copy())
        
        if capture_render and dyn:
            dyn_path.append([d.position.copy() for d in dyn])

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
    reached = env.agent_reached.copy()
    collided = env.agent_collided.copy()
    goals = env.goals.copy()

    flight_dist = np.array([
        float(np.sum(np.linalg.norm(np.diff(path[:, i, :], axis=0), axis=1)))
        for i in range(n_agents)])
    straight = np.array([np.linalg.norm(goals[i] - start_pos[i]) for i in range(n_agents)])

    reached_eff = [min(straight[i] / flight_dist[i], 1.0)
                   for i in range(n_agents)
                   if reached[i] and flight_dist[i] > 1e-8]
    path_eff_reached = float(np.mean(reached_eff)) if reached_eff else np.nan

    closest_appr = float(info.get('min_pair_dist', np.nan))
    near_miss_ratio = float(info.get('near_miss_ratio', 0.0))
    uav_col = np.asarray(info.get('uav_collisions', np.zeros(n_agents, bool)))
    had_encounter = bool(np.isfinite(closest_appr) and closest_appr < ENCOUNTER_DIST)
    conflict_resolved = (np.nan if not had_encounter
                         else (1.0 if not bool(np.any(uav_col)) else 0.0))

    ep = dict(
        seed_scene=seed, steps=len(path) - 1,
        success_rate=float(reached.mean()),
        mission_success=bool(reached.all()),
        collision_rate=float(collided.mean()),
        uav_collision_rate=float(np.mean(uav_col)),
        static_collision_rate=float(sum(c == 'static' for c in coll_type) / n_agents),
        dyn_collision_rate=float(sum(c == 'dynamic' for c in coll_type) / n_agents),
        path_eff_reached=path_eff_reached,
        team_reward=float(cum_reward.sum()),
        closest_approach_m=(closest_appr if np.isfinite(closest_appr) else np.nan),
        near_miss_ratio=near_miss_ratio,
        had_encounter=had_encounter,
        conflict_resolved=conflict_resolved,
        safe_inter_uav_ratio=float(info.get('safe_inter_uav_ratio', np.nan)),
        mean_flight_dist=float(np.mean(flight_dist)),
        mean_flight_time=float(np.mean(steps_taken * dt)),
        mean_time_to_goal=float(np.nanmean(time_to_goal)),
    )
    if capture_render:
        ep['_render_rnd'] = dict(
            path=path, reached=reached, collided=collided, goals=goals,
            dyn_path=np.array(dyn_path) if dyn else None, dyn_r=dyn_r
        )
    return ep


# ===========================================================================
# Statistics — numpy only (Wilson, bootstrap, IQM)
# ===========================================================================
def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _iqm(x):
    x = np.sort(np.asarray(x, float))
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    if n < 4:
        return float(np.mean(x))          # IQM degenerate below 4 points
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(np.mean(x[lo:hi]))


def _bootstrap_ci(values, n_boot=10000, agg=np.mean, ci=0.95, rng=None):
    """Resample the UNIT OF ANALYSIS (seeds) with replacement. With 3 seeds the
    interval is wide by construction — that is the honest uncertainty."""
    v = np.asarray([x for x in values if not (isinstance(x, float) and np.isnan(x))], float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    if n == 1:
        return (float(v[0]), np.nan, np.nan)
    rng = rng or np.random.default_rng(12345)
    boots = np.array([agg(v[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return (float(agg(v)),
            float(np.percentile(boots, 100 * (1 - ci) / 2)),
            float(np.percentile(boots, 100 * (1 + ci) / 2)))


# Metrics aggregated as a per-seed scalar (mean over that seed's scenes).
_SEED_METRICS = ['success_rate', 'mission_success', 'collision_rate',
                 'static_collision_rate', 'dyn_collision_rate',
                 'uav_collision_rate', 'path_eff_reached',
                 'closest_approach_m', 'near_miss_ratio',
                 'safe_inter_uav_ratio', 'mean_flight_dist',
                 'mean_flight_time', 'mean_time_to_goal']


# ===========================================================================
# Two-level aggregation
# ===========================================================================
def aggregate_per_seed(df_ep):
    """scene -> seed. One row per (method, variant, seed, config)."""
    rows = []
    keys = ['method', 'variant', 'seed', 'config_name', 'regime']
    for kv, g in df_ep.groupby(keys, sort=False):
        row = dict(zip(keys, kv))
        row['n_scenes'] = len(g)
        for m in _SEED_METRICS:
            v = g[m].astype(float).values
            row[m] = float(np.nanmean(v)) if len(v) else np.nan
            # within-seed scene SE — what v1 mistakenly reported as "the" SE
            vv = v[~np.isnan(v)]
            row[f'{m}__scene_se'] = (float(vv.std(ddof=1) / np.sqrt(len(vv)))
                                     if len(vv) > 1 else 0.0)
        # encounter-conditioned conflict resolution within this seed
        cr = g['conflict_resolved'].astype(float)
        enc = cr.notna()
        n_enc = int(enc.sum())
        row['n_encounters'] = n_enc
        row['conflict_resolution_rate'] = (float((cr[enc] == 1.0).mean())
                                           if n_enc else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_across_seeds(df_seed):
    """seed -> method. One row per (method, config). Reports the HONEST
    seed-level dispersion next to the within-seed scene SE for contrast."""
    metrics = _SEED_METRICS + ['conflict_resolution_rate']
    rows = []
    for (method, cname), g in df_seed.groupby(['method', 'config_name'], sort=False):
        row = dict(method=method, config_name=cname,
                   variant=g['variant'].iloc[0], regime=g['regime'].iloc[0],
                   n_seeds=len(g))
        for m in metrics:
            seed_vals = g[m].astype(float).values
            sv = seed_vals[~np.isnan(seed_vals)]
            mean, lo, hi = _bootstrap_ci(sv, agg=np.mean)
            row[f'{m}_mean'] = mean
            row[f'{m}_seed_std'] = float(sv.std(ddof=1)) if len(sv) > 1 else 0.0
            row[f'{m}_seed_se'] = (float(sv.std(ddof=1) / np.sqrt(len(sv)))
                                   if len(sv) > 1 else 0.0)
            row[f'{m}_ci_lo'] = lo
            row[f'{m}_ci_hi'] = hi
            row[f'{m}_seeds'] = ";".join(f"{x:.4f}" for x in sv)
            if f'{m}__scene_se' in g.columns:
                row[f'{m}_scene_se_within_seed'] = float(np.nanmean(g[f'{m}__scene_se']))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_method_iqm(df_seed, regimes=('in_dist',)):
    """Robust method-level aggregate: IQM over the (seed x config) score matrix
    for the chosen regime(s), with a stratified bootstrap (resampling SEEDS) CI.
    This is the rliable-style headline number for a few-seed study."""
    rows = []
    sub = df_seed[df_seed['regime'].isin(regimes)]
    for method, g in sub.groupby('method', sort=False):
        seeds = sorted(g['seed'].unique())
        # score matrix: rows = seeds, cols = configs (SR by default)
        for metric in ['success_rate', 'collision_rate', 'conflict_resolution_rate']:
            piv = g.pivot_table(index='seed', columns='config_name',
                                values=metric, aggfunc='mean')
            mat = piv.values  # (n_seeds, n_configs)
            point = _iqm(mat.flatten())
            # stratified bootstrap: resample seeds (rows) with replacement
            rng = np.random.default_rng(2024)
            n_seeds = mat.shape[0]
            boots = []
            for _ in range(10000):
                idx = rng.integers(0, n_seeds, size=n_seeds)
                boots.append(_iqm(mat[idx].flatten()))
            boots = np.array([b for b in boots if not np.isnan(b)])
            lo = float(np.percentile(boots, 2.5)) if len(boots) else np.nan
            hi = float(np.percentile(boots, 97.5)) if len(boots) else np.nan
            rows.append(dict(method=method, metric=metric, regime="+".join(regimes),
                             n_seeds=n_seeds, iqm=point, iqm_ci_lo=lo, iqm_ci_hi=hi))
    return pd.DataFrame(rows)


# ===========================================================================
# Driver
# ===========================================================================
def _seed_label(ckpt_dir, fallback_idx):
    """Parse the TRUE training seed from the checkpoint path (e.g.
    '.../cl_mardpg_seed2/final' -> 2). This is what makes results computed on
    SEPARATE machines mergeable: seeds are globally meaningful, never per-VM
    positional indices. Falls back to the enumeration index if no seed token
    is found (and warns, because that breaks cross-machine merges)."""
    m = re.search(r'seed[_-]?(\d+)', ckpt_dir)
    if m:
        return int(m.group(1))
    print(f"[WARN] no 'seedN' token in '{ckpt_dir}' -> using positional index "
          f"{fallback_idx}. Cross-machine merges will be WRONG; rename the dir "
          f"to include seedN.", flush=True)
    return fallback_idx


def _expand_ckpts(arg):
    """Comma list and/or glob -> sorted list of checkpoint dirs."""
    out = []
    for token in arg.split(','):
        token = token.strip()
        if not token:
            continue
        if any(ch in token for ch in '*?[]'):
            out.extend(sorted(glob.glob(token)))
        else:
            out.append(token)
    if not out:
        raise ValueError(f"No checkpoints resolved from '{arg}'")
    return out


def evaluate(methods, config, episodes, device, outdir, base_seed, quick, wandb_log=False, video=False):
    cfg = load_config(config)
    env_cfg = cfg['environment']
    os.makedirs(outdir, exist_ok=True)
    env = MultiUAVEnv(env_cfg)
    suite = build_suite(quick=quick)

    # Preflight: crossing branch needs assignment._nudge_free (audit Fix A).
    try:
        env.scene_gen.rng.seed(base_seed)
        env.rangefinder.rng.seed(base_seed)
        env.reset(suite[0][2])
    except NameError as e:
        raise SystemExit(
            f"[FATAL] {e}\nThe crossing-assignment branch calls _nudge_free(); "
            "ensure it lives in environment/assignment.py (audit Fix A).")

    ep_records = []
    for name, variant, ckpt_arg in methods:
        if variant == 'apf':
            seed_list = [('apf', None)]
        else:
            seed_list = [(_seed_label(ck, idx), ck)
                         for idx, ck in enumerate(_expand_ckpts(ckpt_arg))]

        for seed_idx, ckpt in seed_list:
            if variant == 'apf':
                provider = APFPolicy(env, name=name)
            else:
                print(f"[load] {name} seed={seed_idx} <- {ckpt}", flush=True)
                agents, _ = load_agents_strict(ckpt, config, device, variant)
                provider = LearnedPolicy(agents, name=name)

            for cname, regime, stage_cfg in suite:
                t0 = time.time()
                for e in range(episodes):
                    scene_seed = base_seed + e        # shared across all -> paired
                    capture = wandb_log and (e < 3)
                    ep = run_episode(env, provider, stage_cfg, env_cfg, scene_seed, capture_render=capture)
                    
                    if capture and '_render_rnd' in ep:
                        rnd = ep.pop('_render_rnd')
                        try:
                            from visualize_eval import plot_trajectory_3d, plot_trajectory_top_down
                            title = f"Traj: {name} | {cname} | ep {e} | reaches {rnd['reached'].sum()}"
                            
                            out_png_3d = os.path.join(outdir, f'traj_{name}_s{seed_idx}_{cname}_ep{e}_3d.png')
                            plot_trajectory_3d(env, env_cfg, rnd, title, out_png_3d)
                            
                            out_png_2d = os.path.join(outdir, f'traj_{name}_s{seed_idx}_{cname}_ep{e}_topdown.png')
                            plot_trajectory_top_down(env, env_cfg, rnd, title, out_png_2d)
                            
                            out_vid = None
                            if video:
                                from visualize_eval import animate
                                out_vid = os.path.join(outdir, f'traj_{name}_s{seed_idx}_{cname}_ep{e}.mp4')
                                animate(env, env_cfg, rnd['path'], rnd['dyn_path'], rnd['dyn_r'], env.goals, title, out_vid)

                            if wandb_log:
                                import wandb
                                log_dict = {
                                    f"eval/traj_3d/{name}_{cname}_ep{e}": wandb.Image(out_png_3d),
                                    f"eval/traj_topdown/{name}_{cname}_ep{e}": wandb.Image(out_png_2d)
                                }
                                if out_vid and os.path.exists(out_vid):
                                    log_dict[f"eval/traj_video/{name}_{cname}_ep{e}"] = wandb.Video(out_vid, format="mp4")
                                elif out_vid and os.path.exists(out_vid.replace('.mp4', '.gif')):
                                    log_dict[f"eval/traj_video/{name}_{cname}_ep{e}"] = wandb.Video(out_vid.replace('.mp4', '.gif'), format="gif")
                                wandb.log(log_dict)
                        except Exception as ex:
                            print(f"[WARN] plot failed: {ex}")
                    
                    ep.update(method=name, variant=variant, seed=seed_idx,
                              checkpoint=str(ckpt), config_name=cname,
                              regime=regime, episode=e)
                    ep_records.append(ep)
                dur = time.time() - t0
                subset = [r for r in ep_records if r['method'] == name and r['seed'] == seed_idx and r['config_name'] == cname]
                sr = np.mean([r['success_rate'] for r in subset])
                coll = np.mean([r['collision_rate'] for r in subset])
                uav_coll = np.mean([r['uav_collision_rate'] for r in subset])
                static_coll = np.mean([r['static_collision_rate'] for r in subset])
                dyn_coll = np.mean([r['dyn_collision_rate'] for r in subset])
                encounter = np.mean([float(r['had_encounter']) for r in subset])
                conflict_res_vals = [r['conflict_resolved'] for r in subset if not np.isnan(r['conflict_resolved'])]
                conflict_res = np.mean(conflict_res_vals) if conflict_res_vals else np.nan
                
                path_eff = np.nanmean([r['path_eff_reached'] for r in subset])
                safe_iu = np.nanmean([r['safe_inter_uav_ratio'] for r in subset])
                min_dist = np.nanmean([r['closest_approach_m'] for r in subset])
                nmr = np.nanmean([r['near_miss_ratio'] for r in subset])
                f_dist = np.nanmean([r['mean_flight_dist'] for r in subset])
                f_time = np.nanmean([r['mean_flight_time'] for r in subset])
                t_goal = np.nanmean([r['mean_time_to_goal'] for r in subset])
                
                print(
                    f"  [{name} s{seed_idx}] {cname:16s} SR {sr:.1%} | coll {coll:.1%} (static: {static_coll:.1%}, dyn: {dyn_coll:.1%}) | "
                    f"uav_coll {uav_coll:.1%} | encounter {encounter:.1%} | conflict_res {conflict_res*100:.1f}%\n"
                    f"    path_eff {path_eff:.1%} | safe_iu {safe_iu:.1%} | min_dist {min_dist:.2f}m | nmr {nmr:.1%} | "
                    f"flight_dist {f_dist:.1f}m | flight_time {f_time:.1f}s | time_to_goal {t_goal:.1f}s  ({dur:.0f}s)"
                , flush=True)

    df_ep = pd.DataFrame(ep_records)
    df_seed = aggregate_per_seed(df_ep)
    df_sum = aggregate_across_seeds(df_seed)
    df_iqm = aggregate_method_iqm(df_seed, regimes=('in_dist',))

    df_ep.to_csv(os.path.join(outdir, 'eval_episodes.csv'), index=False)
    df_seed.to_csv(os.path.join(outdir, 'eval_per_seed.csv'), index=False)
    df_sum.to_csv(os.path.join(outdir, 'eval_summary.csv'), index=False)
    df_iqm.to_csv(os.path.join(outdir, 'eval_method_iqm.csv'), index=False)

    if wandb_log:
        import wandb
        wandb.save(os.path.join(outdir, 'val_*.csv'), base_path=outdir) # actually save exact names
        wandb.save(os.path.join(outdir, 'eval_episodes.csv'), base_path=outdir)
        wandb.save(os.path.join(outdir, 'eval_per_seed.csv'), base_path=outdir)
        wandb.save(os.path.join(outdir, 'eval_summary.csv'), base_path=outdir)
        wandb.save(os.path.join(outdir, 'eval_method_iqm.csv'), base_path=outdir)

    _print_variance_report(df_seed, df_sum)
    print(f"\nWrote eval_episodes / eval_per_seed / eval_summary / "
          f"eval_method_iqm to {outdir}/")
    return df_ep, df_seed, df_sum, df_iqm


def _print_variance_report(df_seed, df_sum):
    """The point of the rewrite, made legible: seed-level SE (honest) vs the
    within-seed scene SE that v1 reported."""
    print("\n" + "=" * 78)
    print("VARIANCE DECOMPOSITION — success_rate  (the unit of analysis is the SEED)")
    print("=" * 78)
    print(f"{'method':12s} {'config':16s} {'n_seed':>6s} {'mean':>7s} "
          f"{'seed_SE':>8s} {'scene_SE':>9s}  seed values")
    print("-" * 78)
    for _, r in df_sum.sort_values(['method', 'config_name']).iterrows():
        seedvals = r.get('success_rate_seeds', '')
        print(f"{r['method']:12s} {r['config_name']:16s} "
              f"{int(r['n_seeds']):6d} "
              f"{r['success_rate_mean']:6.1%} "
              f"{r['success_rate_seed_se']:7.1%} "
              f"{r.get('success_rate_scene_se_within_seed', float('nan')):8.1%}  "
              f"[{seedvals}]")
    print("-" * 78)
    print("seed_SE is the dispersion you must report. scene_SE is within-seed "
          "noise (the\nold ±SE) and is typically much smaller — do NOT use it "
          "for method-level claims.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', action='append', nargs=3, default=[],
                   metavar=('NAME', 'VARIANT', 'CKPTS'),
                   help="repeatable. CKPTS = comma list of seed dirs or a glob. "
                        "VARIANT in {mardpg,maddpg,ind_rdpg,iddpg}")
    p.add_argument('--checkpoint', default=None, help='Legacy: primary method checkpoint dir')
    p.add_argument('--name', default='MARDPG', help='Legacy: display name')
    p.add_argument('--variant', default='mardpg', help='Legacy: variant')
    p.add_argument('--wandb', action='store_true', help='Use wandb tracking')
    p.add_argument('--wandb-project', default='mardpg-uav-eval', help='Wandb project name')
    p.add_argument('--wandb-name', default=None, help='Wandb run name')
    p.add_argument('--apf', action='store_true', help='add reactive APF baseline (no seeds)')
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--episodes', type=int, default=100,
                   help='scenes per (method,seed,config). 100 -> scene SE ~3.5pp at p=.5')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='eval_results')
    p.add_argument('--base-seed', type=int, default=10_000)
    p.add_argument('--suite', choices=['full', 'quick'], default='full')
    p.add_argument('--video', action='store_true', help='Generate video/animation of episodes')
    a = p.parse_args()

    methods = [(nm, var, ck) for nm, var, ck in a.method]
    if a.checkpoint:
        methods.insert(0, (a.name, a.variant, a.checkpoint))
    if a.apf:
        methods.append(('APF', 'apf', ''))
    if not methods:
        raise SystemExit("Provide at least one --method NAME VARIANT CKPTS or --checkpoint.")

    if a.wandb:
        import wandb
        wandb.init(project=a.wandb_project, name=a.wandb_name, config=vars(a))

    df_ep, df_seed, df_sum, df_iqm = evaluate(methods, a.config, a.episodes, a.device, a.outdir,
             a.base_seed, a.suite == 'quick', wandb_log=a.wandb, video=a.video)

    if a.wandb:
        import wandb
        wandb.log({"eval/summary": wandb.Table(dataframe=df_sum)})
        wandb.log({"eval/method_iqm": wandb.Table(dataframe=df_iqm)})
        wandb.finish()


if __name__ == "__main__":
    main()
