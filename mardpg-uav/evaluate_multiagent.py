"""
evaluate_multiagent.py  — paper-faithful, MULTI-AGENT-correct evaluation.

Replaces the outdated evaluate_generalization.py.

Why the old script was wrong
----------------------------
evaluate_generalization.py built its suite WITHOUT `conflict_frac`, so every
config fell back to `conflict_frac = 0.0` in assignment.assign_start_goals. That
reproduced the OLD independent-scatter geometry: paths essentially never crossed,
inter_uav_safe ~ 1.0, and there was no multi-agent interaction in the evaluation
at all. It therefore could not support (or refute) any coordination / no-comm
claim, no matter which checkpoint it scored.

What this script fixes
----------------------
1. The in-distribution anchors REUSE the exact training-curriculum stage configs
   (including their conflict_frac / ring_frac), so the eval MDP == the training
   MDP. OOD configs are derived from the hardest training stages and keep
   conflict_frac = 1.0, so crossings persist out of distribution too.
2. A `sanity_crossings` config (static_obs = 0, conflict_frac = 1.0) isolates
   pure inter-UAV avoidance — the cleanest test of the multi-agent claim.
3. The genuine interaction metrics are reported with statistics:
     - closest_approach_m       : episode min pairwise distance (interaction depth)
     - near_miss_ratio          : fraction of steps any pair was inside the band
     - uav_collision_rate       : per-agent inter-UAV collision rate
     - encounter_rate           : fraction of episodes with a real encounter
     - conflict_resolution_rate : among encounter episodes, fraction with ZERO
                                  inter-UAV collisions (+ Wilson 95% CI).
   conflict_resolution_rate is the headline metric for the no-comm / centralized
   coordination claim.

Retained from the old script (these parts were good)
----------------------------------------------------
  * APF reactive baseline through the SAME rollout/metric code.
  * Per-agent success_rate AND mission-success (all N reach) with Wilson CI.
  * Path efficiency reported BOTH ways (paper ratio-of-sums and reached-only cap).
  * Trapped rate reported BOTH ways (paper partition and progress heuristic).
  * Identical scene seed per episode across methods => paired comparison.

PREREQUISITE
------------
This suite uses conflict_frac > 0, which exercises the crossing branch in
environment/assignment.py. That branch calls _nudge_free(). You MUST have
applied Fix A (relocate _nudge_free from utils/metrics.py into assignment.py)
or this will raise NameError on the first reset. A preflight check below turns
that into a clear message.

HONESTY notes to keep in the thesis
-----------------------------------
  * APF is a potential-field reactive controller, NOT ORCA-3D. Legitimate
    non-learning baseline; do not call it ORCA.
  * Trained reward/hyperparameters deviate from the paper (terminal anchors,
    sigma, lr, buffer). State this when comparing absolute numbers.
  * This script varies SCENE seeds per episode (epistemic over scenes). For
    TRAINING-seed variance, pass several checkpoints of the same method as
    separate --baseline columns and average them.
  * A checkpoint TRAINED on the old scatter MDP will legitimately underperform
    here on the crossing configs — that is an expected distribution shift, not a
    bug. Watch the console for `[WARN] agent i:` lines from load_agents; those
    mean the weights did NOT load (dim mismatch) and the actors are RANDOM, so
    the numbers are meaningless. (APF needs no checkpoint, so a sane APF column
    next to a garbage learned column is the tell.)

Usage
-----
  # MARDPG alone (CPU), with APF baseline, quick smoke:
  python evaluate_multiagent.py --checkpoint checkpoints/final \
      --variant mardpg --device cpu --suite quick --episodes 30 --apf

  # Full 2x2 comparison (checkpoints trained with --variant):
  python evaluate_multiagent.py --checkpoint checkpoints/cl_mardpg_seed0/final \
      --variant mardpg \
      --baseline MADDPG   maddpg   checkpoints/cl_maddpg_seed0/final \
      --baseline Ind-RDPG ind_rdpg checkpoints/cl_ind_rdpg_seed0/final \
      --baseline IDDPG    iddpg    checkpoints/cl_iddpg_seed0/final \
      --apf --episodes 200 --device cpu
"""
import os
import time
import math
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.eval_rollout import load_agents          # variant-aware
from mardpg_uav.train import CURRICULUM                  # reuse exact training stages

# Episode counts as an "encounter" if two live agents came within this distance
# (metres). Matches utils/metrics.ENCOUNTER_DIST.
ENCOUNTER_DIST = 6.0

# Rendering is optional; failure to import must not block the metrics.
try:
    from visualize_eval import plot_trajectory_3d, plot_trajectory_top_down  # noqa: F401
    _HAVE_RENDER = True
except Exception as _e:                                   # pragma: no cover
    _HAVE_RENDER = False
    _RENDER_IMPORT_ERR = _e


# ===========================================================================
# Evaluation suite — multi-agent correct (conflict_frac > 0 everywhere).
# In-distribution anchors are the EXACT training stages so eval MDP == train MDP.
# ===========================================================================
def _stage_cfg(idx_0based: int) -> dict:
    """Copy a curriculum stage config, dropping non-env keys (name/criteria)."""
    c = dict(CURRICULUM[idx_0based])
    c.pop('criteria', None)
    c.pop('name', None)
    return c


def build_suite(quick: bool = False):
    s4 = _stage_cfg(3)     # Moderate Density   (static 7,  conflict 0.8)
    s6 = _stage_cfg(5)     # Max Density        (static 16, conflict 1.0)
    s7 = _stage_cfg(6)     # Dynamic Threats    (static 16, conflict 1.0, dyn)

    def ood_static(**kw):  # distance OOD on the static base (no dynamics)
        c = dict(s6); c.update(kw); return c

    def ood_dyn(**kw):     # dynamic OOD on the dynamic base
        c = dict(s7); c.update(kw); return c

    suite = [
        ('stage4_train',     'in_dist', s4),
        ('stage6_train',     'in_dist', s6),
        ('stage7_train',     'in_dist', s7),
        ('ood_dist_50',      'ood',     ood_static(min_sep=50.0)),
        ('ood_dist_60',      'ood',     ood_static(min_sep=60.0)),
        ('ood_dyn_dense',    'ood',     ood_dyn(dynamic_obs=(3, 4),
                                                dynamic_radius=2.0,
                                                dynamic_speed=(1.0, 2.0))),
        ('ood_dyn_fast',     'ood',     ood_dyn(dynamic_obs=(2, 3),
                                                dynamic_radius=2.5,
                                                dynamic_speed=(2.5, 3.5))),
        ('ood_combined',     'ood',     ood_dyn(min_sep=60.0,
                                                dynamic_obs=(2, 3),
                                                dynamic_radius=2.5,
                                                dynamic_speed=(2.0, 3.0))),
        # Pure inter-UAV avoidance, no obstacles — the cleanest multi-agent test.
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
# Policy providers — uniform interface for learned policies and the APF baseline.
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
# Single episode rollout, policy-agnostic, fully instrumented.
# ===========================================================================
def run_episode(env, policy, stage_cfg, env_cfg, seed):
    n_agents = env_cfg['n_agents']
    dt = env_cfg.get('dt', 0.1)
    iu_min = env_cfg.get('inter_uav_min_dist', 1.0)

    # Reproducible scene + geometry + lidar noise for THIS episode, identical
    # across methods (scene_gen.rng drives crossing assignment too -> paired).
    env.scene_gen.rng.seed(seed)
    env.rangefinder.rng.seed(seed)
    obs = env.reset(stage_cfg)
    policy.reset(env)

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
        dyn_before = (env.agent_dyn_collided.copy()
                      if hasattr(env, 'agent_dyn_collided')
                      else np.zeros(n_agents, bool))

        t0 = time.perf_counter()
        acts = policy.act(env, obs)
        infer_time_total += time.perf_counter() - t0
        n_decisions += int(live_before.sum())

        obs, rewards, done, info = env.step(acts)
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
    dyn_collided = (env.agent_dyn_collided.copy()
                    if hasattr(env, 'agent_dyn_collided') else np.zeros(n_agents, bool))
    goals = env.goals.copy()

    flight_dist = np.zeros(n_agents)
    for i in range(n_agents):
        flight_dist[i] = float(np.sum(np.linalg.norm(np.diff(path[:, i, :], axis=0), axis=1)))

    straight = np.array([np.linalg.norm(goals[i] - start_pos[i]) for i in range(n_agents)])

    # --- Path efficiency, BOTH definitions -------------------------------
    tot_actual = float(flight_dist.sum())
    path_eff_paper = float(straight.sum() / tot_actual) if tot_actual > 1e-8 else np.nan
    reached_eff = []
    for i in range(n_agents):
        if reached[i] and flight_dist[i] > 1e-8:
            reached_eff.append(min(straight[i] / flight_dist[i], 1.0))
    path_eff_reached = float(np.mean(reached_eff)) if reached_eff else np.nan

    # --- Outcome partition (paper-faithful) ------------------------------
    neither = (~reached) & (~collided)
    trapped_rate_paper = float(neither.mean())
    trapped_rate_progress = float(np.mean(info.get('trapped', np.zeros(n_agents, bool))))

    # --- Multi-agent interaction metrics (the point of this rewrite) -----
    closest_appr = float(info.get('min_pair_dist', np.nan))
    near_miss_ratio = float(info.get('near_miss_ratio', 0.0))
    uav_col = np.asarray(info.get('uav_collisions', np.zeros(n_agents, bool)))
    uav_collision_rate = float(np.mean(uav_col))
    had_encounter = bool(np.isfinite(closest_appr) and closest_appr < ENCOUNTER_DIST)
    # 1.0 resolved / 0.0 unresolved / NaN if there was no encounter to resolve.
    if had_encounter:
        conflict_resolved = 1.0 if not bool(np.any(uav_col)) else 0.0
    else:
        conflict_resolved = np.nan

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
        success_rate=float(reached.mean()),                 # paper "success rate"
        mission_success=bool(reached.all()),                # MSR (Bernoulli)
        collision_rate=float(collided.mean()),
        dyn_collision_rate=float(dyn_collided.mean()),
        uav_collision_rate=uav_collision_rate,              # inter-UAV (per agent)
        trapped_rate_paper=trapped_rate_paper,
        trapped_rate_progress=trapped_rate_progress,
        team_reward=float(cum_reward.sum()),
        mean_agent_reward=float(cum_reward.mean()),
        path_eff_paper=path_eff_paper,
        path_eff_reached=path_eff_reached,
        mean_flight_distance_m=float(flight_dist.mean()),
        total_flight_distance_m=float(flight_dist.sum()),
        mean_inference_ms_per_decision=1e3 * infer_time_total / max(1, n_decisions),
        safe_inter_uav_ratio=float(info.get('safe_inter_uav_ratio', 1.0)),
        # interaction block
        closest_approach_m=(closest_appr if np.isfinite(closest_appr) else np.nan),
        near_miss_ratio=near_miss_ratio,
        had_encounter=had_encounter,
        conflict_resolved=conflict_resolved,                # 1.0 / 0.0 / NaN
        n_static_collisions=sum(1 for c in coll_type if c == 'static'),
        n_inter_uav_collisions=sum(1 for c in coll_type if c == 'inter_uav'),
        n_dynamic_collisions=sum(1 for c in coll_type if c == 'dynamic'),
    )
    render = dict(path=path, dyn_path=(np.array(dyn_path) if dyn else None),
                  dyn_r=dyn_r, reached=reached, collided=collided, goals=goals)
    return ep, agent_rows, render


def _episode_score(ep):
    return (1 if ep['mission_success'] else 0,
            ep['success_rate'], ep['team_reward'], -ep['steps'])


# ===========================================================================
# Statistics helpers (numpy-only; no scipy dependency)
# ===========================================================================
def _se(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    return (v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _paired_delta(a, b):
    """Two-sided test on paired per-episode differences a-b (paired by seed)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
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


# ===========================================================================
# Driver
# ===========================================================================
def evaluate_suite(methods, config, episodes, device, outdir,
                   render, base_seed, quick, render_method):
    if not os.path.exists(config):
        fb = os.path.join(os.path.dirname(os.path.abspath(__file__)), config)
        if os.path.exists(fb):
            config = fb
    cfg = yaml.safe_load(open(config))
    env_cfg = cfg['environment']
    n_agents = env_cfg['n_agents']

    os.makedirs(outdir, exist_ok=True)
    env = MultiUAVEnv(env_cfg)
    suite = build_suite(quick=quick)

    # ---- Preflight: catch the missing _nudge_free clearly (Fix A) -------
    try:
        env.scene_gen.rng.seed(base_seed)
        env.rangefinder.rng.seed(base_seed)
        env.reset(suite[0][2])
    except NameError as e:
        raise SystemExit(
            f"[FATAL] {e}\n"
            "This multi-agent suite uses conflict_frac > 0, which exercises the "
            "crossing-assignment branch in environment/assignment.py. That branch "
            "calls _nudge_free(). Apply Fix A first: move _nudge_free from "
            "utils/metrics.py into environment/assignment.py. See "
            "EVAL_MDP_HANDOVER.md.")

    # Build policy providers.
    providers = {}                       # name -> provider
    for name, kind, payload in methods:
        if kind == 'apf':
            providers[name] = APFPolicy(env, name=name)
        else:                            # learned variant: payload=(variant, ckpt)
            variant, ckpt = payload
            agents, _ = load_agents(ckpt, config, device, variant=variant)
            _ = agents[0].select_action(np.zeros(env.obs_dim, np.float32),
                                        np.zeros(env.action_dim, np.float32), evaluate=True)
            agents[0].reset_hidden(batch_size=1, eval_mode=True)
            providers[name] = LearnedPolicy(agents, name=name)

    ep_records, agent_records = [], []
    best = {}     # (method, config) -> (score, seed)

    for name, prov in providers.items():
        for cname, regime, stage_cfg in suite:
            if stage_cfg.get('static_obs', 0) > 16:
                print(f"[WARN] {cname}: static_obs will be clamped to 16 by the grid.")
            print(f"\n=== [{name}] {cname} ({regime}) | {episodes} episodes ===")
            t_cfg = time.time()
            for e in range(episodes):
                seed = base_seed + e          # SAME seed across methods -> paired
                ep, arows, _ = run_episode(env, prov, stage_cfg, env_cfg, seed)
                ep.update(method=name, config_name=cname, regime=regime, episode=e)
                ep_records.append(ep)
                for r in arows:
                    r.update(method=name, config_name=cname, episode=e, seed=seed)
                    agent_records.append(r)
                sc = _episode_score(ep)
                key = (name, cname)
                if key not in best or sc > best[key][0]:
                    best[key] = (sc, seed)
            sub = [r for r in ep_records
                   if r['method'] == name and r['config_name'] == cname]
            sr = np.mean([r['success_rate'] for r in sub])
            sr_se = _se([r['success_rate'] for r in sub])
            msr = np.mean([r['mission_success'] for r in sub])
            cr = np.mean([r['collision_rate'] for r in sub])
            ucr = np.mean([r['uav_collision_rate'] for r in sub])
            enc = np.mean([r['had_encounter'] for r in sub])
            crr_vals = [r['conflict_resolved'] for r in sub
                        if not (isinstance(r['conflict_resolved'], float)
                                and math.isnan(r['conflict_resolved']))]
            crr = (np.mean(crr_vals) if crr_vals else float('nan'))
            dur = time.time() - t_cfg
            print(f"  SR {sr:.1%}±{sr_se:.1%} | MSR {msr:.1%} | coll {cr:.1%} | "
                  f"uav_coll {ucr:.1%} | encounter {enc:.1%} | "
                  f"conflict_res {crr:.1%} | {dur:.0f}s ({dur/max(1,episodes):.2f}s/ep)")

    for r in ep_records:
        r['is_best'] = (r['seed'] == best.get((r['method'], r['config_name']),
                                               (None, None))[1])

    df_ep = pd.DataFrame(ep_records)
    df_ag = pd.DataFrame(agent_records)

    # ---- aggregated summary --------------------------------------------
    def agg(g):
        n = len(g)
        out = dict(n_episodes=n, regime=g['regime'].iloc[0])
        rate_cols = ['success_rate', 'collision_rate', 'dyn_collision_rate',
                     'uav_collision_rate', 'near_miss_ratio',
                     'trapped_rate_paper', 'trapped_rate_progress',
                     'mean_agent_reward', 'team_reward',
                     'path_eff_paper', 'path_eff_reached',
                     'closest_approach_m', 'mean_flight_distance_m', 'flight_time_s',
                     'mean_inference_ms_per_decision', 'safe_inter_uav_ratio']
        for col in rate_cols:
            v = g[col].astype(float)
            out[f'{col}_mean'] = (np.nanmean(v) if len(v) else np.nan)
            out[f'{col}_se'] = _se(v)
        # mission success (Bernoulli per episode)
        k = int(g['mission_success'].sum())
        msr, lo, hi = _wilson(k, n)
        out['mission_success_rate'] = msr
        out['mission_success_ci_lo'] = lo
        out['mission_success_ci_hi'] = hi
        # encounters + conflict resolution (headline multi-agent metric)
        cr = g['conflict_resolved'].astype(float)
        enc_mask = ~cr.isna()
        n_enc = int(enc_mask.sum())
        k_res = int((cr[enc_mask] == 1.0).sum())
        crr, crlo, crhi = _wilson(k_res, n_enc)
        out['encounter_rate'] = float(g['had_encounter'].mean())
        out['n_encounters'] = n_enc
        out['conflict_resolution_rate'] = crr
        out['conflict_res_ci_lo'] = crlo
        out['conflict_res_ci_hi'] = crhi
        out['static_coll_total'] = int(g['n_static_collisions'].sum())
        out['inter_uav_coll_total'] = int(g['n_inter_uav_collisions'].sum())
        out['dynamic_coll_total'] = int(g['n_dynamic_collisions'].sum())
        return pd.Series(out)

    df_sum = (df_ep.groupby(['method', 'config_name'], sort=False)
              .apply(agg).reset_index())

    # ---- pairwise comparison vs the primary method ----------------------
    primary = methods[0][0]
    cmp_rows = []
    if len(providers) > 1:
        for cname, regime, _ in suite:
            base_g = df_ep[(df_ep.method == primary) & (df_ep.config_name == cname)]
            for name in providers:
                if name == primary:
                    continue
                other_g = df_ep[(df_ep.method == name) & (df_ep.config_name == cname)]
                merged = base_g.merge(other_g, on='episode', suffixes=('_p', '_o'))
                for metric in ['success_rate', 'collision_rate', 'uav_collision_rate',
                               'mission_success', 'conflict_resolved']:
                    d = _paired_delta(merged[f'{metric}_p'], merged[f'{metric}_o'])
                    cmp_rows.append(dict(config_name=cname, regime=regime,
                                         primary=primary, baseline=name,
                                         metric=metric, **d))
    df_cmp = pd.DataFrame(cmp_rows)

    df_ep.to_csv(os.path.join(outdir, 'eval_episodes.csv'), index=False)
    df_ag.to_csv(os.path.join(outdir, 'eval_agents.csv'), index=False)
    df_sum.to_csv(os.path.join(outdir, 'eval_summary.csv'), index=False)
    if not df_cmp.empty:
        df_cmp.to_csv(os.path.join(outdir, 'eval_method_comparison.csv'), index=False)
    print(f"\nWrote eval_episodes / eval_agents / eval_summary"
          f"{' / eval_method_comparison' if not df_cmp.empty else ''} to {outdir}/")

    _plot(df_sum, list(providers.keys()),
          os.path.join(outdir, 'multiagent_summary.png'))

    # ---- render best episode of the chosen method per config ------------
    if render and _HAVE_RENDER:
        rm = render_method if render_method in providers else primary
        suite_cfg = {n: s for n, r, s in suite}
        for cname in df_ep['config_name'].unique():
            seed = best[(rm, cname)][1]
            stage_cfg = dict(suite_cfg[cname])
            print(f"Rendering best episode of '{cname}' [{rm}] (seed {seed}) ...")
            _, _, rnd = run_episode(env, providers[rm], stage_cfg, env_cfg, seed)
            title = (f"BEST [{rm}] | {cname} (seed {seed}) | "
                     f"reached {int(rnd['reached'].sum())}/{n_agents}, "
                     f"collided {int(rnd['collided'].sum())}/{n_agents}")
            plot_trajectory_3d(env, env_cfg, rnd, title,
                               os.path.join(outdir, f'best_3d_{rm}_{cname}.png'))
            plot_trajectory_top_down(env, env_cfg, rnd,
                                     title.replace('BEST', 'BEST TOP-DOWN'),
                                     os.path.join(outdir, f'best_top_down_{rm}_{cname}.png'))
    elif render and not _HAVE_RENDER:
        print(f"[render skipped] visualize_eval import failed: {_RENDER_IMPORT_ERR}")

    return df_ep, df_ag, df_sum, df_cmp


def _plot(df_sum, method_order, out_path):
    configs = list(dict.fromkeys(df_sum['config_name'].tolist()))
    nM = len(method_order)
    x = np.arange(len(configs))
    width = 0.8 / max(1, nM)
    cmap = plt.get_cmap('tab10')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(9, 1.3 * len(configs)), 11),
                                        sharex=True)
    for mi, m in enumerate(method_order):
        sub = df_sum[df_sum.method == m].set_index('config_name').reindex(configs)
        off = (mi - (nM - 1) / 2) * width
        ax1.bar(x + off, sub['success_rate_mean'], width,
                yerr=sub['success_rate_se'], capsize=3,
                label=m, color=cmap(mi % 10), alpha=0.9)
        ax2.bar(x + off, sub['collision_rate_mean'], width,
                yerr=sub['collision_rate_se'], capsize=3,
                color=cmap(mi % 10), alpha=0.9)
        ax3.bar(x + off, sub['uav_collision_rate_mean'], width,
                yerr=sub['uav_collision_rate_se'], capsize=3,
                color=cmap(mi % 10), alpha=0.9)
    ax1.axhline(0.80, color='green', ls=':', lw=1.2, label='0.80 ref')
    ax1.set_ylabel('Success rate'); ax1.set_ylim(0, 1.0)
    ax1.set_title('Per-agent success (top), total collision (mid), '
                  'inter-UAV collision (bottom). Error bars = SE.')
    ax1.legend(fontsize=8, ncol=min(nM + 1, 4)); ax1.grid(axis='y', alpha=0.3)
    ax2.set_ylabel('Collision rate')
    ax2.set_ylim(0, max(0.3, float(np.nanmax(df_sum['collision_rate_mean'])) * 1.3))
    ax2.grid(axis='y', alpha=0.3)
    ax3.set_ylabel('Inter-UAV collision rate')
    _ucap = float(np.nanmax(df_sum['uav_collision_rate_mean'])) if len(df_sum) else 0.1
    ax3.set_ylim(0, max(0.1, _ucap * 1.3))
    ax3.set_xticks(x); ax3.set_xticklabels(configs, rotation=30, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")


# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='primary method checkpoint dir')
    p.add_argument('--name', default='MARDPG', help='display name for the primary method')
    p.add_argument('--variant', default='mardpg',
                   choices=['mardpg', 'maddpg', 'ind_rdpg', 'iddpg'],
                   help='architecture of the primary checkpoint')
    p.add_argument('--baseline', action='append', nargs=3, default=[],
                   metavar=('NAME', 'VARIANT', 'CKPT'),
                   help='add a learned baseline; repeatable. '
                        'VARIANT in {mardpg,maddpg,ind_rdpg,iddpg}')
    p.add_argument('--apf', action='store_true', help='add the reactive APF baseline')
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--episodes', type=int, default=100,
                   help='episodes per (method,config). 200 -> SE ~ 3.5pp at p=0.5.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='eval_results')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--render-method', default=None,
                   help='which method to render best episodes for (default: primary)')
    p.add_argument('--base-seed', type=int, default=10_000)
    p.add_argument('--suite', choices=['full', 'quick'], default='full')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', default='mardpg-uav-eval')
    p.add_argument('--wandb-name', default=None)
    a = p.parse_args()

    methods = [(a.name, 'learned', (a.variant, a.checkpoint))]
    for nm, var, ck in a.baseline:
        methods.append((nm, 'learned', (var, ck)))
    if a.apf:
        methods.append(('APF', 'apf', None))

    if a.wandb:
        import wandb
        wandb.init(project=a.wandb_project, name=a.wandb_name, config=vars(a))

    df_ep, df_ag, df_sum, df_cmp = evaluate_suite(
        methods, a.config, a.episodes, a.device, a.outdir,
        not a.no_render, a.base_seed, a.suite == 'quick',
        a.render_method or a.name)

    if a.wandb:
        import wandb
        wandb.log({"eval/summary": wandb.Table(dataframe=df_sum)})
        if not df_cmp.empty:
            wandb.log({"eval/method_comparison": wandb.Table(dataframe=df_cmp)})
        img = os.path.join(a.outdir, 'multiagent_summary.png')
        if os.path.exists(img):
            wandb.log({"eval/multiagent_summary": wandb.Image(img)})
        wandb.finish()


if __name__ == "__main__":
    main()
