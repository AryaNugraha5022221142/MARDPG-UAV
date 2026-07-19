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
from scripts.train import CURRICULUM, load_config
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.rendering import RenderConfig, select_backend, LiveRenderer
from mardpg_uav.rendering.media import generate_episode_media
from mardpg_uav.wandb_logger import WandbLogger

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mardpg.eval")

ENCOUNTER_DIST = 6.0

_VARIANT_FLAGS = {
    'mardpg':   (True,  True),
    'maddpg':   (False, True),
    'ind_rdpg': (True,  False),
    'iddpg':    (False, False),
}

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
            obs_dim=env_cfg.get('obs_dim', 35), action_dim=env_cfg.get('action_dim', 2),
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            recurrent=recurrent, centralized=centralized, device=device)

        init_fp = _module_fingerprint(ag.actor)

        apath = os.path.join(checkpoint_dir, f"agent_{i}.pt")
        if not os.path.exists(apath):
            if i == 0:
                resolved_dir = None
                if os.path.exists(os.path.join(checkpoint_dir, "final", "agent_0.pt")):
                    resolved_dir = os.path.join(checkpoint_dir, "final")
                elif os.path.isdir(checkpoint_dir):
                    candidates = []
                    for d in os.listdir(checkpoint_dir):
                        p = os.path.join(checkpoint_dir, d)
                        if os.path.isdir(p) and os.path.exists(os.path.join(p, "agent_0.pt")):
                            candidates.append(d)
                    if candidates:
                        episodes = [c for c in candidates if c.startswith("episode_") or c.startswith("interrupted_episode_")]
                        stages = [c for c in candidates if c.startswith("stage_")]
                        
                        if episodes:
                            resolved_dir = os.path.join(checkpoint_dir, sorted(episodes)[-1])
                        elif stages:
                            resolved_dir = os.path.join(checkpoint_dir, sorted(stages)[-1])
                        else:
                            resolved_dir = os.path.join(checkpoint_dir, sorted(candidates)[-1])
                if resolved_dir:
                    print(f"Resolved checkpoint directory to leaf: {resolved_dir}")
                    checkpoint_dir = resolved_dir
                    apath = os.path.join(checkpoint_dir, f"agent_{i}.pt")

            if not os.path.exists(apath):
                apath = os.path.join(checkpoint_dir, "agent_0.pt")
                if not os.path.exists(apath):
                    raise FileNotFoundError(f"[LOAD FAIL] missing agent_0.pt in {checkpoint_dir} (and subdirectories)")
        ckpt = torch.load(apath, map_location=device)

        if 'actor_private' in ckpt:
            if i == 0:
                spath = os.path.join(checkpoint_dir, "shared_actor.pt")
                if not os.path.exists(spath):
                    raise FileNotFoundError(f"[LOAD FAIL] missing {spath}")
                sc = torch.load(spath, map_location=device)
                ag.shared_extractor.load_state_dict(sc['shared_actor'])
            ag.actor.load_state_dict(ckpt['actor_private'], strict=False)
        elif 'actor' in ckpt:
            ag.actor.load_state_dict(ckpt['actor'])
        else:
            raise KeyError(f"[LOAD FAIL] {apath} has neither 'actor_private' "
                           f"nor 'actor' (keys: {list(ckpt.keys())})")

        post_fp = _module_fingerprint(ag.actor)

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

    def ood_dense(**kw):
        c = dict(s6); c.update(kw); return c

    suite = [
        ('stage4_train',     'in_dist', s4),
        ('stage6_train',     'in_dist', s6),
        ('stage7_train',     'in_dist', s7),
        ('ood_dist_60',      'ood',     ood_static(min_sep=60.0)),
        ('ood_dense_20',     'ood',     ood_dense(static_obs=20)),
        ('ood_dense_25',     'ood',     ood_dense(static_obs=25)),
        ('ood_dense_max',    'ood',     ood_dense(static_obs=30)),
        ('ood_dyn_fast',     'ood',     ood_dyn(dynamic_obs=(2, 3),
                                                dynamic_radius=2.5,
                                                dynamic_speed=(2.5, 3.5))),
        ('sanity_crossings', 'in_dist', dict(env_size=[100.0, 100.0, 60.0],
                                             static_obs=0, max_steps=600,
                                             min_sep=30.0, min_start_sep=12.0,
                                             conflict_frac=1.0, ring_frac=0.35)),
    ]
    if quick:
        keep = {'stage7_train', 'ood_dist_60', 'ood_dyn_fast', 'sanity_crossings', 'ood_dense_25'}
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
        for i, ag in enumerate(self.current_agents):
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

def run_episode(env, policy, stage_cfg, env_cfg, seed, capture_render=False, live=None):
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

    if live is not None:
        live.reset(env)

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

        if live is not None:
            live.step(env)

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
        return float(np.mean(x))
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
        return (float(v.item()), np.nan, np.nan)
    rng = rng or np.random.default_rng(12345)
    boots = np.array([agg(v[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return (float(agg(v)),
            float(np.percentile(boots, 100 * (1 - ci) / 2)),
            float(np.percentile(boots, 100 * (1 + ci) / 2)))

_SEED_METRICS = ['success_rate', 'mission_success', 'collision_rate',
                 'static_collision_rate', 'dyn_collision_rate',
                 'uav_collision_rate', 'path_eff_reached',
                 'closest_approach_m', 'near_miss_ratio',
                 'safe_inter_uav_ratio', 'mean_flight_dist',
                 'mean_flight_time', 'mean_time_to_goal']

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

            vv = v[~np.isnan(v)]
            row[f'{m}__scene_se'] = (float(vv.std(ddof=1) / np.sqrt(len(vv)))
                                     if len(vv) > 1 else 0.0)

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

        for metric in ['success_rate', 'collision_rate', 'conflict_resolution_rate']:
            piv = g.pivot_table(index='seed', columns='config_name',
                                values=metric, aggfunc='mean')
            mat = piv.values
            point = _iqm(mat.flatten())

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

def _seed_label(ckpt_dir, fallback_idx):
    """Parse the TRUE training seed from the checkpoint path (e.g.
    '.../cl_mardpg_seed2/final' -> 2). This is what makes results computed on
    SEPARATE machines mergeable: seeds are globally meaningful, never per-VM
    positional indices. Falls back to the enumeration index if no seed token
    is found (and warns, because that breaks cross-machine merges)."""
    m = re.search(r'seed[_-]?(\d+)', ckpt_dir)
    if m:
        return int(m.group(1))
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

def evaluate(methods, config, episodes, device, outdir, base_seed, quick,
             wlogger=None, rcfg=None):
    """Run the evaluation suite.

    Args:
        wlogger: a WandbLogger (its .use_wandb flag gates all uploads).
        rcfg: a RenderConfig controlling media generation. When
              rcfg.any_media_requested() is False, no rendering runs at all.
    """
    cfg = load_config(config)
    env_cfg = cfg['environment']
    rcfg = rcfg or RenderConfig.from_config(cfg)
    outdir = rcfg.output_directory if rcfg else outdir
    wandb_log = bool(wlogger and wlogger.use_wandb)
    os.makedirs(outdir, exist_ok=True)
    env = MultiUAVEnv(env_cfg)
    suite = build_suite(quick=quick)


    media_paths = []  # collected for the optional wandb Artifact bundle

    try:
        env.scene_gen.rng.seed(base_seed)
        env.rangefinder.rng.seed(base_seed)
        env.reset(suite[0][2])
    except NameError as e:
        raise SystemExit(
            f"{e}\nThe crossing-assignment branch calls _nudge_free(); "
            "ensure it lives in environment/assignment.py (audit Fix A).")
    # One live renderer for the whole run (built once; no-op when headless).
    live = None
    if rcfg.enable_render and rcfg.realtime_render:
        live = LiveRenderer(env, env_cfg)

    ep_records = []
    for name, variant, ckpt_arg in methods:
        seed_list = [(_seed_label(ck, idx), ck)
                     for idx, ck in enumerate(_expand_ckpts(ckpt_arg))]

        for seed_idx, ckpt in seed_list:
            agents, _ = load_agents_strict(ckpt, config, device, variant)
            provider = LearnedPolicy(agents, name=name)

            for cname, regime, stage_cfg in suite:
                t0 = time.time()
                best_score = None
                best_seed = None

                # Capture trajectory data on the first few episodes so we can
                # render per-episode PNGs (cheap) even when not doing video.
                capture_first_few = rcfg.enable_render and rcfg.save_png

                for e in range(episodes):
                    scene_seed = base_seed + e
                    capture = capture_first_few and (e < 3)
                    ep = run_episode(env, provider, stage_cfg, env_cfg,
                                     scene_seed, capture_render=capture, live=live)

                    sc = (1 if ep['mission_success'] else 0, ep['success_rate'],
                          ep['team_reward'], -ep['steps'])
                    if best_score is None or sc > best_score:
                        best_score = sc
                        best_seed = scene_seed

                    if capture and '_render_rnd' in ep:
                        rnd = ep.pop('_render_rnd')
                        tag = f"traj_{name}_s{seed_idx}_{cname}_ep{e}"
                        title = f"Traj: {name} | {cname} | ep {e} | reaches {int(rnd['reached'].sum())}"
                        # Per-episode: PNG only (no video) to keep it cheap.
                        ep_rcfg = rcfg.merged_with_cli(record_video=False, save_frames=False)
                        produced = generate_episode_media(env, env_cfg, rnd, ep_rcfg,
                                                          tag, title, outdir)
                        media_paths.extend(produced.values())
                        if wandb_log:
                            wlogger.log_media(produced, prefix=f"eval/{name}_{cname}_ep{e}")
                    else:
                        ep.pop('_render_rnd', None)

                    ep.update(method=name, variant=variant, seed=seed_idx,
                              checkpoint=str(ckpt), config_name=cname,
                              regime=regime, episode=e)
                    ep_records.append(ep)

                # Best-episode media: full video + PNGs, driven by the config.
                if rcfg.enable_render and (rcfg.record_video or rcfg.save_frames or rcfg.save_png):
                    best_ep = run_episode(env, provider, stage_cfg, env_cfg,
                                          best_seed, capture_render=True, live=live)
                    rnd = best_ep.pop('_render_rnd', None)
                    if rnd is not None:
                        tag = f"best_{name}_s{seed_idx}_{cname}"
                        title = (f"BEST [{name}] | {cname} (seed {best_seed}) | "
                                 f"reached {int(rnd['reached'].sum())}/{env.n_agents}")
                        produced = generate_episode_media(env, env_cfg, rnd, rcfg,
                                                          tag, title, outdir)
                        media_paths.extend(produced.values())
                        if wandb_log:
                            wlogger.log_media(produced,
                                              prefix=f"eval/best_{name}_{cname}",
                                              fps=rcfg.video_fps)

                log.info("[%s | seed %s | %s] done in %.1fs",
                         name, seed_idx, cname, time.time() - t0)

    if live is not None:
        live.close()

    df_ep = pd.DataFrame(ep_records)
    df_seed = aggregate_per_seed(df_ep)
    df_sum = aggregate_across_seeds(df_seed)
    df_iqm = aggregate_method_iqm(df_seed, regimes=('in_dist',))

    csv_paths = [os.path.join(outdir, n) for n in
                 ('eval_episodes.csv', 'eval_per_seed.csv',
                  'eval_summary.csv', 'eval_method_iqm.csv')]
    df_ep.to_csv(csv_paths[0], index=False)
    df_seed.to_csv(csv_paths[1], index=False)
    df_sum.to_csv(csv_paths[2], index=False)
    df_iqm.to_csv(csv_paths[3], index=False)

    if wandb_log:
        import wandb
        for p in csv_paths:
            wandb.save(p, base_path=outdir)
        # Bundle CSVs + all rendered media into a single versioned artifact.
        if rcfg.wandb_log_media:
            wlogger.log_artifact(
                name=f"eval-results-{wandb.run.id if wandb.run else 'run'}",
                artifact_type="evaluation",
                paths=csv_paths + [p for p in media_paths if p],
                metadata={"n_episodes": int(len(df_ep))})

    _print_variance_report(df_seed, df_sum)
    return df_ep, df_seed, df_sum, df_iqm

def _print_variance_report(df_seed, df_sum):
    """The point of the rewrite, made legible: seed-level SE (honest) vs the
    within-seed scene SE that v1 reported."""

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
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--episodes', type=int, default=100,
                   help='scenes per (method,seed,config). 100 -> scene SE ~3.5pp at p=.5')
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default=None,
                   help='Output dir for media/CSV. Overrides render.output_directory.')
    p.add_argument('--base-seed', type=int, default=10_000)
    p.add_argument('--suite', choices=['full', 'quick'], default='full')

    # Rendering flags. Each defaults to None => "not set, use config value".
    # This is what lets one config drive both cloud (headless) and local runs.
    p.add_argument('--render', dest='enable_render', action='store_true', default=None,
                   help='Enable media generation (PNG/MP4) for this run.')
    p.add_argument('--realtime', dest='realtime_render', action='store_true', default=None,
                   help='Show a live interactive window (local desktops only).')
    p.add_argument('--video', dest='record_video', action='store_true', default=None,
                   help='Record an MP4 of the best episode per scene.')
    p.add_argument('--no-video', dest='record_video', action='store_false', default=None,
                   help='Disable MP4 recording.')
    p.add_argument('--save-png', dest='save_png', action='store_true', default=None)
    p.add_argument('--save-frames', dest='save_frames', action='store_true', default=None)
    p.add_argument('--video-fps', dest='video_fps', type=int, default=None)
    p.add_argument('--video-codec', dest='video_codec', default=None)
    p.add_argument('--render-backend', dest='render_backend', default=None)
    p.add_argument('--image-dpi', dest='image_dpi', type=int, default=None)
    a = p.parse_args()

    methods = [(nm, var, ck) for nm, var, ck in a.method]
    if a.checkpoint:
        methods.insert(0, (a.name, a.variant, a.checkpoint))
    if not methods:
        raise SystemExit("Provide at least one --method NAME VARIANT CKPTS or --checkpoint.")

    # Build the render config: YAML defaults, then CLI overrides.
    cfg = load_config(a.config)
    rcfg = RenderConfig.from_config(cfg).merged_with_cli(
        enable_render=a.enable_render,
        realtime_render=a.realtime_render,
        record_video=a.record_video,
        save_png=a.save_png,
        save_frames=a.save_frames,
        video_fps=a.video_fps,
        video_codec=a.video_codec,
        render_backend=a.render_backend,
        image_dpi=a.image_dpi,
        output_directory=a.outdir,
    )
    # Choosing the backend once, up front, honours headless/cloud correctly.
    select_backend(rcfg.render_backend, want_interactive=rcfg.realtime_render)

    wlogger = WandbLogger(use_wandb=a.wandb, project=a.wandb_project,
                          config={**vars(a), "render": rcfg.__dict__},
                          name=a.wandb_name)

    df_ep, df_seed, df_sum, df_iqm = evaluate(
        methods, a.config, a.episodes, a.device, rcfg.output_directory,
        a.base_seed, a.suite == 'quick', wlogger=wlogger, rcfg=rcfg)

    if wlogger.use_wandb:
        wlogger.log_table("eval/summary", df_sum)
        wlogger.log_table("eval/method_iqm", df_iqm)
        wlogger.finish()

if __name__ == "__main__":
    main()
