import numpy as np

def assign_start_goals(env, stage_cfg):
    """Return (starts, goals), each (N, 3) float32, for the current episode."""
    rng = env.scene_gen.rng
    n = env.n_agents
    env_size = np.asarray(env.cfg['env_size'], dtype=np.float32)
    center = env_size / 2.0
    min_alt = float(env.cfg.get('min_altitude', 0.0))
    ring_r = float(stage_cfg.get('ring_frac', 0.35)) * float(min(env_size[0], env_size[1]))
    conflict_frac = float(stage_cfg.get('conflict_frac', 0.0))
    min_sep = float(stage_cfg.get('min_sep', 20.0))
    min_start_sep = float(stage_cfg.get('min_start_sep', max(env.cfg.get('inter_uav_min_dist', 1.0) * 8, 10.0)))
    goal_min_sep = float(env.cfg.get('goal_min_sep', max(2.0 * env.cfg['goal_threshold'] + env.cfg['inter_uav_min_dist'], 4.0)))
    k = int(round(conflict_frac * n))
    conflict_ids = set(rng.choice(n, size=k, replace=False).tolist()) if k > 0 else set()
    starts = np.zeros((n, 3), dtype=np.float32)
    goals = np.zeros((n, 3), dtype=np.float32)
    (placed_starts, placed_goals) = ([], [])
    base_ang = float(rng.uniform(0.0, 2.0 * np.pi))

    def far_enough(p, placed, d):
        return all((np.linalg.norm(p - q) >= d for q in placed))
    for i in range(n):
        if i in conflict_ids:
            ang = base_ang + 2.0 * np.pi * i / n + float(rng.uniform(-0.12, 0.12))
            z_s = float(rng.uniform(0.3, 0.7) * env_size[2])
            start = center + np.array([ring_r * np.cos(ang), ring_r * np.sin(ang), 0.0], dtype=np.float32)
            start[2] = z_s
            goal = (2.0 * center - start).astype(np.float32)
            goal[2] = float(np.clip(2.0 * center[2] - z_s + rng.uniform(-5.0, 5.0), min_alt + 1.0, env_size[2] - 1.0))
            start = _nudge_free(env, start)
            goal = _nudge_free(env, goal)
        else:
            start = _sample_scatter(env, placed_starts, min_start_sep)
            goal = None
            for _ in range(200):
                g = env._sample_free_position()
                if np.linalg.norm(g - start) >= min_sep and far_enough(g, placed_goals, goal_min_sep) and far_enough(g, placed_starts, min_start_sep):
                    goal = g
                    break
            if goal is None:
                goal = env._sample_free_position()
        starts[i] = start
        goals[i] = goal
        placed_starts.append(start)
        placed_goals.append(goal)
    return (starts, goals)

def _sample_scatter(env, placed, min_start_sep):
    for _ in range(200):
        p = env._sample_free_position()
        if all((np.linalg.norm(p - q) >= min_start_sep for q in placed)):
            return p
    return env._sample_free_position()

def _nudge_free(env, pos, max_tries=200):
    """If a ring/antipodal point lands inside an obstacle, nudge to the nearest
    free sample while approximately preserving the crossing geometry, so we never
    spawn inside geometry (an instant, unavoidable collision)."""
    buf = env.cfg['collision_radius'] + 0.5
    env_size = np.asarray(env.cfg['env_size'], dtype=np.float32)
    min_alt = float(env.cfg.get('min_altitude', 0.0))
    pos = pos.astype(np.float32)
    if not env._inside_obstacles(pos, buffer=buf):
        return pos
    rng = env.scene_gen.rng
    lo = np.array([1.0, 1.0, min_alt + 1.0], dtype=np.float32)
    hi = env_size - 1.0
    for _ in range(max_tries):
        cand = np.clip(pos + rng.uniform(-5.0, 5.0, size=3).astype(np.float32), lo, hi)
        if not env._inside_obstacles(cand, buffer=buf):
            return cand.astype(np.float32)
    return env._sample_free_position()
