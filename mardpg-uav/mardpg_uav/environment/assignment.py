"""
[FIX1] Start/goal assignment for genuine multi-agent interaction.

Drop this file at:  mardpg_uav/environment/assignment.py

Root cause it fixes
-------------------
The old reset() sampled each UAV's start and goal *independently* with only a
minimum-separation constraint, in a 100x100x60 volume. Independent random points
in a large box almost never produce crossing straight-line paths, so the UAVs
flew five unrelated single-agent errands and never came near each other
(inter_uav_safe == 1.00 in every eval). There was no multi-agent interaction to
learn from.

The reference (Xue & Chen 2024, Sec. VI.D.1) instead *intersects* the five
start->goal lines on purpose so that "the avoidance behavior of UAVs towards
environmental obstacles and other UAVs can be clearly represented."

What this does
--------------
A configurable fraction `conflict_frac` of agents are placed on a ring around
the arena centre, each with an ANTIPODAL goal (reflected through the centre), so
their straight-line paths cross the middle of the arena and the agents are forced
to sense and avoid one another. The remaining agents use the legacy independent
scatter sampling. conflict_frac = 0.0 reproduces the old behaviour exactly.

Stage knobs (read from stage_cfg, all optional):
    conflict_frac : float in [0,1]   fraction of agents given crossing pairs
    ring_frac     : float            ring radius as a fraction of min(env_x, env_y)
    min_sep       : float            min straight-line start->goal distance (scatter)
    min_start_sep : float            min separation between any two start points
"""
import numpy as np


def assign_start_goals(env, stage_cfg):
    """Return (starts, goals), each (N, 3) float32, for the current episode."""
    rng      = env.scene_gen.rng
    n        = env.n_agents
    env_size = np.asarray(env.cfg['env_size'], dtype=np.float32)
    center   = env_size / 2.0
    min_alt  = float(env.cfg.get('min_altitude', 0.0))

    ring_r        = float(stage_cfg.get('ring_frac', 0.35)) * float(min(env_size[0], env_size[1]))
    conflict_frac = float(stage_cfg.get('conflict_frac', 0.0))
    min_sep       = float(stage_cfg.get('min_sep', 20.0))
    min_start_sep = float(stage_cfg.get(
        'min_start_sep', max(env.cfg.get('inter_uav_min_dist', 1.0) * 8, 10.0)))
    goal_min_sep  = float(env.cfg.get(
        'goal_min_sep',
        max(2.0 * env.cfg['goal_threshold'] + env.cfg['inter_uav_min_dist'], 4.0)))

    k = int(round(conflict_frac * n))
    conflict_ids = set(rng.choice(n, size=k, replace=False).tolist()) if k > 0 else set()

    starts = np.zeros((n, 3), dtype=np.float32)
    goals  = np.zeros((n, 3), dtype=np.float32)
    placed_starts, placed_goals = [], []
    base_ang = float(rng.uniform(0.0, 2.0 * np.pi))

    def far_enough(p, placed, d):
        return all(np.linalg.norm(p - q) >= d for q in placed)

    for i in range(n):
        if i in conflict_ids:
            # Evenly spaced on a ring, antipodal goal -> path crosses the centre.
            ang = base_ang + 2.0 * np.pi * i / n + float(rng.uniform(-0.12, 0.12))
            z_s = float(rng.uniform(0.30, 0.70) * env_size[2])
            start = center + np.array([ring_r * np.cos(ang),
                                       ring_r * np.sin(ang), 0.0], dtype=np.float32)
            start[2] = z_s
            goal = (2.0 * center - start).astype(np.float32)
            # Vertical jitter so all crossings don't share one altitude band.
            goal[2] = float(np.clip(2.0 * center[2] - z_s + rng.uniform(-5.0, 5.0),
                                    min_alt + 1.0, env_size[2] - 1.0))
            start = _nudge_free(env, start)
            goal  = _nudge_free(env, goal)
        else:
            # Legacy independent scatter, with start/goal separation preserved.
            start = _sample_scatter(env, placed_starts, min_start_sep)
            goal  = None
            for _ in range(200):
                g = env._sample_free_position()
                if (np.linalg.norm(g - start) >= min_sep and
                        far_enough(g, placed_goals, goal_min_sep) and
                        far_enough(g, placed_starts, min_start_sep)):
                    goal = g
                    break
            if goal is None:
                goal = env._sample_free_position()

        starts[i] = start
        goals[i]  = goal
        placed_starts.append(start)
        placed_goals.append(goal)

    return starts, goals


def _sample_scatter(env, placed, min_start_sep):
    for _ in range(200):
        p = env._sample_free_position()
        if all(np.linalg.norm(p - q) >= min_start_sep for q in placed):
            return p
    return env._sample_free_position()
