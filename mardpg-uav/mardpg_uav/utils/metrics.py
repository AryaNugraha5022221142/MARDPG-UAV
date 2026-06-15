"""
Evaluation metrics (v5) — adds graded multi-agent interaction metrics (FIX5)
over v4. Drop-in replacement for mardpg_uav/utils/metrics.py.

New (FIX5) interaction metrics — all NON-gating (CurriculumManager reads only the
keys it checks, so extra keys are ignored and promotion logic is unchanged):
  * min_pair_dist            : mean over episodes of the closest any two live
                               agents came (metres). The real coordination signal,
                               unlike the saturated inter_uav_safe@1m flag.
  * near_miss_ratio          : fraction of steps with any pair inside the soft band.
  * uav_collision_rate       : per-agent inter-UAV collision rate (vs obstacle).
  * encounter_rate           : fraction of episodes in which an encounter happened
                               (min_pair_dist < ENCOUNTER_DIST).
  * conflict_resolution_rate : among encounter episodes, fraction with ZERO
                               inter-UAV collisions. NaN when no encounters
                               occurred in the window. This is the metric that
                               demonstrates the multi-agent / no-comm claim.

v4 behaviour retained: reached-only capped path_efficiency is the GATE metric;
mission_success_rate and path_efficiency_paper are reported (non-gating).
"""
import numpy as np
from typing import Dict

# Episode counts as an "encounter" if two live agents came within this distance
# (metres). Tune per your sensor_range; 6 m ~ 2x the near-miss band default.
ENCOUNTER_DIST = 6.0


def _window_mean(values, window=100, default=0.0):
    if not values:
        return default
    arr = np.asarray(values[-window:], dtype=float)
    if np.all(np.isnan(arr)):
        return default
    return float(np.nanmean(arr))


class MetricsTracker:
    def __init__(self):
        self.episode_rewards   = []
        self.episode_lengths   = []
        self.successes         = []
        self.collisions        = []
        self.dyn_collisions    = []
        self.trapped           = []
        self.timeouts          = []
        self.path_efficiencies = []          # reached-only (GATE metric)
        self.path_eff_paper    = []          # paper definition (non-gating)
        self.mission_successes = []          # all-reached per episode
        self.safe_inter_uav    = []
        # [FIX5] interaction metrics
        self.min_pair_dists    = []
        self.near_miss_ratios  = []
        self.uav_collisions    = []
        self.encounters        = []
        self.resolutions       = []

    def soft_reset(self):
        for lst in (self.episode_rewards, self.episode_lengths, self.successes,
                    self.collisions, self.dyn_collisions, self.trapped,
                    self.timeouts, self.path_efficiencies, self.path_eff_paper,
                    self.mission_successes, self.safe_inter_uav,
                    self.min_pair_dists, self.near_miss_ratios,
                    self.uav_collisions, self.encounters, self.resolutions):
            lst.clear()

    def record_episode(self, length, info, start_pos, goal_pos,
                       path_history, rewards=None):
        n = len(info['reached'])

        if rewards is not None:
            team_reward = sum(rewards)
            self.episode_rewards.append(team_reward / max(1, n))

        self.episode_lengths.append(length)
        self.successes.append(np.mean(info['reached']))
        self.collisions.append(np.mean(info['collisions']))
        self.dyn_collisions.append(np.mean(info.get('dyn_collisions', np.zeros(n))))

        self.mission_successes.append(float(np.all(info['reached'])))

        if 'trapped' in info and info['trapped'] is not None:
            self.trapped.append(np.mean(info['trapped']))
        else:
            trapped_count = sum(1 for r, c in zip(info['reached'], info['collisions'])
                                if not r and not c)
            self.trapped.append(trapped_count / n)

        self.timeouts.append(float(info.get('timeout', False)))
        self.safe_inter_uav.append(info.get('safe_inter_uav_ratio', 1.0))

        # ---- [FIX5] interaction metrics ------------------------------------
        mpd = info.get('min_pair_dist', float('nan'))
        self.min_pair_dists.append(mpd)
        self.near_miss_ratios.append(float(info.get('near_miss_ratio', 0.0)))
        uav_col = info.get('uav_collisions', np.zeros(n))
        self.uav_collisions.append(float(np.mean(uav_col)))
        had_enc = bool(np.isfinite(mpd) and mpd < ENCOUNTER_DIST)
        self.encounters.append(float(had_enc))
        if had_enc:
            self.resolutions.append(float(not bool(np.any(uav_col))))

        # ---- Path efficiency, two definitions ------------------------------
        efficiencies = []
        sum_straight = 0.0
        sum_actual   = 0.0
        if path_history is not None and len(path_history) > 1:
            for i in range(n):
                actual = sum(np.linalg.norm(path_history[t][i] - path_history[t - 1][i])
                             for t in range(1, len(path_history)))
                straight = np.linalg.norm(goal_pos[i] - start_pos[i])
                sum_straight += straight
                sum_actual   += actual
                if bool(info['reached'][i]) and actual > 1e-8:
                    efficiencies.append(min(straight / actual, 1.0))
        self.path_efficiencies.append(np.mean(efficiencies) if efficiencies else np.nan)
        self.path_eff_paper.append(sum_straight / sum_actual if sum_actual > 1e-8 else np.nan)

    def get_stats(self) -> Dict[str, float]:
        return {
            'avg_reward':            _window_mean(self.episode_rewards),
            'success_rate':          _window_mean(self.successes),
            'mission_success_rate':  _window_mean(self.mission_successes),
            'collision_rate':        _window_mean(self.collisions),
            'dyn_collision_rate':    _window_mean(self.dyn_collisions),
            'trapped_rate':          _window_mean(self.trapped),
            'timeout_rate':          _window_mean(self.timeouts),
            'avg_episode_length':    _window_mean(self.episode_lengths),
            'path_efficiency':       _window_mean(self.path_efficiencies),
            'path_efficiency_paper': _window_mean(self.path_eff_paper),
            'inter_uav_safe':        _window_mean(self.safe_inter_uav),
            # [FIX5]
            'min_pair_dist':            _window_mean(self.min_pair_dists),
            'near_miss_ratio':          _window_mean(self.near_miss_ratios),
            'uav_collision_rate':       _window_mean(self.uav_collisions),
            'encounter_rate':           _window_mean(self.encounters),
            'conflict_resolution_rate': _window_mean(self.resolutions, default=float('nan')),
        }

    def get_window_stats(self, window=100) -> dict:
        return {
            'success_rate':          _window_mean(self.successes, window),
            'mission_success_rate':  _window_mean(self.mission_successes, window),
            'collision_rate':        _window_mean(self.collisions, window),
            'dyn_collision_rate':    _window_mean(self.dyn_collisions, window),
            'trapped_rate':          _window_mean(self.trapped, window),
            'timeout_rate':          _window_mean(self.timeouts, window),
            'path_efficiency':       _window_mean(self.path_efficiencies, window),
            'path_efficiency_paper': _window_mean(self.path_eff_paper, window),
            'inter_uav_safe':        _window_mean(self.safe_inter_uav, window),
            # [FIX5]
            'min_pair_dist':            _window_mean(self.min_pair_dists, window),
            'near_miss_ratio':          _window_mean(self.near_miss_ratios, window),
            'uav_collision_rate':       _window_mean(self.uav_collisions, window),
            'encounter_rate':           _window_mean(self.encounters, window),
            'conflict_resolution_rate': _window_mean(self.resolutions, window,
                                                      default=float('nan')),
        }


def _nudge_free(env, pos, max_tries=200):
    """If a ring/antipodal point lands inside an obstacle, nudge to the nearest
    free sample while approximately preserving the crossing geometry. Guarantees
    we never spawn inside geometry (which would be an instant, unavoidable
    collision and would artificially depress success)."""
    buf      = env.cfg['collision_radius'] + 0.5
    env_size = np.asarray(env.cfg['env_size'], dtype=np.float32)
    min_alt  = float(env.cfg.get('min_altitude', 0.0))
    pos      = pos.astype(np.float32)
    if not env._inside_obstacles(pos, buffer=buf):
        return pos
    rng = env.scene_gen.rng
    lo  = np.array([1.0, 1.0, min_alt + 1.0], dtype=np.float32)
    hi  = env_size - 1.0
    for _ in range(max_tries):
        cand = np.clip(pos + rng.uniform(-5.0, 5.0, size=3).astype(np.float32), lo, hi)
        if not env._inside_obstacles(cand, buffer=buf):
            return cand.astype(np.float32)
    return env._sample_free_position()
