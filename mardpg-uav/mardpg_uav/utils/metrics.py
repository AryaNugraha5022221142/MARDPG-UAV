"""
Evaluation metrics (v4) — additive over v3 (audit fix C4 retained).

v3 change (retained): path_efficiency (the curriculum GATE metric) is computed
ONLY over agents that reached their goal, with min(straight/actual, 1.0). Crashed
agents no longer score ~1.0 and inflate the gate.

v4 additions (NON-gating, safe to add — the CurriculumManager reads only the
specific keys it checks, so extra keys are ignored):
  * mission_success_rate : fraction of episodes in which ALL agents reached
    (the paper's "Mission Success", audit M2). Exposed in get_stats /
    get_window_stats so train.py logs it without changing promotion logic.
  * path_efficiency_paper : the reference definition (Xue & Chen 2024, Sec.
    VI.C.e) — ratio of summed straight-line distances to summed actual path
    lengths over ALL agents in an episode. Reported for comparability with the
    paper; it is NOT used to gate the curriculum (it is the inflated definition
    the v3 fix deliberately replaced for gating).

Nothing that the curriculum gate depends on is changed.
"""
import numpy as np
from typing import Dict


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
        self.path_eff_paper    = []          # [v4] paper definition (non-gating)
        self.mission_successes = []          # [v4] all-reached per episode
        self.safe_inter_uav    = []

    def soft_reset(self):
        for lst in (self.episode_rewards, self.episode_lengths, self.successes,
                    self.collisions, self.dyn_collisions, self.trapped,
                    self.timeouts, self.path_efficiencies, self.path_eff_paper,
                    self.mission_successes, self.safe_inter_uav):
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

        # [v4] Mission success = every agent reached.
        self.mission_successes.append(float(np.all(info['reached'])))

        if 'trapped' in info and info['trapped'] is not None:
            self.trapped.append(np.mean(info['trapped']))
        else:
            trapped_count = sum(1 for r, c in zip(info['reached'], info['collisions'])
                                if not r and not c)
            self.trapped.append(trapped_count / n)

        self.timeouts.append(float(info.get('timeout', False)))
        self.safe_inter_uav.append(info.get('safe_inter_uav_ratio', 1.0))

        # ---- Path efficiency, two definitions -------------------------------
        # GATE metric (reached-only, capped): unchanged from v3.
        efficiencies = []
        # Paper metric (ratio of sums over ALL agents): numerator/denominator.
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
            'mission_success_rate':  _window_mean(self.mission_successes),   # [v4]
            'collision_rate':        _window_mean(self.collisions),
            'dyn_collision_rate':    _window_mean(self.dyn_collisions),
            'trapped_rate':          _window_mean(self.trapped),
            'timeout_rate':          _window_mean(self.timeouts),
            'avg_episode_length':    _window_mean(self.episode_lengths),
            'path_efficiency':       _window_mean(self.path_efficiencies),   # gate
            'path_efficiency_paper': _window_mean(self.path_eff_paper),      # [v4]
            'inter_uav_safe':        _window_mean(self.safe_inter_uav),
        }

    def get_window_stats(self, window=100) -> dict:
        return {
            'success_rate':          _window_mean(self.successes, window),
            'mission_success_rate':  _window_mean(self.mission_successes, window),  # [v4]
            'collision_rate':        _window_mean(self.collisions, window),
            'dyn_collision_rate':    _window_mean(self.dyn_collisions, window),
            'trapped_rate':          _window_mean(self.trapped, window),
            'timeout_rate':          _window_mean(self.timeouts, window),
            'path_efficiency':       _window_mean(self.path_efficiencies, window),
            'path_efficiency_paper': _window_mean(self.path_eff_paper, window),     # [v4]
            'inter_uav_safe':        _window_mean(self.safe_inter_uav, window),
        }
