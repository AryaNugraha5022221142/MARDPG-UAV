"""
Evaluation metrics (v3) — audit fix C4.

Change
------
Path efficiency is now computed ONLY over agents that reached their goal.
The previous `min(straight/actual, 1.0)` assigned efficiency ~1.0 to an
agent that flew 2 m and crashed (actual << straight), which inflated the
metric exactly when collision rates were high — and path_efficiency gates
curriculum promotion in stages 1-5. Episodes with zero successful agents
record NaN and are excluded from window averages (NaN-safe means below).
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
        self.path_efficiencies = []
        self.safe_inter_uav    = []

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

        if 'trapped' in info and info['trapped'] is not None:
            self.trapped.append(np.mean(info['trapped']))
        else:
            trapped_count = sum(1 for r, c in zip(info['reached'], info['collisions'])
                                if not r and not c)
            self.trapped.append(trapped_count / n)

        self.timeouts.append(float(info.get('timeout', False)))
        self.safe_inter_uav.append(info.get('safe_inter_uav_ratio', 1.0))

        # [C4] Efficiency is only meaningful for agents that actually reached
        # the goal. Crashed agents previously scored ~1.0 (capped), trapped
        # agents deflated it — both corrupt the curriculum gate.
        efficiencies = []
        if len(path_history) > 1:
            for i in range(n):
                if not bool(info['reached'][i]):
                    continue
                actual = sum(np.linalg.norm(path_history[t][i] - path_history[t - 1][i])
                             for t in range(1, len(path_history)))
                straight = np.linalg.norm(goal_pos[i] - start_pos[i])
                if actual > 1e-8:
                    efficiencies.append(min(straight / actual, 1.0))
        # NaN = "no successful agent this episode" -> excluded from window mean
        self.path_efficiencies.append(np.mean(efficiencies) if efficiencies else np.nan)

    def get_stats(self) -> Dict[str, float]:
        return {
            'avg_reward':         _window_mean(self.episode_rewards),
            'success_rate':       _window_mean(self.successes),
            'collision_rate':     _window_mean(self.collisions),
            'dyn_collision_rate': _window_mean(self.dyn_collisions),
            'trapped_rate':       _window_mean(self.trapped),
            'timeout_rate':       _window_mean(self.timeouts),
            'avg_episode_length': _window_mean(self.episode_lengths),
            'path_efficiency':    _window_mean(self.path_efficiencies),
            'inter_uav_safe':     _window_mean(self.safe_inter_uav),
        }

    def get_window_stats(self, window=100) -> dict:
        return {
            'success_rate':       _window_mean(self.successes, window),
            'collision_rate':     _window_mean(self.collisions, window),
            'dyn_collision_rate': _window_mean(self.dyn_collisions, window),
            'trapped_rate':       _window_mean(self.trapped, window),
            'timeout_rate':       _window_mean(self.timeouts, window),
            'path_efficiency':    _window_mean(self.path_efficiencies, window),
            'inter_uav_safe':     _window_mean(self.safe_inter_uav, window),
        }
