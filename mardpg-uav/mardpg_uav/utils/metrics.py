"""
Evaluation metrics from Section 12.2 of blueprint.
"""
import numpy as np
from typing import Dict, List


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.collisions = []
        self.dyn_collisions = []
        self.trapped = []
        self.timeouts = []
        self.path_efficiencies = []
        self.safe_inter_uav = []
        
    def record_episode(self, length, info, start_pos, goal_pos, path_history, rewards=None):
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
            trapped_count = sum(1 for r, c in zip(info['reached'], info['collisions']) if not r and not c)
            self.trapped.append(trapped_count / n)
            
        self.timeouts.append(float(info.get('timeout', False)))
        
        self.safe_inter_uav.append(info.get('safe_inter_uav_ratio', 1.0))
        
        efficiencies = []
        for i in range(n):
            if len(path_history) > 1:
                actual = sum(np.linalg.norm(path_history[t][i] - path_history[t-1][i]) for t in range(1, len(path_history)))
                straight = np.linalg.norm(goal_pos[i] - start_pos[i])
                if actual > 1e-8:
                    # Cap at 1.0: an agent that barely moved before colliding has
                    # actual << straight, yielding ratio >> 1 — meaningless as efficiency.
                    efficiencies.append(min(straight / actual, 1.0))
        self.path_efficiencies.append(np.mean(efficiencies) if efficiencies else 0.0)

    def get_stats(self) -> Dict[str, float]:
        return {
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'success_rate': np.mean(self.successes[-100:]) if self.successes else 0.0,
            'collision_rate': np.mean(self.collisions[-100:]) if self.collisions else 0.0,
            'dyn_collision_rate': np.mean(self.dyn_collisions[-100:]) if self.dyn_collisions else 0.0,
            'trapped_rate': np.mean(self.trapped[-100:]) if self.trapped else 0.0,
            'timeout_rate': np.mean(self.timeouts[-100:]) if self.timeouts else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'path_efficiency': np.mean(self.path_efficiencies[-100:]) if self.path_efficiencies else 0.0,
            'inter_uav_safe': np.mean(self.safe_inter_uav[-100:]) if self.safe_inter_uav else 0.0
        }
        
    def get_window_stats(self, window=100) -> dict:
        """Get stats specifically for Curriculum evaluation."""
        return {
            'success_rate': np.mean(self.successes[-window:]) if self.successes else 0.0,
            'collision_rate': np.mean(self.collisions[-window:]) if self.collisions else 0.0,
            'dyn_collision_rate': np.mean(self.dyn_collisions[-window:]) if self.dyn_collisions else 0.0,
            'trapped_rate': np.mean(self.trapped[-window:]) if self.trapped else 0.0,
            'timeout_rate': np.mean(self.timeouts[-window:]) if self.timeouts else 0.0,
            'path_efficiency': np.mean(self.path_efficiencies[-window:]) if self.path_efficiencies else 0.0,
            'inter_uav_safe': np.mean(self.safe_inter_uav[-window:]) if self.safe_inter_uav else 0.0
        }
