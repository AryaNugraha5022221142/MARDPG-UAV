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
        self.trapped = []
        self.path_lengths = []
        self.straight_line_dists = []
        
    def record_episode(self, rewards: List[float], length: int,
                      reached: List[bool], per_agent_collided: List[bool],
                      start_pos: List[np.ndarray],
                      goal_pos: List[np.ndarray],
                      path_history: List[List[np.ndarray]]):
        self.episode_rewards.append(sum(rewards))
        self.episode_lengths.append(length)
        
        n_agents = len(reached)
        
        # Calculate per-agent rates instead of all-or-nothing
        success_ratio = sum(reached) / n_agents if n_agents > 0 else 0.0
        self.successes.append(success_ratio)
        
        # Ensure lists
        if isinstance(per_agent_collided, bool):
            per_agent_collided = [per_agent_collided] * n_agents
        if isinstance(reached, bool):
            reached = [reached] * n_agents

        collision_ratio = sum(per_agent_collided) / n_agents if n_agents > 0 else 0.0
        self.collisions.append(collision_ratio)

        # Trapped means not reached and not collided
        trapped_ratio = sum(1 for r, c in zip(reached, per_agent_collided) if not r and not c) / n_agents if n_agents > 0 else 0.0
        self.trapped.append(trapped_ratio)
        
        # Path efficiency
        for i in range(n_agents):
            if len(path_history) > 1:
                actual_path = sum(
                    np.linalg.norm(path_history[t][i] - path_history[t-1][i])
                    for t in range(1, len(path_history))
                )
                straight = np.linalg.norm(goal_pos[i] - start_pos[i])
                if straight > 0:
                    self.path_lengths.append(actual_path)
                    self.straight_line_dists.append(straight)
    
    def get_stats(self) -> Dict[str, float]:
        stats = {
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'success_rate': np.mean(self.successes[-100:]) if len(self.successes) >= 100 else np.mean(self.successes) if self.successes else 0,
            'collision_rate': np.mean(self.collisions[-100:]) if len(self.collisions) >= 100 else np.mean(self.collisions) if self.collisions else 0,
            'trapped_rate': np.mean(self.trapped[-100:]) if len(self.trapped) >= 100 else np.mean(self.trapped) if self.trapped else 0,
            'avg_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
        }
        if self.path_lengths:
            efficiencies = [
                s / max(a, 1e-8)
                for s, a in zip(self.straight_line_dists, self.path_lengths)
                if a > 1e-8
            ]
            stats['path_efficiency'] = np.mean(efficiencies) if efficiencies else 0.0
        return stats
