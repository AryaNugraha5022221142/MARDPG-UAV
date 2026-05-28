import numpy as np
from typing import List, Tuple


class RewardFunction:
    def __init__(self, alpha: float = 3.0, lambda_col: float = 5.0,
                 sigma_col: float = 10.0, lambda_sep: float = 2.0,
                 sigma_sep: float = 5.0, r_free: float = 0.1,
                 r_step: float = -0.4,
                 delta: Tuple[float, float, float, float, float] = (0.40, 0.30, 0.10, 0.10, 0.10),
                 collision_radius: float = 0.5,
                 inter_uav_min: float = 1.0):
        self.alpha = alpha
        self.lambda_col = lambda_col
        self.sigma_col = sigma_col
        self.lambda_sep = lambda_sep
        self.sigma_sep = sigma_sep
        self.r_free = r_free
        self.r_step = r_step
        self.delta = delta
        self.collision_radius = collision_radius
        self.inter_uav_min = inter_uav_min
        
        # Store previous distances for transfer reward
        self.prev_distances = {}

    def reset(self, agent_ids: List[int], distances: dict):
        self.prev_distances = {i: distances[i] for i in agent_ids}

    def compute(self, agent_id: int, position: np.ndarray, goal: np.ndarray,
                rangefinder_raw: np.ndarray, rangefinder_norm: np.ndarray,
                other_positions: List[np.ndarray], obstacles: List) -> float:
        """
        Eq (4): r = δ1*r_trans + δ2*r_col + δ3*r_sep + δ4*r_free + δ5*r_step
        """
        # Eq (2): Transfer reward
        current_dist = np.linalg.norm(position - goal)
        prev_dist = self.prev_distances.get(agent_id, current_dist)
        r_trans = self.alpha * (prev_dist - current_dist)
        self.prev_distances[agent_id] = current_dist

        # Unified d_all_min — Eq.40: min(obstacle_surface_dist, inter_agent_dist) - Rc
        d_obs_surface = self._min_obstacle_surface_distance(position, obstacles)
        d_agent_center = min(
            (np.linalg.norm(position - other) for other in other_positions),
            default=float('inf')
        )

        # Clamp to [0, ∞): penalty is maximised at contact, not inside
        # PDF §9.4 range analysis assumes d_all_min >= 0
        d_all_min = max(0.0, min(d_obs_surface, d_agent_center) - self.collision_radius)
        r_col = -self.lambda_col * np.exp(-self.sigma_col * d_all_min)       # Eq.41

        # Clamp d_sep similarly — PDF Eq.44
        d_sep = max(0.0, d_agent_center - self.inter_uav_min)
        r_sep = -self.lambda_sep * np.exp(-self.sigma_sep * d_sep)           # Eq.44

        # Free space reward
        # r_free fires when ALL 25 beams clear 95% of max range (≥ 9.5 m)
        # With σ_ℓ=0.02 noise, ~86% of truly-max-range scans satisfy this
        r_free = self.r_free if float(np.min(rangefinder_norm)) >= 0.95 else 0.0

        # Step penalty
        r_step = self.r_step

        return (self.delta[0] * r_trans +
                self.delta[1] * r_col +
                self.delta[2] * r_sep +
                self.delta[3] * r_free +
                self.delta[4] * r_step)

    def _surface_distance(self, position, obs):
        p = position
        c = obs.position

        if obs.type == 'sphere':
            return np.linalg.norm(p - c) - obs.size[0]

        if obs.type == 'cylinder':
            r, h = obs.size
            q = np.array([
                np.linalg.norm(p[:2] - c[:2]) - r,
                abs(p[2] - c[2]) - h / 2.0
            ])
            return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1]), 0.0)

        if obs.type == 'box':
            q = np.abs(p - c) - obs.size
            return np.linalg.norm(np.maximum(q, 0.0)) + min(np.max(q), 0.0)

        return float('inf')

    def _min_obstacle_surface_distance(self, position, obstacles):
        if not obstacles:
            return float('inf')
        return min(self._surface_distance(position, obs) for obs in obstacles)

    def _compute_d_sep(self, position: np.ndarray, other_positions: List[np.ndarray]) -> float:
        min_dist = float('inf')
        for other_pos in other_positions:
            dist = np.linalg.norm(position - other_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist - self.inter_uav_min
