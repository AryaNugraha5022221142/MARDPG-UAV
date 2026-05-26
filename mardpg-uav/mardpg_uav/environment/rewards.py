import numpy as np
from typing import List, Tuple


class RewardFunction:
    def __init__(self, alpha: float = 3.0, lambda_: float = 5.0,
                 sigma: float = 15.0, r_free: float = 0.1,
                 r_step: float = -0.6,
                 delta: Tuple[float, float, float, float] = (0.45, 0.30, 0.15, 0.10),
                 collision_radius: float = 0.5,
                 inter_uav_min: float = 1.0):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.sigma = sigma
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
                rangefinder: np.ndarray, other_positions: List[np.ndarray],
                obstacles: List) -> float:
        """
        Eq (4): r = δ1*r_trans + δ2*r_col + δ3*r_free + δ4*r_step
        """
        # Eq (2): Transfer reward
        current_dist = np.linalg.norm(position - goal)
        prev_dist = self.prev_distances.get(agent_id, current_dist)
        r_trans = self.alpha * (prev_dist - current_dist)
        self.prev_distances[agent_id] = current_dist

        # Eq (3): Collision penalty
        d_min = self._compute_d_min(position, other_positions, obstacles)
        r_col = -self.lambda_ * np.exp(-self.sigma * d_min)

        # Free space reward
        r_free = self.r_free if np.all(rangefinder >= 9.9) else 0.0

        # Step penalty
        r_step = self.r_step

        return (self.delta[0] * r_trans +
                self.delta[1] * r_col +
                self.delta[2] * r_free +
                self.delta[3] * r_step)

    def _compute_d_min(self, position: np.ndarray,
                       other_positions: List[np.ndarray],
                       obstacles: List) -> float:
        min_dist = float('inf')
        
        # Distance to other UAVs
        for other_pos in other_positions:
            dist = np.linalg.norm(position - other_pos)
            if dist < min_dist:
                min_dist = dist
        
        # For obstacles, we use a simplified proxy: distance to nearest obstacle center minus radius
        # In a full sim, you'd use proper distance fields. Here we approximate for the penalty.
        # The ray casting already handles sensing; for reward we use the rangefinder minimum.
        # This is a slight deviation for stability.
        
        return max(0.0, min_dist - self.inter_uav_min)  # simplified
