import numpy as np
from typing import List, Tuple


class RewardFunction:
    def __init__(self, alpha: float = 3.0, lambda_col: float = 5.0,
                 sigma_col: float = 15.0,
                 r_free: float = 0.1, r_step: float = -0.6,
                 delta: Tuple[float, float, float, float] = (0.45, 0.30, 0.15, 0.10),
                 collision_radius: float = 0.5,
                 inter_uav_min: float = 1.0,
                 range_max: float = 10.0):
        self.alpha = alpha
        self.lambda_col = lambda_col
        self.sigma_col = sigma_col
        self.r_free = r_free
        self.r_step = r_step
        self.delta = delta
        self.collision_radius = collision_radius
        self.inter_uav_min = inter_uav_min
        self.range_max = range_max
        self.prev_distances = {}

    def reset(self, agent_ids: List[int], distances: dict):
        self.prev_distances = {i: distances[i] for i in agent_ids}

    def compute(self, agent_id: int, position: np.ndarray, goal: np.ndarray,
                rangefinder_raw: np.ndarray, rangefinder_norm: np.ndarray,
                other_positions: List[np.ndarray], obstacles: List,
                obs_centers: np.ndarray = None, obs_max_sizes: np.ndarray = None) -> float:
        """
        Eq. (4): r = delta1*r_trans + delta2*r_col + delta3*r_free + delta4*r_step
        """
        # --- Eq. (2): Transfer reward ---
        current_dist = np.linalg.norm(position - goal)
        prev_dist    = self.prev_distances.get(agent_id, current_dist)
        r_trans      = self.alpha * (prev_dist - current_dist)
        self.prev_distances[agent_id] = current_dist

        # --- Eq. (3): Collision penalty ---
        # Smooth, monotonic penalty that ramps up inside `safe_dist` and saturates at a
        # crash. Unlike the forward-cone sensor reading, d_min here is the TRUE shortest
        # clearance to ANY hazard in ANY direction, and it includes other UAVs. The reward
        # is computed in the environment, so using ground-truth geometry is legitimate
        # (the policy still only sees noisy forward-cone sensors). This restores the
        # cooperative inter-UAV avoidance signal central to the paper.
        sigma = self.lambda_col          # penalty magnitude (paper lambda)
        safe_dist = 2.0                  # start penalizing within 2 m of a surface
        crash_dist = 0.2                 # treat as a physical crash at/under 0.2 m
        k = 3.0                          # steepness of the exponential ramp

        # Surface clearance to nearest static/dynamic obstacle (in metres).
        d_obs = self._min_obstacle_surface_distance(position, obstacles,
                                                    obs_centers, obs_max_sizes)
        # Clearance to nearest other UAV beyond its 2R safety bubble (inter_uav_min = 2R).
        d_uav = min((np.linalg.norm(position - p) for p in other_positions),
                    default=float('inf'))
        d_min = min(d_obs, d_uav - self.inter_uav_min)
        d_min = max(0.0, d_min)

        r_col = 0.0
        if d_min < safe_dist:
            if d_min <= crash_dist:
                r_col = -sigma  # max penalty for an actual crash
            else:
                r_col = -sigma * np.exp(-k * (d_min - crash_dist))
                # Offset so the penalty is continuous and reaches 0 exactly at safe_dist.
                boundary_offset = np.exp(-k * (safe_dist - crash_dist))
                r_col = r_col - (-sigma * boundary_offset)

        # --- Free space reward ---
        # The rangefinder is a 5x5 grid. Indices:
        # 00 01 02 03 04
        # 05 06 07 08 09
        # 10 11 12 13 14  <-- Center row
        # 15 16 17 18 19
        # 20 21 22 23 24
        
        r_free = self.r_free if rangefinder_raw.flatten()[12] >= self.range_max else 0.0

        # --- Step penalty ---
        r_step = self.r_step

        return (self.delta[0] * r_trans +
                self.delta[1] * r_col   +
                self.delta[2] * r_free  +
                self.delta[3] * r_step)

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

    def _min_obstacle_surface_distance(self, position, obstacles, obs_centers=None, obs_max_sizes=None):
        if not obstacles:
            return float('inf')
        
        if obs_centers is not None and obs_max_sizes is not None:
            dists = np.linalg.norm(obs_centers - position, axis=1)
            # Safe culling: only calculate exact surface distance for obstacles that could be within 12m
            mask = dists < (12.0 + obs_max_sizes)
            close_obstacles = [obstacles[i] for i, m in enumerate(mask) if m]
        else:
            close_obstacles = obstacles
            
        if not close_obstacles:
            return float('inf')
        return min(self._surface_distance(position, obs) for obs in close_obstacles)
