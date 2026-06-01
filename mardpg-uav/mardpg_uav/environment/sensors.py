"""
25-beam rangefinder (5 horizontal planes x 5 beams) and goal sensing.
Reference: Section 2.3 and 3.1 of blueprint.
"""
import numpy as np
from typing import List, Tuple
from .obstacles import Obstacle

class Rangefinder:
    def __init__(self, range_max: float = 10.0):
        self.range_max = range_max
        self.n_h = 5
        self.n_v = 5

        # Fixed world-frame azimuths (uniform 360° coverage)
        h_angles = np.linspace(0, 2 * np.pi, self.n_h, endpoint=False)
        # Elevation angles: ±30° symmetric
        v_angles = np.linspace(-np.pi / 6, np.pi / 6, self.n_v)

        yaw_grid = np.broadcast_to(h_angles, (self.n_v, self.n_h)).flatten()
        pitch_grid = np.broadcast_to(
            v_angles[:, None], (self.n_v, self.n_h)).flatten()

        # Pre-compute and cache unit direction vectors (25, 3) — constant
        self._dir_vecs = np.stack([
            np.cos(pitch_grid) * np.cos(yaw_grid),
            np.cos(pitch_grid) * np.sin(yaw_grid),
            np.sin(pitch_grid)
        ], axis=-1).astype(np.float32)   # shape (25, 3)

    def scan(self, position: np.ndarray, obstacles: List[Obstacle], sigma_l: float = 0.02,
             obs_centers: np.ndarray = None, obs_max_sizes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """No heading parameters — world-frame fixed beams."""
        min_dists = np.full(self._dir_vecs.shape[0], self.range_max, dtype=np.float32)

        if obs_centers is not None and obs_max_sizes is not None:
            dists_to_centers = np.linalg.norm(obs_centers - position, axis=1)
            # Only consider obstacles whose bounding sphere intersects the sensor range
            mask = dists_to_centers < (self.range_max + obs_max_sizes)
            close_obstacles = [obstacles[i] for i, m in enumerate(mask) if m]
        else:
            close_obstacles = obstacles

        for obs in close_obstacles:
            if obs.type == 'sphere':
                dist = self._ray_sphere_vec(position, self._dir_vecs, obs.position, obs.size[0])
            elif obs.type == 'cylinder':
                dist = self._ray_cylinder_vec(position, self._dir_vecs, obs.position, obs.size[0], obs.size[1])
            elif obs.type == 'box':
                dist = self._ray_box_vec(position, self._dir_vecs, obs.position, obs.size)
            else:
                continue
            
            mask = dist < min_dists
            min_dists[mask] = dist[mask]

        distances = min_dists.reshape(self.n_v, self.n_h)
        noisy_norm = np.clip(distances / self.range_max
                             + np.random.normal(0.0, sigma_l, distances.shape).astype(np.float32),
                             0.0, 1.0)
        return distances, noisy_norm

    def _ray_sphere_vec(self, o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
        oc = o - c
        a = 1.0  # d is normalized
        b = 2.0 * np.dot(d, oc)
        c_val = np.dot(oc, oc) - r * r
        disc = b * b - 4 * c_val
        
        dist = np.full(d.shape[0], np.inf, dtype=np.float32)
        mask = disc >= 0
        if np.any(mask):
            d_mask = (-b[mask] - np.sqrt(disc[mask])) / 2.0
            p_mask = d_mask > 0
            # assign to dist where mask is True and d_mask > 0
            final_mask = mask.copy()
            final_mask[mask] = p_mask
            dist[final_mask] = d_mask[p_mask]
        return dist

    def _ray_cylinder_vec(self, o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float, h: float) -> np.ndarray:
        oc = o[:2] - c[:2]
        d_xy = d[:, :2]
        a = np.sum(d_xy * d_xy, axis=-1)
        b = 2.0 * np.dot(d_xy, oc)
        c_val = np.dot(oc, oc) - r * r
        disc = b * b - 4 * a * c_val
        
        dist = np.full(d.shape[0], np.inf, dtype=np.float32)
        mask = (disc >= 0) & (a > 1e-6)
        if np.any(mask):
            d_mask = (-b[mask] - np.sqrt(disc[mask])) / (2.0 * a[mask])
            z_hit = o[2] + d_mask * d[mask, 2]
            
            valid = (d_mask > 0) & (z_hit >= c[2] - h/2) & (z_hit <= c[2] + h/2)
            
            final_mask = mask.copy()
            final_mask[mask] = valid
            dist[final_mask] = d_mask[valid]
        return dist

    def _ray_box_vec(self, o: np.ndarray, d: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (c - s - o) / d
            t2 = (c + s - o) / d
            
        # Replace NaN with +/- inf before slab test
        t1 = np.where(np.isnan(t1), -np.inf, t1)
        t2 = np.where(np.isnan(t2), np.inf, t2)
            
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)
        
        t_enter = np.max(t_min, axis=-1)
        t_exit = np.min(t_max, axis=-1)
        
        dist = np.full(d.shape[0], np.inf, dtype=np.float32)
        mask = (t_exit >= t_enter) & (t_enter > 0)
        dist[mask] = t_enter[mask]
        
        # inside case
        mask_in = (t_exit >= t_enter) & (t_enter <= 0) & (t_exit > 0)
        dist[mask_in] = t_exit[mask_in]
        
        return dist


class GoalSensor:
    def __init__(self, arena_diag=89.4):
        self.arena_diag = arena_diag
        # No heading noise — no heading in model

    def compute(self, position: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Returns normalised world-frame displacement [dx/D, dy/D, dz/D]
        Each component ∈ (-1, 1); zero vector when at goal.
        """
        diff = goal - position
        return (diff / self.arena_diag).astype(np.float32)  # shape (3,)

