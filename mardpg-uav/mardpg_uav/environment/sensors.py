"""
25-beam rangefinder (5 horizontal planes x 5 beams) and goal sensing.
Reference: Section 2.3 and 3.1 of blueprint.
"""
import numpy as np
from typing import List, Tuple
from .obstacles import Obstacle

class Rangefinder:
    def __init__(self, range_max: float = 10.0, seed: int = None):
        self.range_max = range_max
        self.n_h = 5
        self.n_v = 5
        # [N3-5] Private stream: lidar noise no longer perturbs the global
        # np.random stream that exploration/scene generation depend on.
        self.rng = np.random.RandomState(seed)

        yaw_angles = np.linspace(-np.pi/3, np.pi/3, self.n_h)
        # Pitch: -30 degrees to +30 degrees (-pi/6 to pi/6)
        pitch_angles = np.linspace(-np.pi/6, np.pi/6, self.n_v)

        yaw_grid = np.broadcast_to(yaw_angles, (self.n_v, self.n_h)).flatten()
        pitch_grid = np.broadcast_to(
            pitch_angles[:, None], (self.n_v, self.n_h)).flatten()

        # Pre-compute and cache unit direction vectors (25, 3) — constant
        self._dir_vecs = np.stack([
            np.cos(pitch_grid) * np.cos(yaw_grid),
            np.cos(pitch_grid) * np.sin(yaw_grid),
            np.sin(pitch_grid)
        ], axis=-1).astype(np.float32)   # shape (25, 3)

    def scan(self, position: np.ndarray, theta: float, phi: float, obstacles: List[Obstacle], sigma_l: float = 0.02,
             obs_centers: np.ndarray = None, obs_max_sizes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate the fixed beam directions by the UAV's current heading (theta, phi).
        theta: yaw in world frame
        phi:   pitch in world frame
        """
        # Rotation matrix: yaw then pitch
        cy, sy = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi),   np.sin(phi)
        
        # Inverted pitch matrix
        R = np.array([
            [ cy*cp, -sy, -cy*sp],
            [ sy*cp,  cy, -sy*sp],
            [ sp,     0.0, cp   ],
        ], dtype=np.float32)   # (3,3)

        rotated_dirs = (R @ self._dir_vecs.T).T   # (25, 3)

        min_dists = np.full(rotated_dirs.shape[0], self.range_max, dtype=np.float32)

        if obs_centers is not None and obs_max_sizes is not None:
            dists_to_centers = np.linalg.norm(obs_centers - position, axis=1)
            # Only consider obstacles whose bounding sphere intersects the sensor range
            mask = dists_to_centers < (self.range_max + obs_max_sizes)
            close_obstacles = [obstacles[i] for i, m in enumerate(mask) if m]
        else:
            close_obstacles = obstacles

        # BATCHED BOX INTERSECTION (Dramatically accelerates wall boundary checks)
        boxes = [obs for obs in close_obstacles if obs.type == 'box']
        if boxes:
            c = np.array([b.position for b in boxes], dtype=np.float32)[:, None, :]  # (M, 1, 3)
            s = np.array([b.size for b in boxes], dtype=np.float32)[:, None, :]      # (M, 1, 3)
            d = rotated_dirs[None, :, :]                                           # (1, 25, 3)
            o = position[None, None, :]                                              # (1, 1, 3)

            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = (c - s - o) / d
                t2 = (c + s - o) / d

            t1 = np.where(np.isnan(t1), -np.inf, t1)
            t2 = np.where(np.isnan(t2), np.inf, t2)
            t_min = np.minimum(t1, t2)
            t_max = np.maximum(t1, t2)
            t_enter = np.max(t_min, axis=-1)  # (M, 25)
            t_exit = np.min(t_max, axis=-1)   # (M, 25)

            valid_mask = (t_exit >= t_enter) & (t_enter > 0)
            t_enter_masked = np.where(valid_mask, t_enter, np.inf)
            min_dists = np.minimum(min_dists, np.min(t_enter_masked, axis=0))

            mask_in = (t_exit >= t_enter) & (t_enter <= 0) & (t_exit > 0)
            t_exit_masked = np.where(mask_in, t_exit, np.inf)
            min_dists = np.minimum(min_dists, np.min(t_exit_masked, axis=0))

        # Fallback to loop for complex geometries (cylinders, spheres)
        other_obs = [obs for obs in close_obstacles if obs.type != 'box']
        for obs in other_obs:
            if obs.type == 'sphere':
                dist = self._ray_sphere_vec(position, rotated_dirs, obs.position, obs.size[0])
            elif obs.type == 'cylinder':
                dist = self._ray_cylinder_vec(position, rotated_dirs, obs.position, obs.size[0], obs.size[1])
            else:
                continue
            
            mask = dist < min_dists
            min_dists[mask] = dist[mask]

        distances = min_dists.reshape(self.n_v, self.n_h)
        noisy_norm = np.clip(distances / self.range_max
                             + self.rng.normal(0.0, sigma_l, distances.shape).astype(np.float32),
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
