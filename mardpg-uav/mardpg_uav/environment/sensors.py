"""
25-beam rangefinder (5 horizontal planes x 5 beams) and goal sensing.
Reference: Section 2.3 and 3.1 of blueprint.
"""
import numpy as np
from typing import List, Tuple
from .obstacles import Obstacle

class Rangefinder:
    def __init__(self, range_max: float = 10.0,
                 h_fov: float = 120.0,  # degrees
                 v_fov: float = 60.0):  # degrees
        self.range_max = range_max
        self.n_h = 5
        self.n_v = 5
        
        self.h_angles = np.linspace(-h_fov/2, h_fov/2, self.n_h) * np.pi / 180
        self.v_angles = np.linspace(-v_fov/2, v_fov/2, self.n_v) * np.pi / 180

    def scan(self, position: np.ndarray, theta: float, phi: float,
             obstacles: List[Obstacle], sigma_l: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        yaw = theta + self.h_angles
        pitch = phi + self.v_angles[:, None]
        
        yaw = np.broadcast_to(yaw, (self.n_v, self.n_h)).flatten()
        pitch = np.broadcast_to(pitch, (self.n_v, self.n_h)).flatten()

        dir_vecs = np.stack([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ], axis=-1)

        min_dists = np.full(dir_vecs.shape[0], self.range_max, dtype=np.float32)

        for obs in obstacles:
            if obs.type == 'sphere':
                dist = self._ray_sphere_vec(position, dir_vecs, obs.position, obs.size[0])
            elif obs.type == 'cylinder':
                dist = self._ray_cylinder_vec(position, dir_vecs, obs.position, obs.size[0], obs.size[1])
            elif obs.type == 'box':
                dist = self._ray_box_vec(position, dir_vecs, obs.position, obs.size)
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
    def __init__(self, arena_diag=89.4, sigma_psi=0.0349):
        self.arena_diag = arena_diag
        self.sigma_psi = sigma_psi

    def compute(self, position, heading_yaw, heading_pitch, goal):
        """Returns [d_norm, Δθ_norm, Δφ_norm] ∈ (-1,1]³  — Eqs.24-27."""
        diff = goal - position
        d_norm = np.linalg.norm(diff) / self.arena_diag              # Eq.25

        noisy_yaw = heading_yaw + np.random.normal(0, self.sigma_psi)
        raw_yaw = np.arctan2(diff[1], diff[0])
        delta_theta = self._wrap(raw_yaw - noisy_yaw) / np.pi       # Eq.26

        d_xy = np.linalg.norm(diff[:2])
        raw_pitch = np.arctan2(diff[2], d_xy)
        delta_phi = self._wrap(raw_pitch - heading_pitch) / np.pi   # Eq.27

        return np.array([d_norm, delta_theta, delta_phi], dtype=np.float32)

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
