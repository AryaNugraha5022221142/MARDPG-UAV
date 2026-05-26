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
        
        # 5 beams in horizontal: -60°, -30°, 0°, 30°, 60°
        self.h_angles = np.linspace(-h_fov/2, h_fov/2, self.n_h) * np.pi / 180
        # 5 vertical layers: -30°, -15°, 0°, 15°, 30°
        self.v_angles = np.linspace(-v_fov/2, v_fov/2, self.n_v) * np.pi / 180

    def scan(self, position: np.ndarray, theta: float, phi: float,
             obstacles: List[Obstacle]) -> np.ndarray:
        """
        Returns 5x5 distance matrix clipped to [0, range_max].
        """
        distances = np.full((self.n_v, self.n_h), self.range_max, dtype=np.float32)
        
        for vi, va in enumerate(self.v_angles):
            for hi, ha in enumerate(self.h_angles):
                # Compute ray direction in world frame
                # yaw rotation (theta) then pitch rotation (phi + va) then horizontal offset (ha)
                yaw = theta + ha
                pitch = phi + va
                
                dir_vec = np.array([
                    np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch)
                ])
                
                dist = self._ray_cast(position, dir_vec, obstacles)
                distances[vi, hi] = min(dist, self.range_max)
        
        return distances

    def _ray_cast(self, origin: np.ndarray, direction: np.ndarray,
                  obstacles: List[Obstacle]) -> float:
        min_dist = self.range_max
        
        for obs in obstacles:
            if obs.type == 'sphere':
                dist = self._ray_sphere(origin, direction, obs.position, obs.size[0])
            elif obs.type == 'cylinder':
                dist = self._ray_cylinder(origin, direction, obs.position, obs.size[0], obs.size[1])
            elif obs.type == 'box':
                dist = self._ray_box(origin, direction, obs.position, obs.size)
            else:
                continue
            if dist is not None and dist < min_dist:
                min_dist = dist
        
        return min_dist

    def _ray_sphere(self, o, d, c, r):
        oc = o - c
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c_val = np.dot(oc, oc) - r * r
        discriminant = b * b - 4 * a * c_val
        if discriminant < 0:
            return None
        dist = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return dist if dist > 0 else None

    def _ray_cylinder(self, o, d, c, r, h):
        # Infinite cylinder intersection in XY, then check Z height
        oc = o[:2] - c[:2]
        d_xy = d[:2]
        a = np.dot(d_xy, d_xy)
        b = 2.0 * np.dot(oc, d_xy)
        c_val = np.dot(oc, oc) - r * r
        discriminant = b * b - 4 * a * c_val
        if discriminant < 0:
            return None
        dist = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if dist <= 0:
            return None
        z_hit = o[2] + dist * d[2]
        if c[2] - h/2 <= z_hit <= c[2] + h/2:
            return dist
        return None

    def _ray_box(self, o, d, c, s):
        # AABB intersection
        min_t = -np.inf
        max_t = np.inf
        for i in range(3):
            if abs(d[i]) < 1e-6:
                if o[i] < c[i] - s[i] or o[i] > c[i] + s[i]:
                    return None
            else:
                t1 = (c[i] - s[i] - o[i]) / d[i]
                t2 = (c[i] + s[i] - o[i]) / d[i]
                min_t = max(min_t, min(t1, t2))
                max_t = min(max_t, max(t1, t2))
        if max_t < 0 or min_t > max_t:
            return None
        return min_t if min_t > 0 else max_t


class GoalSensor:
    def compute(self, position: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Returns [d_g, pi_h, pi_v] as specified in Section 2.3.
        d_g: Euclidean distance
        pi_h: horizontal angle in x-y plane
        pi_v: vertical angle
        """
        diff = goal - position
        d_g = np.linalg.norm(diff)
        
        # Horizontal angle (yaw relative)
        pi_h = np.arctan2(diff[1], diff[0])
        
        # Vertical angle (pitch relative)
        horizontal_dist = np.linalg.norm(diff[:2])
        pi_v = np.arctan2(diff[2], horizontal_dist)
        
        return np.array([d_g, pi_h, pi_v], dtype=np.float32)
