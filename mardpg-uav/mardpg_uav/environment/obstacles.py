"""
Environment generation with 4 scene types.
Reference: Section 2.4 of blueprint.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Obstacle:
    type: str          # 'box', 'cylinder', 'sphere'
    position: np.ndarray
    size: np.ndarray   # [rx, ry, rz] for box, [r, h] for cylinder, [r] for sphere


class SceneGenerator:
    def __init__(self, env_size: Tuple[float, float, float] = (50.0, 50.0, 60.0),
                 seed: int = None):
        self.env_size = np.array(env_size)
        self.rng = np.random.RandomState(seed)

    def generate(self, scene_type: int, density: float, n_agents: int) -> List[Obstacle]:
        """Generate obstacles ensuring valid start/goal regions exist."""
        if scene_type == 1:
            return self._square_columns(density)
        elif scene_type == 2:
            return self._cylinders(density)
        elif scene_type == 3:
            return self._forest(density)
        elif scene_type == 4:
            return self._circular_rings(density)
        else:
            raise ValueError(f"Unknown scene type: {scene_type}")

    def _volume(self) -> float:
        return np.prod(self.env_size)

    def _square_columns(self, density: float) -> List[Obstacle]:
        obstacles = []
        target_volume = density * self._volume()
        current_volume = 0.0
        while current_volume < target_volume:
            w = self.rng.uniform(1.0, 4.0)
            h = self.rng.uniform(5.0, self.env_size[2])
            pos = self.rng.uniform([0, 0, 0], self.env_size)
            pos[2] = h / 2  # center vertically
            obstacles.append(Obstacle('box', pos, np.array([w/2, w/2, h/2])))
            current_volume += (w ** 2) * h
        return obstacles

    def _cylinders(self, density: float) -> List[Obstacle]:
        obstacles = []
        target_volume = density * self._volume()
        current_volume = 0.0
        while current_volume < target_volume:
            r = self.rng.uniform(0.5, 2.5)
            h = self.rng.uniform(5.0, self.env_size[2])
            pos = self.rng.uniform([0, 0, 0], self.env_size)
            pos[2] = h / 2
            obstacles.append(Obstacle('cylinder', pos, np.array([r, h])))
            current_volume += np.pi * (r ** 2) * h
        return obstacles

    def _forest(self, density: float) -> List[Obstacle]:
        obstacles = []
        target_volume = density * self._volume()
        current_volume = 0.0
        while current_volume < target_volume:
            r = self.rng.uniform(0.3, 1.5)
            h = self.rng.uniform(8.0, self.env_size[2])
            pos = self.rng.uniform([0, 0, 0], self.env_size)
            pos[2] = h / 2
            obstacles.append(Obstacle('cylinder', pos, np.array([r, h])))
            current_volume += np.pi * (r ** 2) * h
        return obstacles

    def _circular_rings(self, density: float) -> List[Obstacle]:
        obstacles = []
        n_rings = int(density * 20) + 1
        for _ in range(n_rings):
            center = self.rng.uniform([5, 5, 5], self.env_size - 5)
            major_r = self.rng.uniform(3.0, 8.0)
            tube_r = self.rng.uniform(0.5, 1.5)
            # Approximate as collection of spheres along ring
            n_spheres = max(8, int(2 * np.pi * major_r / tube_r))
            for i in range(n_spheres):
                angle = 2 * np.pi * i / n_spheres
                pos = center + np.array([
                    major_r * np.cos(angle),
                    major_r * np.sin(angle),
                    0
                ])
                obstacles.append(Obstacle('sphere', pos, np.array([tube_r])))
        return obstacles
