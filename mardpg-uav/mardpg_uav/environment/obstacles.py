import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Obstacle:
    type: str          # 'box', 'cylinder', 'sphere'
    position: np.ndarray
    size: np.ndarray   # [rx, ry, rz] for box, [r, h] for cylinder, [r] for sphere
    velocity: np.ndarray = None  # [vx, vy, vz] for dynamic obstacles

class SceneGenerator:
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.env_size = np.array([100.0, 100.0, 60.0]) # Will be updated dynamically

    def generate_stage(self, stage_cfg: dict) -> List[Obstacle]:
        """Generates exactly the obstacles defined in the CL stage."""
        obstacles = []
        
        # 1. Generate Static Buildings (Boxes and Cylinders) on a Manhattan Grid
        n_static = stage_cfg.get('static_obs', 0)
        if n_static > 0:
            # INCREASED to 20.0m to prevent geometric choking and ensure solvable paths
            grid_step = 20.0
            margin = 20.0
            xs = np.arange(margin, self.env_size[0] - margin + 1e-3, grid_step)
            ys = np.arange(margin, self.env_size[1] - margin + 1e-3, grid_step)
            grid_points = [(x, y) for x in xs for y in ys]

            n_static = min(n_static, len(grid_points))
            
            indices = self.rng.choice(len(grid_points), size=n_static, replace=False)
            
            for idx in indices:
                gx, gy = grid_points[idx]
                
                # Fetch max_h from stage config to allow curriculum height scaling (Default: 50.0)
                max_h = stage_cfg.get('max_h', 50.0)
                h = self.rng.uniform(10.0, max_h)
                
                # Randomly mix boxes and cylinders to create a diverse urban environment
                obs_type = self.rng.choice(['box', 'cylinder'])
                
                if obs_type == 'cylinder':
                    # REDUCED radius to 2.0m - 4.0m to guarantee wide enough gaps
                    r = self.rng.uniform(2.0, 4.0)
                    pos = np.array([gx, gy, h / 2])
                    obstacles.append(Obstacle('cylinder', pos, np.array([r, h])))
                    
                elif obs_type == 'box':
                    # Boxes are defined by their "half-extents" (half-width, half-depth, half-height)
                    hx = self.rng.uniform(2.0, 4.0)
                    hy = self.rng.uniform(2.0, 4.0)
                    hz = h / 2.0
                    pos = np.array([gx, gy, hz])
                    obstacles.append(Obstacle('box', pos, np.array([hx, hy, hz])))

        # 2. Generate Dynamic Spheres
        n_dynamic = stage_cfg.get('dynamic_obs', 0)
        
        has_dynamic = False
        if isinstance(n_dynamic, tuple):
            has_dynamic = True
            actual_dyn = self.rng.randint(n_dynamic[0], n_dynamic[1] + 1)
        elif n_dynamic > 0:
            has_dynamic = True
            actual_dyn = n_dynamic

        if has_dynamic:
            r = stage_cfg['dynamic_radius']
            v_range = stage_cfg['dynamic_speed']
            
            for _ in range(actual_dyn):
                pos = self.rng.uniform([r, r, r], self.env_size - r)
                
                # Random velocity direction
                phi = self.rng.uniform(0, 2 * np.pi)
                costheta = self.rng.uniform(-1, 1)
                theta = np.arccos(costheta)
                speed = self.rng.uniform(v_range[0], v_range[1])
                
                vx = speed * np.sin(theta) * np.cos(phi)
                vy = speed * np.sin(theta) * np.sin(phi)
                vz = speed * np.cos(theta)
                
                velocity = np.array([vx, vy, vz], dtype=np.float32)
                obstacles.append(Obstacle('sphere', pos, np.array([r]), velocity=velocity))

        return obstacles