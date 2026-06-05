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
        self.env_size = np.array([50.0, 50.0, 60.0]) # Will be updated dynamically

    def generate_stage(self, stage_cfg: dict) -> List[Obstacle]:
        """Generates exactly the obstacles defined in the CL stage."""
        obstacles = []
        
        # 1. Generate Static Buildings (Boxes)
        n_static = stage_cfg['static_obs']
        if n_static > 0:
            foot_w = stage_cfg['footprint'][0] # (min, max) width
            foot_l = stage_cfg['footprint'][1] # (min, max) length
            h_dist = stage_cfg['height_dist']  # (mean, sigma, clip_min, clip_max)
            
            for _ in range(n_static):
                w = self.rng.uniform(foot_w[0], foot_w[1])
                l = self.rng.uniform(foot_l[0], foot_l[1])
                
                # LogNormal Height
                h = self.rng.lognormal(mean=h_dist[0], sigma=h_dist[1])
                h = np.clip(h, h_dist[2], h_dist[3])
                
                pos = self.rng.uniform([0, 0, 0], self.env_size)
                pos[2] = h / 2  # Ground it
                
                obstacles.append(Obstacle('box', pos, np.array([w/2, l/2, h/2])))

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