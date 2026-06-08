"""
Main Gym-style environment for Multi-UAV Navigation.
Implements Dec-POMDP from Section 1.2.
"""
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List, Optional
from .dynamics import QuadcopterDynamics
from .obstacles import SceneGenerator, Obstacle
from .sensors import Rangefinder, GoalSensor
from .rewards import RewardFunction


class MultiUAVEnv(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.n_agents = config['n_agents']
        
        self.obs_centers = np.zeros((0, 3), dtype=np.float32)
        self.obs_max_sizes = np.zeros(0, dtype=np.float32)
        
        # Subsystems
        self.dynamics = QuadcopterDynamics(
            v=config.get('v', 1.0),
            dt=config['dt'],
            env_size=config['env_size'],
            max_altitude=config['max_altitude'],
            min_altitude=config.get('min_altitude', 0.0),
            max_delta_angle=config.get('max_delta_angle', np.pi / 6)
        )
        self.scene_gen = SceneGenerator(
            seed=config.get('seed', None)
        )
        self.rangefinder = Rangefinder(range_max=config['sensor_range'])
        env_diag = float(np.sqrt(sum(s**2 for s in config['env_size'])))
        self.goal_sensor = GoalSensor(arena_diag=env_diag)
        self.reward_fn = RewardFunction(
            alpha=config['reward']['alpha'],
            lambda_col=config['reward']['lambda_col'],
            sigma_col=config['reward']['sigma_col'],
            r_free=config['reward']['r_free'],
            r_step=config['reward']['r_step'],
            delta=tuple(config['reward']['delta']),
            collision_radius=config['collision_radius'],
            inter_uav_min=config['inter_uav_min_dist'],
            range_max=config['sensor_range']
        )
        
        # Spaces
        self.obs_dim = 30  # [theta(1), phi(1), lidar(25), d5(1), varpi(1), varpi_z(1)]
        self.action_dim = 2  # [rho, tau]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents, self.obs_dim), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self.cfg.get('max_delta_angle', np.pi/6), 
            high=self.cfg.get('max_delta_angle', np.pi/6),
            shape=(self.n_agents, self.action_dim), dtype=np.float32
        )
        
        # State
        self.agents_state = None  # [n_agents, 5] (x, y, z, theta, phi)
        self.goals = None       # [n_agents, 3]
        self.obstacles: List[Obstacle] = []
        self.steps = 0
        
    def reset(self, stage_cfg: dict = None) -> np.ndarray:
        if stage_cfg is None:
            stage_cfg = {'env_size': [50.0, 50.0, 60.0], 'static_obs': 0, 'min_sep': 20.0, 'max_steps': 1500}
            
        self.current_stage_cfg = stage_cfg
        new_env_size = np.array(stage_cfg['env_size'], dtype=np.float32)
        
        # Update Environment Dimensions
        self.cfg['env_size'] = new_env_size
        self.cfg['max_steps_per_episode'] = stage_cfg.get('max_steps', 1500)
        self.dynamics.env_size = new_env_size
        self.dynamics.max_altitude = new_env_size[2]
        self.scene_gen.env_size = new_env_size
        self.goal_sensor.arena_diag = float(np.sqrt(sum(s**2 for s in new_env_size)))
        
        # Generate Obstacles
        self.obstacles = self.scene_gen.generate_stage(stage_cfg)
        self.dynamic_obstacles = [obs for obs in self.obstacles if getattr(obs, 'velocity', None) is not None]
        
        # Inject Wall Boundaries
        t = 5.0
        ex, ey, ez = new_env_size
        min_z = self.cfg.get('min_altitude', 0.0)
        walls = [
            Obstacle('box', np.array([-t, ey/2, ez/2]), np.array([t, ey/2, ez/2])),
            Obstacle('box', np.array([ex+t, ey/2, ez/2]), np.array([t, ey/2, ez/2])),
            Obstacle('box', np.array([ex/2, -t, ez/2]), np.array([ex/2, t, ez/2])),
            Obstacle('box', np.array([ex/2, ey+t, ez/2]), np.array([ex/2, t, ez/2])),
            Obstacle('box', np.array([ex/2, ey/2, min_z-t]), np.array([ex/2, ey/2, t])),
            Obstacle('box', np.array([ex/2, ey/2, ez+t]), np.array([ex/2, ey/2, t]))
        ]
        self.obstacles.extend(walls)
        self._update_obstacle_caches()

        # Sample start/goal positions with min_point_separation
        min_sep = stage_cfg.get('min_sep', 20.0)
        self.agents_state = np.zeros((self.n_agents, 5), dtype=np.float32)
        self.goals = np.zeros((self.n_agents, 3), dtype=np.float32)
        
        positions = []
        for i in range(self.n_agents):
            pos = self._sample_free_position()
            positions.append(pos)
            self.agents_state[i, :3] = pos
            # Random initial heading
            self.agents_state[i, 3] = self.scene_gen.rng.uniform(-np.pi, np.pi)  # theta
            self.agents_state[i, 4] = self.scene_gen.rng.uniform(-np.pi/6, np.pi/6)  # phi
            
            for _ in range(100):
                goal = self._sample_free_position()
                # Check min point separation and goal interference
                if np.linalg.norm(goal - pos) >= min_sep and \
                   all(np.linalg.norm(goal - g) >= self.cfg['inter_uav_min_dist'] for g in self.goals[:i]):
                    break
            self.goals[i] = goal

        self.prev_applied_actions = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.agent_done = np.zeros(self.n_agents, dtype=bool)
        self.agent_reached = np.zeros(self.n_agents, dtype=bool)
        self.agent_collided = np.zeros(self.n_agents, dtype=bool)
        self.agent_dyn_collided = np.zeros(self.n_agents, dtype=bool)
        self._cached_obs = [None] * self.n_agents
        
        dists = {i: np.linalg.norm(self.agents_state[i, :3] - self.goals[i]) for i in range(self.n_agents)}
        self.reward_fn.reset(list(range(self.n_agents)), dists)
        self.steps = 0
        self.safe_inter_uav_steps = 0  # Track safe steps for Stage 2 metric
        
        return self._get_observations()

    def _update_obstacle_caches(self):
        """Helper to refresh bounding boxes for lidar when objects move."""
        self.obs_centers = np.array([obs.position for obs in self.obstacles])
        sizes = []
        for obs in self.obstacles:
            if obs.type == 'box': sizes.append(np.linalg.norm(obs.size))
            elif obs.type == 'cylinder': sizes.append(np.sqrt(obs.size[0]**2 + (obs.size[1]/2)**2))
            else: sizes.append(obs.size[0])
        self.obs_max_sizes = np.array(sizes)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Args:
            actions: [n_agents, 3] in [-v_max, v_max]
        Returns:
            observations: [n_agents, 34]
            rewards: [n_agents]
            done: bool (episode terminates if any agent collides or all reach goals)
            info: dict
        """
        actions = np.clip(np.array(actions, dtype=np.float32),
                          -self.cfg.get('max_delta_angle', np.pi/6),
                           self.cfg.get('max_delta_angle', np.pi/6))
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        collisions = np.zeros(self.n_agents, dtype=bool)
        reached = np.zeros(self.n_agents, dtype=bool)
        
        # Update dynamics
        for i in range(self.n_agents):
            if self.agent_done[i]:
                continue
                
            next_state = self.dynamics.step(self.agents_state[i], actions[i])
            self.agents_state[i] = next_state
            
        # --- NEW: UPDATE DYNAMIC OBSTACLES ---
        if hasattr(self, 'dynamic_obstacles') and len(self.dynamic_obstacles) > 0:
            env_s = self.cfg['env_size']
            for obs in self.dynamic_obstacles:
                obs.position = obs.position + obs.velocity * self.cfg['dt']
                # Bounce logic
                for axis in range(3):
                    if obs.position[axis] - obs.size[0] < 0:
                        obs.position[axis] = obs.size[0]
                        obs.velocity[axis] *= -1
                    elif obs.position[axis] + obs.size[0] > env_s[axis]:
                        obs.position[axis] = env_s[axis] - obs.size[0]
                        obs.velocity[axis] *= -1
            self._update_obstacle_caches()
        
        # Compute observations and rewards
        obs_list = []
        positions = [self.agents_state[i, :3] for i in range(self.n_agents)]
        
        # 1. Pre-calculate all collisions simultaneously
        # Vectorised inter-agent collision
        positions_arr = self.agents_state[:, :3]  # (N, 3)
        diffs = positions_arr[:, None, :] - positions_arr[None, :, :]  # (N, N, 3)
        dist_matrix = np.linalg.norm(diffs, axis=-1)  # (N, N)
        np.fill_diagonal(dist_matrix, np.inf)
        inter_collisions = np.any(dist_matrix < self.cfg['inter_uav_min_dist'], axis=1)  # (N,)

        for i in range(self.n_agents):
            if self.agent_done[i]: continue
            
            if inter_collisions[i]:
                collisions[i] = True
            
            pos = self.agents_state[i, :3]
            
            # Check bounds and obstacles
            if self._out_of_bounds(pos):
                collisions[i] = True
            else:
                # Fast culling & Obstacles
                dists = np.linalg.norm(self.obs_centers - pos, axis=1) if len(self.obstacles) > 0 else []
                mask = dists < (self.cfg['collision_radius'] + self.obs_max_sizes + 0.1) if len(self.obstacles) > 0 else []
                close_obstacles = [self.obstacles[idx] for idx, m in enumerate(mask) if m]
                for obs in close_obstacles:
                    if self.reward_fn._surface_distance(pos, obs) < self.cfg['collision_radius']:
                        collisions[i] = True
                        break

        # Check dynamic collisions
        if hasattr(self, 'dynamic_obstacles'):
            for i in range(self.n_agents):
                if not self.agent_done[i] and len(self.dynamic_obstacles) > 0:
                    pos = self.agents_state[i, :3]
                    for dyn_obs in self.dynamic_obstacles:
                        if self.reward_fn._surface_distance(pos, dyn_obs) < self.cfg['collision_radius']:
                            self.agent_dyn_collided[i] = True
                            collisions[i] = True
                            break

        # Check inter-uav safety for metrics
        if np.all(dist_matrix >= 1.0):
            self.safe_inter_uav_steps += 1

        for i in range(self.n_agents):
            if self.agent_done[i]:
                obs_list.append(self._cached_obs[i])
                
                # FIX: Prevent suicide exploit by punishing collided agents until episode ends
                if self.agent_collided[i]:
                    rewards[i] = self.reward_fn.delta[3] * self.cfg['reward']['r_step']
                else:
                    rewards[i] = 0.0
                continue
                
            pos = self.agents_state[i, :3]
            
            theta = self.agents_state[i, 3]
            phi   = self.agents_state[i, 4]
            # Sensing
            rangefinder_raw, rangefinder_norm = self.rangefinder.scan(
                pos, theta, phi, self.obstacles,
                obs_centers=self.obs_centers, obs_max_sizes=self.obs_max_sizes
            )
            
            # Observe we don't save goal_disp here anymore since observation logic moved to _get_single_observation.
            
            obs = self._get_single_observation(i)
            self._cached_obs[i] = obs
            obs_list.append(obs)
            
            # Reward
            other_pos = [positions[j] for j in range(self.n_agents) if j != i]
            rewards[i] = self.reward_fn.compute(
                i, pos, self.goals[i], rangefinder_raw, rangefinder_norm, other_pos, self.obstacles,
                obs_centers=self.obs_centers, obs_max_sizes=self.obs_max_sizes
            )
            
            # Apply pre-calculated penalties
            if np.linalg.norm(pos - self.goals[i]) < self.cfg['goal_threshold']:
                reached[i] = True
        
        self.steps += 1
        
        self.agent_reached |= reached
        self.agent_collided |= collisions
        
        # Episode termination
        for i in range(self.n_agents):
            self.agent_done[i] = self.agent_done[i] or self.agent_collided[i] or self.agent_reached[i]
            
        timeout = self.steps >= self.cfg['max_steps_per_episode']
        episode_done = timeout or bool(np.all(self.agent_done))
        
        applied_actions = self.prev_applied_actions.copy()
        
        dyn_collisions_copy = self.agent_dyn_collided.copy() if hasattr(self, 'agent_dyn_collided') else np.zeros(self.n_agents, dtype=bool)

        info = {
            'collisions': self.agent_collided.copy(),
            'dyn_collisions': dyn_collisions_copy,
            'reached': self.agent_reached.copy(),
            'safe_inter_uav_ratio': self.safe_inter_uav_steps / max(1, self.steps),
            'step_collisions': collisions.copy(),
            'step_reached': reached.copy(),
            'timeout': timeout,
            'steps': self.steps,
            'agent_done': self.agent_done.copy(),
            'applied_actions': applied_actions,
        }
        
        return np.array(obs_list), rewards, episode_done, info

    def _get_single_observation(self, i: int) -> np.ndarray:
        pos   = self.agents_state[i, :3]
        theta = self.agents_state[i, 3]
        phi   = self.agents_state[i, 4]

        _, lidar_norm = self.rangefinder.scan(
            pos, theta, phi, self.obstacles,
            obs_centers=self.obs_centers,
            obs_max_sizes=self.obs_max_sizes,
        )

        goal_vec = self.goals[i] - pos
        d5       = np.linalg.norm(goal_vec)
        if d5 > 1e-6:
            # 1. Calculate absolute angles
            abs_varpi = np.arctan2(goal_vec[1], goal_vec[0]) 
            abs_varpi_z = np.arctan2(goal_vec[2], np.linalg.norm(goal_vec[:2])) 
            
            # 2. Convert to Relative Angles (Subtract UAV heading)
            varpi = abs_varpi - theta
            varpi_z = abs_varpi_z - phi
            
            # 3. Normalize to [-pi, pi] to prevent continuous rotation buildup
            varpi = (varpi + np.pi) % (2 * np.pi) - np.pi
            varpi_z = (varpi_z + np.pi) % (2 * np.pi) - np.pi
        else:
            varpi = varpi_z = 0.0

        xi  = np.array([d5, varpi, varpi_z], dtype=np.float32)
        obs = np.concatenate([[theta, phi],
                              lidar_norm.flatten(),
                              xi])
        assert obs.shape == (30,), f"obs shape mismatch: {obs.shape}"
        return obs.astype(np.float32)

    def _get_observations(self) -> np.ndarray:
        """Get current observations for all agents."""
        obs_list = []
        for i in range(self.n_agents):
            obs = self._get_single_observation(i)
            if hasattr(self, '_cached_obs'):
                self._cached_obs[i] = obs
            obs_list.append(obs)
        return np.array(obs_list)

    def _sample_free_position(self) -> np.ndarray:
        """Sample position not inside obstacles, with per-agent random fallback."""
        max_attempts = 200  # increase for dense scenes
        env = np.array(self.cfg['env_size'])
        for _ in range(max_attempts):
            # Use a margin from all walls to avoid immediate boundary collision
            margin = min(3.0, np.min(env) * 0.1)
            pos = self.scene_gen.rng.uniform(
                [margin, margin, margin],
                [env[0] - margin, env[1] - margin, env[2] - margin]
            )
            if not self._inside_obstacles(pos, buffer=self.cfg['collision_radius'] + 0.5):
                return pos.astype(np.float32)
        
        # Last resort: jitter a random safe-ish region rather than a fixed point
        # so multiple agents don't share the fallback and immediately collide
        margin_x = min(3.0, env[0] * 0.1)
        margin_y = min(3.0, env[1] * 0.1)
        for attempt in range(50):
            pos = np.array([
                self.scene_gen.rng.uniform(margin_x, env[0] - margin_x),
                self.scene_gen.rng.uniform(margin_y, env[1] - margin_y),
                self.scene_gen.rng.uniform(3.0, env[2] - 3.0),
            ], dtype=np.float32)
            if not self._inside_obstacles(pos, buffer=self.cfg['collision_radius'] + 0.5):
                return pos
        
        # Absolute fallback: log a warning and return centre of arena
        import warnings
        n_obs = self.current_stage_cfg.get('static_obs', 0)
        warnings.warn(f"Could not find free position after all attempts. "
                      f"Arena may be over-saturated (obstacle_count={n_obs}).")
        return self.scene_gen.rng.uniform(
            [5.0, 5.0, 5.0], env - 5.0
        ).astype(np.float32)

    def _inside_obstacles(self, pos: np.ndarray, buffer: float = 0.0) -> bool:
        for obs in self.obstacles:
            if self.reward_fn._surface_distance(pos, obs) < buffer:
                return True
        return False

    def _out_of_bounds(self, pos: np.ndarray) -> bool:
        cr = self.cfg.get('collision_radius', 0.5)
        return (
            pos[0] < cr or pos[0] > self.cfg['env_size'][0] - cr or
            pos[1] < cr or pos[1] > self.cfg['env_size'][1] - cr or
            pos[2] < self.cfg.get('min_altitude', 0.0) + cr or
            pos[2] > self.cfg['max_altitude'] - cr
        )
