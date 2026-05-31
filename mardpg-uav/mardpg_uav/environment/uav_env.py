"""
Main Gym-style environment for Multi-UAV Navigation.
Implements Dec-POMDP from Section 1.2.
"""
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List, Optional
from .dynamics import UAVDynamics
from .obstacles import SceneGenerator, Obstacle
from .sensors import Rangefinder, GoalSensor
from .rewards import RewardFunction


class MultiUAVEnv(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.n_agents = config['n_agents']
        
        # Subsystems
        self.dynamics = UAVDynamics(
            v=config['uav_speed'],
            dt=config['dt'],
            env_size=config['env_size'],
            max_altitude=config['max_altitude'],
            min_altitude=config.get('min_altitude', 0.0)
        )
        self.scene_gen = SceneGenerator(
            env_size=config['env_size'],
            seed=config.get('seed', None)
        )
        self.rangefinder = Rangefinder(range_max=config['sensor_range'])
        self.goal_sensor = GoalSensor()
        self.reward_fn = RewardFunction(
            alpha=config['reward']['alpha'],
            lambda_col=config['reward']['lambda_col'],
            sigma_col=config['reward']['sigma_col'],
            lambda_sep=config['reward']['lambda_sep'],
            sigma_sep=config['reward']['sigma_sep'],
            r_free=config['reward']['r_free'],
            r_step=config['reward']['r_step'],
            delta=tuple(config['reward']['delta']),
            collision_radius=config['collision_radius'],
            inter_uav_min=config['inter_uav_min_dist']
        )
        
        # Spaces
        self.obs_dim = 34  # 4 (att) + 2 (prev_act) + 25 (lidar) + 3 (goal)
        self.action_dim = 2  # rho, tau
        
        # State
        self.agents_state = None  # [n_agents, 5] (x,y,z,theta,phi)
        self.goals = None       # [n_agents, 3]
        self.obstacles: List[Obstacle] = []
        self.steps = 0
        
    def reset(self, scene_type: Optional[int] = None,
              obstacle_density: Optional[float] = None) -> np.ndarray:
        cfg = self.cfg
        density = obstacle_density if obstacle_density is not None else cfg['obstacle_density']
        scene = scene_type if scene_type is not None else self.scene_gen.rng.choice(cfg['scene_types'])
        
        # Generate obstacles
        self.obstacles = self.scene_gen.generate(scene, density, self.n_agents)
        
        # FIX: Inject physical arena boundaries so Lidar can see the walls
        t = 5.0 # Wall thickness
        ex, ey, ez = self.cfg['env_size']
        min_z = self.cfg.get('min_altitude', 0.0)
        cr = self.cfg['collision_radius']
        
        walls = [
            Obstacle('box', np.array([-t - cr, ey/2, ez/2]), np.array([t, ey/2, ez/2])), # X min
            Obstacle('box', np.array([ex+t+cr, ey/2, ez/2]), np.array([t, ey/2, ez/2])), # X max
            Obstacle('box', np.array([ex/2, -t - cr, ez/2]), np.array([ex/2, t, ez/2])), # Y min
            Obstacle('box', np.array([ex/2, ey+t+cr, ez/2]), np.array([ex/2, t, ez/2])), # Y max
            Obstacle('box', np.array([ex/2, ey/2, min_z-t-cr]), np.array([ex/2, ey/2, t])), # Z min (floor)
            Obstacle('box', np.array([ex/2, ey/2, ez+t+cr]), np.array([ex/2, ey/2, t]))  # Z max (ceiling)
        ]
        self.obstacles.extend(walls)

        self.obs_centers = np.array([obs.position for obs in self.obstacles])

        # FIX: Compute true circumscribed bounding radii to prevent corner-clipping
        sizes = []
        for obs in self.obstacles:
            if obs.type == 'box':
                sizes.append(np.linalg.norm(obs.size))
            elif obs.type == 'cylinder':
                sizes.append(np.sqrt(obs.size[0]**2 + (obs.size[1]/2)**2))
            else:
                sizes.append(obs.size[0])
        self.obs_max_sizes = np.array(sizes)
        
        # Sample start/goal positions (ensuring clearance)
        self.agents_state = np.zeros((self.n_agents, 5), dtype=np.float32)
        self.goals = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.prev_applied_actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.agent_done = np.zeros(self.n_agents, dtype=bool)
        self.agent_reached = np.zeros(self.n_agents, dtype=bool)
        self.agent_collided = np.zeros(self.n_agents, dtype=bool)
        
        positions = []
        for i in range(self.n_agents):
            for _ in range(50):
                pos = self._sample_free_position()
                if all(np.linalg.norm(pos - p) >= self.cfg['inter_uav_min_dist'] * 2
                       for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(self._sample_free_position())  # fallback
            self.agents_state[i, :3] = positions[-1]
            self.goals[i] = self._sample_free_position()
            # Random initial orientation
            self.agents_state[i, 3] = self.scene_gen.rng.uniform(-np.pi, np.pi)  # theta
            self.agents_state[i, 4] = self.scene_gen.rng.uniform(-np.pi/4, np.pi/4)  # phi
        
        # Initialize reward function distances
        dists = {i: np.linalg.norm(self.agents_state[i, :3] - self.goals[i])
                 for i in range(self.n_agents)}
        self.reward_fn.reset(list(range(self.n_agents)), dists)
        
        self.steps = 0
        return self._get_observations()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Args:
            actions: [n_agents, 2] in [-π/6, π/6]
        Returns:
            observations: [n_agents, 34]
            rewards: [n_agents]
            done: bool (episode terminates if any agent collides or all reach goals)
            info: dict
        """
        actions = np.array(actions, dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        collisions = np.zeros(self.n_agents, dtype=bool)
        reached = np.zeros(self.n_agents, dtype=bool)
        
        # Update dynamics
        for i in range(self.n_agents):
            if self.agent_done[i]:
                continue
                
            next_state, applied = self.dynamics.step(
                self.agents_state[i], actions[i], self.prev_applied_actions[i])
            self.agents_state[i] = next_state
            self.prev_applied_actions[i] = applied
        
        # Compute observations and rewards
        obs_list = []
        positions = [self.agents_state[i, :3] for i in range(self.n_agents)]
        
        for i in range(self.n_agents):
            if self.agent_done[i]:
                obs_list.append(self._get_single_observation(i))
                rewards[i] = 0.0
                continue
                
            pos = self.agents_state[i, :3]
            theta, phi = self.agents_state[i, 3], self.agents_state[i, 4]
            
            # Sensing
            rangefinder_raw, rangefinder_norm = self.rangefinder.scan(
                pos, theta, phi, self.obstacles,
                obs_centers=self.obs_centers, obs_max_sizes=self.obs_max_sizes
            )
            goal_sensing = self.goal_sensor.compute(pos, theta, phi, self.goals[i])
            
            # Observation vector
            sin_cos_att = np.array([np.sin(theta), np.cos(theta),
                                    np.sin(phi),   np.cos(phi)], dtype=np.float32)
            prev_act_norm = self.prev_applied_actions[i] / (np.pi / 6)
            obs = np.concatenate([sin_cos_att, prev_act_norm, rangefinder_norm.flatten(), goal_sensing])
            obs_list.append(obs)
            
            # Reward
            other_pos = [positions[j] for j in range(self.n_agents) if j != i and not self.agent_done[j]]
            rewards[i] = self.reward_fn.compute(
                i, pos, self.goals[i], rangefinder_raw, rangefinder_norm, other_pos, self.obstacles,
                obs_centers=self.obs_centers, obs_max_sizes=self.obs_max_sizes
            )
            
            # Check collision
            if self._check_collision(i, positions):
                collisions[i] = True
                rewards[i] -= self.cfg['reward']['r_col']  # -250
                
            # Optional: boundary penalty
            WALL_MARGIN = 3.0
            WALL_PENALTY_SCALE = 0.5
            wall_dist = min(
                pos[0], self.cfg['env_size'][0] - pos[0],
                pos[1], self.cfg['env_size'][1] - pos[1],
                pos[2] - self.cfg.get('min_altitude', 0.0),
                self.cfg['max_altitude'] - pos[2]
            )
            if wall_dist < WALL_MARGIN:
                rewards[i] -= WALL_PENALTY_SCALE * (1.0 - wall_dist / WALL_MARGIN)
                
            if self._out_of_bounds(pos):
                rewards[i] -= 5.0  # OOB soft penalty
            
            # Check goal reached
            if np.linalg.norm(pos - self.goals[i]) < self.cfg['goal_threshold']:
                reached[i] = True
                rewards[i] += self.cfg['reward']['r_goal']  # +15
        
        self.steps += 1
        
        self.agent_reached |= reached
        self.agent_collided |= collisions
        
        # Episode termination
        for i in range(self.n_agents):
            self.agent_done[i] = self.agent_done[i] or self.agent_collided[i] or self.agent_reached[i]
            
        timeout = self.steps >= self.cfg['max_steps_per_episode']
        episode_done = bool(np.all(self.agent_done)) or timeout
        
        applied_actions = self.prev_applied_actions.copy()
        
        info = {
            'collisions': self.agent_collided.copy(),
            'reached': self.agent_reached.copy(),
            'step_collisions': collisions.copy(),
            'step_reached': reached.copy(),
            'timeout': timeout,
            'steps': self.steps,
            'agent_done': self.agent_done.copy(),
            'applied_actions': applied_actions,
        }
        
        return np.array(obs_list), rewards, episode_done, info

    def _get_single_observation(self, i: int) -> np.ndarray:
        pos = self.agents_state[i, :3]
        theta, phi = self.agents_state[i, 3], self.agents_state[i, 4]
        rangefinder_raw, rangefinder_norm = self.rangefinder.scan(
            pos, theta, phi, self.obstacles,
            obs_centers=self.obs_centers, obs_max_sizes=self.obs_max_sizes
        )
        goal_sensing = self.goal_sensor.compute(pos, theta, phi, self.goals[i])
        sin_cos_att = np.array([np.sin(theta), np.cos(theta),
                                np.sin(phi),   np.cos(phi)], dtype=np.float32)
        prev_act_norm = self.prev_applied_actions[i] / (np.pi / 6)
        return np.concatenate([sin_cos_att, prev_act_norm, rangefinder_norm.flatten(), goal_sensing])

    def _get_observations(self) -> np.ndarray:
        """Get current observations for all agents."""
        obs_list = []
        for i in range(self.n_agents):
            obs_list.append(self._get_single_observation(i))
        return np.array(obs_list)

    def _sample_free_position(self) -> np.ndarray:
        """Sample position not inside obstacles, with per-agent random fallback."""
        max_attempts = 200  # increase for dense scenes
        env = np.array(self.cfg['env_size'])
        for _ in range(max_attempts):
            # Use a margin from all walls to avoid immediate boundary collision
            margin = 3.0
            pos = self.scene_gen.rng.uniform(
                [margin, margin, 3.0],
                [env[0] - margin, env[1] - margin, env[2] - 3.0]
            )
            if not self._inside_obstacles(pos, buffer=self.cfg['collision_radius'] + 0.5):
                return pos.astype(np.float32)
        
        # Last resort: jitter a random safe-ish region rather than a fixed point
        # so multiple agents don't share the fallback and immediately collide
        for attempt in range(50):
            pos = np.array([
                self.scene_gen.rng.uniform(5.0, env[0] - 5.0),
                self.scene_gen.rng.uniform(5.0, env[1] - 5.0),
                self.scene_gen.rng.uniform(5.0, env[2] - 5.0),
            ], dtype=np.float32)
            if not self._inside_obstacles(pos, buffer=self.cfg['collision_radius'] + 0.5):
                return pos
        
        # Absolute fallback: log a warning and return centre of arena
        import warnings
        warnings.warn(f"Could not find free position after all attempts. "
                      f"Arena may be over-saturated (density={self.cfg['obstacle_density']}).")
        return self.scene_gen.rng.uniform(
            [5.0, 5.0, 5.0], env - 5.0
        ).astype(np.float32)

    def _inside_obstacles(self, pos: np.ndarray, buffer: float = 0.0) -> bool:
        for obs in self.obstacles:
            if self.reward_fn._surface_distance(pos, obs) < buffer:
                return True
        return False

    def _out_of_bounds(self, pos: np.ndarray) -> bool:
        return (
            pos[0] <= 0.0 or pos[0] >= self.cfg['env_size'][0] or
            pos[1] <= 0.0 or pos[1] >= self.cfg['env_size'][1] or
            pos[2] <= self.cfg.get('min_altitude', 0.0) or pos[2] >= self.cfg['max_altitude']
        )

    def _check_collision(self, agent_id: int, positions: List[np.ndarray]) -> bool:
        pos = positions[agent_id]
        
        # Fast culling for obstacles
        if len(self.obstacles) > 0:
            dists = np.linalg.norm(self.obs_centers - pos, axis=1)
            # RC (collision threshold) is small, typically 0.5. Add it to max_size.
            mask = dists < (self.cfg['collision_radius'] + self.obs_max_sizes + 0.1)
            close_obstacles = [self.obstacles[idx] for idx, m in enumerate(mask) if m]
        else:
            close_obstacles = []

        # Obstacle collision
        for obs in close_obstacles:
            if self.reward_fn._surface_distance(pos, obs) < self.cfg['collision_radius']:
                return True
        
        # Inter-UAV collision
        for j, other_pos in enumerate(positions):
            if j != agent_id and not self.agent_done[j]:
                if np.linalg.norm(pos - other_pos) < self.cfg['inter_uav_min_dist']:
                    return True
        return False
