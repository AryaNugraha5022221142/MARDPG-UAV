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
        scene = scene_type if scene_type is not None else np.random.choice(cfg['scene_types'])
        
        # Generate obstacles
        self.obstacles = self.scene_gen.generate(scene, density, self.n_agents)
        
        # Sample start/goal positions (ensuring clearance)
        self.agents_state = np.zeros((self.n_agents, 5), dtype=np.float32)
        self.goals = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.prev_applied_actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.agent_done = np.zeros(self.n_agents, dtype=bool)
        
        for i in range(self.n_agents):
            self.agents_state[i, :3] = self._sample_free_position()
            self.goals[i] = self._sample_free_position()
            # Random initial orientation
            self.agents_state[i, 3] = np.random.uniform(-np.pi, np.pi)  # theta
            self.agents_state[i, 4] = np.random.uniform(-np.pi/4, np.pi/4)  # phi
        
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
            observations: [n_agents, 30]
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
            rangefinder_raw, rangefinder_norm = self.rangefinder.scan(pos, theta, phi, self.obstacles)
            goal_sensing = self.goal_sensor.compute(pos, theta, phi, self.goals[i])
            
            # Observation vector
            sin_cos_att = np.array([np.sin(theta), np.cos(theta),
                                    np.sin(phi),   np.cos(phi)], dtype=np.float32)
            prev_act_norm = self.prev_applied_actions[i] / (np.pi / 6)
            obs = np.concatenate([sin_cos_att, prev_act_norm, rangefinder_norm.flatten(), goal_sensing])
            obs_list.append(obs)
            
            # Reward
            other_pos = [positions[j] for j in range(self.n_agents) if j != i]
            rewards[i] = self.reward_fn.compute(
                i, pos, self.goals[i], rangefinder_raw, other_pos, self.obstacles
            )
            
            # Check collision
            if self._out_of_bounds(pos) or self._check_collision(i, positions):
                collisions[i] = True
                rewards[i] -= self.cfg['reward']['r_col']  # -15
            
            # Check goal reached
            if np.linalg.norm(pos - self.goals[i]) < self.cfg['goal_threshold']:
                reached[i] = True
                rewards[i] += self.cfg['reward']['r_goal']  # +15
        
        self.steps += 1
        
        # Episode termination
        for i in range(self.n_agents):
            self.agent_done[i] = self.agent_done[i] or collisions[i] or reached[i]
            
        timeout = self.steps >= self.cfg['max_steps_per_episode']
        episode_done = bool(np.all(self.agent_done)) or timeout
        
        applied_actions = self.prev_applied_actions.copy()
        
        info = {
            'collisions': collisions,
            'reached': reached,
            'timeout': timeout,
            'steps': self.steps,
            'agent_done': self.agent_done.copy(),
            'applied_actions': applied_actions,
        }
        
        return np.array(obs_list), rewards, episode_done, info

    def _get_single_observation(self, i: int) -> np.ndarray:
        pos = self.agents_state[i, :3]
        theta, phi = self.agents_state[i, 3], self.agents_state[i, 4]
        rangefinder_raw, rangefinder_norm = self.rangefinder.scan(pos, theta, phi, self.obstacles)
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
        """Sample position not inside obstacles."""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = np.random.uniform([0, 0, 2.0], self.cfg['env_size'])
            pos[2] = np.clip(pos[2], 2.0, self.cfg['env_size'][2] - 2.0)
            if not self._inside_obstacles(pos):
                return pos
        return np.array([5.0, 5.0, 5.0])

    def _inside_obstacles(self, pos: np.ndarray) -> bool:
        for obs in self.obstacles:
            if obs.type == 'sphere':
                if np.linalg.norm(pos - obs.position) < obs.size[0]:
                    return True
            elif obs.type == 'cylinder':
                d_xy = np.linalg.norm(pos[:2] - obs.position[:2])
                if d_xy < obs.size[0] and abs(pos[2] - obs.position[2]) < obs.size[1]/2:
                    return True
            elif obs.type == 'box':
                if np.all(np.abs(pos - obs.position) < obs.size):
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
        # Obstacle collision
        for obs in self.obstacles:
            if obs.type == 'sphere':
                if np.linalg.norm(pos - obs.position) < obs.size[0] + self.cfg['collision_radius']:
                    return True
            elif obs.type == 'cylinder':
                d_xy = np.linalg.norm(pos[:2] - obs.position[:2])
                if d_xy < obs.size[0] + self.cfg['collision_radius']:
                    if abs(pos[2] - obs.position[2]) < obs.size[1]/2 + self.cfg['collision_radius']:
                        return True
            elif obs.type == 'box':
                if np.all(np.abs(pos - obs.position) < obs.size + self.cfg['collision_radius']):
                    return True
        
        # Inter-UAV collision
        for j, other_pos in enumerate(positions):
            if j != agent_id:
                if np.linalg.norm(pos - other_pos) < self.cfg['inter_uav_min_dist']:
                    return True
        return False
