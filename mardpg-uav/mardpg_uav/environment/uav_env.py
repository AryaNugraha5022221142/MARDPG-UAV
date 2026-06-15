"""
Main Gym-style environment for Multi-UAV Navigation.
Implements Dec-POMDP from Section 1.2.
"""
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List, Optional
from .dynamics import QuadcopterDynamics
from .obstacles import SceneGenerator, Obstacle
from .sensors import Rangefinder
from .rewards import RewardFunction


class MultiUAVEnv(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        import copy as _copy
        self.cfg = _copy.deepcopy(config)
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
        self.rangefinder = Rangefinder(range_max=config['sensor_range'],
                                       seed=config.get('seed', None))   # [N3-5]
        # GoalSensor removed (N3-6): goal block is computed inline below.
        self.reward_fn = RewardFunction(
            alpha=config['reward']['alpha'],
            lambda_col=config['reward']['lambda_col'],
            sigma_col=config['reward']['sigma_col'],
            sigma_col_uav=config['reward'].get('sigma_col_uav',
                                               config['reward']['sigma_col']),
            r_free=config['reward']['r_free'],
            r_step=config['reward']['r_step'],
            delta=tuple(config['reward']['delta']),
            collision_radius=config['collision_radius'],
            inter_uav_min=config['inter_uav_min_dist'],
            range_max=config['sensor_range']
        )
        
        # [NO-COMM FIX] Neighbour block REMOVED. Other UAVs are perceived only
        # through the onboard rangefinder (injected as spheres; see
        # _other_uav_obstacles), so the policy input is fully decentralized and
        # communication-free — matching the actor input of Xue & Chen (2024):
        # attitude + lidar + goal. Other live UAVs are sensed as spheres of this
        # radius (>= body radius, so a neighbour's safety bubble is detectable
        # rather than a 0.5 m point that falls between beams).
        self.uav_sense_radius = float(config.get('uav_sense_radius',
                                                  config.get('inter_uav_min_dist', 1.0)))
        # attitude(4) + lidar(25) + goal(5) + alive(1)
        self.obs_dim    = 4 + 25 + 5 + 1  # = 35
        self.action_dim = 2
        
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

        # [FIX1] Start/goal assignment delegated to assignment.assign_start_goals,
        # which produces intersecting (crossing) start->goal pairs for a
        # configurable fraction of agents (stage_cfg['conflict_frac']) so that
        # straight-line paths cross and genuine inter-UAV avoidance is exercised.
        # conflict_frac == 0.0 reproduces the legacy scatter behaviour exactly.
        from .assignment import assign_start_goals
        self.agents_state = np.zeros((self.n_agents, 5), dtype=np.float32)
        self.goals = np.zeros((self.n_agents, 3), dtype=np.float32)

        starts, goals = assign_start_goals(self, stage_cfg)
        self.agents_state[:, :3] = starts
        self.goals[:] = goals
        for i in range(self.n_agents):
            self.agents_state[i, 3] = self.scene_gen.rng.uniform(-np.pi, np.pi)      # theta
            self.agents_state[i, 4] = self.scene_gen.rng.uniform(-np.pi/6, np.pi/6)  # phi

        self.agent_done = np.zeros(self.n_agents, dtype=bool)
        self.agent_reached = np.zeros(self.n_agents, dtype=bool)
        self.agent_collided = np.zeros(self.n_agents, dtype=bool)
        self.agent_dyn_collided = np.zeros(self.n_agents, dtype=bool)
        self._cached_obs = [None] * self.n_agents
        
        self.agent_progress_history = [[] for _ in range(self.n_agents)]
        
        dists = {i: np.linalg.norm(self.agents_state[i, :3] - self.goals[i]) for i in range(self.n_agents)}
        self.reward_fn.reset(list(range(self.n_agents)), dists)
        self.steps = 0
        self.safe_inter_uav_steps = 0  # Track safe steps for Stage 2 metric
        # [FIX5] graded interaction accumulators
        self.min_pair_dist = np.inf
        self.near_miss_steps = 0
        self.agent_uav_collided = np.zeros(self.n_agents, dtype=bool)

        return self._get_observations()

    def _world_to_body(self, vec: np.ndarray, theta: float, phi: float) -> np.ndarray:
        """[N3-1] Rotate a world-frame vector into the UAV body frame.
        Transpose of the body->world rotation used by the rangefinder, so goal
        bearing and neighbor features now live in the SAME (body) frame."""
        cy, sy = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi),   np.sin(phi)
        Rt = np.array([[ cy*cp,  sy*cp,  sp ],
                       [-sy,      cy,     0.0],
                       [-cy*sp, -sy*sp,  cp ]], dtype=np.float32)
        return (Rt @ vec.astype(np.float32)).astype(np.float32)

    def _update_obstacle_caches(self):
        """Helper to refresh bounding boxes for lidar when objects move."""
        self.obs_centers = np.array([obs.position for obs in self.obstacles])
        sizes = []
        for obs in self.obstacles:
            if obs.type == 'box': sizes.append(np.linalg.norm(obs.size))
            elif obs.type == 'cylinder': sizes.append(np.sqrt(obs.size[0]**2 + (obs.size[1]/2)**2))
            else: sizes.append(obs.size[0])
        self.obs_max_sizes = np.array(sizes)

    def _other_uav_obstacles(self, i: int):
        """[NO-COMM FIX] Other LIVE UAVs as spherical lidar targets so a UAV can
        sense its neighbours onboard (no communication). Radius =
        uav_sense_radius (>= body radius). SENSING only; physical inter-UAV
        collision is still tested at inter_uav_min_dist in step()."""
        r = float(self.uav_sense_radius)
        done = getattr(self, 'agent_done', np.zeros(self.n_agents, dtype=bool))
        return [Obstacle('sphere', self.agents_state[j, :3].copy(),
                         np.array([r], dtype=np.float32))
                for j in range(self.n_agents) if j != i and not done[j]]

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Args:
            actions: [n_agents, 2] angle increments in [-max_delta_angle, max_delta_angle]
        Returns:
            observations: [n_agents, 33]
            rewards: [n_agents]
            done: bool (timeout, or all agents terminal)
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
        # [N-1] Done agents are NOT physical collision hazards: a reached agent has
        # landed and a crashed agent is removed. A live agent cannot sense them
        # (lidar ignores UAVs) and must not die to a frozen "ghost". Mask done
        # agents out as collision TARGETS (columns). dist_matrix itself is left
        # intact for the safe_inter_uav metric, which already slices active agents.
        ghost = self.agent_done  # frozen from prior steps; both-die-this-step pairs are still live here
        dist_for_collision = dist_matrix.copy()
        dist_for_collision[:, ghost] = np.inf
        inter_collisions = np.any(dist_for_collision < self.cfg['inter_uav_min_dist'], axis=1)  # (N,)
        # [FIX5] flag inter-UAV collisions separately from obstacle collisions
        self.agent_uav_collided |= (inter_collisions & ~self.agent_done)

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

        # Check inter-uav safety for metrics, counting only active (non-done) agents.
        # Done agents are frozen in place; including them would permanently penalise the
        # metric for the rest of the episode after any collision.
        active = ~self.agent_done
        n_active = active.sum()
        if n_active >= 2:
            active_dists = dist_matrix[np.ix_(active, active)]  # diagonal is inf
            md = float(active_dists.min())
            self.min_pair_dist = min(self.min_pair_dist, md)
            near_band = float(self.cfg.get('near_miss_band', 3.0))
            if md < near_band:
                self.near_miss_steps += 1
            if np.all(active_dists >= 1.0):
                self.safe_inter_uav_steps += 1
        elif n_active <= 1:
            # Fewer than 2 active agents — no inter-UAV risk; count as safe
            self.safe_inter_uav_steps += 1

        for i in range(self.n_agents):
            if not self.agent_done[i]:
                dist = np.linalg.norm(self.agents_state[i, :3] - self.goals[i])
                self.agent_progress_history[i].append(dist)
                
            if self.agent_done[i]:
                obs_list.append(self._cached_obs[i])
                
                # Done agents receive 0 reward (ignored by sequence mask anyway)
                rewards[i] = 0.0
                continue
                
            pos = self.agents_state[i, :3]
            
            theta = self.agents_state[i, 3]
            phi   = self.agents_state[i, 4]
            # Sensing — other live UAVs injected as spheres so the lidar
            # perceives them (decentralized, comms-free inter-UAV avoidance).
            rangefinder_raw, rangefinder_norm = self.rangefinder.scan(
                pos, theta, phi, self.obstacles,
                obs_centers=self.obs_centers, obs_max_sizes=self.obs_max_sizes,
                extra_obstacles=self._other_uav_obstacles(i)
            )
            
            # Observe we don't save goal_disp here anymore since observation logic moved to _get_single_observation.
            
            obs = self._get_single_observation(i, lidar_norm=rangefinder_norm)
            self._cached_obs[i] = obs
            obs_list.append(obs)
            
            # Reward
            # [N-1] Separation penalty only over agents that physically exist.
            other_pos = [positions[j] for j in range(self.n_agents)
                         if j != i and not self.agent_done[j]]
            rewards[i] = self.reward_fn.compute(
                i, pos, self.goals[i], rangefinder_raw, rangefinder_norm, other_pos, self.obstacles,
                obs_centers=self.obs_centers, obs_max_sizes=self.obs_max_sizes
            )
            
            # Apply pre-calculated penalties
            if np.linalg.norm(pos - self.goals[i]) < self.cfg['goal_threshold']:
                reached[i] = True
                rewards[i] += self.cfg['reward'].get('r_goal', 10.0)  # Add terminal anchor
                
            # [N-3/M-2] Terminal collision anchor, applied UNSCALED so it is
            # symmetric with the unscaled +r_goal success anchor. This is the
            # raw reward decrement, NOT routed through the proximity weight
            # delta[1]. With gamma=0.99 the discounted value of wandering in the
            # proximity band is bounded well below this magnitude, so immediate
            # death stays value-dominated. (At 30.0 this reproduces the previous
            # effective penalty exactly: 0.30 * 100 == 30.)
            elif collisions[i]:
                rewards[i] -= self.cfg['reward'].get('r_collision_terminal', 30.0)
        
        self.steps += 1
        
        self.agent_reached |= reached
        self.agent_collided |= collisions
        
        # Episode termination
        for i in range(self.n_agents):
            self.agent_done[i] = self.agent_done[i] or self.agent_collided[i] or self.agent_reached[i]
            
            if self.agent_done[i]:
                obs_list[i][-1] = 0.0
                if self._cached_obs[i] is not None:
                    self._cached_obs[i][-1] = 0.0
            
        timeout = self.steps >= self.cfg['max_steps_per_episode']
        episode_done = timeout or bool(np.all(self.agent_done))
        
        dyn_collisions_copy = self.agent_dyn_collided.copy() if hasattr(self, 'agent_dyn_collided') else np.zeros(self.n_agents, dtype=bool)

        trapped_agents = self.get_trapped_agents() if episode_done else np.zeros(self.n_agents, dtype=bool)

        info = {
            'collisions': self.agent_collided.copy(),
            'dyn_collisions': dyn_collisions_copy,
            'reached': self.agent_reached.copy(),
            'safe_inter_uav_ratio': self.safe_inter_uav_steps / max(1, self.steps),
            # [FIX5] graded interaction metrics
            'min_pair_dist': (float(self.min_pair_dist)
                              if np.isfinite(self.min_pair_dist) else float('nan')),
            'near_miss_ratio': self.near_miss_steps / max(1, self.steps),
            'uav_collisions': self.agent_uav_collided.copy(),
            'step_collisions': collisions.copy(),
            'step_reached': reached.copy(),
            'trapped': trapped_agents,
            'timeout': timeout,
            'steps': self.steps,
            'agent_done': self.agent_done.copy()
        }
        
        return np.array(obs_list), rewards, episode_done, info

    def get_trapped_agents(self, progress_window: int = 50, progress_threshold: float = 0.5) -> np.ndarray:
        """
        Returns boolean array: True if agent is trapped.
        Trapped = not reached, not collided, and failed to reduce
        distance to goal by > progress_threshold over the last
        progress_window steps.
        
        Args:
            progress_window: number of steps to look back
            progress_threshold: minimum distance reduction (meters) to avoid
                                being classified as trapped
        """
        trapped = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            if self.agent_reached[i] or self.agent_collided[i]:
                continue  # not trapped — they had a definitive outcome
            
            hist = self.agent_progress_history[i]
            if len(hist) >= progress_window:
                dist_start = hist[-progress_window]
                dist_end   = hist[-1]
                progress   = dist_start - dist_end  # positive = moving toward goal
                if progress < progress_threshold:
                    trapped[i] = True
        return trapped

    def _get_single_observation(self, i: int, lidar_norm: np.ndarray = None) -> np.ndarray:
        pos   = self.agents_state[i, :3]
        theta = self.agents_state[i, 3]
        phi   = self.agents_state[i, 4]

        if lidar_norm is None:
            _, lidar_norm = self.rangefinder.scan(
                pos, theta, phi, self.obstacles,
                obs_centers=self.obs_centers,
                obs_max_sizes=self.obs_max_sizes,
                extra_obstacles=self._other_uav_obstacles(i),
            )

        arena_diag = float(np.sqrt(sum(s**2 for s in self.cfg['env_size'])))

        # ---- Goal block: distance + sin/cos of body-frame bearing (wrap-free) [N-9] ----
        goal_vec = self.goals[i] - pos
        d5       = np.linalg.norm(goal_vec)
        if d5 > 1e-6:
            abs_varpi   = np.arctan2(goal_vec[1], goal_vec[0])
            abs_varpi_z = np.arctan2(goal_vec[2], np.linalg.norm(goal_vec[:2]))
            varpi   = abs_varpi   - theta
            varpi_z = abs_varpi_z - phi
        else:
            varpi = varpi_z = 0.0
        d5_norm = d5 / arena_diag
        goal_block = np.array([d5_norm,
                               np.sin(varpi),   np.cos(varpi),
                               np.sin(varpi_z), np.cos(varpi_z)], dtype=np.float32)

        # [NO-COMM FIX] Neighbour block removed: other UAVs are perceived only
        # through the onboard rangefinder (injected as sphere returns in the scan
        # call above and in step()), so the policy input is communication-free.
        done = getattr(self, 'agent_done', np.zeros(self.n_agents, dtype=bool))
        alive = 1.0 if not done[i] else 0.0

        obs = np.concatenate([
            [np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)],  # 0:4
            lidar_norm.flatten(),                                      # 4:29
            goal_block,                                                # 29:34
            [alive],                                                   # 34
        ]).astype(np.float32)
        assert obs.shape == (self.obs_dim,), f"obs shape mismatch: {obs.shape}"
        return obs

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
