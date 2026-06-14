"""
Artificial Potential Field (APF) baseline controller for MultiUAVEnv.

Non-learning, reactive. Per live UAV it builds a desired world-frame heading
from: an attractive pull toward the goal, a repulsive push from the nearest
obstacle surfaces sensed by the SAME 25-beam rangefinder the RL agent uses, and
a repulsive push from other live UAVs. Because speed is constant, the desired
heading is converted into the env's action space -- bounded yaw/pitch
increments [rho, tau] in [-max_delta, max_delta].

Expected (and scientifically useful) failure mode: pure APF is prone to local
minima / oscillation in dense or concave obstacle fields, which should surface
as a higher trapped-rate than the learned policy. That contrast is the point of
including it.

Place at: mardpg_uav/apf.py
"""
import numpy as np


class APFController:
    def __init__(self, env,
                 k_att: float = 1.0,
                 k_rep_obs: float = 2.0,
                 k_rep_uav: float = 2.0,
                 d0_obs: float = 6.0,      # obstacle influence radius (m)
                 d0_uav: float = 4.0):     # inter-UAV influence radius (m)
        self.env = env
        self.k_att = k_att
        self.k_rep_obs = k_rep_obs
        self.k_rep_uav = k_rep_uav
        self.d0_obs = d0_obs
        self.d0_uav = d0_uav
        self.max_delta = env.dynamics.max_delta

    def _beam_world_dirs(self, theta, phi):
        # Mirrors Rangefinder.scan's rotation of the cached unit beam vectors.
        cy, sy = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi),   np.sin(phi)
        R = np.array([[cy * cp, -sy, -cy * sp],
                      [sy * cp,  cy, -sy * sp],
                      [sp,      0.0,  cp     ]], dtype=np.float32)
        return (R @ self.env.rangefinder._dir_vecs.T).T   # (25, 3)

    def act(self):
        env = self.env
        actions = np.zeros((env.n_agents, 2), dtype=np.float32)
        positions = env.agents_state[:, :3]
        for i in range(env.n_agents):
            if env.agent_done[i]:
                continue
            pos   = positions[i]
            theta = env.agents_state[i, 3]
            phi   = env.agents_state[i, 4]

            # --- Attractive (toward goal) ---
            to_goal = env.goals[i] - pos
            dist = np.linalg.norm(to_goal)
            f = self.k_att * (to_goal / dist) if dist > 1e-6 else np.zeros(3, np.float32)

            # --- Repulsive: obstacles via raw lidar beams ---
            dists, _ = env.rangefinder.scan(
                pos, theta, phi, env.obstacles,
                obs_centers=env.obs_centers, obs_max_sizes=env.obs_max_sizes)
            dists = dists.flatten()
            dirs  = self._beam_world_dirs(theta, phi)
            for d, bdir in zip(dists, dirs):
                if 1e-3 < d < self.d0_obs:
                    mag = self.k_rep_obs * (1.0 / d - 1.0 / self.d0_obs) / (d * d)
                    f = f - mag * bdir          # push away from the beam hit

            # --- Repulsive: other live UAVs ---
            for j in range(env.n_agents):
                if j == i or env.agent_done[j]:
                    continue
                diff = pos - positions[j]
                dj = np.linalg.norm(diff)
                if 1e-6 < dj < self.d0_uav:
                    mag = self.k_rep_uav * (1.0 / dj - 1.0 / self.d0_uav) / (dj * dj)
                    f = f + mag * (diff / dj)

            nf = np.linalg.norm(f)
            if nf < 1e-6:
                continue
            f = f / nf
            theta_des = np.arctan2(f[1], f[0])
            phi_des   = np.arctan2(f[2], np.linalg.norm(f[:2]))
            d_theta = (theta_des - theta + np.pi) % (2 * np.pi) - np.pi
            d_phi   = phi_des - phi
            actions[i, 0] = np.clip(d_theta, -self.max_delta, self.max_delta)
            actions[i, 1] = np.clip(d_phi,   -self.max_delta, self.max_delta)
        return actions
