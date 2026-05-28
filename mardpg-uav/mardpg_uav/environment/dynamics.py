"""
UAV Kinematics from Eq (1) of the blueprint.
Implements deterministic state transition with fixed speed.
"""
import numpy as np

DELTA_MAX = (np.pi / 6) / 3   # ≈ 0.1745 rad/step — Eq.11

class UAVDynamics:
    def __init__(self, v=3.0, dt=0.1, env_size=(50.0, 50.0, 60.0), max_altitude=60.0, min_altitude=0.0):
        self.env_size = np.asarray(env_size, dtype=np.float32)
        self.v = v
        self.dt = dt
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.action_bounds = (-np.pi / 6, np.pi / 6)

    @staticmethod
    def _wrap(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi   # Eq.9

    def step(self, state: np.ndarray, action_raw: np.ndarray,
             action_prev_applied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (next_state, action_applied) — Eq.10, 4-8.
        """
        # Rate limiting — Eq.10
        delta = np.clip(action_raw - action_prev_applied, -DELTA_MAX, DELTA_MAX)
        action_applied = np.clip(
            action_prev_applied + delta,
            self.action_bounds[0],
            self.action_bounds[1],
        )

        x, y, z, theta, phi = state
        rho, tau = action_applied

        theta_next = self._wrap(theta + rho * self.dt)            # Eq.4
        phi_next   = np.clip(phi + tau * self.dt, -np.pi/2, np.pi/2)  # Eq.5
        x_next = x + self.v * np.cos(theta_next) * np.cos(phi_next) * self.dt
        y_next = y + self.v * np.sin(theta_next) * np.cos(phi_next) * self.dt
        x_next = np.clip(x_next, 0.0, self.env_size[0])
        y_next = np.clip(y_next, 0.0, self.env_size[1])
        z_next = np.clip(z + self.v * np.sin(phi_next) * self.dt,
                         self.min_altitude, self.max_altitude)

        return np.array([x_next, y_next, z_next, theta_next, phi_next],
                        dtype=np.float32), action_applied
