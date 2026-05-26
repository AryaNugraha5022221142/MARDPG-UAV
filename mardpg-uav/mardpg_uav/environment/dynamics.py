"""
UAV Kinematics from Eq (1) of the blueprint.
Implements deterministic state transition with fixed speed.
"""
import numpy as np
from typing import Tuple


class UAVDynamics:
    def __init__(self, v: float = 3.0, dt: float = 0.1,
                 max_altitude: float = 60.0, min_altitude: float = 0.0):
        self.v = v              # fixed speed (m/s)
        self.dt = dt            # control timestep (s)
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.action_bounds = (-np.pi / 6, np.pi / 6)  # rad/s

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Args:
            state: [x, y, z, theta, phi]
            action: [rho, tau] angular velocities (rad/s)
        Returns:
            next_state: [x, y, z, theta, phi]
        """
        x, y, z, theta, phi = state
        rho, tau = np.clip(action, *self.action_bounds)

        # Eq (1) with explicit dt
        theta_next = theta + rho * self.dt
        phi_next = phi + tau * self.dt

        x_next = x + self.v * np.cos(theta_next) * np.cos(phi_next) * self.dt
        y_next = y + self.v * np.sin(theta_next) * np.cos(phi_next) * self.dt
        z_next = z + self.v * np.sin(phi_next) * self.dt

        # Altitude bounds
        z_next = np.clip(z_next, self.min_altitude, self.max_altitude)

        return np.array([x_next, y_next, z_next, theta_next, phi_next], dtype=np.float32)
