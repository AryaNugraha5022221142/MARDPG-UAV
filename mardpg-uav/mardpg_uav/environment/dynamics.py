"""
UAV Kinematics — faithful to Eq. (1) of Xue & Chen (2024).
State : [x, y, z, theta, phi]   (3 position + 2 directional angles)
Action: [rho, tau]               (angle increments, clipped to ±pi/6)
"""
import numpy as np

class QuadcopterDynamics:
    def __init__(self, v: float = 1.0, dt: float = 0.1,
                 env_size=(100.0, 100.0, 60.0),
                 max_altitude: float = 60.0,
                 min_altitude: float = 0.0,
                 max_delta_angle: float = np.pi / 6):
        self.v              = v                         # constant speed (m/s)
        self.dt             = dt
        self.env_size       = np.asarray(env_size, dtype=np.float32)
        self.max_altitude   = max_altitude
        self.min_altitude   = min_altitude
        self.max_delta      = max_delta_angle           # pi/6

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        state  : [x, y, z, theta, phi]
        action : [rho, tau]  in [-pi/6, pi/6]
        returns: next_state  [x, y, z, theta, phi]
        """
        x, y, z, theta, phi = state

        rho = np.clip(action[0], -self.max_delta, self.max_delta)
        tau = np.clip(action[1], -self.max_delta, self.max_delta)

        # Eq. (1)
        theta_new = theta + rho
        phi_new   = phi   + tau

        # Keep angles in [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        phi_new   = np.clip(phi_new, -np.pi / 2, np.pi / 2)   # physical elevation limit

        x_new = x + self.v * self.dt * np.cos(theta_new) * np.cos(phi_new)
        y_new = y + self.v * self.dt * np.sin(theta_new) * np.cos(phi_new)
        z_new = z + self.v * self.dt * np.sin(phi_new)

        # Boundary absorption
        x_new = np.clip(x_new, 0.0, self.env_size[0])
        y_new = np.clip(y_new, 0.0, self.env_size[1])
        z_new = np.clip(z_new, self.min_altitude, self.max_altitude)

        return np.array([x_new, y_new, z_new, theta_new, phi_new], dtype=np.float32)
