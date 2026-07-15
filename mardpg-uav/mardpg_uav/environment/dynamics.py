import numpy as np

class QuadcopterDynamics:

    def __init__(self, v: float=1.0, dt: float=0.1, env_size=(100.0, 100.0, 60.0), max_altitude: float=60.0, min_altitude: float=0.0, max_delta_angle: float=np.pi / 6):
        self.v = v
        self.dt = dt
        self.env_size = np.asarray(env_size, dtype=np.float32)
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.max_delta = max_delta_angle

    def step(self, state: np.ndarray, action: np.ndarray, current_v: float=None) -> np.ndarray:
        """
        state  : [x, y, z, theta, phi]
        action : [rho, tau, (optional) delta_v]
        returns: next_state  [x, y, z, theta, phi]
        """
        (x, y, z, theta, phi) = state
        rho = np.clip(action, -self.max_delta, self.max_delta)
        tau = np.clip(action, -self.max_delta, self.max_delta)
        theta_new = theta + rho
        phi_new = phi + tau
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        phi_new = np.clip(phi_new, -np.pi / 2, np.pi / 2)
        v = current_v if current_v is not None else self.v
        x_new = x + v * self.dt * np.cos(theta_new) * np.cos(phi_new)
        y_new = y + v * self.dt * np.sin(theta_new) * np.cos(phi_new)
        z_new = z + v * self.dt * np.sin(phi_new)
        x_new = np.clip(x_new, 0.0, self.env_size)
        y_new = np.clip(y_new, 0.0, self.env_size)
        z_new = np.clip(z_new, self.min_altitude, self.max_altitude)
        return np.array([x_new, y_new, z_new, theta_new, phi_new], dtype=np.float32)