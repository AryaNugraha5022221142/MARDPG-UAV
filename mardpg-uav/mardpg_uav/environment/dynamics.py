"""
UAV Kinematics.
Implements QuadcopterDynamics.
"""
import numpy as np

class QuadcopterDynamics:
    def __init__(self, v_max=3.0, dt=0.1, tau_v=0.3,
                 env_size=(50.0, 50.0, 60.0),
                 max_altitude=60.0, min_altitude=0.0,
                 action_bounds=(-3.0, 3.0)):
        self.v_max = v_max
        self.dt = dt
        self.alpha = dt / tau_v          # smoothing coefficient ∈ (0,1)
        assert 0 < self.alpha < 1, "tau_v must satisfy tau_v > dt"
        self.env_size = np.asarray(env_size, dtype=np.float32)
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.action_bounds = action_bounds  # (v_min, v_max) m/s

    def step(self, state: np.ndarray, v_cmd: np.ndarray,
             _unused_prev_applied: np.ndarray) -> tuple:
        """
        state:   [x, y, z, vx, vy, vz]
        v_cmd:   [vx_cmd, vy_cmd, vz_cmd] in [action_bounds]
        Returns: (next_state, v_applied)
        """
        v_cmd_clipped = np.clip(v_cmd,
                                self.action_bounds[0],
                                self.action_bounds[1])

        v_actual = state[3:6]
        v_next = v_actual + self.alpha * (v_cmd_clipped - v_actual)

        pos = state[:3]
        pos_next = pos + v_next * self.dt

        # Boundary clamping — also zero the velocity component at wall
        for axis in range(3):
            lo = (self.min_altitude if axis == 2 else 0.0)
            hi = (self.max_altitude if axis == 2 else self.env_size[axis])
            if pos_next[axis] < lo:
                pos_next[axis] = lo
                v_next[axis] = 0.0      # absorbing wall
            elif pos_next[axis] > hi:
                pos_next[axis] = hi
                v_next[axis] = 0.0

        next_state = np.concatenate([pos_next, v_next]).astype(np.float32)
        return next_state, v_next   # v_next is the "applied" action
