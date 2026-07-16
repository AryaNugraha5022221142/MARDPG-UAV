"""SceneFrameRenderer: turn a trajectory into a stream of RGB frames.

This is what feeds the video recorder and the optional per-frame PNG dump. It
is deliberately *offscreen* (Agg canvas), so it works identically on a
headless VM and a local machine, and it never opens a window.

Performance model -- the fix for "rendering is unusably slow":
  * The figure, the 3D axes, the static obstacles, and every per-agent line /
    head / trail artist are created ONCE in __init__.
  * render_frame(t) only calls set_data / set_3d_properties on existing
    artists and draws the canvas once. No figure or artist is recreated per
    frame, which is what the old plt.pause loop did implicitly.
  * A frame is read straight out of the Agg buffer as a numpy array; we never
    round-trip through savefig()/PNG for video frames.
"""

from __future__ import annotations

import numpy as np

from .scene import (
    agent_color, HAZARD_COLOR, draw_static_obstacles_3d, draw_sphere,
)


class SceneFrameRenderer:
    """Reusable offscreen renderer for one scene's trajectory.

    Args:
        env: the environment (used only for its static obstacle geometry).
        env_cfg: environment config dict (needs 'env_size', 'n_agents').
        rnd: render bundle dict with keys path[T,N,3], goals[N,3],
             reached[N], collided[N], dyn_path[T,K,3] or None, dyn_r[K].
        width, height, dpi: frame geometry.
        tail: number of trailing steps kept visible behind each agent.
    """

    def __init__(self, env, env_cfg, rnd, width=1280, height=960, dpi=100, tail=60):
        import matplotlib.pyplot as plt

        self.path = np.asarray(rnd["path"])
        self.goals = np.asarray(rnd["goals"])
        self.reached = np.asarray(rnd["reached"])
        self.collided = np.asarray(rnd["collided"])
        self.dyn_path = rnd.get("dyn_path")
        self.dyn_r = rnd.get("dyn_r", [])
        self.T = self.path.shape[0]
        self.n_agents = int(env_cfg["n_agents"])
        self.tail = tail
        ex, ey, ez = env_cfg["env_size"]

        # figsize in inches so width/height px come out exactly at this dpi.
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(0, ex)
        self.ax.set_ylim(0, ey)
        self.ax.set_zlim(0, ez)
        self.ax.set_box_aspect((ex, ey, ez))
        self.ax.view_init(elev=22, azim=-58)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # Static world + goals: drawn ONCE.
        draw_static_obstacles_3d(self.ax, env, max_z=ez)
        for i in range(self.n_agents):
            self.ax.scatter(*self.goals[i], color=agent_color(i), marker="*",
                            s=200, edgecolor="k", linewidth=0.5, depthshade=False)

        # Per-agent artists, created ONCE and updated in place.
        self.trails = []
        self.heads = []
        for i in range(self.n_agents):
            c = agent_color(i)
            (trail,) = self.ax.plot([], [], [], color=c, lw=2.2, alpha=0.9)
            (head,) = self.ax.plot([], [], [], color=c, marker="o", ms=6, ls="")
            self.trails.append(trail)
            self.heads.append(head)

        # Dynamic obstacle path lines (spheres are re-drawn per frame since a
        # 3D surface artist cannot be cheaply repositioned; kept minimal).
        self.dyn_lines = []
        self._dyn_spheres = []
        if self.dyn_path is not None and np.asarray(self.dyn_path).shape[1] > 0:
            for _ in range(self.dyn_path.shape[1]):
                (dl,) = self.ax.plot([], [], [], color=HAZARD_COLOR, lw=1.5, ls="--", alpha=0.6)
                self.dyn_lines.append(dl)
                self._dyn_spheres.append(None)

        self.title = self.ax.set_title("")
        self.fig.canvas.draw()  # prime the Agg buffer once

    def render_frame(self, t: int) -> np.ndarray:
        """Update artists for timestep t and return an (H, W, 3) uint8 array."""
        t = int(np.clip(t, 0, self.T - 1))
        lo = max(0, t - self.tail)
        for i in range(self.n_agents):
            seg = self.path[lo:t + 1, i, :]
            self.trails[i].set_data(seg[:, 0], seg[:, 1])
            self.trails[i].set_3d_properties(seg[:, 2])
            self.heads[i].set_data([self.path[t, i, 0]], [self.path[t, i, 1]])
            self.heads[i].set_3d_properties([self.path[t, i, 2]])

        if self.dyn_lines:
            for k, dl in enumerate(self.dyn_lines):
                seg = self.dyn_path[lo:t + 1, k, :]
                dl.set_data(seg[:, 0], seg[:, 1])
                dl.set_3d_properties(seg[:, 2])
                if self._dyn_spheres[k] is not None:
                    self._dyn_spheres[k].remove()
                r = self.dyn_r[k] if k < len(self.dyn_r) else 1.0
                self._dyn_spheres[k] = draw_sphere(
                    self.ax, self.dyn_path[t, k, :], r, color=HAZARD_COLOR, alpha=0.35)

        self.title.set_text(f"step {t}/{self.T - 1}")
        self.fig.canvas.draw()
        return self._buffer_to_rgb()

    def _buffer_to_rgb(self) -> np.ndarray:
        """Read the current canvas as an (H, W, 3) uint8 RGB array.

        buffer_rgba() is the portable path across matplotlib versions
        (tostring_rgb was removed in 3.10); we drop the alpha channel.
        """
        canvas = self.fig.canvas
        w, h = canvas.get_width_height()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)
        return buf[:, :, :3].copy()

    def frame_size(self):
        """(width, height) in pixels of the frames this renderer emits."""
        return self.fig.canvas.get_width_height()

    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)
