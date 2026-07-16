with open("mardpg-uav/mardpg_uav/rendering/live.py", "r") as f:
    content = f.read()

new_content = """\"\"\"LiveRenderer: smooth on-screen real-time visualisation.

Task 2 ("Fix evaluation rendering") root cause: the patched real-time path
called, every single timestep, `plt.pause(0.001)` on a full 3D axes with a
fresh `scatter(*goal)` added each step. plt.pause forces a full-figure
redraw + GUI event flush, `scatter` leaks a new artist per step, and 3D axes
are expensive to fully redraw -- together giving the freezing, plummeting FPS
described.

Fixes here:
  * Build the figure/axes/obstacles/goals ONCE (see step()).
  * Reuse one line artist per agent for trail + one for head; only update data.
  * Use the canvas blit path: cache the static background, then on each step
    restore it and re-draw only the agent artists. This avoids re-rendering
    the static obstacles every frame.
  * Throttle GUI event processing with flush_events()/start_event_loop rather
    than plt.pause, and cap redraw rate so the sim stays responsive.

If the active backend is Agg (headless), the LiveRenderer becomes a no-op: it
never tries to open a window, so the same eval code runs on the VM unchanged.
\"\"\"
from __future__ import annotations

import time
import logging
import numpy as np

from .backend import is_interactive_backend
from .scene import agent_color, draw_static_obstacles_3d

log = logging.getLogger("mardpg.render")


class LiveRenderer:
    def __init__(self, env, env_cfg, max_fps: float = 30.0, tail: int = 60):
        self.enabled = is_interactive_backend()
        self.env_cfg = env_cfg
        self.n_agents = int(env_cfg["n_agents"])
        self.tail = tail
        self.min_dt = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_draw = 0.0
        self._history = [[] for _ in range(self.n_agents)]
        self.fig = None
        self.ax = None
        self._bg = None
        self.trails = []
        self.heads = []

        if not self.enabled:
            log.info("LiveRenderer disabled (headless backend); realtime view skipped.")
            return

        import matplotlib.pyplot as plt
        plt.ion()
        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        
        self.trails, self.heads = [], []
        for i in range(self.n_agents):
            c = agent_color(i)
            (tr,) = self.ax.plot([], [], [], color=c, lw=2.0)
            (hd,) = self.ax.plot([], [], [], color=c, marker="o", ms=6, ls="")
            self.trails.append(tr)
            self.heads.append(hd)
            
        self.fig.show()

        if hasattr(env, 'goals') and env.goals is not None and len(env.goals) > 0:
            self._setup_scene(env)

    def _setup_scene(self, env):
        if not self.enabled or self.ax is None:
            return
            
        self.ax.clear()
        ex, ey, ez = self.env_cfg["env_size"]
        self.ax.set_xlim(0, ex)
        self.ax.set_ylim(0, ey)
        self.ax.set_zlim(0, ez)
        self.ax.set_box_aspect((ex, ey, ez))
        self.ax.view_init(elev=22, azim=-58)

        draw_static_obstacles_3d(self.ax, env, max_z=ez)

        if hasattr(env, 'goals') and env.goals is not None:
            for i in range(self.n_agents):
                if i < len(env.goals):
                    self.ax.scatter(*env.goals[i], color=agent_color(i), marker="*",
                                    s=180, edgecolor="k", linewidth=0.5)

        # We need to re-add trails and heads since ax.clear() removed them
        self.trails, self.heads = [], []
        for i in range(self.n_agents):
            c = agent_color(i)
            (tr,) = self.ax.plot([], [], [], color=c, lw=2.0)
            (hd,) = self.ax.plot([], [], [], color=c, marker="o", ms=6, ls="")
            self.trails.append(tr)
            self.heads.append(hd)

        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def reset(self, env):
        self._history = [[] for _ in range(self.n_agents)]
        if self.enabled:
            self._setup_scene(env)

    def step(self, env):
        \"\"\"Push the current agent positions and refresh the window.\"\"\"
        if not self.enabled:
            return

        pos = env.agents_state[:, :3]
        for i in range(self.n_agents):
            if not env.agent_done[i]:
                self._history[i].append(pos[i].copy())

        now = time.time()
        if now - self._last_draw < self.min_dt:
            return
        self._last_draw = now
        
        if self._bg is None:
            return

        self.fig.canvas.restore_region(self._bg)
        for i in range(self.n_agents):
            hist = self._history[i]
            if not hist:
                continue
            arr = np.asarray(hist[-self.tail:])
            self.trails[i].set_data(arr[:, 0], arr[:, 1])
            self.trails[i].set_3d_properties(arr[:, 2])
            self.heads[i].set_data([arr[-1, 0]], [arr[-1, 1]])
            self.heads[i].set_3d_properties([arr[-1, 2]])
            self.ax.draw_artist(self.trails[i])
            self.ax.draw_artist(self.heads[i])
            
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def close(self):
        if not self.enabled or self.fig is None:
            return
        import matplotlib.pyplot as plt
        plt.close(self.fig)
        self.fig = None
"""

with open("mardpg-uav/mardpg_uav/rendering/live.py", "w") as f:
    f.write(new_content)
