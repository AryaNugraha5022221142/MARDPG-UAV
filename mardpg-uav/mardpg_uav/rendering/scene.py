"""Shared, side-effect-free scene primitives.

These helpers draw obstacles/goals onto a given matplotlib Axes. They are the
one place obstacle drawing lives, so the live viewer, the video frames and the
static publication plots all render an identical world (previously this logic
was copy-pasted and drifted between visualize_eval.py and the render patch).
"""

from __future__ import annotations

import numpy as np

# Colour-blind-safe palette (Okabe-Ito), matching the original figures.
AGENT_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
HAZARD_COLOR = "#D55E00"
OBSTACLE_GRAY = "#9AA0A6"


def agent_color(i: int) -> str:
    return AGENT_COLORS[i % len(AGENT_COLORS)]


def static_obstacles(env):
    """The static obstacles only (drop the 6 boundary walls appended in reset)."""
    obs = getattr(env, "obstacles", []) or []
    return obs[:-6] if len(obs) >= 6 else obs


def draw_cylinder(ax, pos, r, h, color=OBSTACLE_GRAY, alpha=0.35, resolution=24):
    z0, z1 = pos[2] - h / 2.0, pos[2] + h / 2.0
    theta = np.linspace(0, 2 * np.pi, resolution)
    zc = np.linspace(z0, z1, 2)
    tg, zg = np.meshgrid(theta, zc)
    xg = pos[0] + r * np.cos(tg)
    yg = pos[1] + r * np.sin(tg)
    ax.plot_surface(xg, yg, zg, color=color, alpha=alpha, linewidth=0, shade=True)


def draw_box(ax, pos, half, color=OBSTACLE_GRAY, alpha=0.35):
    x = [pos[0] - half[0], pos[0] + half[0]]
    y = [pos[1] - half[1], pos[1] + half[1]]
    z = [pos[2] - half[2], pos[2] + half[2]]
    xx, yy = np.meshgrid(x, y)
    for zz in z:
        ax.plot_surface(xx, yy, np.full_like(xx, zz), color=color, alpha=alpha, linewidth=0)
    xx, zz = np.meshgrid(x, z)
    for yy_ in y:
        ax.plot_surface(xx, np.full_like(xx, yy_), zz, color=color, alpha=alpha, linewidth=0)
    yy, zz = np.meshgrid(y, z)
    for xx_ in x:
        ax.plot_surface(np.full_like(yy, xx_), yy, zz, color=color, alpha=alpha, linewidth=0)


def draw_sphere(ax, center, r, color, alpha=0.5, resolution=12):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = r * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def draw_static_obstacles_3d(ax, env, max_z=60.0, cmap_name="plasma"):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    for ob in static_obstacles(env):
        if ob.type == "cylinder":
            h = float(ob.size[1])
        elif ob.type == "box":
            h = float(ob.size[2]) * 2
        else:
            h = float(ob.size[0]) * 2
        c = cmap(np.clip(h / max_z, 0, 1))
        if ob.type == "cylinder":
            draw_cylinder(ax, ob.position, float(ob.size[0]), float(ob.size[1]), color=c, alpha=0.45)
        elif ob.type == "box":
            draw_box(ax, ob.position, ob.size, color=c, alpha=0.45)


def draw_static_obstacles_2d(ax, env, max_z=60.0, cmap_name="plasma"):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    for ob in static_obstacles(env):
        if ob.type == "cylinder":
            h = float(ob.size[1])
        elif ob.type == "box":
            h = float(ob.size[2]) * 2
        else:
            h = float(ob.size[0]) * 2
        c = cmap(np.clip(h / max_z, 0, 1))
        if ob.type == "cylinder":
            ax.add_patch(plt.Circle(ob.position[:2], float(ob.size[0]), color=c, alpha=0.55, linewidth=0))
        elif ob.type == "box":
            ax.add_patch(plt.Rectangle(
                (ob.position[0] - ob.size[0], ob.position[1] - ob.size[1]),
                float(ob.size[0]) * 2, float(ob.size[1]) * 2, color=c, alpha=0.55, linewidth=0))
