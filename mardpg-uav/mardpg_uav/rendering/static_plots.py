"""Publication-quality static trajectory figures (Task 4: PNG rendering).

Migrated and de-duplicated from the old scripts/visualize_eval.py. Every
function here saves ONE figure and closes it (the old code double-closed and
occasionally leaked figures). DPI is caller-controlled so a thesis run can
request 300 DPI without editing source.
"""

from __future__ import annotations

import os
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .scene import (
    agent_color, AGENT_COLORS, HAZARD_COLOR,
    draw_static_obstacles_3d, draw_static_obstacles_2d, draw_sphere,
)


def _segments(path_i):
    pts = path_i.reshape(-1, 1, path_i.shape[-1])
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def _alpha_ramp(color, n, a0=0.2, a1=1.0):
    rgb = to_rgb(color)
    return np.array([(rgb[0], rgb[1], rgb[2], a) for a in np.linspace(a0, a1, n)])


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


def plot_trajectory_3d(env, env_cfg, rnd, title, out_path, dpi=200, elev=22, azim=-58):
    import matplotlib.pyplot as plt
    _ensure_dir(out_path)
    n_agents = env_cfg["n_agents"]
    ex, ey, ez = env_cfg["env_size"]
    path = np.asarray(rnd["path"])
    goals = np.asarray(rnd["goals"])
    reached = np.asarray(rnd["reached"])
    collided = np.asarray(rnd["collided"])
    dyn_path = rnd.get("dyn_path")
    dyn_r = rnd.get("dyn_r", [])
    T = len(path)

    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    draw_static_obstacles_3d(ax, env, max_z=ez)

    if dyn_path is not None:
        for k in range(dyn_path.shape[1]):
            dp = dyn_path[:, k, :]
            ax.plot(dp[:, 0], dp[:, 1], dp[:, 2], color=HAZARD_COLOR, lw=1.6, alpha=0.6, ls="--")
            draw_sphere(ax, dp[0], dyn_r[k], HAZARD_COLOR, alpha=0.35)

    handles = []
    for i in range(n_agents):
        c = agent_color(i)
        lc = Line3DCollection(_segments(path[:, i, :]),
                              colors=_alpha_ramp(c, T - 1), linewidths=2.4)
        ax.add_collection3d(lc)
        em = "o" if reached[i] else "X" if collided[i] else "P"
        ax.scatter(*path[0, i], color=c, marker="s", s=60, edgecolor="k", linewidth=0.6, depthshade=False)
        ax.scatter(*path[-1, i], color=c, marker=em, s=85, edgecolor="k", linewidth=0.7, depthshade=False)
        ax.scatter(*goals[i], color=c, marker="*", s=260, edgecolor="k", linewidth=0.7, depthshade=False)
        ax.plot(path[:, i, 0], path[:, i, 1], np.zeros(T), color=c, ls=":", alpha=0.2, lw=1)
        status = "reached" if reached[i] else "collided" if collided[i] else "timeout"
        handles.append(Line2D([0], [0], color=c, lw=2.6, marker=em, markersize=9,
                              markeredgecolor="k", label=f"Agent {i + 1} - {status}"))

    ax.set_xlim(0, ex); ax.set_ylim(0, ey); ax.set_zlim(0, ez)
    ax.set_box_aspect((ex, ey, ez))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trajectory_top_down(env, env_cfg, rnd, title, out_path, dpi=200):
    import matplotlib.pyplot as plt
    _ensure_dir(out_path)
    n_agents = env_cfg["n_agents"]
    ex, ey, ez = env_cfg["env_size"]
    path = np.asarray(rnd["path"])
    goals = np.asarray(rnd["goals"])
    reached = np.asarray(rnd["reached"])
    collided = np.asarray(rnd["collided"])
    dyn_path = rnd.get("dyn_path")

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    draw_static_obstacles_2d(ax, env, max_z=ez)
    for i in range(n_agents):
        c = agent_color(i)
        ax.plot(path[:, i, 0], path[:, i, 1], color=c, lw=1.8, alpha=0.7)
        ax.plot([path[0, i, 0], goals[i, 0]], [path[0, i, 1], goals[i, 1]],
                color=c, lw=1.2, alpha=0.3, ls="--")
        ax.scatter(*path[0, i, :2], color=c, marker="s", s=80, edgecolor="k", linewidth=0.5)
        ax.scatter(*goals[i, :2], color=c, marker="*", s=300, edgecolor="k", linewidth=0.5)
        em = "o" if reached[i] else "X" if collided[i] else "P"
        ax.scatter(*path[-1, i, :2], color=c, marker=em, s=120, edgecolor="k", linewidth=1.5, zorder=5)
    if dyn_path is not None:
        for k in range(dyn_path.shape[1]):
            ax.plot(dyn_path[:, k, 0], dyn_path[:, k, 1], color=HAZARD_COLOR, ls="--", alpha=0.5, lw=2)
    ax.set_xlim(0, ex); ax.set_ylim(0, ey); ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.grid(alpha=0.3)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary(env, env_cfg, rnd, stage_name, out_path, dpi=200):
    """Four-panel diagnostic: 3D, top-down, swarm separation, goal progress."""
    import matplotlib.pyplot as plt
    _ensure_dir(out_path)
    n_agents = env_cfg["n_agents"]
    ex, ey, ez = env_cfg["env_size"]
    path = np.asarray(rnd["path"])
    goals = np.asarray(rnd["goals"])
    reached = np.asarray(rnd["reached"])
    collided = np.asarray(rnd["collided"])
    dyn_path = rnd.get("dyn_path")
    dyn_r = rnd.get("dyn_r", [])
    T = len(path)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axtop = fig.add_subplot(2, 2, 2)
    axsep = fig.add_subplot(2, 2, 3)
    axgoal = fig.add_subplot(2, 2, 4)

    draw_static_obstacles_3d(ax3d, env, max_z=ez)
    if dyn_path is not None:
        for k in range(dyn_path.shape[1]):
            dp = dyn_path[:, k, :]
            ax3d.plot(dp[:, 0], dp[:, 1], dp[:, 2], color=HAZARD_COLOR, lw=1.5, alpha=0.5, ls="--")
            draw_sphere(ax3d, dp[0], dyn_r[k], HAZARD_COLOR, alpha=0.35)
    for i in range(n_agents):
        c = agent_color(i)
        lc = Line3DCollection(_segments(path[:, i, :]), colors=_alpha_ramp(c, T - 1), linewidths=2.2)
        ax3d.add_collection3d(lc)
        ax3d.scatter(*path[0, i], color=c, marker="s", s=55, edgecolor="k", linewidth=0.5)
        em = "o" if reached[i] else "X" if collided[i] else "P"
        ax3d.scatter(*path[-1, i], color=c, marker=em, s=70, edgecolor="k", linewidth=0.6)
        ax3d.scatter(*goals[i], color=c, marker="*", s=220, edgecolor="k", linewidth=0.6)
    ax3d.set_xlim(0, ex); ax3d.set_ylim(0, ey); ax3d.set_zlim(0, ez)
    ax3d.set_box_aspect((ex, ey, ez))
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_title("3D trajectories (square=start, star=goal, o=reached, X=collided)")

    draw_static_obstacles_2d(axtop, env, max_z=ez)
    for i in range(n_agents):
        c = agent_color(i)
        segs = _segments(path[:, i, :2])
        axtop.add_collection(LineCollection(segs, colors=_alpha_ramp(c, T - 1), linewidths=2.0))
        axtop.scatter(path[0, i, 0], path[0, i, 1], color=c, marker="s", s=45,
                      edgecolor="k", linewidth=0.5, label=f"Agent {i + 1}")
        axtop.scatter(goals[i, 0], goals[i, 1], color=c, marker="*", s=180, edgecolor="k", linewidth=0.5)
    axtop.set_xlim(0, ex); axtop.set_ylim(0, ey); axtop.set_aspect("equal")
    axtop.set_xlabel("X (m)"); axtop.set_ylabel("Y (m)")
    axtop.set_title("Top-down (X-Y) - formation & path crossing")
    axtop.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axtop.grid(alpha=0.3)

    floor = env_cfg.get("inter_uav_min_dist", 1.0)
    min_sep = np.full(T, np.inf)
    for t in range(T):
        for a in range(n_agents):
            for b in range(a + 1, n_agents):
                min_sep[t] = min(min_sep[t], np.linalg.norm(path[t, a] - path[t, b]))
    axsep.plot(min_sep, color="#333333", lw=1.8)
    axsep.axhline(floor, color=HAZARD_COLOR, ls="--", lw=1.5, label=f"safety floor ({floor:.1f} m)")
    axsep.set_xlabel("step"); axsep.set_ylabel("min pairwise distance (m)")
    axsep.set_title("Swarm separation - dips toward the floor = near-misses")
    axsep.legend(fontsize=8); axsep.grid(alpha=0.3)

    for i in range(n_agents):
        c = agent_color(i)
        d2g = np.linalg.norm(path[:, i, :] - goals[i], axis=1)
        axgoal.plot(d2g, color=c, lw=1.8, label=f"Agent {i + 1}")
    axgoal.axhline(env_cfg.get("goal_threshold", 1.0), color="gray", ls="--", lw=1, label="goal threshold")
    axgoal.set_xlabel("step"); axgoal.set_ylabel("distance to goal (m)")
    axgoal.set_title("Progress - flat tail = trapped; sharp stop = reached/collided")
    axgoal.legend(fontsize=8); axgoal.grid(alpha=0.3)

    n_reached, n_col = int(reached.sum()), int(collided.sum())
    fig.suptitle(f"MARDPG-UAV eval | {stage_name} | {T} steps | "
                 f"reached {n_reached}/{n_agents}, collided {n_col}/{n_agents}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
