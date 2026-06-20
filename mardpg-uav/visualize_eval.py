"""
visualize_eval.py — improved evaluation visualization for MARDPG-UAV.

Drop-in replacement for visualize.py. Key differences are documented inline
and in the accompanying notes. Highlights:

  * FIXED cylinder rendering (old code drew cylinders 1.5x too tall, spanning
    z in [pos_z - h, pos_z + h] instead of [pos_z - h/2, pos_z + h/2]).
  * Correct 3D aspect ratio via set_box_aspect (arena is 100x100x60; default
    matplotlib distorts this, which distorts apparent climb/descent angles).
  * Time is encoded along each trajectory (alpha ramp start->end) so direction
    and temporal order are legible from a still image.
  * Colorblind-safe agent palette (Okabe-Ito), with red/vermillion RESERVED
    for hazards so Agent 5 is no longer confused with dynamic threats.
  * Dynamic obstacles are drawn along their SWEPT PATH, not only their final
    position (the old plot showed where a moving threat ended, implying empty
    space the UAV actually had to avoid).
  * Stage is selectable (old code hardcoded the hardest stage-7 scene, which
    is out-of-distribution for any checkpoint not trained that far — making
    healthy models look chaotic).
  * Two 2D coordination diagnostics that communicate swarm behavior far better
    than a single 3D view: pairwise minimum inter-UAV separation vs the safety
    floor, and per-agent distance-to-goal over time.
  * Optional animation (--animate) for true dynamics.

Usage:
    python visualize_eval.py --checkpoint checkpoints/final --stage 2
    python visualize_eval.py --checkpoint checkpoints/stage_1_cleared --stage 1 --animate
"""
import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.train import CURRICULUM

# Okabe-Ito: vermillion (#D55E00) and a second red are reserved for hazards,
# so no agent color collides with the dynamic-threat color.
AGENT_COLORS = ['#0072B2', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']
HAZARD_COLOR = '#D55E00'
OBSTACLE_GRAY = '#9AA0A6'


# ---------------------------------------------------------------------------
# Obstacle drawing helpers (geometry matches obstacles.py / rewards.py exactly)
# ---------------------------------------------------------------------------
def _draw_cylinder(ax, pos, r, h, color=OBSTACLE_GRAY, alpha=0.30):
    # Cylinder spans z in [pos_z - h/2, pos_z + h/2]; obstacles.py places the
    # center at h/2 so the true extent is [0, h]. (The old visualize.py used
    # [pos_z - h, pos_z + h] => 1.5x height, dipping below the floor.)
    z0, z1 = pos[2] - h / 2.0, pos[2] + h / 2.0
    theta = np.linspace(0, 2 * np.pi, 32)
    zc = np.linspace(z0, z1, 2)
    tg, zg = np.meshgrid(theta, zc)
    xg = pos[0] + r * np.cos(tg)
    yg = pos[1] + r * np.sin(tg)
    ax.plot_surface(xg, yg, zg, color=color, alpha=alpha, linewidth=0, shade=True)


def _draw_box(ax, pos, half, color=OBSTACLE_GRAY, alpha=0.30):
    x = [pos[0] - half[0], pos[0] + half[0]]
    y = [pos[1] - half[1], pos[1] + half[1]]
    z = [pos[2] - half[2], pos[2] + half[2]]
    xx, yy = np.meshgrid(x, y)
    for zz in z:
        ax.plot_surface(xx, yy, np.full_like(xx, zz), color=color, alpha=alpha, linewidth=0)
    xx, zz = np.meshgrid(x, z)
    for yy in y:
        ax.plot_surface(xx, np.full_like(xx, yy), zz, color=color, alpha=alpha, linewidth=0)
    yy, zz = np.meshgrid(y, z)
    for xx in x:
        ax.plot_surface(np.full_like(yy, xx), yy, zz, color=color, alpha=alpha, linewidth=0)


def _draw_sphere(ax, center, r, color, alpha=0.5):
    u = np.linspace(0, 2 * np.pi, 16)
    v = np.linspace(0, np.pi, 16)
    x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = r * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def _draw_static_obstacles(ax, env, max_z=60.0):
    # env.obstacles ends with 6 boundary walls; skip them.
    cmap = plt.cm.plasma
    for ob in env.obstacles[:-6]:
        # Color based on relative height
        h = ob.size[1] if ob.type == 'cylinder' else (ob.size[2] * 2 if ob.type == 'box' else ob.size[0] * 2)
        p_z = h / max_z
        c = cmap(np.clip(p_z, 0, 1))

        if ob.type == 'cylinder':
            _draw_cylinder(ax, ob.position, ob.size[0], ob.size[1], color=c, alpha=0.45)
        elif ob.type == 'box':
            _draw_box(ax, ob.position, ob.size, color=c, alpha=0.45)
        # dynamic spheres are drawn separately along their swept path


def _time_graded_segments(path_i):
    """Return (segments, rgba_with_alpha_ramp_indices) for a single agent path."""
    pts = path_i.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    return segs


def _alpha_ramp_colors(color, n, a0=0.20, a1=1.0):
    rgb = to_rgb(color)
    return np.array([(rgb[0], rgb[1], rgb[2], a) for a in np.linspace(a0, a1, n)])


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------
def _load_agents(checkpoint_dir, env, env_cfg, net_cfg, algo_cfg, device):
    n_agents = env_cfg['n_agents']
    agents = []
    for i in range(n_agents):
        ag = MARDPGAgent(
            agent_id=i, n_agents=n_agents,
            obs_dim=env.obs_dim, action_dim=env.action_dim,
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            tau=algo_cfg['tau'], gamma=algo_cfg['gamma'],
            burn_in=algo_cfg['burn_in'], device=device)
        try:
            ckpt = torch.load(f"{checkpoint_dir}/agent_{i}.pt", map_location=device)
            if 'actor_private' in ckpt:
                if i == 0:
                    sc = torch.load(f"{checkpoint_dir}/shared_actor.pt", map_location=device)
                    ag.shared_extractor.load_state_dict(sc['shared_actor'])
                ag.actor.load_state_dict(ckpt['actor_private'], strict=False)
            else:
                ag.actor.load_state_dict(ckpt['actor'])
            print(f"Loaded agent {i}")
        except Exception as e:
            print(f"[WARN] agent {i}: {e}. Using untrained weights.")
        agents.append(ag)
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    return agents


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Static multi-panel figure
# ---------------------------------------------------------------------------
def plot_static(env, env_cfg, path, dyn_path, dyn_r, reached, collided, goals,
                stage_name, out_path):
    n_agents = env_cfg['n_agents']
    ex, ey, ez = env_cfg['env_size']
    T = len(path)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')

    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    axtop = fig.add_subplot(2, 2, 2)
    axsep = fig.add_subplot(2, 2, 3)
    axgoal = fig.add_subplot(2, 2, 4)

    # ---- 3D isometric ----
    _draw_static_obstacles(ax3d, env, max_z=ez)
    if dyn_path is not None:
        for k in range(dyn_path.shape[1]):
            dp = dyn_path[:, k, :]
            ax3d.plot(dp[:, 0], dp[:, 1], dp[:, 2], color=HAZARD_COLOR,
                      lw=1.5, alpha=0.5, ls='--')
            ax3d.scatter(dp[0, 0], dp[0, 1], dp[0, 2], color=HAZARD_COLOR, marker='o', s=40, facecolors='none', linewidth=1.2)
            _draw_sphere(ax3d, dp[-1], dyn_r[k], HAZARD_COLOR, alpha=0.35)

    for i in range(n_agents):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        segs = _time_graded_segments(path[:, i, :])
        lc = Line3DCollection(segs, colors=_alpha_ramp_colors(c, len(segs)), linewidths=2.2)
        ax3d.add_collection3d(lc)
        ax3d.scatter(*path[0, i], color=c, marker='s', s=55, edgecolor='k', linewidth=0.5)
        end_marker = 'o' if reached[i] else ('X' if collided[i] else 'P')
        ax3d.scatter(*path[-1, i], color=c, marker=end_marker, s=70, edgecolor='k', linewidth=0.6)
        ax3d.scatter(*goals[i], color=c, marker='*', s=220, edgecolor='k', linewidth=0.6)
        ax3d.plot(path[:, i, 0], path[:, i, 1], np.zeros(T), color=c, ls=':', alpha=0.25, lw=1)

    ax3d.set_xlim(0, ex); ax3d.set_ylim(0, ey); ax3d.set_zlim(0, ez)
    ax3d.set_box_aspect((ex, ey, ez))         # <-- true proportions, not distorted
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_title('3D trajectories (square=start, star=goal, o=reached, X=collided)')

    # ---- top-down (clearest for formation/coordination) ----
    for i in range(n_agents):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        pts = path[:, i, :2].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, colors=_alpha_ramp_colors(c, len(segs)), linewidths=2.0)
        axtop.add_collection(lc)
        axtop.scatter(path[0, i, 0], path[0, i, 1], color=c, marker='s', s=45,
                      edgecolor='k', linewidth=0.5, label=f'Agent {i+1}')
        axtop.scatter(goals[i, 0], goals[i, 1], color=c, marker='*', s=180,
                      edgecolor='k', linewidth=0.5)
    cmap = plt.cm.plasma
    for ob in env.obstacles[:-6]:
        h = ob.size[1] if ob.type == 'cylinder' else (ob.size[2] * 2 if ob.type == 'box' else ob.size[0] * 2)
        p_z = h / ez
        c = cmap(np.clip(p_z, 0, 1))
        if ob.type == 'cylinder':
            axtop.add_patch(plt.Circle(ob.position[:2], ob.size[0], color=c, alpha=0.55))
        elif ob.type == 'box':
            axtop.add_patch(plt.Rectangle(ob.position[:2] - ob.size[:2],
                                          2 * ob.size[0], 2 * ob.size[1],
                                          color=c, alpha=0.55))
    if dyn_path is not None:
        for k in range(dyn_path.shape[1]):
            axtop.plot(dyn_path[:, k, 0], dyn_path[:, k, 1], color=HAZARD_COLOR, ls='--', alpha=0.5)
            axtop.scatter(dyn_path[0, k, 0], dyn_path[0, k, 1], color=HAZARD_COLOR, marker='o', facecolors='none', s=40, linewidths=1.2)
    axtop.set_xlim(0, ex); axtop.set_ylim(0, ey); axtop.set_aspect('equal')
    axtop.set_xlabel('X (m)'); axtop.set_ylabel('Y (m)')
    axtop.set_title('Top-down (X-Y) — formation & path crossing')
    axtop.legend(loc='upper right', fontsize=8, framealpha=0.9)
    axtop.grid(alpha=0.3)

    # ---- pairwise minimum inter-UAV separation vs safety floor ----
    floor = env_cfg.get('inter_uav_min_dist', 1.0)
    min_sep = np.full(T, np.inf)
    for t in range(T):
        for a in range(n_agents):
            for b in range(a + 1, n_agents):
                min_sep[t] = min(min_sep[t], np.linalg.norm(path[t, a] - path[t, b]))
    axsep.plot(min_sep, color='#333333', lw=1.8)
    axsep.axhline(floor, color=HAZARD_COLOR, ls='--', lw=1.5,
                  label=f'safety floor ({floor:.1f} m)')
    axsep.set_xlabel('step'); axsep.set_ylabel('min pairwise distance (m)')
    axsep.set_title('Swarm separation — dips toward the floor = near-misses')
    axsep.legend(fontsize=8); axsep.grid(alpha=0.3)

    # ---- per-agent distance to goal over time ----
    for i in range(n_agents):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        d2g = np.linalg.norm(path[:, i, :] - goals[i], axis=1)
        axgoal.plot(d2g, color=c, lw=1.8, label=f'Agent {i+1}')
    axgoal.axhline(env_cfg.get('goal_threshold', 1.0), color='gray', ls='--', lw=1,
                   label='goal threshold')
    axgoal.set_xlabel('step'); axgoal.set_ylabel('distance to goal (m)')
    axgoal.set_title('Progress — flat tail = trapped; sharp stop = reached/collided')
    axgoal.legend(fontsize=8); axgoal.grid(alpha=0.3)

    n_reached = int(reached.sum()); n_col = int(collided.sum())
    fig.suptitle(f'MARDPG-UAV eval | {stage_name} | {T} steps | '
                 f'reached {n_reached}/{n_agents}, collided {n_col}/{n_agents}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure -> {out_path}")


# ---------------------------------------------------------------------------
# Focused 3D-only trajectory render (the side panels are derivable from CSV).
# Legend is split into two parts: agent identity+outcome, and marker semantics.
# ---------------------------------------------------------------------------
def plot_trajectory_top_down(env, env_cfg, render, title, out_path):
    n_agents = env_cfg['n_agents']
    ex, ey, ez = env_cfg['env_size']
    path = render['path']
    dyn_path = render['dyn_path']
    dyn_r = render['dyn_r']
    reached = render['reached']
    collided = render['collided']
    goals = render['goals']
    T = len(path)

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    axtop = fig.add_subplot(111)

    # background circles/boxes
    cmap = plt.cm.plasma
    for ob in env.obstacles[:-6]:
        h = ob.size[1] if ob.type == 'cylinder' else (ob.size[2] * 2 if ob.type == 'box' else ob.size[0] * 2)
        p_z = h / ez
        c = cmap(np.clip(p_z, 0, 1))
        if ob.type == 'cylinder':
            circle = plt.Circle((ob.position[0], ob.position[1]), ob.size[0], color=c, alpha=0.55, linewidth=0)
            axtop.add_patch(circle)
        elif ob.type == 'box':
            box = plt.Rectangle((ob.position[0] - ob.size[0], ob.position[1] - ob.size[1]), 
                                ob.size[0]*2, ob.size[1]*2, color=c, alpha=0.55, linewidth=0)
            axtop.add_patch(box)

    for i in range(n_agents):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        # path
        axtop.plot(path[:, i, 0], path[:, i, 1], color=c, lw=1.8, alpha=0.6)
        # straight line from start to goal
        axtop.plot([path[0, i, 0], goals[i, 0]], [path[0, i, 1], goals[i, 1]], color=c, lw=1.2, alpha=0.3, ls='--')
        # start
        axtop.scatter(*path[0, i, :2], color=c, marker='s', s=80, edgecolor='k', linewidth=0.5)
        # goal
        axtop.scatter(*goals[i, :2], color=c, marker='*', s=300, edgecolor='k', linewidth=0.5)
        # end
        em = 'o' if reached[i] else ('X' if collided[i] else 'P')
        axtop.scatter(*path[-1, i, :2], color=c, marker=em, s=120, edgecolor='k', linewidth=1.5, zorder=5)

    if dyn_path is not None:
        for k in range(dyn_path.shape[1]):
            axtop.plot(dyn_path[:, k, 0], dyn_path[:, k, 1], color=HAZARD_COLOR, ls='--', alpha=0.5, lw=2)
            axtop.scatter(dyn_path[0, k, 0], dyn_path[0, k, 1], color=HAZARD_COLOR, marker='o', facecolors='none', s=40, linewidths=1.2)

    axtop.set_xlim(0, ex); axtop.set_ylim(0, ey); axtop.set_aspect('equal')
    axtop.set_xlabel('X (m)'); axtop.set_ylabel('Y (m)')
    axtop.set_title(title, fontsize=12, fontweight='bold', pad=14)
    axtop.grid(alpha=0.3)

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_trajectory_3d(env, env_cfg, render, title, out_path, elev=22, azim=-58):
    n_agents = env_cfg['n_agents']
    ex, ey, ez = env_cfg['env_size']
    path = render['path']
    dyn_path = render['dyn_path']
    dyn_r = render['dyn_r']
    reached = render['reached']
    collided = render['collided']
    goals = render['goals']
    T = len(path)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    _draw_static_obstacles(ax, env, max_z=ez)
    if dyn_path is not None:
        for k in range(dyn_path.shape[1]):
            dp = dyn_path[:, k, :]
            ax.plot(dp[:, 0], dp[:, 1], dp[:, 2], color=HAZARD_COLOR,
                    lw=1.6, alpha=0.6, ls='--')
            ax.scatter(dp[0, 0], dp[0, 1], dp[0, 2], color=HAZARD_COLOR, marker='o', s=40, facecolors='none', linewidth=1.2)
            _draw_sphere(ax, dp[-1], dyn_r[k], HAZARD_COLOR, alpha=0.35)

    agent_handles = []
    for i in range(n_agents):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        segs = _time_graded_segments(path[:, i, :])
        lc = Line3DCollection(segs, colors=_alpha_ramp_colors(c, len(segs)), linewidths=2.4)
        ax.add_collection3d(lc)
        em = 'o' if reached[i] else ('X' if collided[i] else 'P')
        ax.scatter(*path[0, i], color=c, marker='s', s=60,
                   edgecolor='k', linewidth=0.6, depthshade=False)
        ax.scatter(*path[-1, i], color=c, marker=em, s=85,
                   edgecolor='k', linewidth=0.7, depthshade=False)
        ax.scatter(*goals[i], color=c, marker='*', s=260,
                   edgecolor='k', linewidth=0.7, depthshade=False)
        # straight line from start to goal
        ax.plot([path[0, i, 0], goals[i, 0]], [path[0, i, 1], goals[i, 1]], [path[0, i, 2], goals[i, 2]], color=c, lw=1.2, alpha=0.3, ls='--')
        ax.plot(path[:, i, 0], path[:, i, 1], np.zeros(T), color=c, ls=':', alpha=0.2, lw=1)
        status = 'reached' if reached[i] else ('collided' if collided[i] else 'timeout')
        agent_handles.append(Line2D([0], [0], color=c, lw=2.6, marker=em,
                                    markersize=9, markeredgecolor='k',
                                    label=f'Agent {i+1} \u2014 {status}'))

    ax.set_xlim(0, ex); ax.set_ylim(0, ey); ax.set_zlim(0, ez)
    ax.set_box_aspect((ex, ey, ez))
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=14)

    # Legend part 1: agent identity (color) + outcome (end marker).
    leg1 = ax.legend(handles=agent_handles, loc='upper left',
                     bbox_to_anchor=(-0.02, 1.0), fontsize=9, framealpha=0.92,
                     title='Agent (color = identity)')
    ax.add_artist(leg1)

    # Legend part 2: marker / line semantics (neutral grey).
    sem = [
        Line2D([0], [0], color='0.35', marker='s', ls='None', markersize=9,
               markeredgecolor='k', label='start'),
        Line2D([0], [0], color='0.35', marker='*', ls='None', markersize=13,
               markeredgecolor='k', label='goal'),
        Line2D([0], [0], color='0.35', marker='o', ls='None', markersize=9,
               markeredgecolor='k', label='end (reached)'),
        Line2D([0], [0], color='0.35', marker='X', ls='None', markersize=9,
               markeredgecolor='k', label='end (collided)'),
        Line2D([0], [0], color='0.7', lw=6, alpha=0.5, label='obstacle'),
    ]
    if dyn_path is not None:
        sem.append(Line2D([0], [0], color=HAZARD_COLOR, lw=1.8, ls='--',
                          label='dynamic threat path'))
    ax.legend(handles=sem, loc='upper right', bbox_to_anchor=(1.02, 1.0),
              fontsize=9, framealpha=0.92, title='Markers')

    fig.text(0.5, 0.015,
             'Line opacity increases start \u2192 end (direction of travel).  '
             'Dotted line = ground (X\u2013Y) projection.',
             ha='center', fontsize=9, color='0.3')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Optional animation
# ---------------------------------------------------------------------------
def animate(env, env_cfg, path, dyn_path, dyn_r, goals, stage_name, out_path, tail=40):
    from matplotlib.animation import FuncAnimation
    n_agents = env_cfg['n_agents']
    ex, ey, ez = env_cfg['env_size']
    T = len(path)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    _draw_static_obstacles(ax, env, max_z=ez)
    ax.set_xlim(0, ex); ax.set_ylim(0, ey); ax.set_zlim(0, ez)
    ax.set_box_aspect((ex, ey, ez))
    ax.view_init(elev=22, azim=-58)
    for i in range(n_agents):
        ax.scatter(*goals[i], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                   marker='*', s=200, edgecolor='k', linewidth=0.5)

    lines = [ax.plot([], [], [], color=AGENT_COLORS[i % len(AGENT_COLORS)], lw=2)[0]
             for i in range(n_agents)]
    heads = [ax.plot([], [], [], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                     marker='o', ms=6)[0] for i in range(n_agents)]

    # Dynamic obstacles
    dyn_lines = []
    dyn_sphere_colls = []
    if dyn_path is not None and dyn_path.shape[1] > 0:
        for k in range(dyn_path.shape[1]):
            # Path line
            dyn_lines.append(ax.plot([], [], [], color=HAZARD_COLOR, lw=1.5, ls='--', alpha=0.5)[0])
            dyn_sphere_colls.append(None)

    def update(t):
        lo = max(0, t - tail)
        for i in range(n_agents):
            seg = path[lo:t + 1, i, :]
            lines[i].set_data(seg[:, 0], seg[:, 1]); lines[i].set_3d_properties(seg[:, 2])
            heads[i].set_data([path[t, i, 0]], [path[t, i, 1]])
            heads[i].set_3d_properties([path[t, i, 2]])
            
        ret = lines + heads
        
        if dyn_path is not None and dyn_path.shape[1] > 0:
            for k in range(dyn_path.shape[1]):
                seg = dyn_path[lo:t + 1, k, :]
                dyn_lines[k].set_data(seg[:, 0], seg[:, 1])
                dyn_lines[k].set_3d_properties(seg[:, 2])
                
                # Update sphere shape
                if dyn_sphere_colls[k] is not None:
                    dyn_sphere_colls[k].remove()
                dyn_sphere_colls[k] = _draw_sphere(ax, dyn_path[t, k, :], dyn_r[k], color=HAZARD_COLOR, alpha=0.35)
                
                ret.append(dyn_lines[k])
                
        ax.set_title(f'{stage_name} — step {t}/{T - 1}')
        return ret

    anim = FuncAnimation(fig, update, frames=T, interval=40, blit=False)
    try:
        anim.save(out_path, writer='ffmpeg', dpi=120)
    except Exception:
        anim.save(out_path.replace('.mp4', '.gif'), writer='pillow', dpi=90)
    print(f"Saved animation -> {out_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints')
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--stage', type=int, default=len(CURRICULUM),
                   help='Curriculum stage scene to evaluate in (1..N). '
                        'Use the stage the checkpoint was actually trained to.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='trajectory_eval.png')
    p.add_argument('--animate', action='store_true')
    args = p.parse_args()

    from mardpg_uav.eval_rollout import load_agents
    
    if not os.path.exists(args.config):
        fb = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
        if os.path.exists(fb):
            args.config = fb
    
    agents, cfg = load_agents(args.checkpoint, args.config, args.device)
    env_cfg = cfg['environment']

    env = MultiUAVEnv(env_cfg)

    from mardpg_uav.eval_rollout import run_eval, make_learned_act_fn
    
    stage_cfg = dict(CURRICULUM[max(0, min(args.stage, len(CURRICULUM)) - 1)])
    stage_cfg['max_steps'] = env_cfg.get('max_steps_per_episode', stage_cfg.get('max_steps', 1500))
    stage_name = stage_cfg.get('name', f'stage {args.stage}')

    act_fn, on_start = make_learned_act_fn(agents, env)
    _, m = run_eval(env, stage_cfg, act_fn, n_episodes=1, base_seed=42, on_episode_start=on_start, collect_paths=True)
    
    info = m.episodes[0]['info']
    path = np.array(m.episodes[0]['path_history'])
    
    dyn_path = info.get('dyn_path', None)
    dyn_r = info.get('dyn_r', [])
    reached = info.get('reached', np.zeros(env.n_agents))
    collided = info.get('collisions', np.zeros(env.n_agents))
    goals = env.goals.copy()

    plot_static(env, env_cfg, path, dyn_path, dyn_r, reached, collided, goals,
                stage_name, args.out)
    if args.animate:
        animate(env, env_cfg, path, dyn_path, dyn_r, goals, stage_name,
                args.out.replace('.png', '.mp4'))


if __name__ == "__main__":
    main()
