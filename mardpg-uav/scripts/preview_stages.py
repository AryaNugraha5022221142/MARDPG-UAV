import os
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.train import CURRICULUM
from mardpg_uav.environment.uav_env import MultiUAVEnv

def plot_all_stages(save_path='stages_preview.png'):
    import os
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'default.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    env = MultiUAVEnv(cfg['environment'])
    stages_to_plot = list(range(len(CURRICULUM)))
    fig = plt.figure(figsize=(24, 38))
    fig.patch.set_facecolor('#FAFAFA')
    max_z = 60.0
    (all_handles, all_labels) = ([], [])
    for (i, stage_idx) in enumerate(stages_to_plot):
        ax = fig.add_subplot(4, 2, i + 1, projection='3d')
        ax.set_facecolor('#FAFAFA')
        stage_cfg = CURRICULUM[stage_idx].copy()
        env.reset(stage_cfg)
        for ob in env.obstacles[:-6]:
            if ob.type == 'cylinder':
                ob_pos = ob.position
                ob_rad = ob.sizeob_h = ob.size
                zC = np.linspace(ob_pos - ob_h / 2, ob_pos + ob_h / 2, 20)
                thetaC = np.linspace(0, 2 * np.pi, 20)
                (theta_grid, z_grid) = np.meshgrid(thetaC, zC)
                x_grid = ob_pos + ob_rad * np.cos(theta_grid)
                y_grid = ob_pos + ob_rad * np.sin(theta_grid)
                ax.plot_surface(x_grid, y_grid, z_grid, color='#CCCCCC', shade=True, alpha=0.4, linewidth=0)
                rC = np.linspace(0, ob_rad, 5)
                thetaC2 = np.linspace(0, 2 * np.pi, 20)
                (r_grid_top, th_grid_top) = np.meshgrid(rC, thetaC2)
                x_top = ob_pos + r_grid_top * np.cos(th_grid_top)
                y_top = ob_pos + r_grid_top * np.sin(th_grid_top)
                z_top = np.full_like(x_top, ob_pos + ob_h / 2)
                ax.plot_surface(x_top, y_top, z_top, color='#CCCCCC', shade=True, alpha=0.4, linewidth=0)
            elif ob.type == 'sphere':
                ob_pos = ob.position
                ob_rad = ob.size[0]
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = ob_rad * np.outer(np.cos(u), np.sin(v)) + ob_pos[0]
                y = ob_rad * np.outer(np.sin(u), np.sin(v)) + ob_pos[1]
                z = ob_rad * np.outer(np.ones(np.size(u)), np.cos(v)) + ob_pos[2]
                surf = ax.plot_surface(x, y, z, color='#E74C3C', alpha=0.8, shade=True)
                if 'Dynamic Threat' not in all_labels:
                    surf._edgecolors2d = surf._edgecolor3d
                    surf._facecolors2d = surf._facecolor3d
                    all_handles.append(surf)
                    all_labels.append('Dynamic Threat')
        uav_colors = ['#2980B9', '#27AE60', '#8E44AD', '#D35400', '#C0392B']
        for j in range(env.n_agents):
            goal = env.goals[j]
            start = env.agents_state[j, :3]
            color = uav_colors[j % len(uav_colors)]
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            gr = 3.0
            gx = gr * np.outer(np.cos(u), np.sin(v)) + goal[0]
            gy = gr * np.outer(np.sin(u), np.sin(v)) + goal[1]
            gz = gr * np.outer(np.ones(np.size(u)), np.cos(v)) + goal[2]
            ax.plot_surface(gx, gy, gz, color=color, alpha=0.3, shade=True)
            scatter = ax.scatter(goal[0], goal[1], goal[2], color=color, s=150, marker='*')
            if f'UAV {j + 1} Goal' not in all_labels:
                all_handles.append(scatter)
                all_labels.append(f'UAV {j + 1} Goal')
            n_waypoints = 15
            t = np.linspace(0, 1, n_waypoints)
            curve_x = np.sin(t * np.pi) * 15.0 * (1 if j % 2 == 0 else -1)
            curve_y = np.sin(t * np.pi) * 20.0 * (1 if j % 3 == 0 else -1)
            curve_z = np.sin(t * np.pi) * 10.0
            way_x = start + (goal - start) * t + curve_x
            way_y = start + (goal - start) * t + curve_y
            way_z = start + (goal - start) * t + curve_z
            (line,) = ax.plot(way_x, way_y, way_z, color=color, linestyle='-', linewidth=2.5)
            if f'UAV {j + 1} Trajectory' not in all_labels:
                all_handles.append(line)
                all_labels.append(f'UAV {j + 1} Trajectory')
            ax.plot(way_x, way_y, np.zeros_like(way_z), color=color, linestyle=':', alpha=0.5, linewidth=1.5)
        ax.set_xlim([0, stage_cfg['env_size']])
        ax.set_ylim([0, stage_cfg['env_size']])
        ax.set_zlim([0, stage_cfg['env_size']])
        ax.set_xlabel('X (m)', labelpad=15, fontweight='bold')
        ax.set_ylabel('Y (m)', labelpad=15, fontweight='bold')
        ax.set_zlabel('Z (m)', labelpad=15, fontweight='bold')
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"Stage {stage_idx + 1}: {stage_cfg['name']}", fontweight='bold', fontsize=18, pad=20)
        ax.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
    fig.legend(all_handles, all_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=14)
    plt.subplots_adjust(wspace=0.2, hspace=0.3, bottom=0.15)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
if __name__ == '__main__':
    plot_all_stages()