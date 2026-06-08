import yaml
import numpy as np
import matplotlib.pyplot as plt

# Enable strict LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mardpg_uav.train import CURRICULUM
from mardpg_uav.environment.uav_env import MultiUAVEnv

def plot_all_stages(save_path="stages_preview.png"):
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    env = MultiUAVEnv(cfg['environment'])
    
    stages_to_plot = list(range(len(CURRICULUM)))
    # 6 stages: we can plot in a 3x2 grid
    fig = plt.figure(figsize=(24, 28))
    fig.patch.set_facecolor('#FAFAFA')
    
    max_z = 60.0 # Most stages have max height around 60m

    all_handles, all_labels = [], []

    for i, stage_idx in enumerate(stages_to_plot):
        ax = fig.add_subplot(3, 2, i + 1, projection='3d')
        ax.set_facecolor('#FAFAFA')
        
        stage_cfg = CURRICULUM[stage_idx].copy()
        
        env.reset(stage_cfg)
        
        # Plot obstacles
        for ob in env.obstacles[:-6]:  # Skip walls
            if ob.type == 'cylinder':
                ob_pos = ob.position
                ob_rad = ob.size[0]
                ob_h = ob.size[1]
                
                # Cylinder generator
                zC = np.linspace(ob_pos[2] - ob_h/2, ob_pos[2] + ob_h/2, 20)
                thetaC = np.linspace(0, 2*np.pi, 20)
                theta_grid, z_grid = np.meshgrid(thetaC, zC)
                x_grid = ob_pos[0] + ob_rad * np.cos(theta_grid)
                y_grid = ob_pos[1] + ob_rad * np.sin(theta_grid)
                
                # Side faces
                ax.plot_surface(x_grid, y_grid, z_grid, color='#CCCCCC', shade=True, alpha=0.4, linewidth=0)
                
                # Top face (cap)
                rC = np.linspace(0, ob_rad, 5)
                thetaC2 = np.linspace(0, 2*np.pi, 20)
                r_grid_top, th_grid_top = np.meshgrid(rC, thetaC2)
                x_top = ob_pos[0] + r_grid_top * np.cos(th_grid_top)
                y_top = ob_pos[1] + r_grid_top * np.sin(th_grid_top)
                z_top = np.full_like(x_top, ob_pos[2] + ob_h/2)
                
                ax.plot_surface(x_top, y_top, z_top, color='#CCCCCC', shade=True, alpha=0.4, linewidth=0)
                
            elif ob.type == 'sphere':
                # Dynamic threats
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
        # Plot Goals and optional Trajectory
        for j in range(env.n_agents):
            goal = env.goals[j]
            start = env.agents_state[j, :3]
            color = uav_colors[j % len(uav_colors)]
            
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            gr = 3.0 # slightly smaller goal radius
            gx = gr * np.outer(np.cos(u), np.sin(v)) + goal[0]
            gy = gr * np.outer(np.sin(u), np.sin(v)) + goal[1]
            gz = gr * np.outer(np.ones(np.size(u)), np.cos(v)) + goal[2]
            
            # Semi-transparent red sphere for the spatial boundary
            ax.plot_surface(gx, gy, gz, color=color, alpha=0.3, shade=True)
            
            # And a star in the middle!
            scatter = ax.scatter(goal[0], goal[1], goal[2], color=color, s=150, marker='*')
            if f'UAV {j+1} Goal' not in all_labels:
                all_handles.append(scatter)
                all_labels.append(f'UAV {j+1} Goal')
            
            # Trajectory
            n_waypoints = 15
            t = np.linspace(0, 1, n_waypoints)
            curve_x = np.sin(t * np.pi) * 15.0 * (1 if j % 2 == 0 else -1)
            curve_y = np.sin(t * np.pi) * 20.0 * (1 if j % 3 == 0 else -1)
            curve_z = np.sin(t * np.pi) * 10.0
            
            way_x = start[0] + (goal[0] - start[0]) * t + curve_x
            way_y = start[1] + (goal[1] - start[1]) * t + curve_y
            way_z = start[2] + (goal[2] - start[2]) * t + curve_z
            
            line, = ax.plot(way_x, way_y, way_z, color=color, linestyle='-', linewidth=2.5)
            if f'UAV {j+1} Trajectory' not in all_labels:
                all_handles.append(line)
                all_labels.append(f'UAV {j+1} Trajectory')
            
            # Shadow projection on the floor
            ax.plot(way_x, way_y, np.zeros_like(way_z), color=color, linestyle=':', alpha=0.5, linewidth=1.5)

        # Environment styling
        ax.set_xlim([0, stage_cfg['env_size'][0]])
        ax.set_ylim([0, stage_cfg['env_size'][1]])
        ax.set_zlim([0, stage_cfg['env_size'][2]])
        
        ax.set_xlabel('X (m)', labelpad=15, fontweight='bold')
        ax.set_ylabel('Y (m)', labelpad=15, fontweight='bold')
        ax.set_zlabel('Z (m)', labelpad=15, fontweight='bold')
        
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"Stage {stage_idx+1}: {stage_cfg['name']}", fontweight='bold', fontsize=18, pad=20)
        
        # Grid customization
        ax.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')

    fig.legend(all_handles, all_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=14)

    plt.subplots_adjust(wspace=0.2, hspace=0.3, bottom=0.15)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved all stages preview to {save_path}")

if __name__ == '__main__':
    plot_all_stages()
