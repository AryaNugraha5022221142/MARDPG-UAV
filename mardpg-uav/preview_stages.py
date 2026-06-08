import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mardpg_uav.train import CURRICULUM
from mardpg_uav.environment.uav_env import MultiUAVEnv

def plot_all_stages(save_path="stages_preview.png"):
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    env = MultiUAVEnv(cfg['environment'])
    
    stages_to_plot = list(range(len(CURRICULUM)))
    # 7 stages: we can plot in a 4x2 grid
    fig = plt.figure(figsize=(24, 32))
    fig.patch.set_facecolor('#FAFAFA')
    
    cmap = cm.viridis
    max_z = 60.0 # Most stages have max height around 60m

    for i, stage_idx in enumerate(stages_to_plot):
        ax = fig.add_subplot(4, 2, i + 1, projection='3d')
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
                
                # Map Z to color (Height-Based)
                Z_norm = z_grid / max_z
                colors = cmap(Z_norm)
                
                # Side faces
                ax.plot_surface(x_grid, y_grid, z_grid, facecolors=colors, shade=True, alpha=0.9, linewidth=0)
                
                # Top face (cap)
                rC = np.linspace(0, ob_rad, 5)
                thetaC2 = np.linspace(0, 2*np.pi, 20)
                r_grid_top, th_grid_top = np.meshgrid(rC, thetaC2)
                x_top = ob_pos[0] + r_grid_top * np.cos(th_grid_top)
                y_top = ob_pos[1] + r_grid_top * np.sin(th_grid_top)
                z_top = np.full_like(x_top, ob_pos[2] + ob_h/2)
                
                top_color = cmap((ob_pos[2] + ob_h/2) / max_z)
                ax.plot_surface(x_top, y_top, z_top, color=top_color, shade=True, alpha=0.9, linewidth=0)
                
            elif ob.type == 'sphere':
                # Dynamic threats
                ob_pos = ob.position
                ob_rad = ob.size[0]
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = ob_rad * np.outer(np.cos(u), np.sin(v)) + ob_pos[0]
                y = ob_rad * np.outer(np.sin(u), np.sin(v)) + ob_pos[1]
                z = ob_rad * np.outer(np.ones(np.size(u)), np.cos(v)) + ob_pos[2]
                
                ax.plot_surface(x, y, z, color='#E67E22', alpha=0.8, shade=True, label='Dynamic Threat')

        # Plot Goals and optional Trajectory
        for j in range(env.n_agents):
            goal = env.goals[j]
            start = env.agents_state[j, :3]
            
            # The red objects previously seen were goals. Let's make them red stars to avoid blocky spheres.
            # or fine high-resolution spheres
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            gr = 3.0 # slightly smaller goal radius
            gx = gr * np.outer(np.cos(u), np.sin(v)) + goal[0]
            gy = gr * np.outer(np.sin(u), np.sin(v)) + goal[1]
            gz = gr * np.outer(np.ones(np.size(u)), np.cos(v)) + goal[2]
            
            # Semi-transparent red sphere for the spatial boundary
            surf = ax.plot_surface(gx, gy, gz, color='#E74C3C', alpha=0.4, shade=True)
            # Add an explicitly defined legend proxy
            surf._edgecolors2d = surf._edgecolor3d
            surf._facecolors2d = surf._facecolor3d
            
            # And a star in the middle!
            ax.scatter(goal[0], goal[1], goal[2], color='#C0392B', s=150, marker='*', label='Goal' if j == 0 else "")
            
            # Trajectory
            n_waypoints = 15
            t = np.linspace(0, 1, n_waypoints)
            curve_x = np.sin(t * np.pi) * 15.0 * (1 if j % 2 == 0 else -1)
            curve_y = np.sin(t * np.pi) * 20.0 * (1 if j % 3 == 0 else -1)
            curve_z = np.sin(t * np.pi) * 10.0
            
            way_x = start[0] + (goal[0] - start[0]) * t + curve_x
            way_y = start[1] + (goal[1] - start[1]) * t + curve_y
            way_z = start[2] + (goal[2] - start[2]) * t + curve_z
            
            ax.plot(way_x, way_y, way_z, 'k--', marker='.', markersize=8, alpha=0.8, linewidth=1.5, label='UAV Trajectory' if j == 0 else "")

        # Environment styling
        ax.set_xlim([0, stage_cfg['env_size'][0]])
        ax.set_ylim([0, stage_cfg['env_size'][1]])
        ax.set_zlim([0, stage_cfg['env_size'][2]])
        
        ax.set_xlabel('X (m)', labelpad=15, fontweight='bold')
        ax.set_ylabel('Y (m)', labelpad=15, fontweight='bold')
        ax.set_zlabel('Z (m)', labelpad=15, fontweight='bold')
        
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"Stage {stage_idx}: {stage_cfg['name']}", fontweight='bold', fontsize=18, pad=20)
        
        # Grid customization
        ax.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
        
        # Legend (only set up proxies correctly to avoid errors)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Colorbar on the side of the whole figure
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, max_z])
    cbar = fig.colorbar(m, cax=cbar_ax)
    cbar.set_label("Obstacle Height (m)", rotation=270, labelpad=25, fontweight='bold', fontsize=16)

    # Instead of plt.tight_layout(), we use subplots_adjust to preserve the right space for cbar
    plt.subplots_adjust(wspace=0.2, hspace=0.3, right=0.88)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved all stages preview to {save_path}")

if __name__ == '__main__':
    plot_all_stages()
