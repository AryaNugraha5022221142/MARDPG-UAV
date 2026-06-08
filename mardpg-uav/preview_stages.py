import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mardpg_uav.train import CURRICULUM
from mardpg_uav.environment.uav_env import MultiUAVEnv

def plot_single_scene(save_path="stages_preview.png"):
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    env = MultiUAVEnv(cfg['environment'])
    
    # We will preview a dense stage (Stage 6) customized for the plot
    stage_idx = 6
    stage_cfg = CURRICULUM[stage_idx].copy()
    
    # Specific dimensions as requested
    stage_cfg['env_size'] = [200.0, 200.0, 100.0]
    stage_cfg['static_obs'] = 30 # Nice dense scene
    
    env.reset(stage_cfg)
    
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#FAFAFA')
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_facecolor('#FAFAFA')
    
    # Set continuous colormap
    cmap = cm.viridis
    
    max_z = stage_cfg['env_size'][2]
    
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
            # The top height is roughly ob_pos[2]+ob_h/2. Let's color solid by total height of the obstacle!
            # The prompt says "where the RGB color is determined by the obstacle's height (Z-axis value)"
            top_height = ob_h # Or total Z extent. ob_h is the height. So color by height!
            obs_color = cmap(top_height / max_z)
            
            # Alternatively use a gradient across the cylinder:
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
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x = ob_rad * np.outer(np.cos(u), np.sin(v)) + ob_pos[0]
            y = ob_rad * np.outer(np.sin(u), np.sin(v)) + ob_pos[1]
            z = ob_rad * np.outer(np.ones(np.size(u)), np.cos(v)) + ob_pos[2]
            
            ax.plot_surface(x, y, z, color='#E67E22', alpha=0.8, shade=True)

    # Plot Goals and optional Trajectory
    # "Add a distinct red cylinder (or semi-transparent red sphere) at the goal location"
    for i in range(env.n_agents):
        goal = env.goals[i]
        start = env.agents_state[i, :3]
        
        # Red Sphere Goal
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        gr = 4.0 # goal radius
        gx = gr * np.outer(np.cos(u), np.sin(v)) + goal[0]
        gy = gr * np.outer(np.sin(u), np.sin(v)) + goal[1]
        gz = gr * np.outer(np.ones(np.size(u)), np.cos(v)) + goal[2]
        
        ax.plot_surface(gx, gy, gz, color='#E74C3C', alpha=0.6, shade=True)
        
        # Fake structured trajectory like the reference style
        n_waypoints = 15
        t = np.linspace(0, 1, n_waypoints)
        # Add smooth curve to the path
        curve_x = np.sin(t * np.pi) * 20.0 * (1 if i % 2 == 0 else -1)
        curve_y = np.sin(t * np.pi) * 30.0 * (1 if i % 3 == 0 else -1)
        curve_z = np.sin(t * np.pi) * 15.0
        
        way_x = start[0] + (goal[0] - start[0]) * t + curve_x
        way_y = start[1] + (goal[1] - start[1]) * t + curve_y
        way_z = start[2] + (goal[2] - start[2]) * t + curve_z
        
        ax.plot(way_x, way_y, way_z, 'k--', marker='*', markersize=8, alpha=0.8, linewidth=1.5, label='UAV Trajectory' if i == 0 else "")

    # Environment styling
    ax.set_xlim([0, stage_cfg['env_size'][0]])
    ax.set_ylim([0, stage_cfg['env_size'][1]])
    ax.set_zlim([0, stage_cfg['env_size'][2]])
    
    ax.set_xlabel('X (m)', labelpad=15, fontweight='bold')
    ax.set_ylabel('Y (m)', labelpad=15, fontweight='bold')
    ax.set_zlabel('Z (m)', labelpad=15, fontweight='bold')
    
    ax.view_init(elev=25, azim=-60)
    ax.set_title("UAV Simulation Environment", fontweight='bold', fontsize=18, pad=30)
    
    # Grid customization
    ax.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
    
    # Colorbar
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, max_z])
    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label("Obstacle Height (m)", rotation=270, labelpad=20, fontweight='bold')
    
    # Optional legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved highly styled preview to {save_path}")

if __name__ == '__main__':
    plot_single_scene()
