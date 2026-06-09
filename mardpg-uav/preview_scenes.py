import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mardpg_uav.environment.uav_env import MultiUAVEnv

def plot_scene(ax, env, title):
    print(f"Plotting {title} - {len(env.obstacles)} total objects (including walls)")
    # Obstacles definition
    for ob in env.obstacles[:-6]:  # Skip the last 6 border walls
        if ob.type == 'sphere':
            ob_pos = ob.position
            ob_rad = ob.size[0]
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = ob_rad * np.outer(np.cos(u), np.sin(v)) + ob_pos[0]
            y = ob_rad * np.outer(np.sin(u), np.sin(v)) + ob_pos[1]
            z = ob_rad * np.outer(np.ones(np.size(u)), np.cos(v)) + ob_pos[2]
            ax.plot_surface(x, y, z, color='gray', alpha=0.6, shade=True)
        elif ob.type == 'box':
            ob_pos = ob.position
            hl = ob.size
            x = [ob_pos[0]-hl[0], ob_pos[0]+hl[0]]
            y = [ob_pos[1]-hl[1], ob_pos[1]+hl[1]]
            z = [ob_pos[2]-hl[2], ob_pos[2]+hl[2]]
            xx, yy = np.meshgrid(x, y)
            for zz in z: ax.plot_surface(xx, yy, np.full_like(xx, zz), color='#555555', alpha=1.0, shade=True)
            xx, zz = np.meshgrid(x, z)
            for yy in y: ax.plot_surface(xx, np.full_like(xx, yy), zz, color='#555555', alpha=1.0, shade=True)
            yy, zz = np.meshgrid(y, z)
            for xx in x: ax.plot_surface(np.full_like(yy, xx), yy, zz, color='#555555', alpha=1.0, shade=True)
        elif ob.type == 'cylinder':
            ob_pos = ob.position
            ob_rad = ob.size[0]
            ob_h = ob.size[1]
            zC = np.linspace(ob_pos[2]-ob_h/2, ob_pos[2]+ob_h/2, 10)
            thetaC = np.linspace(0, 2*np.pi, 20)
            theta_grid, z_grid = np.meshgrid(thetaC, zC)
            x_grid = ob_pos[0] + ob_rad * np.cos(theta_grid)
            y_grid = ob_pos[1] + ob_rad * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.6, shade=True)
            
    ax.set_title(title, pad=0)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, 60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def main():
    import os
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "default.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    env = MultiUAVEnv(cfg['environment'])
    
    fig = plt.figure(figsize=(15, 12))
    scenes = [
        (1, "Scene 1: Square Columns"),
        (2, "Scene 2: Cylinders"),
        (3, "Scene 3: Forest (Thin Cylinders)"),
        (4, "Scene 4: Circular Rings")
    ]
    
    stage_cfgs = {
        1: {'env_size': [100., 100., 60.], 'static_obs': 4,  'min_sep': 20., 'max_steps': 500},
        2: {'env_size': [100., 100., 60.], 'static_obs': 8,  'min_sep': 30., 'max_steps': 500},
        3: {'env_size': [100., 100., 60.], 'static_obs': 12, 'min_sep': 40., 'max_steps': 500},
        4: {'env_size': [100., 100., 60.], 'static_obs': 16, 'min_sep': 40., 'max_steps': 500},
    }
    
    for i, (scene_type, title) in enumerate(scenes):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        env.reset(stage_cfgs[scene_type])
        plot_scene(ax, env, title)
        
    plt.tight_layout()
    plt.savefig("scenes_preview.png", dpi=200, bbox_inches='tight')
    print("Saved preview to scenes_preview.png")

if __name__ == "__main__":
    main()
