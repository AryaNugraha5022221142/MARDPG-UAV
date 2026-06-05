import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mardpg_uav.train import CURRICULUM
from mardpg_uav.environment.uav_env import MultiUAVEnv

def plot_scene(ax, env, title):
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
            
            color = 'red' if getattr(ob, 'velocity', None) is not None else 'gray'
            ax.plot_surface(x, y, z, color=color, alpha=0.6, shade=True)
            
        elif ob.type == 'box':
            ob_pos = ob.position
            hl = ob.size
            x = [ob_pos[0]-hl[0], ob_pos[0]+hl[0]]
            y = [ob_pos[1]-hl[1], ob_pos[1]+hl[1]]
            z = [ob_pos[2]-hl[2], ob_pos[2]+hl[2]]
            xx, yy = np.meshgrid(x, y)
            for zz in z: ax.plot_surface(xx, yy, np.full_like(xx, zz), color='gray', alpha=0.6, shade=True)
            xx, zz = np.meshgrid(x, z)
            for yy in y: ax.plot_surface(xx, np.full_like(xx, yy), zz, color='gray', alpha=0.6, shade=True)
            yy, zz = np.meshgrid(y, z)
            for xx in x: ax.plot_surface(np.full_like(yy, xx), yy, zz, color='gray', alpha=0.6, shade=True)
            
    ax.set_title(title, pad=0)
    ax.set_xlim(0, env.cfg['env_size'][0])
    ax.set_ylim(0, env.cfg['env_size'][1])
    ax.set_zlim(0, env.cfg['env_size'][2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def main():
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    env = MultiUAVEnv(cfg['environment'])
    
    # We will preview Stages 1, 3, 5, 6
    stages_to_plot = [1, 3, 5, 6]
    fig = plt.figure(figsize=(20, 16))
    
    for i, idx in enumerate(stages_to_plot):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        stage_cfg = CURRICULUM[idx]
        env.reset(stage_cfg)
        plot_scene(ax, env, f"Stage {idx}: {stage_cfg['name']}\n(Scale: {stage_cfg['env_size'][0]}m)")
        
    plt.tight_layout()
    plt.savefig("stages_preview.png", dpi=200, bbox_inches='tight')
    print("Saved preview to stages_preview.png")

if __name__ == "__main__":
    main()
