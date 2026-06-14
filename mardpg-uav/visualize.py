import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.algorithm.mardpg import MARDPGAgent

def visualize(checkpoint_dir, config_path="config/default.yaml", device="cpu"):
    import os
    if not os.path.exists(config_path):
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        if os.path.exists(fallback):
            config_path = fallback
    from mardpg_uav.eval_rollout import load_agents
    agents, cfg = load_agents(checkpoint_dir, config_path, device)
    env_cfg = cfg['environment']
    
    env = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    
    from mardpg_uav.eval_rollout import run_eval, make_learned_act_fn
       
    stage_cfg = {'env_size': [100.0, 100.0, 60.0], 'static_obs': 16,
                 'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
                 'min_sep': 40.0, 'max_steps': 1500}
    obs = env.reset(stage_cfg)
    
    act_fn, on_start = make_learned_act_fn(agents, env)
    _, m = run_eval(env, stage_cfg, act_fn, n_episodes=1, base_seed=42, on_episode_start=on_start, collect_paths=True)
    
    path_history = m.episodes[0]['path_history']
    path_history = np.array(path_history) # shape: (T, N, 3)

    # Plotting 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#2980B9', '#27AE60', '#8E44AD', '#D35400', '#C0392B']
    
    # Plot obstacle
    print(f"Total simulated obstacles: {len(env.obstacles)}")
    # Skip the last 6 obstacles as they are the arena boundary walls
    for ob in env.obstacles[:-6]:
        if ob.type == 'sphere':
            ob_pos = ob.position
            ob_rad = ob.size[0]
            
            # Gambar bola rintangan sederhana
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = ob_rad * np.outer(np.cos(u), np.sin(v)) + ob_pos[0]
            y = ob_rad * np.outer(np.sin(u), np.sin(v)) + ob_pos[1]
            z = ob_rad * np.outer(np.ones(np.size(u)), np.cos(v)) + ob_pos[2]
            
            ax.plot_surface(x, y, z, color='#E74C3C', alpha=0.8, shade=True)
            
        elif ob.type == 'box':
            ob_pos = ob.position
            hl = ob.size
            # Draw box limits
            x = [ob_pos[0]-hl[0], ob_pos[0]+hl[0]]
            y = [ob_pos[1]-hl[1], ob_pos[1]+hl[1]]
            z = [ob_pos[2]-hl[2], ob_pos[2]+hl[2]]
            
            xx, yy = np.meshgrid(x, y)
            for zz in z: ax.plot_surface(xx, yy, np.full_like(xx, zz), color='#CCCCCC', alpha=0.4, shade=True)
            xx, zz = np.meshgrid(x, z)
            for yy in y: ax.plot_surface(xx, np.full_like(xx, yy), zz, color='#CCCCCC', alpha=0.4, shade=True)
            yy, zz = np.meshgrid(y, z)
            for xx in x: ax.plot_surface(np.full_like(yy, xx), yy, zz, color='#CCCCCC', alpha=0.4, shade=True)
            
        elif ob.type == 'cylinder':
            ob_pos = ob.position
            ob_rad = ob.size[0]
            ob_h = ob.size[1]
            
            # Cylinder up to height
            zC = np.linspace(max(0, ob_pos[2] - ob_h), ob_pos[2] + ob_h, 5)
            thetaC = np.linspace(0, 2*np.pi, 20)
            theta_grid, z_grid = np.meshgrid(thetaC, zC)
            x_grid = ob_pos[0] + ob_rad * np.cos(theta_grid)
            y_grid = ob_pos[1] + ob_rad * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, color='#CCCCCC', alpha=0.4, shade=True)
        
    for i in range(n_agents):
        c = colors[i % len(colors)]
        xs = path_history[:, i, 0]
        ys = path_history[:, i, 1]
        zs = path_history[:, i, 2]
        
        # Plot trajectory line
        ax.plot(xs, ys, zs, color=c, linewidth=2.5, label=f'Agent {i+1}')
        # Setup shadow
        ax.plot(xs, ys, np.zeros_like(zs), color=c, linestyle=':', alpha=0.5, linewidth=1.5)
        # Plot Start
        ax.scatter(xs[0], ys[0], zs[0], color=c, marker='o', s=50)
        # Plot Goal
        ax.scatter(env.goals[i, 0], env.goals[i, 1], env.goals[i, 2], color=c, marker='*', s=150)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'UAV Trajectories (Completed in {len(path_history)} steps)')
    ax.set_xlim(0, env_cfg['env_size'][0])
    ax.set_ylim(0, env_cfg['env_size'][1])
    ax.set_zlim(0, env_cfg['env_size'][2])
    ax.legend()
    
    # Save the output
    plt.savefig('trajectory_3d.png', dpi=300)
    print("Trajectory plot saved to 'trajectory_3d.png'")
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints', help='Path to checkpoint directory (default: checkpoints)')
    parser.add_argument('--config', default='config/default.yaml')
    args = parser.parse_args()
    
    visualize(args.checkpoint, args.config)
