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
    cfg = yaml.safe_load(open(config_path))
    env_cfg = cfg['environment']
    net_cfg = cfg['network']
    algo_cfg = cfg['algorithm']
    
    env = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    
    # Load agents
    agents = []
    for i in range(n_agents):
        agent = MARDPGAgent(
            agent_id=i, 
            n_agents=n_agents,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            lr_actor=algo_cfg['lr_actor'],
            lr_critic=algo_cfg['lr_critic'],
            tau=algo_cfg['tau'], gamma=algo_cfg['gamma'],
            burn_in=algo_cfg['burn_in'],
            gradient_clip=algo_cfg['gradient_clip'],
            device=device
        )
        try:
            ckpt = torch.load(f"{checkpoint_dir}/agent_{i}.pt", map_location=device)
            
            # FIX: Handle split shared/private dictionaries
            if 'actor_private' in ckpt:
                if i == 0:
                    shared_ckpt = torch.load(f"{checkpoint_dir}/shared_actor.pt", map_location=device)
                    agent.shared_extractor.load_state_dict(shared_ckpt['shared_actor'])
                agent.actor.load_state_dict(ckpt['actor_private'], strict=False)
            else:
                agent.actor.load_state_dict(ckpt['actor'])
                
            print(f"Loaded agent {i} weights")
        except Exception as e:
            print(f"Warning: Failed to load agent {i}: {e}. Using untrained weights.")
        agents.append(agent)
    
    # FIX: Enforce parameter sharing for inference
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    
       
    stage_cfg = {'env_size': [100.0, 100.0, 60.0], 'static_obs': 16,
                 'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
                 'min_sep': 40.0, 'max_steps': 1500}
    obs = env.reset(stage_cfg)
    for agent in agents:
        agent.actor.eval()
        agent.reset_hidden(batch_size=1, eval_mode=True)
    
    # Simpan histori [step, agent, koordinat]
    # Batch function
    def select_actions_batch_eval(agents, obs_all, v_max, agent_done, prev_actions, action_dim=2):
        actions = []
        for i, agent in enumerate(agents):
            if agent_done[i]:
                actions.append(np.zeros(action_dim))
            else:
                action = agent.select_action(obs_all[i], prev_actions[i], evaluate=True)
                action = np.clip(action, -agent.actor.action_bound, agent.actor.action_bound)
                actions.append(action)
        return actions
        
    prev_actions = [np.zeros(env.action_dim, dtype=np.float32) for _ in range(n_agents)]
    path_history = [env.agents_state[:, :3].copy()]
    for step in range(env_cfg['max_steps_per_episode']):
        actions = select_actions_batch_eval(agents, obs, env_cfg.get('v_max', 3.0), env.agent_done, prev_actions, env.action_dim)
        
        obs, rewards, done, info = env.step(actions)
        prev_actions = actions.copy()
        
        path_history.append(env.agents_state[:, :3].copy())
        
        if done:
            break
            
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
