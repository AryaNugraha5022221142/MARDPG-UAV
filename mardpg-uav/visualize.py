import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.algorithm.mardpg import MARDPGAgent

def visualize(checkpoint_dir, config_path="config/default.yaml", device="cpu"):
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
            agent_id=i, n_agents=n_agents,
            hidden_dim=net_cfg['actor']['lstm_hidden'],
            lr_actor=algo_cfg['lr_actor'],
            lr_critic=algo_cfg['lr_critic'],
            tau=algo_cfg['tau'], gamma=algo_cfg['gamma'],
            burn_in=algo_cfg['burn_in'],
            gradient_clip=algo_cfg['gradient_clip'],
            device=device
        )
        try:
            ckpt = torch.load(f"{checkpoint_dir}/agent_{i}.pt", map_location=device)
            agent.actor.load_state_dict(ckpt['actor'])
            print(f"Loaded agent {i} weights")
        except FileNotFoundError:
            print(f"Warning: Checkpoint for agent {i} not found. Using untrained weights.")
        agents.append(agent)
    
       
    obs = env.reset()
    for agent in agents:
        agent.reset_hidden(batch_size=1, eval_mode=True)
    
    # Simpan histori [step, agent, koordinat]
    path_history = [env.agents_state[:, :3].copy()]
    
    for step in range(env_cfg['max_steps_per_episode']):
        actions = []
        for i, agent in enumerate(agents):
            action = agent.select_action(obs[i], evaluate=True)
            actions.append(action)
        
        obs, rewards, done, info = env.step(np.array(actions))
        path_history.append(env.agents_state[:, :3].copy())
        
        if done:
            break
            
    path_history = np.array(path_history) # shape: (T, N, 3)

    # Plotting 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Plot obstacle
    for obs_cfg in env_cfg['obstacles']:
        ob_pos = obs_cfg['position']
        ob_rad = obs_cfg['radius']
        
        # Gambar bola rintangan sederhana
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = ob_rad * np.outer(np.cos(u), np.sin(v)) + ob_pos[0]
        y = ob_rad * np.outer(np.sin(u), np.sin(v)) + ob_pos[1]
        z = ob_rad * np.outer(np.ones(np.size(u)), np.cos(v)) + ob_pos[2]
        
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)
        
    for i in range(n_agents):
        c = colors[i % len(colors)]
        xs = path_history[:, i, 0]
        ys = path_history[:, i, 1]
        zs = path_history[:, i, 2]
        
        # Plot trajectory line
        ax.plot(xs, ys, zs, color=c, label=f'Agent {i}')
        # Plot Start
        ax.scatter(xs[0], ys[0], zs[0], color=c, marker='o', s=50)
        # Plot Goal
        ax.scatter(env.goals[i, 0], env.goals[i, 1], env.goals[i, 2], color=c, marker='*', s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'UAV Trajectories (Completed in {len(path_history)} steps)')
    ax.set_xlim(0, env_cfg['arena_size'][0])
    ax.set_ylim(0, env_cfg['arena_size'][1])
    ax.set_zlim(0, env_cfg['arena_size'][2])
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
