"""
Evaluation script for MARDPG-NAV.
Reference: Section 12 of blueprint.
"""
import yaml
import torch
import numpy as np
from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.utils.metrics import MetricsTracker


def evaluate(checkpoint_dir: str, config_path: str = "config/default.yaml",
             n_eval_episodes: int = 250, device: str = 'cpu'):
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
            obs_dim=env.obs_dim, action_dim=env.action_dim,
            hidden_dim=net_cfg['actor']['lstm_hidden'],
            lr_actor=algo_cfg['lr_actor'],
            lr_critic=algo_cfg['lr_critic'],
            tau=algo_cfg['tau'], gamma=algo_cfg['gamma'],
            gradient_clip=algo_cfg['gradient_clip'],
            device=device
        )
        ckpt = torch.load(f"{checkpoint_dir}/agent_{i}.pt", map_location=device)
        if 'actor_private' in ckpt:
            if i == 0:
                try:
                    shared_ckpt = torch.load(f"{checkpoint_dir}/shared_actor.pt", map_location=device)
                    agent.shared_extractor.load_state_dict(shared_ckpt['shared_actor'])
                except Exception as e:
                    print(f"Warning: Could not load shared actor: {e}")
            agent.actor.load_state_dict(ckpt['actor_private'], strict=False)
        else:
            agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
        agent._hard_update(agent.actor_target, agent.actor)
        agent._hard_update(agent.critic_target, agent.critic)
        agents.append(agent)
    
    # Share parameters
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    
    metrics = MetricsTracker()
    
    for ep in range(n_eval_episodes):
        obs = env.reset()
        
        for agent in agents:
            agent.reset_hidden(batch_size=1, eval_mode=True)
        
        episode_reward = 0
        path_history = [env.agents_state[:, :3].copy()]
        reached = [False] * n_agents
        collision = False
        
        for step in range(env_cfg['max_steps_per_episode']):
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(obs[i], evaluate=True)
                actions.append(action)
            
            obs, rewards, done, info = env.step(np.array(actions))
            episode_reward += sum(rewards)
            path_history.append(env.agents_state[:, :3].copy())
            
            if isinstance(info['reached'], np.ndarray):
                reached = info['reached'].tolist()
            else:
                reached = info['reached']
            
            if isinstance(info['collisions'], np.ndarray):
                collision = any(info['collisions'])
            else:
                collision = any(info['collisions']) if isinstance(info['collisions'], list) else info['collisions']
            
            if done:
                break
        
        metrics.record_episode(
            [episode_reward], step+1, reached, collision,
            [path_history[0][i] for i in range(n_agents)],
            [env.goals[i] for i in range(n_agents)],
            path_history
        )
    
    stats = metrics.get_stats()
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success Rate:  {stats['success_rate']:.2%}")
    print(f"Collision Rate: {stats['collision_rate']:.2%}")
    print(f"Trapped Rate:   {stats['trapped_rate']:.2%}")
    print(f"Avg Reward:     {stats['avg_reward']:.2f}")
    if 'path_efficiency' in stats:
        print(f"Path Efficiency: {stats['path_efficiency']:.3f}")
    print("=" * 50)
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint directory')
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--episodes', type=int, default=250)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.config, args.episodes, args.device)
