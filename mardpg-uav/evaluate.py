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


def select_actions_batch_eval(agents, obs_all, v_max, agent_done, prev_actions, action_dim=2):
    actions = []
    for i, agent in enumerate(agents):
        if agent_done[i]:
            actions.append(np.zeros(action_dim))
        else:
            action = agent.select_action(obs_all[i], prev_actions[i], evaluate=True)
            action = np.clip(action, -agent.actor.action_bound, agent.actor.action_bound)
            actions.append(action)
    return np.array(actions)

def evaluate(checkpoint_dir: str, config_path: str = "config/default.yaml",
             n_eval_episodes: int = 250, device: str = 'cpu', static_obs: int = 16):
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
            agent_id=i, n_agents=n_agents,
            obs_dim=env.obs_dim, action_dim=2,
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            device=device
        )
        ckpt = torch.load(f"{checkpoint_dir}/agent_{i}.pt", map_location=device)
        if 'actor_private' in ckpt:
            if i == 0:
                try:
                    shared_ckpt = torch.load(f"{checkpoint_dir}/shared_actor.pt", map_location=device)
                    agent.shared_extractor.load_state_dict(shared_ckpt['shared_actor'])
                except Exception as e:
                    print(f"Error: Could not load shared actor: {e}")
                    raise SystemExit(1)
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
    
    final_stage_cfg = {
        'env_size': [100.0, 100.0, 60.0], 'static_obs': static_obs,
        'min_sep': 40.0, 'max_steps': 1500
    }
    
    for ep in range(n_eval_episodes):
        obs = env.reset(final_stage_cfg)
        
        for agent in agents:
            agent.actor.eval()
            agent.reset_hidden(batch_size=1, eval_mode=True)
        
        episode_reward = 0
        path_history = [env.agents_state[:, :3].copy()]
        reached = [False] * n_agents
        per_agent_collided = [False] * n_agents
        
        prev_actions = [np.zeros(env.action_dim, dtype=np.float32) for _ in range(n_agents)]

        for step in range(env_cfg['max_steps_per_episode']):
            actions = select_actions_batch_eval(agents, obs, env_cfg.get('v_max', 3.0), env.agent_done, prev_actions, env.action_dim)
            
            obs, rewards, done, info = env.step(actions)
            prev_actions = actions.copy()
            
            episode_reward += sum(rewards)
            path_history.append(env.agents_state[:, :3].copy())
            
            if isinstance(info['reached'], np.ndarray):
                reached = info['reached'].tolist()
            else:
                reached = info['reached']
            
            if isinstance(info['collisions'], np.ndarray):
                per_agent_collided = info['collisions'].tolist()
            else:
                per_agent_collided = info['collisions']
            
            if done:
                break
        
        metrics.record_episode(
            length=step + 1,
            info={'reached': np.array(reached), 'collisions': np.array(per_agent_collided),
                  'dyn_collisions': info.get('dyn_collisions', np.zeros(n_agents)), 
                  'safe_inter_uav_ratio': info.get('safe_inter_uav_ratio', 1.0)},
            start_pos=[path_history[0][i] for i in range(n_agents)],
            goal_pos=[env.goals[i] for i in range(n_agents)],
            path_history=path_history,
            rewards=[episode_reward]
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
    parser.add_argument('--static-obs', type=int, default=16,
                        help='16 = final training stage; 20 = out-of-distribution '
                             'generalization test (report separately).')
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.config, args.episodes, args.device, args.static_obs)
