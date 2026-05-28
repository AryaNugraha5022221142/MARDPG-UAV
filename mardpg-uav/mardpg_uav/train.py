"""
Main training loop for MARDPG-NAV.
Reference: Section 14.2 and Algorithm 1 of blueprint.
"""
import os
import yaml
import torch
import numpy as np
from typing import List
from .environment.uav_env import MultiUAVEnv
from .algorithm.mardpg import MARDPGAgent
from .algorithm.replay_buffer import EpisodeReplayBuffer, Episode
from .algorithm.noise import OUNoise
from .utils.metrics import MetricsTracker
from tqdm import tqdm


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(config_path: str = "config/default.yaml", device: str = 'cpu'):
    cfg = load_config(config_path)
    env_cfg = cfg['environment']
    algo_cfg = cfg['algorithm']
    net_cfg = cfg['network']
    
    # Initialize environment
    env = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    
    # Initialize agents with parameter sharing
    agents: List[MARDPGAgent] = []
    for i in range(n_agents):
        agent = MARDPGAgent(
            agent_id=i,
            n_agents=n_agents,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_dim=net_cfg['actor']['lstm_hidden'],
            lr_actor=algo_cfg['lr_actor'],
            lr_critic=algo_cfg['lr_critic'],
            tau=algo_cfg['tau'],
            gamma=algo_cfg['gamma'],
            burn_in=algo_cfg['burn_in'],
            gradient_clip=algo_cfg['gradient_clip'],
            device=device
        )
        agents.append(agent)
    
    # Share parameters in lower layers (Section 10.1)
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    
    # Replay buffer and noise
    buffer = EpisodeReplayBuffer(
        capacity=algo_cfg['replay_capacity'],
        seq_len=algo_cfg['seq_len']
    )
    noise = OUNoise(
        n_agents=n_agents,
        action_dim=env.action_dim,
        kappa=algo_cfg['exploration']['ou_kappa'],
        sigma0=algo_cfg['exploration']['noise_std_start'],
        sigma_inf=algo_cfg['exploration']['noise_std_end'],
        anneal_steps=algo_cfg['exploration']['noise_anneal_steps']
    )
    
    metrics = MetricsTracker()
    global_step = 0
    
    print("=" * 60)
    print("MARDPG-NAV Training")
    print(f"Agents: {n_agents}, Device: {device}")
    print("=" * 60)
    
    for episode in tqdm(range(algo_cfg['n_episodes']), desc="Training progress"):
        # Sequence length curriculum (ramp from 20 to 50 over 3000 episodes)
        buffer.seq_len = int(20 + min(30, (episode / 3000.0) * 30))
        
        obs = env.reset()
        
        # Reset hidden states at episode boundaries (Section 7.2)
        for agent in agents:
            agent.reset_hidden(batch_size=1, eval_mode=False)
        
        noise.reset()
        
        episode_data = Episode()
        episode_reward = 0
        path_history = [env.agents_state[:, :3].copy()]
        
        for step in range(env_cfg['max_steps_per_episode']):
            # Decentralized execution with exploration (Algorithm 1, line 8-9)
            noise_val = noise.sample()
            
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(obs[i], evaluate=False)
                # Add noise during training (always added, OU anneals naturally)
                action += noise_val[i]
                action = np.clip(action, -np.pi/6, np.pi/6)
                actions.append(action)
            
            actions = np.array(actions)
            
            # Execute joint action
            next_obs, rewards, done, info = env.step(actions)
            global_step += 1
            
            # Store transition
            episode_data.append(obs.copy(), info['applied_actions'].copy(), rewards.copy(), info['agent_done'])
            episode_reward += sum(rewards)
            path_history.append(env.agents_state[:, :3].copy())
            
            obs = next_obs
            
            if done:
                break
        
        # Store episode
        buffer.add_episode(episode_data)
        
        # Record metrics
        metrics.record_episode(
            [sum(episode_data.rewards[t]) for t in range(episode_data.length())],
            episode_data.length(),
            info['reached'].tolist() if isinstance(info['reached'], np.ndarray) else info['reached'],
            any(info['collisions']) if isinstance(info['collisions'], np.ndarray) else any(info['collisions']),
            [path_history[0][i] for i in range(n_agents)],
            [env.goals[i] for i in range(n_agents)],
            path_history
        )
        
        # Update agents (Algorithm 1, line 28-35)
        if episode % algo_cfg['update_freq'] == 0 \
           and len(buffer) >= algo_cfg['warmup_episodes']:
            
            batch = buffer.sample(algo_cfg['batch_size'])
            if batch is not None:
                batch_obs, batch_obs_next, batch_actions, batch_rewards, batch_dones = batch
                
                for i, agent in enumerate(agents):
                    c_loss, a_loss = agent.update(
                        batch_obs, batch_obs_next, batch_actions, batch_rewards, batch_dones, agents
                    )
        
        # Logging
        if episode % 100 == 0 and episode > 0:
            stats = metrics.get_stats()
            print(f"Episode {episode:5d} | "
                  f"AvgReward: {stats['avg_reward']:7.2f} | "
                  f"Success: {stats['success_rate']:.2%} | "
                  f"Collision: {stats['collision_rate']:.2%} | "
                  f"Length: {stats['avg_episode_length']:5.1f}")
        
        # Save checkpoints
        if episode % 1000 == 0 and episode > 0:
            save_dir = f"checkpoints/episode_{episode}"
            os.makedirs(save_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                }, f"{save_dir}/agent_{i}.pt")
    
    print("\nTraining complete. Final stats:")
    print(metrics.get_stats())
    return agents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    train(args.config, args.device)
