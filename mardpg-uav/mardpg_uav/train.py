"""
Main training loop for MARDPG-NAV.
Reference: Section 14.2 and Algorithm 1 of blueprint.
"""
import os
import yaml
import torch
import numpy as np
import random
from typing import List
from .environment.uav_env import MultiUAVEnv
from .algorithm.mardpg import MARDPGAgent
from .algorithm.replay_buffer import EpisodeReplayBuffer, Episode
from .algorithm.noise import GaussianNoise
from .utils.metrics import MetricsTracker
from tqdm import tqdm


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(config_path: str = "config/default.yaml", device: str = None, resume_dir: str = None, seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(config_path)
    if 'seed' in cfg:
        seed = cfg['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
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
    
    shared_optimizer = torch.optim.Adam(agents[0].shared_extractor.parameters(), lr=algo_cfg['lr_actor'])
        
    start_episode = 0
    if resume_dir:
        print(f"Resuming training from checkpoint: {resume_dir}")
        for i, agent in enumerate(agents):
            try:
                checkpoint = torch.load(f"{resume_dir}/agent_{i}.pt", map_location=device, weights_only=True)
            except Exception:
                # Older torch versions might not support weights_only flag or might be dictionary
                checkpoint = torch.load(f"{resume_dir}/agent_{i}.pt", map_location=device)
            
            # Handle new save format if present
            if 'actor' in checkpoint:
                if i == 0:
                    agent.actor.load_state_dict(checkpoint['actor'])
                else:
                    private = {k: v for k, v in checkpoint['actor'].items()
                               if k.startswith(('lstm.', 'fc_out.'))}
                    agent.actor.load_state_dict(private, strict=False)
            elif 'actor_private' in checkpoint:
                if i == 0:
                    try:
                        shared_ckpt = torch.load(f"{resume_dir}/shared_actor.pt", map_location=device)
                        agent.shared_extractor.load_state_dict(shared_ckpt['shared_actor'])
                    except Exception:
                        pass
                agent.actor.load_state_dict(checkpoint['actor_private'], strict=False)
                
            agent.critic.load_state_dict(checkpoint['critic'])
            # Soft update target networks to match exactly
            for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                target_param.data.copy_(param.data)
        print("Successfully loaded checkpoints!")
        
        import re
        match = re.search(r'episode_(\d+)', resume_dir)
        if match:
            start_episode = int(match.group(1))
            print(f"Continuing from episode {start_episode}")
    
    # Replay buffer and noise
    buffer = EpisodeReplayBuffer(
        capacity=algo_cfg['replay_capacity'],
        seq_len=algo_cfg['seq_len']
    )
    noise = GaussianNoise(
        n_agents=n_agents,
        action_dim=env.action_dim,
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
    
    try:
        for episode in tqdm(range(start_episode, algo_cfg['n_episodes']), desc="Training progress", initial=start_episode, total=algo_cfg['n_episodes']):
            
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
                
                # FIX (Bug 10): Store applied actions, not raw commanded actions
                episode_data.append(obs.copy(), info['applied_actions'].copy(), rewards.copy(), info['agent_done'].copy())
                episode_reward += sum(rewards)
                path_history.append(env.agents_state[:, :3].copy())
                
                obs = next_obs
                
                if done:
                    # Append terminal state to allow BPTT sampling of the final transition
                    episode_data.append(
                        obs.copy(), 
                        np.zeros_like(actions), 
                        np.zeros_like(rewards), 
                        np.ones_like(info['agent_done'], dtype=bool)
                    )
                    break
            
            # Store episode
            buffer.add_episode(episode_data)
            
            # Record metrics
            metrics.record_episode(
                [sum(episode_data.rewards[t]) for t in range(episode_data.length())],
                episode_data.length(),
                info['reached'].tolist() if isinstance(info['reached'], np.ndarray) else info['reached'],
                info['collisions'].tolist() if isinstance(info['collisions'], np.ndarray) else info['collisions'],
                [path_history[0][i] for i in range(n_agents)],
                [env.goals[i] for i in range(n_agents)],
                path_history
            )
            
            # UPDATE BLOCK (Fixing Bugs 1, 3, 4, 6, 9)
            if episode > algo_cfg['warmup_episodes'] and global_step % algo_cfg['update_freq'] == 0 and len(buffer) >= algo_cfg['batch_size']:
                
                for _ in range(algo_cfg.get('grad_steps_per_update', 1)):
                    batch = buffer.sample(algo_cfg['batch_size'])
                    if batch is None: break
                    batch_obs, batch_obs_next, batch_actions, batch_rewards, batch_dones = [b.to(device) for b in batch]
                    batch_size, seq_len, _, obs_dim = batch_obs.shape
                    
                    # 1. Forward passes for all agents (Online and Target)
                    agent_hiddens, next_agent_hiddens, target_actions = [], [], []
                    for i, agent in enumerate(agents):
                        feat = agent.actor.shared(batch_obs[:, :, i, :].flatten(0,1)).view(batch_size, seq_len, -1)
                        h_out, _ = agent.actor.lstm(feat, None)
                        agent_hiddens.append(h_out)
                        
                        with torch.no_grad():
                            feat_next = agent.actor_target.shared(batch_obs_next[:, :, i, :].flatten(0,1)).view(batch_size, seq_len, -1)
                            h_next, _ = agent.actor_target.lstm(feat_next, None)
                            next_agent_hiddens.append(h_next)
                            next_act = agent.actor_target.tanh(agent.actor_target.fc_out(h_next)) * agent.actor_target.action_bound
                            target_actions.append(next_act)

                    # 2. Prepare Critic Tensors (DETACHED to fix Bug 2)
                    detached_hidden_all = torch.stack([h.detach().reshape(batch_size*seq_len, -1) for h in agent_hiddens], dim=1)
                    next_hidden_all = torch.stack([h.reshape(batch_size*seq_len, -1) for h in next_agent_hiddens], dim=1)
                    next_act_all = torch.stack([a.reshape(batch_size*seq_len, -1) for a in target_actions], dim=1)
                    act_all = batch_actions.reshape(batch_size * seq_len, n_agents, -1)

                    # FIX (Bug 9): Explicit padding mask to prevent zero-state leak
                    burn_mask = torch.arange(seq_len, device=device).unsqueeze(0) >= algo_cfg['burn_in']
                    done_mask = ~torch.cat([torch.zeros_like(batch_dones[:, :1, 0]), batch_dones[:, :-1, 0]], dim=1)
                    mask = burn_mask & done_mask

                    # 3. Update Critics (Independent graph)
                    for i, agent in enumerate(agents):
                        agent.update_critic(detached_hidden_all, act_all, next_hidden_all, next_act_all, 
                                            batch_rewards, batch_dones, seq_len, mask)

                    # 4. Update Actors & Shared Extractor (Fix Bug 6 - Order of zero_grad)
                    shared_optimizer.zero_grad()
                    for agent in agents: agent.actor_optimizer.zero_grad()
                    
                    actor_losses = []
                    for i, agent in enumerate(agents):
                        # Construct joint actions where only current agent retains local gradient
                        actor_actions = []
                        for j, other_agent in enumerate(agents):
                            a_j = other_agent.actor.tanh(other_agent.actor.fc_out(agent_hiddens[j])) * other_agent.actor.action_bound
                            if i != j: a_j = a_j.detach() 
                            actor_actions.append(a_j.view(batch_size * seq_len, -1))
                        
                        actor_act_all = torch.stack(actor_actions, dim=1)
                        # Hiddens WITH gradients
                        hidden_all_with_grad = torch.stack([h.reshape(batch_size*seq_len, -1) for h in agent_hiddens], dim=1)
                        actor_losses.append(agent.compute_actor_loss(hidden_all_with_grad, actor_act_all, mask))

                    # Aggregate actor losses and backprop ONE time through shared components
                    total_actor_loss = sum(actor_losses) / n_agents
                    total_actor_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(agents[0].shared_extractor.parameters(), algo_cfg['gradient_clip'])
                    shared_optimizer.step()
                    for agent in agents:
                        torch.nn.utils.clip_grad_norm_(agent.private_actor_params, algo_cfg['gradient_clip'])
                        agent.actor_optimizer.step()

                    # 5. Soft Updates
                    for agent in agents:
                        agent._soft_update(agent.actor_target, agent.actor)
                        agent._soft_update(agent.critic_target, agent.critic)
                        # Fix: Shared target update
                        agent._soft_update(agent.actor_target.shared, agent.actor.shared)
            
            # Logging
            if episode % 100 == 0 and episode > 0:
                stats = metrics.get_stats()
                current_noise = noise.get_sigma()
                print(f"Episode {episode:5d} | "
                      f"AvgReward: {stats['avg_reward']:7.2f} | "
                      f"Success: {stats['success_rate']:.2%} | "
                      f"Collision: {stats['collision_rate']:.2%} | "
                      f"Length: {stats['avg_episode_length']:5.1f} | "
                      f"Noise: {current_noise:.3f}")
            
            # Save checkpoints
            if episode % 1000 == 0 and episode > 0:
                save_dir = f"checkpoints/episode_{episode}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save({'shared_actor': agents[0].shared_extractor.state_dict()},
                           f"{save_dir}/shared_actor.pt")
                for i, agent in enumerate(agents):
                    private = {k: v for k, v in agent.actor.state_dict().items()
                               if k.startswith(('lstm.', 'fc_out.'))}
                    torch.save({
                        'actor_private': private,
                        'critic': agent.critic.state_dict(),
                    }, f"{save_dir}/agent_{i}.pt")
                    
            # Continuous JSON logging every 100 episodes
            if episode % 100 == 0 and episode > 0:
                os.makedirs("checkpoints", exist_ok=True)
                log_file = "checkpoints/training_log.csv"
                is_new = not os.path.exists(log_file)
                with open(log_file, 'a') as f:
                    if is_new:
                        f.write("episode,avg_reward,success_rate,collision_rate,trapped_rate,avg_episode_length\n")
                    f.write(f"{episode},{stats['avg_reward']},{stats['success_rate']},{stats['collision_rate']},{stats.get('trapped_rate', 0)},{stats['avg_episode_length']}\n")
                    
    except KeyboardInterrupt:
        print("\n[KEYBOARD INTERRUPT] Training forcefully stopped by user.")
        save_dir = f"checkpoints/interrupted_episode_{episode}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving emergency checkpoints to {save_dir} ...")
        torch.save({'shared_actor': agents[0].shared_extractor.state_dict()},
                   f"{save_dir}/shared_actor.pt")
        for i, agent in enumerate(agents):
            private = {k: v for k, v in agent.actor.state_dict().items()
                       if k.startswith(('lstm.', 'fc_out.'))}
            torch.save({
                'actor_private': private,
                'critic': agent.critic.state_dict(),
            }, f"{save_dir}/agent_{i}.pt")
        
        print("\nInterrupted Training Stats:")
        stats = metrics.get_stats()
        print(stats)
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_stats = {}
        for k, v in stats.items():
            if isinstance(v, (np.floating, float)):
                serializable_stats[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                serializable_stats[k] = int(v)
            else:
                serializable_stats[k] = v
                
        with open(f"{save_dir}/interrupted_stats.json", 'w') as f:
            json.dump(serializable_stats, f, indent=4)
        print(f"Saved stats to {save_dir}/interrupted_stats.json")
        return agents
    
    print("\nTraining complete. Final stats:")
    stats = metrics.get_stats()
    print(stats)
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    serializable_stats = {}
    for k, v in stats.items():
        if isinstance(v, (np.floating, float)):
            serializable_stats[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            serializable_stats[k] = int(v)
        else:
            serializable_stats[k] = v
            
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/final_stats.json", 'w') as f:
        json.dump(serializable_stats, f, indent=4)
    print("Saved stats to checkpoints/final_stats.json")
    return agents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--device', default=None, choices=['cpu', 'cuda'])
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint directory to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    train(args.config, args.device, args.resume, args.seed)
