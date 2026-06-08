"""
Main training loop for MARDPG-NAV.
Reference: Section 14.2 and Algorithm 1 of blueprint.
"""
import os
print("Loading Machine Learning libraries (PyTorch, WandB, etc.). Please wait, this may take up to a minute...")
import yaml

import torch
import numpy as np
import random
import datetime
import wandb
from typing import List
from .environment.uav_env import MultiUAVEnv
from .algorithm.mardpg import MARDPGAgent
from .algorithm.replay_buffer import SequenceReplayBuffer
from .algorithm.noise import GaussianNoise
from .utils.metrics import MetricsTracker
from tqdm import tqdm

CURRICULUM = [
    { # Stage 0
        'name': 'Pre-Training Warm-Up', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 300, 
        'static_obs': 0, 'min_sep': 20.0,
        'criteria': {'success_rate': 0.95, 'path_efficiency': 0.95}
    },
    { # Stage 1
        'name': 'Sparse Familiarisation', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 500,
        'static_obs': 4, 'min_sep': 30.0,
        'criteria': {'success_rate': 0.90, 'collision_rate': 0.02, 'path_efficiency': 0.85, 'operator': 'less_col'}
    },
    { # Stage 2
        'name': 'Coordination Under Pressure', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 700,
        'static_obs': 8, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.85, 'collision_rate': 0.05, 'inter_uav_safe': 0.95, 'operator': 'less_col_greater_safe'}
    },
    { # Stage 3
        'name': 'First Full-Scale Test', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1000,
        'static_obs': 12, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.80, 'trapped_rate': 0.05, 'path_efficiency': 0.80, 'operator': 'less_trap'}
    },
    { # Stage 4
        'name': 'Saturated Navigation', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1200,
        'static_obs': 16, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.75, 'collision_rate': 0.10, 'path_efficiency': 0.75, 'operator': 'less_col'}
    },
    { # Stage 5
        'name': 'Max Density Stress Test', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 20, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.70, 'path_efficiency': 0.70}
    },
    { # Stage 6
        'name': 'Dynamic Threats', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 20, 'min_sep': 40.0,
        'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
        'criteria': {'success_rate': 0.70, 'dyn_collision_rate': 0.05, 'path_efficiency': 0.70, 'operator': 'less_dyn'}
    }
]

class CurriculumManager:
    def __init__(self, stages):
        self.stages = stages
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        
    def get_current_config(self):
        return self.stages[self.current_stage_idx]
        
    def evaluate_promotion(self, stats):
        # Need at least 100 episodes in current stage to evaluate
        if self.episodes_in_stage < 100:
            return False
            
        c = self.stages[self.current_stage_idx]['criteria']
        op = c.get('operator', 'standard')
        
        # Base criteria
        passed = (stats['success_rate'] >= c['success_rate']) and \
                 (stats.get('path_efficiency', 0) >= c.get('path_efficiency', 0))
                 
        if op == 'less_col':
            passed = passed and (stats['collision_rate'] <= c['collision_rate'])
        elif op == 'less_trap':
            passed = passed and (stats['trapped_rate'] <= c['trapped_rate'])
        elif op == 'less_col_greater_safe':
            passed = passed and (stats['collision_rate'] <= c['collision_rate']) and \
                     (stats['inter_uav_safe'] >= c['inter_uav_safe'])
        elif op == 'less_dyn':
            passed = passed and (stats['dyn_collision_rate'] <= c['dyn_collision_rate'])

        if passed and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.episodes_in_stage = 0
            print(f"\n🚀 PROMOTED TO STAGE {self.current_stage_idx}: {self.stages[self.current_stage_idx]['name']} 🚀\n")
            return True
        return False



def load_config(path: str = "config/default.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def select_actions_batch(agents, obs_all, noise_val, v_max, agent_done, action_dim=2):
    # obs_all: (n_agents, obs_dim)
    n = len(agents)
    obs_tensor = torch.FloatTensor(obs_all).unsqueeze(1).to(agents[0].device)
    
    with torch.no_grad():
        flat = obs_tensor.view(n, -1)
        shared_feat = agents[0].actor.shared(flat)
        shared_feat = shared_feat.unsqueeze(1)
        
        actions = []
        for i, agent in enumerate(agents):
            if not agent_done[i]:
                feat_i = shared_feat[i:i+1]
                h_in = agent.actor_hidden if agent.actor_hidden is not None else (
                    torch.zeros(1, 1, agent.actor.lstm.hidden_size).to(agent.device),
                    torch.zeros(1, 1, agent.actor.lstm.hidden_size).to(agent.device)
                )
                lstm_out, h_out = agent.actor.lstm(feat_i, h_in)
                act = agent.actor.tanh(agent.actor.fc_out(lstm_out[:, -1, :])) * agent.actor.action_bound
                agent.actor_hidden = h_out
                
                action = act[0].cpu().numpy()
                action += noise_val[i]
                action = np.clip(action, -agent.action_bound, agent.action_bound)
            else:
                action = np.zeros(action_dim, dtype=np.float32)
            actions.append(action)
            
    return np.array(actions)


def train(config_path: str = "config/default.yaml", device: str = None, resume_dir: str = None, seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    cfg = load_config(config_path)
    if device is None:
        device = cfg.get('algorithm', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.", flush=True)
        device = 'cpu'
    if 'seed' in cfg:
        seed = cfg['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("Initializing Weights & Biases (W&B)...")
    print("If you are not logged in, W&B might prompt you for an API key or choice.")
    print("You can press '3' to run offline if prompted.")
    
    wandb.init(
        project="mardpg-uav",
        entity="aryanugraha0412-institut-teknologi-sepuluh-nopember",
        name=f"run_{run_id}_seed_{seed}",
        config=cfg
    )

    
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
            obs_dim=30,
            action_dim=2,
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            lr_actor=algo_cfg['lr_actor'],
            lr_critic=algo_cfg['lr_critic'],
            tau=algo_cfg['tau'],
            gamma=algo_cfg['gamma'],
            gradient_clip=algo_cfg['gradient_clip'],
            burn_in=algo_cfg['burn_in'],
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
                        if i == 0 and 'shared_opt' in shared_ckpt:
                            shared_optimizer.load_state_dict(shared_ckpt['shared_opt'])
                    except Exception:
                        pass
                agent.actor.load_state_dict(checkpoint['actor_private'], strict=False)
                
            agent.critic.load_state_dict(checkpoint['critic'])
            
            if 'actor_opt' in checkpoint:
                agent.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
            if 'critic_opt' in checkpoint:
                agent.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
                
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
    buffer = SequenceReplayBuffer(
        capacity=algo_cfg['replay_capacity'],
        seq_len=algo_cfg['seq_len'],
        n_agents=n_agents,
        obs_dim=30,
        action_dim=2
    )
    noise = GaussianNoise(
        n_agents=n_agents,
        action_dim=2,
        sigma=algo_cfg.get('noise_std', 0.1)
    )
    
    metrics = MetricsTracker()
    cl_manager = CurriculumManager(CURRICULUM)
    global_step = 0
    
    # Restore global step and noise scheduling if resuming
    if resume_dir:
        algo_cfg['warmup_episodes'] = max(algo_cfg['warmup_episodes'], start_episode + 50)
        try:
            shared_ckpt = torch.load(f"{resume_dir}/shared_actor.pt", map_location=device)
            if 'global_step' in shared_ckpt:
                global_step = shared_ckpt['global_step']
                # Fast-forwarding noise state not needed for constant GaussianNoise
            print(f"Restored global_step: {global_step}")
        except Exception as e:
            print(f"Warning: Could not load global_step from checkpoint: {e}")

    print("=" * 60)
    print("MARDPG-NAV Training")
    print(f"Agents: {n_agents}, Device: {device}")
    print("=" * 60)
    
    # STORE ORIGINAL DIMENSIONS BEFORE THE LOOP
    original_env_size = list(cfg['environment']['env_size'])
    last_update_step = 0

    try:
        for episode in range(start_episode, algo_cfg['n_episodes']):
            stage_cfg = cl_manager.get_current_config()
            
            # Reset Env with Stage Config
            obs = env.reset(stage_cfg)
            cl_manager.episodes_in_stage += 1
            
            # Reset hidden states at episode boundaries (Section 7.2)
            for agent in agents:
                agent.reset_hidden(batch_size=1, eval_mode=False)
            
            noise.reset()
            
            episode_reward = 0
            ep_len = 0
            path_history = [env.agents_state[:, :3].copy()]
            
            for step in range(stage_cfg['max_steps']):
                noise_val = noise.sample()
                # Gaussian noise is constant so no need to log its changing sigma to W&B
                v_max = env_cfg.get('v_max', 3.0)
                actions = select_actions_batch(agents, obs, noise_val, v_max, env.agent_done, env.action_dim)
                
                next_obs, rewards, done, info = env.step(actions)
                global_step += 1
                
                buffer.add_transition(obs.copy(), actions.copy(), rewards.copy(), info['agent_done'].copy())
                episode_reward += sum(rewards)
                path_history.append(env.agents_state[:, :3].copy())
                ep_len += 1
                obs = next_obs
                
                if done:
                    # Append terminal state to allow BPTT sampling of the final transition
                    buffer.add_transition(
                        obs.copy(), 
                        np.zeros_like(actions), 
                        np.zeros_like(rewards), 
                        np.ones_like(info['agent_done'], dtype=bool)
                    )
                    break
            
            buffer.end_episode()
            
            # Record to metrics
            metrics.record_episode(
                length=ep_len, 
                info=info, 
                start_pos=[path_history[0][i] for i in range(n_agents)],
                goal_pos=[env.goals[i] for i in range(n_agents)], 
                path_history=path_history,
                rewards=[episode_reward]
            )
            
            # Evaluate Promotion
            stats = metrics.get_window_stats(100)
            promoted = cl_manager.evaluate_promotion(stats)
            # GaussianNoise doesn't use scheduled annealing like the old version, so we just continue
            
            stats = metrics.get_stats()
            
            # FIX FOR BUG 9: Log both raw and smoothed metrics to match CSV exactly
            wandb.log({
                "episode": episode,
                "ep_reward_raw": float(episode_reward),
                "ep_length_raw": ep_len,
                "success_rate_raw": np.mean(info['reached']),
                "collision_rate_raw": np.mean(info['collisions']),
                "dyn_collision_rate_raw": np.mean(info.get('dyn_collisions', [])),
                "smoothed_success_rate": stats['success_rate'],
                "smoothed_collision_rate": stats['collision_rate'],
                "smoothed_dyn_collision_rate": stats.get('dyn_collision_rate', 0),
                "smoothed_trapped_rate": stats['trapped_rate'],
                "smoothed_avg_reward": stats['avg_reward'],
                "stage": cl_manager.current_stage_idx
            }, step=global_step)
            
            # UPDATE BLOCK (Fixing Bugs 1, 3, 4, 6, 9)
            if episode < algo_cfg['warmup_episodes'] or len(buffer) < algo_cfg['batch_size']:
                last_update_step = global_step
            else:
                updates_to_do = (global_step - last_update_step) // algo_cfg['update_freq']
                total_grad_steps = updates_to_do * algo_cfg.get('grad_steps_per_update', 1)
                
                if updates_to_do > 0:
                    last_update_step += updates_to_do * algo_cfg['update_freq']
                
                grad_metrics = {
                    "actor_loss": [], "critic_loss": [], "q_vals": [],
                    "critic_grad_norm": [], "shared_grad_norm": [], "actor_grad_norm": []
                }
                for _ in range(total_grad_steps):
                    batch = buffer.sample(algo_cfg['batch_size'])
                    if batch is None: break
                    batch_obs, batch_obs_next, batch_actions, batch_rewards, batch_dones = [b.to(device) for b in batch]
                    batch_size, seq_len, _, obs_dim = batch_obs.shape
                    
                    # Define environment limit (matches dynamics.py)
                    tau_v = env_cfg.get('tau_v', 0.3)
                    v_max  = env_cfg.get('v_max', 3.0)

                    # 1. Forward passes for all agents (Online and Target)
                    agent_hiddens, next_agent_hiddens, target_actions = [], [], []

                    # Batched shared feature extraction (Bug 2 Fix)
                    all_obs_flat = batch_obs.permute(0, 2, 1, 3).reshape(batch_size * n_agents * seq_len, obs_dim)
                    all_feats = agents[0].actor.shared(all_obs_flat).view(batch_size, n_agents, seq_len, -1)

                    with torch.no_grad():
                        all_obs_next_flat = batch_obs_next.permute(0, 2, 1, 3).reshape(batch_size * n_agents * seq_len, obs_dim)
                        all_feats_next = agents[0].actor_target.shared(all_obs_next_flat).view(batch_size, n_agents, seq_len, -1)

                    for i, agent in enumerate(agents):
                        feat = all_feats[:, i, :, :]
                        h_out, _ = agent.actor.lstm(feat, None)
                        agent_hiddens.append(h_out)
                        
                        with torch.no_grad():
                            feat_next = all_feats_next[:, i, :, :]
                            h_next, _ = agent.actor_target.lstm(feat_next, None)
                            next_agent_hiddens.append(h_next)
                            
                            # 1. Get raw unconstrained target action
                            next_act_raw = agent.actor_target.tanh(agent.actor_target.fc_out(h_next)) * agent.actor_target.action_bound
                            
                            # 2. Add clipped exploration noise (TD3 smoothing)
                            target_noise = torch.randn_like(next_act_raw) * algo_cfg['policy_noise']
                            target_noise = torch.clamp(target_noise, -algo_cfg['noise_clip'], algo_cfg['noise_clip'])
                            bnd = agent.actor_target.action_bound
                            final_next_act = torch.clamp(next_act_raw + target_noise, -bnd, bnd)
                            
                            target_actions.append(final_next_act)

                    # 2. Prepare Critic Tensors (Now passing observations directly)
                    obs_all = batch_obs.reshape(batch_size * seq_len, n_agents, -1)
                    next_obs_all = batch_obs_next.reshape(batch_size * seq_len, n_agents, -1)
                    next_act_all = torch.stack([a.reshape(batch_size*seq_len, -1) for a in target_actions], dim=1)
                    act_all = batch_actions.reshape(batch_size * seq_len, n_agents, -1)

                    # Explicit padding mask to prevent zero-state leak
                    burn_mask = torch.arange(seq_len, device=device).view(1, -1, 1) >= algo_cfg['burn_in']
                    
                    done_mask = ~torch.cat([
                        torch.zeros(batch_size, 1, n_agents, device=device, dtype=torch.bool),
                        batch_dones[:, :-1, :]
                    ], dim=1)
                    
                    is_terminal = batch_dones & done_mask
                    burn_mask = burn_mask | is_terminal
                    
                    agent_mask = burn_mask & done_mask

                    # 3. Update Critics
                    c_losses = []
                    q_vals = []
                    c_grad_norms = []
                    for i, agent in enumerate(agents):
                        mask_i = agent_mask[:, :, i]
                        # Capture criticize loss
                        c_loss, q_val, c_grad = agent.update_critic(obs_all, act_all, next_obs_all, next_act_all, 
                                            batch_rewards, batch_dones, seq_len, mask_i)
                        c_losses.append(c_loss)
                        q_vals.append(q_val)
                        c_grad_norms.append(c_grad)

                    # 4. Update Actors & Shared Extractor (Fix Bug 6 - Order of zero_grad)
                    shared_optimizer.zero_grad()
                    for agent in agents: agent.actor_optimizer.zero_grad()
                    
                    actor_losses = []
                    for i, agent in enumerate(agents):
                        mask_i = agent_mask[:, :, i]
                        # Construct joint actions where only current agent retains local gradient
                        actor_actions = []
                        for j, other_agent in enumerate(agents):
                            a_j = other_agent.actor.tanh(other_agent.actor.fc_out(agent_hiddens[j])) * other_agent.actor.action_bound
                            if i != j: a_j = a_j.detach() 
                            actor_actions.append(a_j.view(batch_size * seq_len, -1))
                        
                        actor_act_all = torch.stack(actor_actions, dim=1)
                        # Pass obs_all directly to actor_loss
                        actor_losses.append(agent.compute_actor_loss(obs_all, actor_act_all, mask_i))

                    # Aggregate actor losses and backprop ONE time through shared components
                    total_actor_loss = sum(actor_losses) / n_agents
                    total_actor_loss.backward()
                    
                    shared_grad_norm = torch.nn.utils.clip_grad_norm_(agents[0].shared_extractor.parameters(), algo_cfg['gradient_clip'])
                    shared_optimizer.step()
                    
                    actor_grad_norms = []
                    for agent in agents:
                        a_grad = torch.nn.utils.clip_grad_norm_(agent.private_actor_params, algo_cfg['gradient_clip'])
                        actor_grad_norms.append(a_grad.item())
                        agent.actor_optimizer.step()

                    # 5. Soft Updates
                    agents[0]._soft_update(agents[0].actor_target.shared, agents[0].actor.shared)
                    for agent in agents:
                        agent._soft_update(agent.actor_target.lstm, agent.actor.lstm)
                        agent._soft_update(agent.actor_target.fc_out, agent.actor.fc_out)
                        agent._soft_update(agent.critic_target, agent.critic)
                        
                    grad_metrics["actor_loss"].append(total_actor_loss.item())
                    grad_metrics["critic_loss"].append(np.mean(c_losses))
                    grad_metrics["q_vals"].append(np.mean(q_vals))
                    grad_metrics["critic_grad_norm"].append(np.mean(c_grad_norms))
                    grad_metrics["shared_grad_norm"].append(shared_grad_norm.item())
                    grad_metrics["actor_grad_norm"].append(np.mean(actor_grad_norms))
                
                if total_grad_steps > 0:
                    wandb.log({k: np.mean(v) for k, v in grad_metrics.items()}, step=global_step)
            
            # Logging
            if episode % 100 == 0 and episode > 0:
                stats = metrics.get_stats()
                current_noise = noise.sigma
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
                torch.save({
                    'shared_actor': agents[0].shared_extractor.state_dict(),
                    'shared_opt': shared_optimizer.state_dict(),
                    'global_step': global_step
                }, f"{save_dir}/shared_actor.pt")
                for i, agent in enumerate(agents):
                    private = {k: v for k, v in agent.actor.state_dict().items()
                               if k.startswith(('lstm.', 'fc_out.'))}
                    torch.save({
                        'actor_private': private,
                        'critic': agent.critic.state_dict(),
                        'actor_opt': agent.actor_optimizer.state_dict(),
                        'critic_opt': agent.critic_optimizer.state_dict(),
                    }, f"{save_dir}/agent_{i}.pt")
                    
            # Continuous JSON logging every 100 episodes
            if episode % 100 == 0 and episode > 0:
                os.makedirs("checkpoints", exist_ok=True)
                log_file = f"checkpoints/training_log_{run_id}.csv"
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
        torch.save({
            'shared_actor': agents[0].shared_extractor.state_dict(),
            'shared_opt': shared_optimizer.state_dict(),
            'global_step': global_step
        }, f"{save_dir}/shared_actor.pt")
        for i, agent in enumerate(agents):
            private = {k: v for k, v in agent.actor.state_dict().items()
                       if k.startswith(('lstm.', 'fc_out.'))}
            torch.save({
                'actor_private': private,
                'critic': agent.critic.state_dict(),
                'actor_opt': agent.actor_optimizer.state_dict(),
                'critic_opt': agent.critic_optimizer.state_dict(),
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
