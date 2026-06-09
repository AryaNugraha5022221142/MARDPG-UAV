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
    { # Stage 1 — Free Space: no obstacles, large separation.
      # Thresholds are achievable after ~300-500 episodes for an initializing policy.
      # Goal: learn basic goal-directed flight and avoid peer agents.
        'name': 'Free Space Coordination', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 400,
        'static_obs': 0, 'min_sep': 15.0, 'min_start_sep': 12.0,
        'criteria': {'success_rate': 0.30, 'collision_rate': 0.60, 'path_efficiency': 0.40, 'operator': 'less_col'}
    },
    { # Stage 2 — First obstacles, closer starts. Agents must share space with buildings.
      # After Stage 1 policy, ~20-30% collision reduction and basic avoidance expected.
        'name': 'Sparse Obstacles', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 700,
        'static_obs': 3, 'max_h': 20.0, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.50, 'collision_rate': 0.40, 'inter_uav_safe': 0.70, 'operator': 'less_col_greater_safe'}
    },
    { # Stage 3 — Moderate obstacle density. Path planning becomes necessary.
        'name': 'Moderate Density', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1000,
        'static_obs': 7, 'max_h': 20.0, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.60, 'trapped_rate': 0.15, 'path_efficiency': 0.55, 'operator': 'less_trap'}
    },
    { # Stage 4 — High-density urban-like environment. Tall obstacles; full 3D navigation.
        'name': 'Dense Urban', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1200,
        'static_obs': 12, 'max_h': 50.0, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.65, 'collision_rate': 0.20, 'path_efficiency': 0.60, 'operator': 'less_col'}
    },
    { # Stage 5 — Maximum static density. Agents must plan long detours.
        'name': 'Max Density Stress Test', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.60, 'path_efficiency': 0.55}
    },
    { # Stage 6 — Dynamic threats added on top of max static density.
        'name': 'Dynamic Threats', 'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
        'criteria': {'success_rate': 0.55, 'dyn_collision_rate': 0.10, 'path_efficiency': 0.50, 'operator': 'less_dyn'}
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
            print(f"\n🚀 PROMOTED TO STAGE {self.current_stage_idx}: {self.stages[self.current_stage_idx]['name']} 🚀\n", flush=True)
            return True
        return False



def load_config(path: str = "config/default.yaml") -> dict:
    if not os.path.exists(path):
        fallback = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        if os.path.exists(fallback):
            path = fallback
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def select_actions_batch(agents, obs_all, prev_actions, noise_val, v_max, agent_done, action_dim=2):
    # obs_all: (n_agents, obs_dim)
    n = len(agents)
    obs_tensor = torch.FloatTensor(obs_all).unsqueeze(1).to(agents[0].device)
    prev_act_tensor = torch.FloatTensor(np.array(prev_actions)).unsqueeze(1).to(agents[0].device)
    
    with torch.no_grad():
        flat = obs_tensor.view(n, -1)
        shared_feat = agents[0].actor.shared(flat)
        shared_feat = shared_feat.unsqueeze(1)
        
        actions = []
        for i, agent in enumerate(agents):
            if not agent_done[i]:
                feat_i = shared_feat[i:i+1]
                prev_act_i = prev_act_tensor[i:i+1]
                x_i = torch.cat([feat_i, prev_act_i], dim=-1)
                
                h_in = agent.actor_hidden if agent.actor_hidden is not None else (
                    torch.zeros(1, 1, agent.actor.lstm.hidden_size).to(agent.device),
                    torch.zeros(1, 1, agent.actor.lstm.hidden_size).to(agent.device)
                )
                lstm_out, h_out = agent.actor.lstm(x_i, h_in)
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
        name=f"run_{run_id}_seed_{seed}",
        config=cfg
    )

    
    env_cfg = cfg['environment']
    env_cfg['seed'] = seed
    algo_cfg = cfg['algorithm']
    net_cfg = cfg['network']
    
    # Initialize environment
    env = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    
    obs_dim = 32
    action_dim = env.action_dim
    
    # Initialize agents with parameter sharing
    agents: List[MARDPGAgent] = []
    for i in range(n_agents):
        agent = MARDPGAgent(
            agent_id=i,
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
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

    # One dedicated optimizer for the shared feature extractor. After share_parameters,
    # every agent.actor.shared points at agents[0].shared_extractor, so this single
    # optimizer is the ONLY thing that updates the encoder -> exactly one update per step.
    shared_optimizer = torch.optim.Adam(agents[0].shared_extractor.parameters(),
                                        lr=algo_cfg['lr_actor'])

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
                        if 'shared_opt' in shared_ckpt:
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
        seq_len=algo_cfg['seq_len'] + algo_cfg.get('burn_in', 10),
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim
    )
    noise = GaussianNoise(
        n_agents=n_agents,
        action_dim=action_dim,
        sigma=algo_cfg.get('noise_std', 0.1)
    )
    
    metrics = MetricsTracker()
    cl_manager = CurriculumManager(CURRICULUM)
    global_step = 0
    
    # Restore global step and noise scheduling if resuming
    if resume_dir:
        algo_cfg['warmup_steps'] = max(algo_cfg.get('warmup_steps', 2000), global_step + 2000)
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
    
    import time
    # STORE ORIGINAL DIMENSIONS BEFORE THE LOOP
    original_env_size = list(cfg['environment']['env_size'])
    last_update_step = 0
    start_time = time.time()
    steps_done = global_step
    # window_episode_start tracks the beginning of the most recent 100-ep window
    window_start_time = time.time()
    window_start_episode = start_episode

    print("=" * 60)
    print(f"MARDPG-NAV Training | Stage: {cl_manager.current_stage_idx + 1}/6")
    print("=" * 60, flush=True)

    episode_start_time = time.time()

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
            
            prev_actions = [np.zeros(env.action_dim, dtype=np.float32) for _ in range(n_agents)]
            
            for step in range(stage_cfg['max_steps']):
                noise_val = noise.sample()
                v_max = env_cfg.get('v_max', 3.0)
                
                if global_step < algo_cfg.get('warmup_steps', 2000):
                    actions = []
                    for i in range(n_agents):
                        if env.agent_done[i]:
                            actions.append(np.zeros(env.action_dim, dtype=np.float32))
                        else:
                            actions.append(env.action_space.sample()[i])
                    actions = np.array(actions)
                else:
                    actions = select_actions_batch(agents, obs, prev_actions, noise_val, v_max, env.agent_done, action_dim)
                
                next_obs, rewards, done, info = env.step(actions)
                global_step += 1
                
                buffer.add_transition(obs.copy(), prev_actions.copy(), actions.copy(), rewards.copy(), info['agent_done'].copy())
                episode_reward += sum(rewards)
                path_history.append(env.agents_state[:, :3].copy())
                ep_len += 1
                obs = next_obs
                prev_actions = actions.copy()
                
                if done:
                    # Append terminal state to allow BPTT sampling of the final transition
                    buffer.add_transition(
                        obs.copy(), 
                        prev_actions.copy(),
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
            if episode % 10 == 0:
                wandb.log({
                    "episode": episode,
                    "ep_reward_raw": float(episode_reward),
                    "ep_length_raw": ep_len,
                    "success_rate_raw": np.mean(info['reached']),
                    "collision_rate_raw": np.mean(info['collisions']),
                    "dyn_collision_rate_raw": float(np.mean(info['dyn_collisions'])) if len(info.get('dyn_collisions', [])) > 0 else 0.0,
                    "smoothed_success_rate": stats['success_rate'],
                    "smoothed_collision_rate": stats['collision_rate'],
                    "smoothed_dyn_collision_rate": stats.get('dyn_collision_rate', 0),
                    "smoothed_trapped_rate": stats['trapped_rate'],
                    "smoothed_avg_reward": stats['avg_reward'],
                    "stage": cl_manager.current_stage_idx
                }, step=global_step)
            
            # UPDATE BLOCK (Fixing Bugs 1, 3, 4, 6, 9)
            if global_step < algo_cfg.get('warmup_steps', 2000) or len(buffer) < algo_cfg['batch_size']:
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
                    batch_obs, batch_obs_next, batch_prev_actions, batch_actions, batch_rewards, batch_dones = [b.to(device) for b in batch]
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
                        prev_act = batch_prev_actions[:, :, i, :]
                        x = torch.cat([feat, prev_act], dim=-1)
                        h_out, _ = agent.actor.lstm(x, None)
                        agent_hiddens.append(h_out)
                        
                        with torch.no_grad():
                            feat_next = all_feats_next[:, i, :, :]
                            prev_act_next = batch_actions[:, :, i, :]
                            x_next = torch.cat([feat_next, prev_act_next], dim=-1)
                            h_next, _ = agent.actor_target.lstm(x_next, None)
                            next_agent_hiddens.append(h_next)
                            
                            final_next_act = agent.actor_target.tanh(
                                agent.actor_target.fc_out(h_next)
                            ) * agent.actor_target.action_bound
                            
                            target_actions.append(final_next_act)

                    # 2. Prepare critic tensors. The centralized critic consumes RAW
                    #    observations of all agents (paper §V.B), so no feature tensors here.
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

                    # 4. Update actors and the shared extractor.
                    #    zero_grad the single shared optimizer AND every actor optimizer.
                    shared_optimizer.zero_grad()
                    for agent in agents:
                        agent.actor_optimizer.zero_grad()

                    actor_losses = []
                    prev_act_all = batch_prev_actions.reshape(batch_size * seq_len, n_agents, -1)
                    for i, agent in enumerate(agents):
                        mask_i = agent_mask[:, :, i]
                        # Critic consumes raw obs; only this agent's action carries gradient.
                        actor_losses.append(agent.compute_actor_loss(obs_all, act_all, prev_act_all, mask_i))

                    # Aggregate and backprop ONCE through the shared components.
                    total_actor_loss = sum(actor_losses) / n_agents
                    total_actor_loss.backward()

                    # Step the shared feature extractor EXACTLY ONCE.
                    shared_grad_norm = torch.nn.utils.clip_grad_norm_(
                        agents[0].shared_extractor.parameters(), algo_cfg['gradient_clip'])
                    shared_optimizer.step()

                    # Step each agent's PRIVATE actor params (LSTM + output head) once.
                    actor_grad_norms = []
                    for agent in agents:
                        a_grad = torch.nn.utils.clip_grad_norm_(
                            agent.actor_private_params, algo_cfg['gradient_clip'])
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
            
            # Logging and update
            if episode > 0 and episode % 100 == 0:
                stats = metrics.get_stats()

                now = time.time()
                window_elapsed = max(now - window_start_time, 1e-6)
                eps_per_sec = (episode - window_start_episode) / window_elapsed
                steps_per_sec = (global_step - steps_done) / window_elapsed

                window_start_time = now
                window_start_episode = episode
                start_time = now
                steps_done = global_step

                remaining_eps = algo_cfg['n_episodes'] - episode
                eta_hours = remaining_eps / max(eps_per_sec * 3600, 1e-6)
                
                print(
                    f"Ep {episode:6d}/{algo_cfg['n_episodes']} | "
                    f"Stage {cl_manager.current_stage_idx + 1} ({cl_manager.episodes_in_stage:4d} eps) | "
                    f"Reward: {stats['avg_reward']:7.2f} | "
                    f"Success: {stats['success_rate']:.2%} | "
                    f"Collision: {stats['collision_rate']:.2%} | "
                    f"Trapped: {stats.get('trapped_rate', 0):.2%} | "
                    f"Timeout: {stats.get('timeout_rate', 0):.2%} | "
                    f"Len: {stats['avg_episode_length']:5.1f} | "
                    f"Buf: {len(buffer):6d} | "
                    f"Noise: {noise.sigma:.3f} | "
                    f"{eps_per_sec:.2f} ep/s | "
                    f"ETA: {eta_hours:.1f}h",
                    flush=True
                )
            
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
                
                gm_vals = {'actor_loss': 0.0, 'critic_loss': 0.0, 'q_vals': 0.0, 'critic_grad_norm': 0.0, 'shared_grad_norm': 0.0, 'actor_grad_norm': 0.0}
                if 'grad_metrics' in locals() and grad_metrics.get('actor_loss'):
                    gm_vals = {k: (float(np.mean(v)) if v else 0.0) for k, v in grad_metrics.items()}
                
                with open(log_file, 'a') as f:
                    if is_new:
                        f.write("episode,avg_reward,success_rate,collision_rate,trapped_rate,avg_episode_length,actor_loss,critic_loss,q_mean,critic_grad,shared_grad,actor_grad\n")
                    f.write(f"{episode},{stats['avg_reward']},{stats['success_rate']},{stats['collision_rate']},"
                            f"{stats.get('trapped_rate',0)},{stats['avg_episode_length']},{gm_vals['actor_loss']},"
                            f"{gm_vals['critic_loss']},{gm_vals['q_vals']},{gm_vals['critic_grad_norm']},"
                            f"{gm_vals['shared_grad_norm']},{gm_vals['actor_grad_norm']}\n")
                    
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
