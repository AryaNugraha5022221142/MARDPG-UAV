"""
Main training loop for MARDPG-NAV (v3 — post-audit fixes).

Architecture: independent centralized critic per agent (own critic, target
critic, optimizer) + shared actor feature extractor + private actor
LSTM/head per agent.

Changes vs v2
-------------
[C1] The fake "episode-end padding transition" (zero action, zero reward,
     dones=True) is no longer appended; it was sampled and trained on as a
     genuine terminal, regressing Q(s_T, 0) -> 0 for every timed-out agent.
     The buffer now stores next_obs explicitly, so the last real step of a
     timed-out episode bootstraps correctly (truncation != termination).
[C2] next_obs comes from the buffer, never from obs[idx+1] reads that could
     cross episode/overwrite boundaries.
[C3] Checkpoints now persist and resume the FULL training state: curriculum
     stage + episodes_in_stage, exploration sigma, episode counter,
     global_step, target critics, and (optionally / on interrupt) the
     replay buffer. Previously a crash at stage 5 silently restarted the
     run at stage 1 with sigma=0.3 and an empty buffer.
[I2] Episodes shorter than seq_len+burn_in are right-padded inside the
     buffer (pad steps masked out of all losses) instead of being invisible
     to learning.
[I3] Curriculum promotion is now decided on periodic NOISE-FREE evaluation
     episodes (eval_every / eval_episodes in config) instead of noisy
     training rollouts. Eval episodes do not touch the replay buffer,
     training hidden states, or the noise schedule.
[minor] The dead `(dones & alive)` OR-branch in the validity mask is
     removed (it was always sliced off by the burn-in cut in the losses);
     the mask is now simply: past-burn-in AND alive-at-step-start AND
     not-a-pad-step.
"""
import os
print("Loading ML libraries…")
import time
import yaml, torch, numpy as np, random, datetime
import wandb
from typing import List

from .environment.uav_env     import MultiUAVEnv
from .algorithm.mardpg        import MARDPGAgent
from .algorithm.replay_buffer import SequenceReplayBuffer
from .algorithm.noise         import GaussianNoise
from .utils.metrics           import MetricsTracker


# ---------------------------------------------------------------------------
# Curriculum  (7 stages — difficulty axes separated to avoid cliff)
# ---------------------------------------------------------------------------
CURRICULUM = [
    {   # 1 — Free space, NEAR goals. Goal-seeking + peer avoidance only.
        'name': 'Free Space (near)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 400,
        'static_obs': 0, 'min_sep': 15.0, 'min_start_sep': 12.0,
        'criteria': {'success_rate': 0.35, 'collision_rate': 0.55,
                     'path_efficiency': 0.40, 'operator': 'less_col'}},
    {   # 2 — Free space, FAR goals. Horizon extension BEFORE obstacles.
        'name': 'Free Space (far)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 600,
        'static_obs': 0, 'min_sep': 30.0, 'min_start_sep': 12.0,
        'criteria': {'success_rate': 0.45, 'collision_rate': 0.45,
                     'path_efficiency': 0.45, 'operator': 'less_col'}},
    {   # 3 — FEW obstacles at the SHORTER goal distance.
        'name': 'Sparse Obstacles (near)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 700,
        'static_obs': 3, 'max_h': 20.0, 'min_sep': 30.0, 'min_start_sep': 12.0,
        'criteria': {'success_rate': 0.45, 'collision_rate': 0.40,
                     'inter_uav_safe': 0.70, 'operator': 'less_col_greater_safe'}},
    {   # 4 — Moderate density + longer goals.
        'name': 'Moderate Density',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1000,
        'static_obs': 7, 'max_h': 20.0, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.55, 'trapped_rate': 0.15,
                     'path_efficiency': 0.55, 'operator': 'less_trap'}},
    {   # 5 — Dense urban, tall obstacles, full 3D.
        'name': 'Dense Urban',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1200,
        'static_obs': 12, 'max_h': 50.0, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.60, 'collision_rate': 0.20,
                     'path_efficiency': 0.60, 'operator': 'less_col'}},
    {   # 6 — Max static density.
        'name': 'Max Density Stress Test',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'criteria': {'success_rate': 0.60, 'path_efficiency': 0.55}},
    {   # 7 — Dynamic threats on top of max static density.
        'name': 'Dynamic Threats',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
        'criteria': {'success_rate': 0.55, 'dyn_collision_rate': 0.10,
                     'path_efficiency': 0.50, 'operator': 'less_dyn'}},
]

N_STAGES = len(CURRICULUM)


# ---------------------------------------------------------------------------
# Curriculum manager  (promotion now driven by noise-free eval stats — I3)
# ---------------------------------------------------------------------------
class CurriculumManager:
    def __init__(self, stages):
        self.stages            = stages
        self.current_stage_idx = 0
        self.episodes_in_stage = 0

    def get_current_config(self):
        return self.stages[self.current_stage_idx]

    def evaluate_promotion(self, stats):
        """stats: dict from a NOISE-FREE eval block. Caller gates how often
        and after how many in-stage episodes this is invoked."""
        c  = self.stages[self.current_stage_idx]['criteria']
        op = c.get('operator', 'standard')
        passed = (stats['success_rate']             >= c['success_rate'] and
                  stats.get('path_efficiency', 0.0) >= c.get('path_efficiency', 0.0))
        if op == 'less_col':
            passed = passed and stats['collision_rate'] <= c['collision_rate']
        elif op == 'less_trap':
            passed = passed and stats['trapped_rate']   <= c['trapped_rate']
        elif op == 'less_col_greater_safe':
            passed = (passed and
                      stats['collision_rate'] <= c['collision_rate'] and
                      stats['inter_uav_safe'] >= c['inter_uav_safe'])
        elif op == 'less_dyn':
            passed = passed and stats['dyn_collision_rate'] <= c['dyn_collision_rate']

        if passed and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.episodes_in_stage  = 0
            name = self.stages[self.current_stage_idx]['name']
            print(f"\n🚀 PROMOTED TO STAGE {self.current_stage_idx + 1}/{N_STAGES}: "
                  f"{name} 🚀\n", flush=True)
            return True
        return False


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(path: str = "config/default.yaml") -> dict:
    if not os.path.exists(path):
        fallback = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        if os.path.exists(fallback):
            path = fallback
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Action selection (training roll-out, with exploration noise)
# ---------------------------------------------------------------------------
def select_actions_batch(agents, obs_all, prev_actions, noise_val,
                         agent_done, action_dim=2):
    n          = len(agents)
    obs_t      = torch.FloatTensor(obs_all).unsqueeze(1).to(agents[0].device)
    prev_act_t = torch.FloatTensor(np.array(prev_actions)).unsqueeze(1).to(agents[0].device)

    with torch.no_grad():
        shared_feat = agents[0].actor.shared(obs_t.view(n, -1)).unsqueeze(1)
        actions = []
        for i, ag in enumerate(agents):
            if not agent_done[i]:
                x_i  = torch.cat([shared_feat[i:i+1], prev_act_t[i:i+1]], dim=-1)
                h_in = ag.actor_hidden if ag.actor_hidden is not None else (
                    torch.zeros(1, 1, ag.actor.lstm.hidden_size).to(ag.device),
                    torch.zeros(1, 1, ag.actor.lstm.hidden_size).to(ag.device))
                out, h_out = ag.actor.lstm(x_i, h_in)
                act = ag.actor.tanh(ag.actor.fc_out(out[:, -1, :])) * ag.actor.action_bound
                ag.actor_hidden = h_out
                action = act[0].cpu().numpy() + noise_val[i]
                action = np.clip(action, -ag.action_bound, ag.action_bound)
            else:
                action = np.zeros(action_dim, dtype=np.float32)
            actions.append(action)
    return np.array(actions)


# ---------------------------------------------------------------------------
# Noise-free evaluation block (I3) — used for curriculum promotion.
# Does NOT write to the replay buffer, training hidden states, training
# metrics, or the noise schedule.
# ---------------------------------------------------------------------------
def run_eval(env, agents, stage_cfg, n_episodes, action_dim, n_agents):
    em = MetricsTracker()
    for _ in range(n_episodes):
        obs = env.reset(stage_cfg)
        for ag in agents:
            ag.reset_hidden(batch_size=1, eval_mode=True)
        prev_actions = [np.zeros(action_dim, dtype=np.float32)
                        for _ in range(n_agents)]
        path_history = [env.agents_state[:, :3].copy()]
        ep_reward, ep_len = 0.0, 0
        info = {}
        for _ in range(stage_cfg['max_steps']):
            acts = []
            for i, ag in enumerate(agents):
                if env.agent_done[i]:
                    acts.append(np.zeros(action_dim, dtype=np.float32))
                else:
                    a = ag.select_action(obs[i], prev_actions[i], evaluate=True)
                    acts.append(np.clip(a, -ag.action_bound, ag.action_bound))
            acts = np.array(acts, dtype=np.float32)
            obs, rewards, done, info = env.step(acts)
            prev_actions = acts.copy()
            ep_reward += float(sum(rewards))
            ep_len    += 1
            path_history.append(env.agents_state[:, :3].copy())
            if done:
                break
        em.record_episode(
            length=ep_len, info=info,
            start_pos=[path_history[0][i] for i in range(n_agents)],
            goal_pos=[env.goals[i] for i in range(n_agents)],
            path_history=path_history, rewards=[ep_reward])
    return em.get_window_stats(n_episodes)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train(config_path: str = "config/default.yaml",
          device: str = None,
          resume_dir: str = None,
          seed: int = 42):

    cfg      = load_config(config_path)
    env_cfg  = cfg['environment']; env_cfg['seed'] = seed
    algo_cfg = cfg['algorithm']
    net_cfg  = cfg['network']

    # --- Seeding ---
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # [I4] cuDNN-deterministic LSTM kernels are slow on GPU; only enable
    # when explicitly requested. (Env timing is nondeterministic anyway.)
    torch.backends.cudnn.deterministic = bool(
        algo_cfg.get('deterministic_cudnn', False))

    if device is None:
        device = algo_cfg.get('device',
                              'cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available — falling back to CPU.", flush=True)
        device = 'cpu'

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(project="mardpg-uav",
               name=f"run_{run_id}_seed_{seed}",
               config=cfg)

    env        = MultiUAVEnv(env_cfg)
    n_agents   = env_cfg['n_agents']
    obs_dim    = 32
    action_dim = env.action_dim

    # --- Agents ---
    agents: List[MARDPGAgent] = []
    for i in range(n_agents):
        ag = MARDPGAgent(
            agent_id=i, n_agents=n_agents,
            obs_dim=obs_dim, action_dim=action_dim,
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            lr_actor=algo_cfg['lr_actor'], lr_critic=algo_cfg['lr_critic'],
            tau=algo_cfg['tau'], gamma=algo_cfg['gamma'],
            gradient_clip=algo_cfg['gradient_clip'],
            burn_in=algo_cfg['burn_in'],
            huber_beta=algo_cfg.get('huber_beta', 10.0),
            device=device)
        agents.append(ag)

    # Share lower encoder layers
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])

    shared_optimizer = torch.optim.Adam(
        agents[0].shared_extractor.parameters(), lr=algo_cfg['lr_actor'])

    # --- Buffer / noise / metrics / curriculum ---
    buffer = SequenceReplayBuffer(
        capacity=algo_cfg['replay_capacity'],
        seq_len=algo_cfg['seq_len'] + algo_cfg.get('burn_in', 10),
        n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim)

    noise   = GaussianNoise(n_agents=n_agents, action_dim=action_dim,
                            sigma=algo_cfg.get('noise_std', 0.3),
                            sigma_min=algo_cfg.get('noise_min', 0.05),
                            decay=algo_cfg.get('noise_decay', 0.9995))
    metrics = MetricsTracker()
    cl      = CurriculumManager(CURRICULUM)

    # --- Optional resume (C3: full training state) ---
    start_episode = 0
    global_step   = 0
    if resume_dir:
        print(f"Resuming from {resume_dir}")
        try:
            sc = torch.load(f"{resume_dir}/shared_actor.pt", map_location=device)
            agents[0].shared_extractor.load_state_dict(sc['shared_actor'])
            if 'shared_opt' in sc:
                shared_optimizer.load_state_dict(sc['shared_opt'])
            global_step = sc.get('global_step', 0)
            # Full state (new checkpoints); fall back gracefully for old ones
            if 'episode' in sc:
                start_episode = int(sc['episode'])
            else:
                import re
                m = re.search(r'episode_(\d+)', resume_dir)
                start_episode = int(m.group(1)) if m else 0
            cl.current_stage_idx = int(sc.get('stage_idx', 0))
            cl.episodes_in_stage = int(sc.get('episodes_in_stage', 0))
            noise.sigma          = float(sc.get('noise_sigma', noise.sigma))
        except Exception as e:
            print(f"  shared_actor.pt not loaded: {e}")

        for i, ag in enumerate(agents):
            try:
                ckpt = torch.load(f"{resume_dir}/agent_{i}.pt",
                                  map_location=device, weights_only=True)
            except Exception:
                ckpt = torch.load(f"{resume_dir}/agent_{i}.pt", map_location=device)
            if 'actor_private' in ckpt:
                ag.actor.load_state_dict(ckpt['actor_private'], strict=False)
            if 'actor_opt' in ckpt:
                ag.actor_optimizer.load_state_dict(ckpt['actor_opt'])
            if 'critic' in ckpt:
                ag.critic.load_state_dict(ckpt['critic'])
                # Prefer the saved target critic; fall back to hard copy
                ag.critic_target.load_state_dict(
                    ckpt.get('critic_target', ckpt['critic']))
            if 'critic_opt' in ckpt:
                ag.critic_optimizer.load_state_dict(ckpt['critic_opt'])
            for tp, sp in zip(ag.actor_target.parameters(), ag.actor.parameters()):
                tp.data.copy_(sp.data)

        replay_path = f"{resume_dir}/replay.npz"
        if os.path.exists(replay_path):
            buffer.load(replay_path)
            print(f"  Replay buffer restored ({len(buffer)} transitions)")

        print(f"  Resumed at episode {start_episode}, global_step {global_step}, "
              f"stage {cl.current_stage_idx + 1}, noise {noise.sigma:.3f}")

    last_update_step     = global_step
    window_start_time    = time.time()
    window_start_episode = start_episode

    eval_every       = int(algo_cfg.get('eval_every', 200))
    eval_episodes    = int(algo_cfg.get('eval_episodes', 20))
    promo_min_eps    = int(algo_cfg.get('promotion_min_episodes', 200))
    save_replay_flag = bool(algo_cfg.get('save_replay_on_checkpoint', False))

    print("=" * 60)
    print(f"MARDPG-NAV Training  |  Agents: {n_agents}  |  Device: {device}")
    print(f"Stage: {cl.current_stage_idx + 1}/{N_STAGES}")
    print("=" * 60, flush=True)

    episode = start_episode
    try:
        for episode in range(start_episode, algo_cfg['n_episodes']):
            stage_cfg = cl.get_current_config()
            obs = env.reset(stage_cfg)
            cl.episodes_in_stage += 1

            for ag in agents:
                ag.reset_hidden(batch_size=1, eval_mode=False)
            noise.reset()

            episode_reward = 0.0
            ep_len         = 0
            path_history   = [env.agents_state[:, :3].copy()]
            prev_actions   = [np.zeros(action_dim, dtype=np.float32)
                              for _ in range(n_agents)]

            # ---- Roll-out ------------------------------------------------
            for step in range(stage_cfg['max_steps']):
                noise_val = noise.sample()
                if global_step < algo_cfg.get('warmup_steps', 2000):
                    actions = np.array([
                        env.action_space.sample()[i] if not env.agent_done[i]
                        else np.zeros(action_dim, dtype=np.float32)
                        for i in range(n_agents)])
                else:
                    actions = select_actions_batch(
                        agents, obs, prev_actions, noise_val,
                        env.agent_done, action_dim)

                next_obs, rewards, done, info = env.step(actions)
                global_step += 1

                # [C1/C2] next_obs stored explicitly; NO terminal padding
                # transition is appended any more.
                buffer.add_transition(obs.copy(), np.array(prev_actions).copy(),
                                      actions.copy(), rewards.copy(),
                                      next_obs.copy(),
                                      info['agent_done'].copy())
                episode_reward += sum(rewards)
                path_history.append(env.agents_state[:, :3].copy())
                ep_len += 1
                obs = next_obs
                prev_actions = actions.copy()

                if done:
                    break

            buffer.end_episode()          # pads short episodes internally (I2)
            noise.decay_sigma()
            metrics.record_episode(
                length=ep_len, info=info,
                start_pos=[path_history[0][i] for i in range(n_agents)],
                goal_pos=[env.goals[i] for i in range(n_agents)],
                path_history=path_history, rewards=[episode_reward])

            stats = metrics.get_stats()

            if episode % 10 == 0:
                wandb.log({
                    "episode": episode,
                    "ep_reward_raw": float(episode_reward),
                    "ep_length_raw": ep_len,
                    "success_rate_raw": float(np.mean(info['reached'])),
                    "collision_rate_raw": float(np.mean(info['collisions'])),
                    "smoothed_success_rate": stats['success_rate'],
                    "smoothed_collision_rate": stats['collision_rate'],
                    "smoothed_trapped_rate": stats['trapped_rate'],
                    "smoothed_avg_reward": stats['avg_reward'],
                    "noise_sigma": noise.sigma,
                    "stage": cl.current_stage_idx,
                }, step=global_step)

            # ---- Update --------------------------------------------------
            grad_metrics = {k: [] for k in
                            ["actor_loss", "critic_loss", "q_vals",
                             "critic_grad_norm", "shared_grad_norm",
                             "actor_grad_norm"]}

            skip_update = (global_step < algo_cfg.get('warmup_steps', 2000) or
                           len(buffer) < algo_cfg['batch_size'])
            if skip_update:
                last_update_step = global_step
            else:
                updates_todo = (global_step - last_update_step) // algo_cfg['update_freq']
                if updates_todo > 0:
                    last_update_step += updates_todo * algo_cfg['update_freq']

                for _ in range(updates_todo * algo_cfg.get('grad_steps_per_update', 1)):
                    batch = buffer.sample(algo_cfg['batch_size'])
                    if batch is None:
                        break
                    (batch_obs, batch_next_obs, batch_prev_actions,
                     batch_actions, batch_rewards, batch_dones,
                     batch_pads) = [b.to(device) for b in batch]

                    b_sz, seq_len, _, o_dim = batch_obs.shape
                    bi = algo_cfg['burn_in']

                    # ---- Validity mask (simplified — the old
                    # `| (dones & alive)` branch was dead code: terminals
                    # inside burn-in were sliced off in the losses anyway).
                    # valid := past-burn-in AND alive-at-step-start AND not-pad
                    burn_mask  = (torch.arange(seq_len, device=device)
                                  .view(1, -1, 1) >= bi)
                    alive_prev = ~torch.cat([
                        torch.zeros(b_sz, 1, n_agents, device=device, dtype=torch.bool),
                        batch_dones[:, :-1, :]], dim=1)
                    agent_mask = burn_mask & alive_prev & (~batch_pads).unsqueeze(-1)

                    obs_all      = batch_obs.reshape(b_sz * seq_len, n_agents, -1)
                    next_obs_all = batch_next_obs.reshape(b_sz * seq_len, n_agents, -1)
                    act_all      = batch_actions.reshape(b_sz * seq_len, n_agents, -1)

                    # ---- Target next-actions (no grad) -------------------
                    # prev-action for obs_{t+1} is action_t == batch_actions[t]
                    target_actions = []
                    with torch.no_grad():
                        all_nf = agents[0].actor_target.shared(
                            batch_next_obs.permute(0, 2, 1, 3)
                            .reshape(b_sz * n_agents * seq_len, o_dim)
                        ).view(b_sz, n_agents, seq_len, -1)
                        for i, ag in enumerate(agents):
                            xn = torch.cat([all_nf[:, i],
                                            batch_actions[:, :, i, :]], dim=-1)
                            hn, _ = ag.actor_target.lstm(xn, None)
                            target_actions.append(
                                ag.actor_target.tanh(ag.actor_target.fc_out(hn))
                                * ag.actor_target.action_bound)

                    next_act_all = torch.stack(
                        [a.reshape(b_sz * seq_len, -1) for a in target_actions], dim=1)

                    # ---- Independent critic updates ----------------------
                    c_loss_vals, q_mean_list, c_grads = [], [], []

                    for i, ag in enumerate(agents):
                        ag.critic_optimizer.zero_grad()
                        cl_i, q_mean_i, valid_steps = ag.compute_critic_loss(
                            obs_all, act_all, next_obs_all, next_act_all,
                            batch_rewards[:, :, i], batch_dones[:, :, i],
                            agent_mask[:, :, i])
                        if valid_steps > 0:
                            cl_i.backward()
                            c_grad_i = torch.nn.utils.clip_grad_norm_(
                                ag.critic.parameters(), algo_cfg['gradient_clip'])
                            ag.critic_optimizer.step()
                            c_loss_vals.append(cl_i.item())
                            q_mean_list.append(q_mean_i)
                            c_grads.append(c_grad_i.item())

                    c_loss_mean = float(np.mean(c_loss_vals)) if c_loss_vals else 0.0
                    c_grad      = float(np.mean(c_grads))     if c_grads else 0.0
                    q_mean      = float(np.mean(q_mean_list)) if q_mean_list else 0.0

                    # ---- Actor update (critics frozen) -------------------
                    for ag in agents:
                        for p in ag.critic.parameters():
                            p.requires_grad = False

                    shared_optimizer.zero_grad()
                    for ag in agents:
                        ag.actor_optimizer.zero_grad()

                    prev_act_all = batch_prev_actions.reshape(
                        b_sz * seq_len, n_agents, -1)

                    actor_results = [
                        ag.compute_actor_loss(obs_all, act_all, prev_act_all,
                                              agent_mask[:, :, ag.agent_id])
                        for ag in agents]
                    actor_losses = [res[0] for res in actor_results if res[1] > 0]

                    ag_grads = []
                    sh_grad  = torch.zeros(1)
                    if actor_losses:
                        total_actor_loss = torch.stack(actor_losses).mean()
                        total_actor_loss.backward()

                        sh_grad = torch.nn.utils.clip_grad_norm_(
                            agents[0].shared_extractor.parameters(),
                            algo_cfg['gradient_clip'])
                        shared_optimizer.step()

                        for ag in agents:
                            g = torch.nn.utils.clip_grad_norm_(
                                ag.actor_private_params, algo_cfg['gradient_clip'])
                            ag_grads.append(g.item())
                            ag.actor_optimizer.step()
                    else:
                        total_actor_loss = torch.zeros(1)
                        ag_grads = [0.0] * n_agents

                    for ag in agents:
                        for p in ag.critic.parameters():
                            p.requires_grad = True

                    # ---- Soft target updates ----------------------------
                    agents[0]._soft_update(agents[0].actor_target.shared,
                                           agents[0].actor.shared)
                    for ag in agents:
                        ag._soft_update(ag.actor_target.lstm,   ag.actor.lstm)
                        ag._soft_update(ag.actor_target.fc_out, ag.actor.fc_out)
                        ag._soft_update(ag.critic_target, ag.critic)

                    # ---- Accumulate grad metrics -------------------------
                    grad_metrics["actor_loss"].append(total_actor_loss.item())
                    grad_metrics["critic_loss"].append(c_loss_mean)
                    grad_metrics["q_vals"].append(q_mean)
                    grad_metrics["critic_grad_norm"].append(c_grad)
                    grad_metrics["shared_grad_norm"].append(sh_grad.item())
                    grad_metrics["actor_grad_norm"].append(float(np.mean(ag_grads)))

                if grad_metrics["actor_loss"]:
                    wandb.log({k: np.mean(v) for k, v in grad_metrics.items()},
                              step=global_step)

            # ---- Periodic NOISE-FREE eval & curriculum promotion (I3) ----
            if (cl.episodes_in_stage >= promo_min_eps and
                    cl.episodes_in_stage % eval_every == 0 and
                    global_step >= algo_cfg.get('warmup_steps', 2000)):
                eval_stats = run_eval(env, agents, stage_cfg,
                                      eval_episodes, action_dim, n_agents)
                log_payload = {f"eval/{k}": v for k, v in eval_stats.items()}
                log_payload["stage"] = cl.current_stage_idx
                wandb.log(log_payload, step=global_step)
                print(f"[EVAL @ ep {episode}] stage {cl.current_stage_idx + 1} | "
                      f"success {eval_stats['success_rate']:.2%} | "
                      f"collision {eval_stats['collision_rate']:.2%} | "
                      f"trapped {eval_stats['trapped_rate']:.2%} | "
                      f"path_eff {eval_stats['path_efficiency']:.2f}", flush=True)
                cl.evaluate_promotion(eval_stats)

            # ---- Console log every 100 episodes -------------------------
            if episode > 0 and episode % 100 == 0:
                now     = time.time()
                elapsed = max(now - window_start_time, 1e-6)
                eps_ps  = (episode - window_start_episode) / elapsed
                eta_h   = (algo_cfg['n_episodes'] - episode) / max(eps_ps * 3600, 1e-6)
                window_start_time    = now
                window_start_episode = episode

                print(
                    f"Ep {episode:6d}/{algo_cfg['n_episodes']} | "
                    f"Stage {cl.current_stage_idx + 1}/{N_STAGES} "
                    f"({cl.episodes_in_stage:4d} eps) | "
                    f"Reward: {stats['avg_reward']:7.2f} | "
                    f"Success: {stats['success_rate']:.2%} | "
                    f"Collision: {stats['collision_rate']:.2%} | "
                    f"Trapped: {stats.get('trapped_rate', 0):.2%} | "
                    f"Timeout: {stats.get('timeout_rate', 0):.2%} | "
                    f"Len: {stats['avg_episode_length']:5.1f} | "
                    f"Buf: {len(buffer):6d} | "
                    f"Noise: {noise.sigma:.3f} | "
                    f"{eps_ps:.2f} ep/s | ETA: {eta_h:.1f}h",
                    flush=True)

            # ---- Checkpoint every 1000 episodes -------------------------
            if episode % 1000 == 0 and episode > 0:
                _save_checkpoint(f"checkpoints/episode_{episode}",
                                 agents, shared_optimizer, global_step,
                                 episode, cl, noise,
                                 buffer=buffer, save_buffer=save_replay_flag)

            # ---- CSV every 100 episodes ---------------------------------
            if episode % 100 == 0 and episode > 0:
                _write_csv(f"checkpoints/training_log_{run_id}.csv",
                           episode, stats, grad_metrics)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving emergency checkpoint (incl. replay buffer)…")
        _save_checkpoint(f"checkpoints/interrupted_episode_{episode}",
                         agents, shared_optimizer, global_step,
                         episode, cl, noise, buffer=buffer, save_buffer=True)
        print("Done.")
        return agents

    print("\nTraining complete.")
    _save_checkpoint("checkpoints/final", agents, shared_optimizer,
                     global_step, episode, cl, noise,
                     buffer=buffer, save_buffer=save_replay_flag)
    return agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_checkpoint(save_dir, agents, shared_opt, global_step,
                     episode, curriculum, noise,
                     buffer=None, save_buffer=False):
    """[C3] Persist the FULL training state required to resume a long run."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'shared_actor':      agents[0].shared_extractor.state_dict(),
                'shared_opt':        shared_opt.state_dict(),
                'global_step':       global_step,
                'episode':           episode,
                'stage_idx':         curriculum.current_stage_idx,
                'episodes_in_stage': curriculum.episodes_in_stage,
                'noise_sigma':       noise.sigma},
               f"{save_dir}/shared_actor.pt")
    for i, ag in enumerate(agents):
        private = {k: v for k, v in ag.actor.state_dict().items()
                   if k.startswith(('lstm.', 'fc_out.'))}
        torch.save({'actor_private': private,
                    'actor_opt':     ag.actor_optimizer.state_dict(),
                    'critic':        ag.critic.state_dict(),
                    'critic_target': ag.critic_target.state_dict(),
                    'critic_opt':    ag.critic_optimizer.state_dict()},
                   f"{save_dir}/agent_{i}.pt")
    if save_buffer and buffer is not None:
        buffer.save(f"{save_dir}/replay.npz")


def _write_csv(path, episode, stats, grad_metrics):
    os.makedirs("checkpoints", exist_ok=True)
    is_new = not os.path.exists(path)
    gm = {k: (float(np.mean(v)) if v else 0.0) for k, v in grad_metrics.items()}
    with open(path, 'a') as f:
        if is_new:
            f.write("episode,avg_reward,success_rate,collision_rate,"
                    "trapped_rate,avg_episode_length,"
                    "actor_loss,critic_loss,q_mean,"
                    "critic_grad,shared_grad,actor_grad\n")
        f.write(f"{episode},{stats['avg_reward']},{stats['success_rate']},"
                f"{stats['collision_rate']},{stats.get('trapped_rate', 0)},"
                f"{stats['avg_episode_length']},{gm['actor_loss']},"
                f"{gm['critic_loss']},{gm['q_vals']},"
                f"{gm['critic_grad_norm']},{gm['shared_grad_norm']},"
                f"{gm['actor_grad_norm']}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config',  default='config/default.yaml')
    p.add_argument('--device',  default=None, choices=['cpu', 'cuda'])
    p.add_argument('--resume',  default=None)
    p.add_argument('--seed',    type=int, default=42)
    a = p.parse_args()
    train(a.config, a.device, a.resume, a.seed)
