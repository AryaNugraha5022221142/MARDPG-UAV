import os
import time
import itertools
import yaml, torch, numpy as np, random, datetime, copy
from collections import deque
from mardpg_uav.wandb_logger import WandbLogger
from typing import List

from mardpg_uav.environment.uav_env     import MultiUAVEnv
from mardpg_uav.algorithm.mardpg        import MARDPGAgent
from mardpg_uav.algorithm.replay_buffer import SequenceReplayBuffer
from mardpg_uav.algorithm.noise         import GaussianNoise
from mardpg_uav.utils.metrics           import MetricsTracker

CURRICULUM = [
    {
        'name': 'Free Space (near)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 400,
        'static_obs': 0, 'min_sep': 15.0, 'min_start_sep': 12.0,
        'conflict_frac': 0.0, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.35, 'collision_rate': 0.40,
                     'path_efficiency': 0.40, 'operator': 'less_col'}},
    {
        'name': 'Free Space (crossings)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 600,
        'static_obs': 0, 'min_sep': 30.0, 'min_start_sep': 12.0,
        'conflict_frac': 0.6, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.45, 'collision_rate': 0.35,
                     'inter_uav_safe': 0.70, 'operator': 'less_col_greater_safe'}},
    {
        'name': 'Sparse Obstacles (crossings)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 700,
        'static_obs': 3, 'max_h': 20.0, 'min_sep': 30.0, 'min_start_sep': 12.0,
        'conflict_frac': 0.6, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.45, 'collision_rate': 0.30,
                     'inter_uav_safe': 0.70, 'operator': 'less_col_greater_safe'}},
    {
        'name': 'Moderate Density',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1000,
        'static_obs': 7, 'max_h': 20.0, 'min_sep': 40.0,
        'conflict_frac': 0.8, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.55, 'trapped_rate': 0.15,
                     'path_efficiency': 0.55, 'operator': 'less_trap'}},
    {
        'name': 'Dense Urban',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1200,
        'static_obs': 12, 'max_h': 50.0, 'min_sep': 40.0,
        'conflict_frac': 1.0, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.60, 'collision_rate': 0.20,
                     'path_efficiency': 0.60, 'operator': 'less_col'}},
    {
        'name': 'Max Density Stress Test',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'conflict_frac': 1.0, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.60, 'path_efficiency': 0.55}},
    {
        'name': 'Dynamic Threats',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'conflict_frac': 1.0, 'ring_frac': 0.35,
        'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
        'criteria': {'success_rate': 0.55, 'dyn_collision_rate': 0.10,
                     'path_efficiency': 0.50, 'operator': 'less_dyn'}},
]

N_STAGES = len(CURRICULUM)

class CurriculumManager:
    def __init__(self, stages,
                 required_consecutive_passes=2,
                 fast_consecutive_passes=1,
                 promotion_margin=0.0,
                 start_stage_idx=0,
                 freeze=False):
        self.stages             = stages
        self.current_stage_idx  = start_stage_idx
        self.freeze             = freeze
        self.episodes_in_stage  = 0
        self.consecutive_passes = 0
        self.required_consecutive_passes = int(required_consecutive_passes)

        self.fast_consecutive_passes = max(1, int(fast_consecutive_passes))
        self.promotion_margin        = float(promotion_margin)

    def get_current_config(self):
        return self.stages[self.current_stage_idx]

    def _check(self, stats):
        """Return (passed, margin_ok).
        `passed`    : meets every active criterion at the exact bar.
        `margin_ok` : meets every active criterion with `promotion_margin` to spare.
        direction +1 => 'value >= bar' (higher better); -1 => 'value <= bar' (lower better).
        """
        c  = self.stages[self.current_stage_idx]['criteria']
        op = c.get('operator', 'standard')
        m  = self.promotion_margin

        checks = [(stats['success_rate'], c['success_rate'], +1)]
        if 'path_efficiency' in c:
            checks.append((stats.get('path_efficiency', 0.0), c['path_efficiency'], +1))
        if op == 'less_col':
            checks.append((stats['collision_rate'], c['collision_rate'], -1))
        elif op == 'less_trap':
            checks.append((stats['trapped_rate'], c['trapped_rate'], -1))
        elif op == 'less_col_greater_safe':
            checks.append((stats['collision_rate'], c['collision_rate'], -1))
            checks.append((stats['inter_uav_safe'], c['inter_uav_safe'], +1))
        elif op == 'less_dyn':
            checks.append((stats['dyn_collision_rate'], c['dyn_collision_rate'], -1))

        passed = margin = True
        for val, bar, direction in checks:
            if direction > 0:
                passed &= (val >= bar)
                margin &= (val >= bar + m)
            else:
                passed &= (val <= bar)
                margin &= (val <= bar - m)
        return bool(passed), bool(passed and margin)

    def evaluate_promotion(self, stats):
        if self.freeze:
            return False
        passed, margin_ok = self._check(stats)

        if passed:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        needed = (self.fast_consecutive_passes if margin_ok
                  else self.required_consecutive_passes)
        is_last = self.current_stage_idx >= len(self.stages) - 1

        if passed and self.consecutive_passes >= needed and not is_last:
            self.current_stage_idx += 1
            self.episodes_in_stage  = 0
            self.consecutive_passes = 0
            name = self.stages[self.current_stage_idx]['name']
            return True
        elif passed:
            pass
        return False

def hazard_shift(prev_stage_cfg: dict, new_stage_cfg: dict) -> bool:
    """True iff the promotion ENTERS a stage whose hazard distribution shifts
    relative to the stage just trained: more static obstacles, or dynamic
    obstacles appearing for the first time. Used to pick the larger replay
    flush fraction at promotion."""
    return (new_stage_cfg.get('static_obs', 0) >
            prev_stage_cfg.get('static_obs', 0)) or           ('dynamic_obs' in new_stage_cfg and
            'dynamic_obs' not in prev_stage_cfg)

def load_config(path: str = "config/default.yaml") -> dict:
    if not os.path.exists(path):
        fallback = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        if os.path.exists(fallback):
            path = fallback
    with open(path) as f:
        return yaml.safe_load(f)

def select_actions_batch(agents, obs_all, prev_actions, noise_val,
                         agent_done, action_dim=2):
    obs_t      = torch.FloatTensor(obs_all).unsqueeze(1).to(agents[0].device)
    prev_act_t = torch.FloatTensor(np.array(prev_actions)).unsqueeze(1).to(agents[0].device)

    with torch.no_grad():
        actions = []
        for i, ag in enumerate(agents):
            if not agent_done[i]:
                h_in = ag.actor_hidden if ag.actor_hidden is not None else (
                    torch.zeros(1, 1, ag.actor.lstm.hidden_size).to(ag.device),
                    torch.zeros(1, 1, ag.actor.lstm.hidden_size).to(ag.device))
                act, h_out = ag.actor(obs_t[i:i+1], prev_act_t[i:i+1], h_in)
                ag.actor_hidden = h_out
                action = act[0, -1].cpu().numpy() + noise_val[i]
                action = np.clip(action, -ag.action_bound, ag.action_bound)
            else:
                action = np.zeros(action_dim, dtype=np.float32)
            actions.append(action)
    return np.array(actions)

def run_eval(env, agents, stage_cfg, n_episodes, action_dim, n_agents, gamma,
             base_eval_seed=10_000):
    em = MetricsTracker()
    disc_returns = []
    q0_means     = []
    for _ep in range(n_episodes):

        _es = base_eval_seed + _ep
        env.scene_gen.rng.seed(_es)
        env.rangefinder.rng.seed(_es)
        np.random.seed(_es)

        obs = env.reset(stage_cfg)

        for ag in agents:
            ag.reset_hidden(batch_size=1, eval_mode=True)
        prev_actions = [np.zeros(action_dim, dtype=np.float32) for _ in range(n_agents)]
        path_history = [env.agents_state[:, :3].copy()]
        ep_reward, ep_len, disc = 0.0, 0, 0.0
        info = {}
        for t in range(stage_cfg['max_steps']):
            acts = []
            for i, ag in enumerate(agents):
                if env.agent_done[i]:
                    acts.append(np.zeros(action_dim, dtype=np.float32))
                else:
                    a = ag.select_action(obs[i], prev_actions[i], evaluate=True)
                    acts.append(np.clip(a, -ag.action_bound, ag.action_bound))
            acts = np.array(acts, dtype=np.float32)

            if t == 0:
                with torch.no_grad():
                    o_t = torch.FloatTensor(obs).unsqueeze(0).to(agents[0].device)
                    a_t = torch.FloatTensor(acts).unsqueeze(0).to(agents[0].device)
                    q0 = []
                    for ag in agents:
                        oi, ai = ag._critic_inputs(o_t, a_t)
                        q0.append(float(ag.critic(oi, ai, hidden=None, seq_len=1).item()))
                q0_means.append(float(np.mean(q0)))

            obs, rewards, done, info = env.step(acts)
            prev_actions = acts.copy()
            step_team = float(sum(rewards))
            ep_reward += step_team
            disc += (gamma ** t) * (step_team / n_agents)
            ep_len += 1
            path_history.append(env.agents_state[:, :3].copy())
            if done:
                break
        disc_returns.append(disc)
        em.record_episode(
            length=ep_len, info=info,
            start_pos=[path_history[i] for i in range(n_agents)],
            goal_pos=[env.goals[i] for i in range(n_agents)],
            path_history=path_history, rewards=[ep_reward])
    stats = em.get_window_stats(n_episodes)

    stats['realized_return'] = float(np.mean(disc_returns)) if disc_returns else 0.0
    stats['q_eval_start']    = float(np.mean(q0_means))     if q0_means     else 0.0
    return stats

def train(config_path: str = "config/default.yaml",
          device: str = None,
          resume_dir: str = None,
          seed: int = 42,
          out_dir: str = "checkpoints",
          run_name: str = None,
          max_episodes: int = -1,
          use_wandb: bool = True,
          recurrent: bool = True,
          centralized: bool = True,
          no_curriculum: bool = False):

    cfg      = load_config(config_path)
    env_cfg  = cfg['environment']; env_cfg['seed'] = seed
    algo_cfg = cfg['algorithm']
    net_cfg  = cfg['network']

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


    if device is None:
        device = algo_cfg.get('device', 'cpu')
    if device != 'cpu' and torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(algo_cfg.get('deterministic_cudnn', False))
    else:
        device = 'cpu'

    if device == 'cpu':
        msg = ("Running on CPU. Measured stage-1 throughput was 0.01-0.03 ep/s "
               "(ETA 330-450 h at n_episodes=20000); later stages are 2.5-3.75x "
               "longer per episode. Run benchmark.py to confirm the bottleneck, "
               "move to a GPU instance, or set algorithm.abort_if_cpu: false to "
               "bypass this check.")
        if bool(algo_cfg.get('abort_if_cpu', False)):
            raise RuntimeError(f"{msg}")

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = WandbLogger(use_wandb=use_wandb, project="mardpg-uav", name=run_name if run_name else f"run_{run_id}_seed_{seed}", config=cfg)

    env        = MultiUAVEnv(env_cfg)
    env.action_space.seed(seed)
    n_agents   = env_cfg['n_agents']
    obs_dim    = env.obs_dim
    action_dim = env.action_dim

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
            twin_critic=bool(algo_cfg.get('twin_critic', False)),
            recurrent=recurrent, centralized=centralized,
            device=device)
        agents.append(ag)

    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])

    _shared_lr = algo_cfg.get('lr_shared_actor', algo_cfg['lr_actor'])
    if not centralized:
        _shared_lr = min(_shared_lr, algo_cfg['lr_actor'])
    shared_optimizer = torch.optim.Adam(
        agents[0].shared_extractor.parameters(), lr=_shared_lr)
    _shared_clip = float(algo_cfg.get('shared_gradient_clip', algo_cfg['gradient_clip']))
    if not centralized:
        _shared_clip = min(_shared_clip, 1.0)

    buffer = SequenceReplayBuffer(
        capacity=algo_cfg['replay_capacity'],
        seq_len=algo_cfg['seq_len'] + algo_cfg.get('burn_in', 10),
        n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim,
        tail_pad=algo_cfg.get('tail_pad', None))

    noise   = GaussianNoise(n_agents=n_agents, action_dim=action_dim,
                            sigma=algo_cfg.get('noise_std', 0.3),
                            sigma_min=algo_cfg.get('noise_min', 0.05),
                            decay=algo_cfg.get('noise_decay', 0.9995))
    metrics = MetricsTracker()

    scaled_curriculum = []
    for stage in CURRICULUM:
        new_stage = copy.deepcopy(stage)
        if n_agents > 5:
            scale_factor = (n_agents / 5.0) ** 0.5
            orig_env_size = new_stage['env_size']
            new_stage['env_size'] = [
                float(orig_env_size[0] * scale_factor),
                float(orig_env_size[1] * scale_factor),
                float(orig_env_size[2])] # Z-axis kept constant

            if 'static_obs' in new_stage and new_stage['static_obs'] > 0:
                new_stage['static_obs'] = int(new_stage['static_obs'] * (n_agents / 5.0))
            if 'dynamic_obs' in new_stage:
                dyn_obs = new_stage['dynamic_obs']
                if isinstance(dyn_obs, tuple):
                    new_stage['dynamic_obs'] = (
                        int(dyn_obs[0] * (n_agents / 5.0)),
                        int(dyn_obs[1] * (n_agents / 5.0))
                    )
                else:
                    new_stage['dynamic_obs'] = int(dyn_obs * (n_agents / 5.0))
            if 'dynamic_speed' in new_stage:
                dyn_speed = new_stage['dynamic_speed']
                if isinstance(dyn_speed, tuple):
                    new_stage['dynamic_speed'] = (
                        float(dyn_speed[0] * scale_factor),
                        float(dyn_speed[1] * scale_factor)
                    )
        scaled_curriculum.append(new_stage)

    cl = CurriculumManager(
        scaled_curriculum,
        required_consecutive_passes=algo_cfg.get('promotion_consecutive_evals', 2),
        fast_consecutive_passes=algo_cfg.get('promotion_fast_consecutive_evals', 1),
        promotion_margin=algo_cfg.get('promotion_margin', 0.10),
        start_stage_idx=(N_STAGES - 1 if no_curriculum else 0),
        freeze=no_curriculum)

    flush_frac       = float(algo_cfg.get('replay_flush_frac_on_promotion', 0.0))
    flush_frac_shift = float(algo_cfg.get('replay_flush_frac_on_obstacle_shift', 0.5))
    tgt_noise        = float(algo_cfg.get('target_noise', 0.1))
    tgt_noise_clip   = float(algo_cfg.get('target_noise_clip', 0.2))

    GRAD_WIN = 200
    grad_window = {k: deque(maxlen=GRAD_WIN) for k in
                   ["actor_loss", "critic_loss", "q_vals",
                    "critic_grad_norm", "shared_grad_norm", "actor_grad_norm"]}

    start_episode = 0
    global_step   = 0
    if resume_dir:
        try:
            sc = torch.load(f"{resume_dir}/shared_actor.pt", map_location=device)
            agents[0].shared_extractor.load_state_dict(sc['shared_actor'])
            if 'shared_opt' in sc:
                shared_optimizer.load_state_dict(sc['shared_opt'])
            global_step = sc.get('global_step', 0)

            if 'episode' in sc:
                start_episode = int(sc['episode'])
            else:
                import re
                m = re.search(r'episode_(\d+)', resume_dir)
                start_episode = int(m.group(1)) if m else 0
            cl.current_stage_idx = int(sc.get('stage_idx', 0))
            cl.episodes_in_stage = int(sc.get('episodes_in_stage', 0))
            cl.consecutive_passes = int(sc.get('consecutive_passes', 0))
            noise.sigma          = float(sc.get('noise_sigma', noise.sigma))
        except Exception as e:
            pass

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

                ag.critic_target.load_state_dict(
                    ckpt.get('critic_target', ckpt['critic']))
            if 'critic_opt' in ckpt:
                ag.critic_optimizer.load_state_dict(ckpt['critic_opt'])
            if getattr(ag, 'twin_critic', False) and 'critic2' in ckpt:
                ag.critic2.load_state_dict(ckpt['critic2'])
                ag.critic2_target.load_state_dict(ckpt.get('critic2_target', ckpt['critic2']))
                if 'critic2_opt' in ckpt:
                    ag.critic2_optimizer.load_state_dict(ckpt['critic2_opt'])
            for tp, sp in zip(ag.actor_target.parameters(), ag.actor.parameters()):
                tp.data.copy_(sp.data)

        replay_path = f"{resume_dir}/replay.npz"
        if os.path.exists(replay_path):
            try:
                buffer.load(replay_path)
                pass
            except Exception as e:
                pass


    last_update_step     = global_step
    window_start_time    = time.time()
    window_start_episode = start_episode

    eval_every       = int(algo_cfg.get('eval_every', 200))
    eval_episodes    = int(algo_cfg.get('eval_episodes', 30))

    eval_late_thresh   = float(algo_cfg.get('eval_episodes_late_threshold', 0.50))
    eval_episodes_late = int(algo_cfg.get('eval_episodes_late', max(eval_episodes, 50)))
    promo_min_eps    = int(algo_cfg.get('promotion_min_episodes', 200))
    save_replay_flag = bool(algo_cfg.get('save_replay_on_checkpoint', False))

    _variant = ('MARDPG' if (recurrent and centralized)
                else 'MADDPG' if (centralized and not recurrent)
                else 'IND-RDPG' if (recurrent and not centralized)
                else 'IDDPG' if (not recurrent and not centralized)
                else 'custom')
    _cl = 'OFF (direct stage 7)' if no_curriculum else 'ON'

    q_mean = c_loss_mean = c_grad = 0.0
    q_div_streak = 0

    episode = start_episode
    try:
        for episode in itertools.count(start_episode):
            if max_episodes != -1 and episode >= max_episodes:
                break
            stage_cfg = cl.get_current_config()

            obs = env.reset(stage_cfg)

            cl.episodes_in_stage += 1

            for ag in agents:
                ag.reset_hidden(batch_size=1, eval_mode=False)

            episode_reward = 0.0
            ep_len         = 0
            path_history   = [env.agents_state[:, :3].copy()]
            prev_actions   = [np.zeros(action_dim, dtype=np.float32)
                              for _ in range(n_agents)]

            for step in range(stage_cfg['max_steps']):
                noise_val = noise.sample()
                if global_step < algo_cfg.get('warmup_steps', 2000):
                    actions = env.action_space.sample()
                    actions[env.agent_done] = 0.0
                else:
                    actions = select_actions_batch(
                        agents, obs, prev_actions, noise_val,
                        env.agent_done, action_dim)

                next_obs, rewards, done, info = env.step(actions)
                global_step += 1

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

            buffer.end_episode()
            noise.decay_sigma()
            metrics.record_episode(
                length=ep_len, info=info,
                start_pos=[path_history[i] for i in range(n_agents)],
                goal_pos=[env.goals[i] for i in range(n_agents)],
                path_history=path_history, rewards=[episode_reward])

            stats = metrics.get_stats()

            if episode % 10 == 0:
                logger.log({
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

                    burn_mask  = (torch.arange(seq_len, device=device)
                                  .view(1, -1, 1) >= bi)
                    alive_prev = ~torch.cat([
                        torch.zeros(b_sz, 1, n_agents, device=device, dtype=torch.bool),
                        batch_dones[:, :-1, :]], dim=1)
                    agent_mask = burn_mask & alive_prev & (~batch_pads).unsqueeze(-1)

                    obs_all      = batch_obs.reshape(b_sz * seq_len, n_agents, -1)
                    next_obs_all = batch_next_obs.reshape(b_sz * seq_len, n_agents, -1)
                    act_all      = batch_actions.reshape(b_sz * seq_len, n_agents, -1)

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
                            a = (ag.actor_target.tanh(ag.actor_target.fc_out(hn))
                                 * ag.actor_target.action_bound)

                            if tgt_noise > 0.0:
                                eps = (torch.randn_like(a) * tgt_noise).clamp(
                                    -tgt_noise_clip, tgt_noise_clip)
                                a = (a + eps).clamp(-ag.actor_target.action_bound,
                                                    ag.actor_target.action_bound)
                            target_actions.append(a)

                    next_act_all = torch.stack(
                        [a.reshape(b_sz * seq_len, -1) for a in target_actions], dim=1)

                    next_act_all = next_act_all.masked_fill(batch_dones.reshape(b_sz * seq_len, n_agents, 1), 0.0)

                    c_loss_vals, q_mean_list, c_grads = [], [], []

                    for i, ag in enumerate(agents):
                        ag.critic_optimizer.zero_grad()
                        if ag.twin_critic:
                            ag.critic2_optimizer.zero_grad()
                        cl_i, q_mean_i, valid_steps = ag.compute_critic_loss(
                            obs_all, act_all, next_obs_all, next_act_all,
                            batch_rewards[:, :, i], batch_dones[:, :, i],
                            agent_mask[:, :, i])
                        if valid_steps > 0:
                            cl_i.backward()
                            c_grad_i = torch.nn.utils.clip_grad_norm_(
                                ag.critic.parameters(), algo_cfg['gradient_clip'])
                            ag.critic_optimizer.step()
                            if ag.twin_critic:
                                torch.nn.utils.clip_grad_norm_(
                                    ag.critic2.parameters(), algo_cfg['gradient_clip'])
                                ag.critic2_optimizer.step()
                            c_loss_vals.append(cl_i.item())
                            q_mean_list.append(q_mean_i)
                            c_grads.append(c_grad_i.item())

                    c_loss_mean = float(np.mean(c_loss_vals)) if c_loss_vals else 0.0
                    c_grad      = float(np.mean(c_grads))     if c_grads else 0.0
                    q_mean      = float(np.mean(q_mean_list)) if q_mean_list else 0.0

                    for ag in agents:
                        for p in ag.critic.parameters():
                            p.requires_grad = False
                        if ag.twin_critic:
                            for p in ag.critic2.parameters():
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
                            _shared_clip)
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
                        if ag.twin_critic:
                            for p in ag.critic2.parameters():
                                p.requires_grad = True

                    agents[0]._soft_update(agents[0].actor_target.shared,
                                           agents[0].actor.shared)
                    for ag in agents:
                        ag._soft_update(ag.actor_target.lstm,   ag.actor.lstm)
                        ag._soft_update(ag.actor_target.fc_out, ag.actor.fc_out)
                        ag._soft_update(ag.critic_target, ag.critic)
                        if ag.twin_critic:
                            ag._soft_update(ag.critic2_target, ag.critic2)

                    grad_window["actor_loss"].append(total_actor_loss.item())
                    grad_window["critic_loss"].append(c_loss_mean)
                    grad_window["q_vals"].append(q_mean)
                    grad_window["critic_grad_norm"].append(c_grad)
                    grad_window["shared_grad_norm"].append(sh_grad.item())
                    grad_window["actor_grad_norm"].append(float(np.mean(ag_grads)))

                if grad_window["actor_loss"]:
                    logger.log({k: float(np.mean(v)) for k, v in grad_window.items()
                               if len(v) > 0}, step=global_step)

            if (cl.episodes_in_stage >= promo_min_eps and
                    cl.episodes_in_stage % eval_every == 0 and
                    global_step >= algo_cfg.get('warmup_steps', 2000)):

                _scene_state  = env.scene_gen.rng.get_state()
                _lidar_state  = env.rangefinder.rng.get_state()
                _global_state = np.random.get_state()

                bar = float(stage_cfg['criteria'].get('success_rate', 0.0))
                n_eval_now = (eval_episodes_late if bar >= eval_late_thresh
                              else eval_episodes)

                eval_stats = run_eval(env, agents, stage_cfg,
                                      n_eval_now, action_dim, n_agents,
                                      algo_cfg['gamma'])

                env.scene_gen.rng.set_state(_scene_state)
                env.rangefinder.rng.set_state(_lidar_state)
                np.random.set_state(_global_state)

                rr        = eval_stats.get('realized_return', 0.0)
                q_cal     = eval_stats.get('q_eval_start', 0.0)
                q_gap     = q_cal - rr
                q_gap_rep = q_mean - rr
                div_warn  = float(algo_cfg.get('q_divergence_warn', 15.0))
                if q_gap > div_warn:
                    q_div_streak += 1
                else:
                    q_div_streak = 0
                logger.log({"eval/q_gap_signed": q_gap,
                           "eval/q_eval_start": q_cal,
                           "eval/q_divergence": abs(q_gap_rep),
                           "eval/q_div_streak": q_div_streak}, step=global_step)

                log_payload = {f"eval/{k}": v for k, v in eval_stats.items()}
                log_payload["stage"] = cl.current_stage_idx
                logger.log(log_payload, step=global_step)
                _crr = eval_stats.get('conflict_resolution_rate', float('nan'))

                _write_eval_csv(f"{out_dir}/eval_log_{run_id}.csv",
                                episode, cl.current_stage_idx + 1, n_eval_now,
                                eval_stats, q_mean)

                promoted = cl.evaluate_promotion(eval_stats)
                if promoted:
                    metrics.soft_reset()

                    _save_checkpoint(
                        f"{out_dir}/stage_{cl.current_stage_idx}_cleared",
                        agents, shared_optimizer, global_step, episode,
                        cl, noise, buffer=buffer, save_buffer=False)

                    new_stage_cfg = cl.get_current_config()
                    shift = hazard_shift(stage_cfg, new_stage_cfg)
                    frac = max(flush_frac, flush_frac_shift) if shift                        else flush_frac
                    dropped = buffer.invalidate_oldest(frac)
                    reboost = float(algo_cfg.get('noise_reboost', 0.15))
                    if noise.sigma < reboost:
                        noise.sigma = reboost

            if episode > 0 and episode % 100 == 0:
                now     = time.time()
                elapsed = max(now - window_start_time, 1e-6)
                eps_ps  = (episode - window_start_episode) / elapsed
                window_start_time    = now
                window_start_episode = episode


            if episode % 1000 == 0 and episode > 0:
                _save_checkpoint(f"{out_dir}/episode_{episode}",
                                 agents, shared_optimizer, global_step,
                                 episode, cl, noise,
                                 buffer=buffer, save_buffer=save_replay_flag)

            if episode % 100 == 0 and episode > 0:
                _write_csv(f"{out_dir}/training_log_{run_id}.csv",
                           episode, stats, grad_window)

    except KeyboardInterrupt:
        _save_checkpoint(f"{out_dir}/interrupted_episode_{episode}",
                         agents, shared_optimizer, global_step,
                         episode, cl, noise, buffer=buffer, save_buffer=True)
        logger.finish()
        return agents

    _save_checkpoint(f"{out_dir}/final", agents, shared_optimizer,
                     global_step, episode, cl, noise,
                     buffer=buffer, save_buffer=save_replay_flag)
    logger.finish()
    return agents

def _save_checkpoint(save_dir, agents, shared_opt, global_step,
                     episode, curriculum, noise,
                     buffer=None, save_buffer=False):
    """Persist the FULL training state required to resume a long run."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'shared_actor':      agents[0].shared_extractor.state_dict(),
                'shared_opt':        shared_opt.state_dict(),
                'global_step':       global_step,
                'episode':           episode,
                'stage_idx':         curriculum.current_stage_idx,
                'episodes_in_stage': curriculum.episodes_in_stage,
                'consecutive_passes': curriculum.consecutive_passes,
                'noise_sigma':       noise.sigma},
               f"{save_dir}/shared_actor.pt")
    for i, ag in enumerate(agents):
        private = {k: v for k, v in ag.actor.state_dict().items()
                   if k.startswith(('lstm.', 'fc_out.'))}
        save_obj = {'actor_private': private,
                    'actor_opt':     ag.actor_optimizer.state_dict(),
                    'critic':        ag.critic.state_dict(),
                    'critic_target': ag.critic_target.state_dict(),
                    'critic_opt':    ag.critic_optimizer.state_dict()}
        if getattr(ag, 'twin_critic', False):
            save_obj.update({'critic2':        ag.critic2.state_dict(),
                             'critic2_target': ag.critic2_target.state_dict(),
                             'critic2_opt':    ag.critic2_optimizer.state_dict()})
        torch.save(save_obj, f"{save_dir}/agent_{i}.pt")
    if save_buffer and buffer is not None:
        buffer.save(f"{save_dir}/replay.npz")

def _write_eval_csv(path, episode, stage_1based, n_eval, s, q_mean_replay):
    """One row per promotion eval — the authoritative held-out record.
    `s` is the eval_stats dict; `stage_1based` is
    the stage being evaluated (before any promotion this eval triggers)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    g = lambda k, d=0.0: s.get(k, d)
    with open(path, 'a') as f:
        if is_new:
            f.write("episode,stage,n_eval,success_rate,mission_success_rate,"
                    "collision_rate,uav_collision_rate,trapped_rate,"
                    "path_efficiency,inter_uav_safe,min_pair_dist,near_miss_ratio,"
                    "conflict_resolution_rate,realized_return,q_eval_start,"
                    "q_mean_replay\n")
        f.write(f"{episode},{stage_1based},{n_eval},{g('success_rate')},"
                f"{g('mission_success_rate')},{g('collision_rate')},"
                f"{g('uav_collision_rate')},{g('trapped_rate')},"
                f"{g('path_efficiency')},{g('inter_uav_safe')},"
                f"{g('min_pair_dist', float('nan'))},{g('near_miss_ratio')},"
                f"{g('conflict_resolution_rate', float('nan'))},"
                f"{g('realized_return')},{g('q_eval_start')},{q_mean_replay}\n")

def _write_csv(path, episode, stats, grad_window):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    gm = {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in grad_window.items()}
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
    p.add_argument('--out-dir', default='checkpoints')
    p.add_argument('--run-name', default=None)
    p.add_argument('--max-episodes', type=int, default=-1)
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--variant', default='mardpg',
                   choices=['mardpg', 'maddpg', 'ind_rdpg', 'iddpg'])
    p.add_argument('--no-curriculum', action='store_true')
    a = p.parse_args()

    rec, cen = {'mardpg': (True, True), 'maddpg': (False, True),
                'ind_rdpg': (True, False), 'iddpg': (False, False)}[a.variant]
    train(a.config, a.device, a.resume, a.seed,
          out_dir=a.out_dir, run_name=a.run_name,
          max_episodes=a.max_episodes, use_wandb=False,
          recurrent=rec, centralized=cen, no_curriculum=a.no_curriculum)

