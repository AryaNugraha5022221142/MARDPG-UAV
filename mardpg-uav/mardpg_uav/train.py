"""
Main training loop for MARDPG-NAV (v4 — final-audit fixes).

Architecture: independent centralized critic per agent (own critic, target
critic, optimizer) + shared actor feature extractor + private actor
LSTM/head per agent.

Changes vs v3 (final audit, 2026-06-11)
---------------------------------------
[N4-1] Critic calibration is now SIGNED and STATE-MATCHED. The old B3
       diagnostic compared abs(q_mean - realized_return), where q_mean is
       averaged over REPLAY batches (old data, old policy, exploratory
       actions) and realized_return is the CURRENT noise-free policy's
       eval return — incommensurable distributions, and abs() discards
       the only information that matters (overestimation vs benign lag).
       It fired a false "overestimation" warning at ep 199 when the critic
       was in fact UNDERestimating (-1.18 vs +17.75). run_eval now probes
       Q_i(s0, a0) on the eval episodes themselves (zero LSTM state, which
       is in-distribution: training windows also start from zero hidden),
       and the warning escalates ONLY on persistent POSITIVE gap.
[N4-2] Pre-registered replay-flush policy at promotions that shift the
       hazard distribution (static_obs increases, or dynamic obstacles
       first appear). Until v3 this was a manual knob defaulting to 0.0;
       the stage 2->3 transition would otherwise train the lidar conv on
       ~300k transitions in which the lidar block is uniformly maxed.
[N4-3] Adaptive eval size at tight promotion gates: n=30 gives SE ~ ±9pp
       at p=0.5; stages gating at 0.55-0.60 success would flap pass/fail
       and reset the consecutive-pass counter (promotion latency).
       Gates with success bars >= threshold now use eval_episodes_late.
[N4-4] CPU runs now warn loudly (or abort, if configured) with the
       measured stage-1 throughput. The 2026-06-11 run executed on CPU at
       0.01-0.03 ep/s (ETA 330-450 h) despite config device: cuda.
[not changed] Episode-granular update bursts (audit item C2) are left
       as-is deliberately: the validated learning dynamics were produced
       under this schedule, no metric implicates it, and interleaving
       updates into the rollout changes data/update ordering for zero
       demonstrated benefit. Revisit only alongside env parallelisation.

Changes vs v2 (retained)
------------------------
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
import itertools
import yaml, torch, numpy as np, random, datetime
from collections import deque
import wandb
from mardpg_uav.wandb_logger import WandbLogger
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
    {   # 1 — Free space, NEAR goals. Pure goal-seeking, no crossings yet.
        'name': 'Free Space (near)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 400,
        'static_obs': 0, 'min_sep': 15.0, 'min_start_sep': 12.0,
        'conflict_frac': 0.0, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.35, 'collision_rate': 0.40,
                     'path_efficiency': 0.40, 'operator': 'less_col'}},
    {   # 2 — Free space + CROSSINGS. Inter-UAV avoidance before obstacles.
        'name': 'Free Space (crossings)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 600,
        'static_obs': 0, 'min_sep': 30.0, 'min_start_sep': 12.0,
        'conflict_frac': 0.6, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.45, 'collision_rate': 0.35,
                     'inter_uav_safe': 0.70, 'operator': 'less_col_greater_safe'}},
    {   # 3 — Sparse obstacles + crossings.
        'name': 'Sparse Obstacles (crossings)',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 700,
        'static_obs': 3, 'max_h': 20.0, 'min_sep': 30.0, 'min_start_sep': 12.0,
        'conflict_frac': 0.6, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.45, 'collision_rate': 0.30,
                     'inter_uav_safe': 0.70, 'operator': 'less_col_greater_safe'}},
    {   # 4 — Moderate density + more crossings.
        'name': 'Moderate Density',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1000,
        'static_obs': 7, 'max_h': 20.0, 'min_sep': 40.0,
        'conflict_frac': 0.8, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.55, 'trapped_rate': 0.15,
                     'path_efficiency': 0.55, 'operator': 'less_trap'}},
    {   # 5 — Dense urban, full crossings.
        'name': 'Dense Urban',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1200,
        'static_obs': 12, 'max_h': 50.0, 'min_sep': 40.0,
        'conflict_frac': 1.0, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.60, 'collision_rate': 0.20,
                     'path_efficiency': 0.60, 'operator': 'less_col'}},
    {   # 6 — Max static density, full crossings.
        'name': 'Max Density Stress Test',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'conflict_frac': 1.0, 'ring_frac': 0.35,
        'criteria': {'success_rate': 0.60, 'path_efficiency': 0.55}},
    {   # 7 — Dynamic threats on top of max static density, full crossings.
        'name': 'Dynamic Threats',
        'env_size': [100.0, 100.0, 60.0], 'max_steps': 1500,
        'static_obs': 16, 'max_h': 50.0, 'min_sep': 40.0,
        'conflict_frac': 1.0, 'ring_frac': 0.35,
        'dynamic_obs': (1, 2), 'dynamic_radius': 2.0, 'dynamic_speed': (1.0, 2.0),
        'criteria': {'success_rate': 0.55, 'dyn_collision_rate': 0.10,
                     'path_efficiency': 0.50, 'operator': 'less_dyn'}},
]

N_STAGES = len(CURRICULUM)


# ---------------------------------------------------------------------------
# Curriculum manager  (promotion now driven by noise-free eval stats — I3)
# ---------------------------------------------------------------------------
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
        # Fast path: promote after this many consecutive passes when the LATEST
        # eval clears every active criterion by `promotion_margin`. Easy stages
        # (which beat the bar by a wide margin) promote immediately; marginal
        # stages still require the full count for robustness.
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

        # Fewer passes required when the margin is comfortable (statistically
        # safe to move sooner); the full count when it only just cleared.
        needed = (self.fast_consecutive_passes if margin_ok
                  else self.required_consecutive_passes)
        is_last = self.current_stage_idx >= len(self.stages) - 1

        if passed and self.consecutive_passes >= needed and not is_last:
            self.current_stage_idx += 1
            self.episodes_in_stage  = 0
            self.consecutive_passes = 0
            name = self.stages[self.current_stage_idx]['name']
            print(f"\n🚀 PROMOTED TO STAGE {self.current_stage_idx + 1}/{N_STAGES}: "
                  f"{name} 🚀  [{'fast' if margin_ok else 'standard'} promote, "
                  f"{needed} consecutive pass(es)]\n", flush=True)
            return True
        elif passed:
            print(f"\n⏳ QUALIFIED FOR PROMOTION: {self.consecutive_passes}/{needed} "
                  f"consecutive passes (margin={'yes' if margin_ok else 'no'}).\n",
                  flush=True)
        return False


# ---------------------------------------------------------------------------
# [N4-2] Hazard-distribution shift predicate (module-level so it is testable)
# ---------------------------------------------------------------------------
def hazard_shift(prev_stage_cfg: dict, new_stage_cfg: dict) -> bool:
    """True iff the promotion ENTERS a stage whose hazard distribution shifts
    relative to the stage just trained: more static obstacles, or dynamic
    obstacles appearing for the first time. Used to pick the larger replay
    flush fraction at promotion."""
    return (new_stage_cfg.get('static_obs', 0) >
            prev_stage_cfg.get('static_obs', 0)) or \
           ('dynamic_obs' in new_stage_cfg and
            'dynamic_obs' not in prev_stage_cfg)


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


# ---------------------------------------------------------------------------
# Noise-free evaluation block (I3) — used for curriculum promotion.
# Does NOT write to the replay buffer, training hidden states, training
# metrics, or the noise schedule.
# [N4-1] Additionally probes Q_i(s0, a0) on each eval episode for a signed,
# state- and policy-matched critic-calibration metric (q_eval_start).
# ---------------------------------------------------------------------------
def run_eval(env, agents, stage_cfg, n_episodes, action_dim, n_agents, gamma,
             base_eval_seed=10_000):
    em = MetricsTracker()
    disc_returns = []
    q0_means     = []   # [N4-1] critic Q(s0, a0) on the EVAL distribution
    for _ep in range(n_episodes):
        # [FIX6] Fixed held-out eval scenes: every promotion eval at every
        # checkpoint sees the SAME n_episodes scenarios (comparable across time,
        # not drawn from the training stream). Training RNG is restored by the
        # caller's snapshot/restore around this call.
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

            # [N4-1] Calibration probe at episode start. Zero LSTM state is
            # in-distribution for the critic (training windows also start
            # from zero hidden), the state is the CURRENT policy's visit
            # distribution, and the action is the noise-free policy action —
            # so Q_i(s0, a0) is directly comparable to the realized per-agent
            # discounted return of the same episode, unlike the replay-batch
            # q_mean (old data, old policy, exploratory actions).
            if t == 0:
                with torch.no_grad():
                    o_t = torch.FloatTensor(obs).unsqueeze(0).to(agents[0].device)
                    a_t = torch.FloatTensor(acts).unsqueeze(0).to(agents[0].device)
                    q0 = []
                    for ag in agents:
                        oi, ai = ag._critic_inputs(o_t, a_t)
                        q0.append(float(ag.critic(oi, ai, hidden=None, seq_len=1)[0].item()))
                q0_means.append(float(np.mean(q0)))

            obs, rewards, done, info = env.step(acts)
            prev_actions = acts.copy()
            step_team = float(sum(rewards))
            ep_reward += step_team
            disc += (gamma ** t) * (step_team / n_agents)   # per-agent discounted return
            ep_len += 1
            path_history.append(env.agents_state[:, :3].copy())
            if done:
                break
        disc_returns.append(disc)
        em.record_episode(
            length=ep_len, info=info,
            start_pos=[path_history[0][i] for i in range(n_agents)],
            goal_pos=[env.goals[i] for i in range(n_agents)],
            path_history=path_history, rewards=[ep_reward])
    stats = em.get_window_stats(n_episodes)
    # [N3-7] Realized discounted return — compare against q_eval_start to
    # detect critic overestimation before it spirals.
    stats['realized_return'] = float(np.mean(disc_returns)) if disc_returns else 0.0
    stats['q_eval_start']    = float(np.mean(q0_means))     if q0_means     else 0.0  # [N4-1]
    return stats


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
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

    # [N4-4] CPU guard. Measured on the 2026-06-11 run: stage 1 (the easiest
    # stage: 400-step cap, 0 obstacles) ran at 0.01-0.03 ep/s on CPU with
    # grad_steps_per_update=8 — ETA 330-450 h for n_episodes=20000, and
    # stages 4-7 are 2.5-3.75x longer per episode. A full curriculum run on
    # CPU is not a realistic plan; this is the single biggest practical
    # blocker found by the final audit.
    if device == 'cpu':
        msg = ("Running on CPU. Measured stage-1 throughput was 0.01-0.03 ep/s "
               "(ETA 330-450 h at n_episodes=20000); later stages are 2.5-3.75x "
               "longer per episode. Run benchmark.py to confirm the bottleneck, "
               "move to a GPU instance, or set algorithm.abort_if_cpu: false to "
               "bypass this check.")
        if bool(algo_cfg.get('abort_if_cpu', False)):
            raise RuntimeError(f"[N4-4] {msg}")
        print(f"[N4-4 WARNING] {msg}", flush=True)

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = WandbLogger(use_wandb=use_wandb, project="mardpg-uav", name=run_name if run_name else f"run_{run_id}_seed_{seed}", config=cfg)

    env        = MultiUAVEnv(env_cfg)
    env.action_space.seed(seed)          # [C5] reproducible warmup-phase actions
    n_agents   = env_cfg['n_agents']
    obs_dim    = env.obs_dim
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
            twin_critic=bool(algo_cfg.get('twin_critic', False)),   # [N3-7]
            recurrent=recurrent, centralized=centralized,
            device=device)
        agents.append(ag)

    # Share lower encoder layers
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])

    # [FIX3] Independent-critic variants drive the SHARED encoder with N
    # mutually-inconsistent policy gradients (each from a critic that sees only
    # its own agent), which destabilises the shared lower layers. Damp the
    # shared-encoder learning rate and gradient clip for non-centralized variants.
    # Centralized variants keep the validated values exactly.
    _shared_lr = algo_cfg.get('lr_shared_actor', algo_cfg['lr_actor'])
    if not centralized:
        _shared_lr = min(_shared_lr, algo_cfg['lr_actor'])
    shared_optimizer = torch.optim.Adam(
        agents[0].shared_extractor.parameters(), lr=_shared_lr)
    _shared_clip = float(algo_cfg.get('shared_gradient_clip', algo_cfg['gradient_clip']))
    if not centralized:
        _shared_clip = min(_shared_clip, 1.0)

    # --- Buffer / noise / metrics / curriculum ---
    buffer = SequenceReplayBuffer(
        capacity=algo_cfg['replay_capacity'],
        seq_len=algo_cfg['seq_len'] + algo_cfg.get('burn_in', 10),
        n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim,
        tail_pad=algo_cfg.get('tail_pad', None))   # [B1] None -> full coverage

    noise   = GaussianNoise(n_agents=n_agents, action_dim=action_dim,
                            sigma=algo_cfg.get('noise_std', 0.3),
                            sigma_min=algo_cfg.get('noise_min', 0.05),
                            decay=algo_cfg.get('noise_decay', 0.9995))
    metrics = MetricsTracker()
    
    # Automatically scale environment size and obstacle counts for larger multi-agent settings
    scaled_curriculum = []
    import copy
    for stage in CURRICULUM:
        new_stage = copy.deepcopy(stage)
        if n_agents > 5:
            scale_factor = (n_agents / 5.0) ** 0.5  # Scale area linearly with number of agents
            new_stage['env_size'] = [
                float(new_stage['env_size'][0] * scale_factor),
                float(new_stage['env_size'][1] * scale_factor),
                new_stage['env_size'][2]
            ]
            # Scale static obstacles to maintain density
            if 'static_obs' in new_stage and new_stage['static_obs'] > 0:
                new_stage['static_obs'] = int(new_stage['static_obs'] * (n_agents / 5.0))
            if 'dynamic_obs' in new_stage:
                if isinstance(new_stage['dynamic_obs'], tuple):
                    new_stage['dynamic_obs'] = (
                        int(new_stage['dynamic_obs'][0] * (n_agents / 5.0)),
                        int(new_stage['dynamic_obs'][1] * (n_agents / 5.0))
                    )
                else:
                    new_stage['dynamic_obs'] = int(new_stage['dynamic_obs'] * (n_agents / 5.0))
        scaled_curriculum.append(new_stage)

    cl = CurriculumManager(
        scaled_curriculum,
        required_consecutive_passes=algo_cfg.get('promotion_consecutive_evals', 2),
        fast_consecutive_passes=algo_cfg.get('promotion_fast_consecutive_evals', 1),
        promotion_margin=algo_cfg.get('promotion_margin', 0.10),
        start_stage_idx=(N_STAGES - 1 if no_curriculum else 0),
        freeze=no_curriculum)

    flush_frac       = float(algo_cfg.get('replay_flush_frac_on_promotion', 0.0))      # N3-4
    flush_frac_shift = float(algo_cfg.get('replay_flush_frac_on_obstacle_shift', 0.5)) # [N4-2]
    tgt_noise        = float(algo_cfg.get('target_policy_noise', 0.0))                 # N3-7
    tgt_noise_clip   = float(algo_cfg.get('target_noise_clip', 0.2))                   # N3-7

    # [N3-6b] Rolling windows so CSV/console grad columns reflect the logging
    # window, not just the most recent episode's 1-3 updates.
    GRAD_WIN = 200
    grad_window = {k: deque(maxlen=GRAD_WIN) for k in
                   ["actor_loss", "critic_loss", "q_vals",
                    "critic_grad_norm", "shared_grad_norm", "actor_grad_norm"]}

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
            cl.consecutive_passes = int(sc.get('consecutive_passes', 0))
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
                print(f"  Replay buffer restored ({len(buffer)} transitions)")
            except Exception as e:
                print(f"  [Warning] Could not load replay buffer from {replay_path}: {e}")
                print("  Starting with an empty replay buffer instead.")

        print(f"  Resumed at episode {start_episode}, global_step {global_step}, "
              f"stage {cl.current_stage_idx + 1}, noise {noise.sigma:.3f}")

    last_update_step     = global_step
    window_start_time    = time.time()
    window_start_episode = start_episode

    eval_every       = int(algo_cfg.get('eval_every', 200))
    eval_episodes    = int(algo_cfg.get('eval_episodes', 30))
    # [N4-3] Promotion gates with success bars >= this threshold get a larger
    # eval set. At n=30, SE ~ ±9pp at p=0.5; stages 4-6 gate at 0.55-0.60,
    # where ±9pp causes pass/fail flapping and consecutive-pass resets
    # (promotion latency, easily misread as a learning failure).
    eval_late_thresh   = float(algo_cfg.get('eval_episodes_late_threshold', 0.50))
    eval_episodes_late = int(algo_cfg.get('eval_episodes_late', max(eval_episodes, 50)))
    promo_min_eps    = int(algo_cfg.get('promotion_min_episodes', 200))
    save_replay_flag = bool(algo_cfg.get('save_replay_on_checkpoint', False))

    print("=" * 60)
    _variant = ('MARDPG' if (recurrent and centralized)
                else 'MADDPG' if (centralized and not recurrent)
                else 'IND-RDPG' if (recurrent and not centralized)
                else 'IDDPG' if (not recurrent and not centralized)
                else 'custom')
    _cl = 'OFF (direct stage 7)' if no_curriculum else 'ON'
    print(f"{_variant}-NAV Training  |  Agents: {n_agents}  |  Device: {device}  "
          f"|  Curriculum: {_cl}")
    print(f"Stage: {cl.current_stage_idx + 1}/{N_STAGES}")
    print("=" * 60, flush=True)

    q_mean = c_loss_mean = c_grad = 0.0   # [C6] defined before any eval print
    q_div_streak = 0                       # [B3/N4-1] POSITIVE-gap streak only

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

            # ---- Roll-out ------------------------------------------------
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
                            a = (ag.actor_target.tanh(ag.actor_target.fc_out(hn))
                                 * ag.actor_target.action_bound)
                            # [N3-7] clipped target-policy smoothing (off if 0)
                            if tgt_noise > 0.0:
                                eps = (torch.randn_like(a) * tgt_noise).clamp(
                                    -tgt_noise_clip, tgt_noise_clip)
                                a = (a + eps).clamp(-ag.actor_target.action_bound,
                                                    ag.actor_target.action_bound)
                            target_actions.append(a)

                    next_act_all = torch.stack(
                        [a.reshape(b_sz * seq_len, -1) for a in target_actions], dim=1)

                    # Zero target actions for agents that are done
                    next_act_all = next_act_all.masked_fill(batch_dones.reshape(b_sz * seq_len, n_agents, 1), 0.0)

                    # ---- Independent critic updates ----------------------
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

                    # ---- Actor update (critics frozen) -------------------
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

                    # ---- Soft target updates ----------------------------
                    agents[0]._soft_update(agents[0].actor_target.shared,
                                           agents[0].actor.shared)
                    for ag in agents:
                        ag._soft_update(ag.actor_target.lstm,   ag.actor.lstm)
                        ag._soft_update(ag.actor_target.fc_out, ag.actor.fc_out)
                        ag._soft_update(ag.critic_target, ag.critic)
                        if ag.twin_critic:
                            ag._soft_update(ag.critic2_target, ag.critic2)

                    # ---- Accumulate grad metrics -------------------------
                    grad_window["actor_loss"].append(total_actor_loss.item())
                    grad_window["critic_loss"].append(c_loss_mean)
                    grad_window["q_vals"].append(q_mean)
                    grad_window["critic_grad_norm"].append(c_grad)
                    grad_window["shared_grad_norm"].append(sh_grad.item())
                    grad_window["actor_grad_norm"].append(float(np.mean(ag_grads)))

                if grad_window["actor_loss"]:
                    logger.log({k: float(np.mean(v)) for k, v in grad_window.items()
                               if len(v) > 0}, step=global_step)

            # ---- Periodic NOISE-FREE eval & curriculum promotion (I3) ----
            if (cl.episodes_in_stage >= promo_min_eps and
                    cl.episodes_in_stage % eval_every == 0 and
                    global_step >= algo_cfg.get('warmup_steps', 2000)):
                # [N3-5] Snapshot ALL stochastic state eval touches: scene RNG,
                # the rangefinder's private RNG, and the global numpy stream.
                _scene_state  = env.scene_gen.rng.get_state()
                _lidar_state  = env.rangefinder.rng.get_state()
                _global_state = np.random.get_state()

                # [N4-3] adaptive eval size near tight promotion gates
                bar = float(stage_cfg['criteria'].get('success_rate', 0.0))
                n_eval_now = (eval_episodes_late if bar >= eval_late_thresh
                              else eval_episodes)

                eval_stats = run_eval(env, agents, stage_cfg,
                                      n_eval_now, action_dim, n_agents,
                                      algo_cfg['gamma'])

                env.scene_gen.rng.set_state(_scene_state)
                env.rangefinder.rng.set_state(_lidar_state)
                np.random.set_state(_global_state)

                # [N4-1/B3] Critic calibration — signed and state-matched.
                # q_eval_start = mean_i Q_i(s0, a0) on the eval episodes just
                # run (current noise-free policy, zero hidden), directly
                # comparable to realized_return (per-agent discounted return
                # of the SAME episodes).
                #   gap > +warn : on-distribution OVERestimation — the
                #                 dangerous direction; escalate.
                #   gap < -warn : critic lagging an improving policy —
                #                 expected during fast learning; benign.
                # The old abs(q_mean - realized_return) compared replay-batch
                # Q (old data/policy/noise) against current-policy eval return
                # and fired a false "overestimation" warning while the critic
                # was underestimating by 19. q_gap_replay is kept as a
                # secondary, clearly-labelled diagnostic only.
                rr        = eval_stats.get('realized_return', 0.0)
                q_cal     = eval_stats.get('q_eval_start', 0.0)
                q_gap     = q_cal - rr
                q_gap_rep = q_mean - rr
                div_warn  = float(algo_cfg.get('q_divergence_warn', 15.0))
                if q_gap > div_warn:
                    q_div_streak += 1
                    print(f"[B3 WARNING] q_eval_start - realized_return = "
                          f"+{q_gap:.2f} > {div_warn} ({q_div_streak} consecutive "
                          f"eval(s)): critic OVERestimation on the eval "
                          f"distribution. Contingency (documented in config): "
                          f"lower grad_steps_per_update and/or enable "
                          f"twin_critic, then restart the stage.", flush=True)
                else:
                    if q_gap < -div_warn:
                        print(f"[B3 info] q_eval_start - realized_return = "
                              f"{q_gap:.2f}: critic lagging an improving policy "
                              f"(benign; expected during fast learning).",
                              flush=True)
                    q_div_streak = 0
                logger.log({"eval/q_gap_signed": q_gap,          # [N4-1] primary
                           "eval/q_eval_start": q_cal,          # [N4-1]
                           "eval/q_gap_replay": q_gap_rep,      # legacy, signed
                           "eval/q_divergence": abs(q_gap_rep), # legacy key (dashboards)
                           "eval/q_div_streak": q_div_streak}, step=global_step)

                log_payload = {f"eval/{k}": v for k, v in eval_stats.items()}
                log_payload["stage"] = cl.current_stage_idx
                logger.log(log_payload, step=global_step)
                _crr = eval_stats.get('conflict_resolution_rate', float('nan'))
                print(f"[EVAL @ ep {episode}] stage {cl.current_stage_idx + 1} | "
                      f"n_eval {n_eval_now} | "
                      f"success {eval_stats['success_rate']:.2%} | "
                      f"mission {eval_stats.get('mission_success_rate', 0.0):.2%} | "
                      f"collision {eval_stats['collision_rate']:.2%} | "
                      f"uav_col {eval_stats.get('uav_collision_rate', 0.0):.2%} | "
                      f"trapped {eval_stats['trapped_rate']:.2%} | "
                      f"path_eff {eval_stats['path_efficiency']:.2f} | "
                      f"min_pair {eval_stats.get('min_pair_dist', float('nan')):.2f} | "
                      f"near_miss {eval_stats.get('near_miss_ratio', 0.0):.2f} | "
                      f"conflict_res {_crr:.2f} | "
                      f"realized_return {eval_stats['realized_return']:.2f} | "
                      f"q_eval_start {eval_stats['q_eval_start']:.2f} | "
                      f"q_mean(replay) {q_mean:.2f}", flush=True)

                _write_eval_csv(f"{out_dir}/eval_log_{run_id}.csv",
                                episode, cl.current_stage_idx + 1, n_eval_now,
                                eval_stats, q_mean)

                promoted = cl.evaluate_promotion(eval_stats)
                if promoted:
                    metrics.soft_reset()                      # [N3-6] no cross-stage blend
                    # Best-per-stage insurance: snapshot the policy that just
                    # CLEARED the previous stage, before the harder stage can
                    # degrade it. Independent of the every-1000-ep cadence, so a
                    # later collapse can never cost you the good model.
                    _save_checkpoint(
                        f"{out_dir}/stage_{cl.current_stage_idx}_cleared",
                        agents, shared_optimizer, global_step, episode,
                        cl, noise, buffer=buffer, save_buffer=False)

                    # [N4-2] Pre-registered flush policy. If the NEW stage
                    # shifts the hazard distribution — static obstacle count
                    # increases (e.g. stage 2->3: lidar goes from uniformly
                    # maxed to informative) or dynamic obstacles first appear
                    # (stage 6->7) — the buffer is dominated by transitions in
                    # which the new hazard channel carries no signal. Flush
                    # the larger of the two configured fractions; same-
                    # geometry promotions (1->2) keep the base fraction.
                    new_stage_cfg = cl.get_current_config()
                    shift = hazard_shift(stage_cfg, new_stage_cfg)
                    frac = max(flush_frac, flush_frac_shift) if shift \
                        else flush_frac
                    dropped = buffer.invalidate_oldest(frac)   # [N3-4]/[N4-2]
                    if dropped:
                        print(f"[N4-2] Flushed {dropped} stale windows on "
                              f"promotion (frac={frac:.2f}, "
                              f"hazard_shift={shift}).", flush=True)
                    reboost = float(algo_cfg.get('noise_reboost', 0.15))
                    if noise.sigma < reboost:
                        noise.sigma = reboost
                        print(f"[N-8] Exploration re-boosted: sigma -> {noise.sigma:.3f} "
                              f"on promotion to stage {cl.current_stage_idx + 1}", flush=True)

            # ---- Console log every 100 episodes -------------------------
            if episode > 0 and episode % 100 == 0:
                now     = time.time()
                elapsed = max(now - window_start_time, 1e-6)
                eps_ps  = (episode - window_start_episode) / elapsed
                window_start_time    = now
                window_start_episode = episode

                print(
                    f"Ep {episode:6d} (open-ended) | "
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
                    f"{eps_ps:.2f} ep/s",
                    flush=True)

            # ---- Checkpoint every 1000 episodes -------------------------
            if episode % 1000 == 0 and episode > 0:
                _save_checkpoint(f"{out_dir}/episode_{episode}",
                                 agents, shared_optimizer, global_step,
                                 episode, cl, noise,
                                 buffer=buffer, save_buffer=save_replay_flag)

            # ---- CSV every 100 episodes ---------------------------------
            if episode % 100 == 0 and episode > 0:
                _write_csv(f"{out_dir}/training_log_{run_id}.csv",
                           episode, stats, grad_window)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving emergency checkpoint (incl. replay buffer)…")
        _save_checkpoint(f"{out_dir}/interrupted_episode_{episode}",
                         agents, shared_optimizer, global_step,
                         episode, cl, noise, buffer=buffer, save_buffer=True)
        print("Done.")
        logger.finish()
        return agents

    print("\nTraining complete.")
    _save_checkpoint(f"{out_dir}/final", agents, shared_optimizer,
                     global_step, episode, cl, noise,
                     buffer=buffer, save_buffer=save_replay_flag)
    logger.finish()
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
    """[FIX6e] One row per promotion eval — the authoritative held-out record.
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
          max_episodes=a.max_episodes, use_wandb=not a.no_wandb,
          recurrent=rec, centralized=cen, no_curriculum=a.no_curriculum)
