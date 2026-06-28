import numpy as np
import torch
import os
from mardpg_uav.utils.metrics import MetricsTracker
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.train import load_config


# Map the training --variant names to (recurrent, centralized) so that an
# evaluation can reconstruct the SAME actor architecture a checkpoint was
# trained with. This is required to load MADDPG / IDDPG baselines: their actors
# use a feed-forward core (_FFCore), not an LSTM, so building a recurrent Actor
# and loading their weights would raise a state_dict mismatch.
_VARIANT_FLAGS = {
    'mardpg':   (True,  True),    # recurrent   + centralized critic  (proposed)
    'maddpg':   (False, True),    # feed-forward + centralized critic  (paper)
    'ind_rdpg': (True,  False),   # recurrent   + independent critic   (paper: Ind-RDPG)
    'iddpg':    (False, False),   # feed-forward + independent critic  (bonus, full 2x2)
}


def load_agents(checkpoint_dir, config_path, device='cpu',
                variant='mardpg', recurrent=None, centralized=None):
    """Load a set of agents for evaluation.

    variant : one of {'mardpg','maddpg','iddpg'} (sets recurrent/centralized).
    recurrent / centralized : explicit overrides; if given they win over `variant`.

    Only the ACTOR is needed for action selection at eval time, so the critic
    architecture (centralized vs independent) does not affect rollouts — but we
    still build it consistently so checkpoints that bundle critic weights load
    without warnings. `recurrent` DOES matter: it selects LSTM vs feed-forward
    actor core, and must match how the checkpoint was trained.
    """
    if recurrent is None or centralized is None:
        v_rec, v_cen = _VARIANT_FLAGS.get(variant, (True, True))
        recurrent = v_rec if recurrent is None else recurrent
        centralized = v_cen if centralized is None else centralized

    cfg = load_config(config_path)
    env_cfg = cfg['environment']
    net_cfg = cfg['network']
    n_agents = env_cfg['n_agents']

    trained_agents = 0
    while os.path.exists(os.path.join(checkpoint_dir, f"agent_{trained_agents}.pt")):
        trained_agents += 1
    
    if trained_agents == 0:
        trained_agents = n_agents

    agents = []
    for i in range(n_agents):
        ag = MARDPGAgent(
            agent_id=i, n_agents=n_agents,
            obs_dim=env_cfg.get('obs_dim', 35), action_dim=env_cfg.get('action_dim', 2),   # 49 -> 35 (no-comm obs)
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            recurrent=recurrent, centralized=centralized,
            device=device
        )
        try:
            load_idx = i % trained_agents
            ckpt = torch.load(f"{checkpoint_dir}/agent_{load_idx}.pt", map_location=device)
            if 'actor_private' in ckpt:
                if i == 0:
                    sc = torch.load(f"{checkpoint_dir}/shared_actor.pt", map_location=device)
                    ag.shared_extractor.load_state_dict(sc['shared_actor'])
                ag.actor.load_state_dict(ckpt['actor_private'], strict=False)
            else:
                ag.actor.load_state_dict(ckpt['actor'])
        except Exception as e:
            if trained_agents == n_agents and i == 0:
                print(f"[ERROR] Could not load agent {i}. Checkpoint directory {checkpoint_dir} contents:")
                if os.path.exists(checkpoint_dir):
                    print(os.listdir(checkpoint_dir))
                else:
                    print(f"Directory {checkpoint_dir} DOES NOT EXIST!")
            print(f"[WARN] agent {i}: {e}")
        agents.append(ag)

    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    return agents, cfg


def run_eval(env, stage_cfg, act_fn, n_episodes=50, base_seed=0,
             on_episode_start=None, collect_paths=True):
    m = MetricsTracker()

    for ep in range(n_episodes):
        ep_seed = base_seed + ep
        np.random.seed(ep_seed)
        env.action_space.seed(ep_seed)
        if hasattr(env, 'scene_gen') and hasattr(env.scene_gen, 'rng'):
            env.scene_gen.rng = np.random.RandomState(ep_seed)
        if hasattr(env, 'rangefinder') and hasattr(env.rangefinder, 'rng'):
            env.rangefinder.rng = np.random.RandomState(ep_seed)

        env.reset(stage_cfg)

        if on_episode_start:
            on_episode_start(env)

        ph = [env.agents_state[:, :3].copy()] if collect_paths else []
        dyn = getattr(env, 'dynamic_obstacles', [])
        dyn_path = [[d.position.copy() for d in dyn]] if dyn and collect_paths else []
        dyn_r = [d.size[0] for d in dyn] if dyn else []

        info = {}
        ep_r, L = 0.0, 0
        for _t in range(stage_cfg["max_steps"]):
            acts = act_fn()
            _, r, done, info = env.step(acts)
            ep_r += float(sum(r))
            L += 1
            if collect_paths:
                ph.append(env.agents_state[:, :3].copy())
                if dyn:
                    dyn_path.append([d.position.copy() for d in dyn])
            if done:
                break
        if collect_paths:
            info['dyn_path'] = np.array(dyn_path) if dyn else None
            info['dyn_r'] = dyn_r

        m.record_episode(length=L, info=info,
                         start_pos=[ph[0][i] for i in range(env.n_agents)] if collect_paths else None,
                         goal_pos=[env.goals[i] for i in range(env.n_agents)],
                         path_history=ph if collect_paths else None,
                         rewards=[ep_r])
    return m.get_window_stats(n_episodes), m


def make_apf_act_fn(env, apf_ctrl):
    def act_fn():
        return apf_ctrl.act()
    return act_fn


def make_learned_act_fn(agents, env):
    state = {'prev_acts': np.zeros((env.n_agents, env.action_dim), dtype=np.float32)}

    def on_start(env):
        for ag in agents:
            ag.hidden = None
            if hasattr(ag.actor, 'eval'):
                ag.actor.eval()
            ag.reset_hidden(batch_size=1, eval_mode=True)
        state['prev_acts'] = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)

    def act_fn():
        obs = env._get_observations()
        acts = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
        for i, ag in enumerate(agents):
            if env.agent_done[i]:
                continue
            action = ag.select_action(obs[i], state['prev_acts'][i], evaluate=True)
            action = np.clip(action, -ag.actor.action_bound, ag.actor.action_bound)
            acts[i] = action
        state['prev_acts'] = acts.copy()
        return acts

    return act_fn, on_start
