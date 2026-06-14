import numpy as np
import torch
from mardpg_uav.utils.metrics import MetricsTracker
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.train import load_config

def load_agents(checkpoint_dir, config_path, device='cpu'):
    cfg = load_config(config_path)
    env_cfg = cfg['environment']
    net_cfg = cfg['network']
    n_agents = env_cfg['n_agents']
    
    agents = []
    for i in range(n_agents):
        ag = MARDPGAgent(
            agent_id=i, n_agents=n_agents,
            obs_dim=env_cfg.get('obs_dim', 49), action_dim=2,
            action_bound=env_cfg.get('max_delta_angle', 0.5236),
            lstm_hidden=net_cfg.get('actor_lstm_hidden', 128),
            fc_hidden=net_cfg.get('critic_lstm_hidden', 128),
            device=device
        )
        try:
            ckpt = torch.load(f"{checkpoint_dir}/agent_{i}.pt", map_location=device)
            if 'actor_private' in ckpt:
                if i == 0:
                    sc = torch.load(f"{checkpoint_dir}/shared_actor.pt", map_location=device)
                    ag.shared_extractor.load_state_dict(sc['shared_actor'])
                ag.actor.load_state_dict(ckpt['actor_private'], strict=False)
            else:
                ag.actor.load_state_dict(ckpt['actor'])
        except Exception as e:
            print(f"[WARN] agent {i}: {e}")
        agents.append(ag)
        
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    return agents, cfg

def run_eval(env, stage_cfg, act_fn, n_episodes=50, base_seed=0, on_episode_start=None, collect_paths=True):
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
    import torch
    def on_start(env):
        for ag in agents:
            ag.hidden = None

    def act_fn():
        obs = env.get_obs()
        acts = np.zeros((env.n_agents, 2))
        for i, ag in enumerate(agents):
            if env.agent_done[i]: continue
            o = torch.FloatTensor(obs[i]).unsqueeze(0).to(ag.device)
            p = torch.FloatTensor(env.agents_state[i, 7:9]).unsqueeze(0).to(ag.device)
            # Unsqueeze for sequence length -> (1, 1, dim)
            o = o.unsqueeze(1)
            p = p.unsqueeze(1)
            with torch.no_grad():
                a, ag.hidden = ag.actor(o, p, ag.hidden)
            acts[i] = a.cpu().numpy()[0]
        return acts
        
    return act_fn, on_start
