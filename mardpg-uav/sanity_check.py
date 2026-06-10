"""
Run before any paid GPU session.
Checks shapes, forward pass, one full update cycle for independent critics.
"""
import torch, numpy as np, yaml
from mardpg_uav.environment.uav_env   import MultiUAVEnv
from mardpg_uav.algorithm.mardpg      import MARDPGAgent
from mardpg_uav.algorithm.replay_buffer import SequenceReplayBuffer

CFG_PATH = "config/default.yaml"

def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    env_cfg  = cfg['environment']
    algo_cfg = cfg['algorithm']
    device   = 'cpu'

    env      = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    obs_dim  = 32; act_dim = env.action_dim

    # Build agents
    agents = []
    for i in range(n_agents):
        ag = MARDPGAgent(agent_id=i, n_agents=n_agents,
                         obs_dim=obs_dim, action_dim=act_dim, device=device)
        agents.append(ag)
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])

    shared_opt  = torch.optim.Adam(agents[0].shared_extractor.parameters(), lr=1e-3)

    # 1. Observation shape
    obs = env.reset({'env_size': [100.,100.,60.], 'static_obs': 0,
                     'min_sep': 15., 'max_steps': 400, 'min_start_sep': 12.})
    assert obs.shape == (n_agents, obs_dim), f"Bad obs shape: {obs.shape}"
    print(f"[PASS] obs shape {obs.shape}")

    # 2. Fill replay buffer with random transitions
    buf = SequenceReplayBuffer(capacity=10_000,
                               seq_len=algo_cfg['seq_len'] + algo_cfg['burn_in'],
                               n_agents=n_agents, obs_dim=obs_dim, action_dim=act_dim)
    for _ in range(200):
        a  = env.action_space.sample()
        no, r, d, info = env.step(a)
        buf.add_transition(obs.copy(), [np.zeros(act_dim)]*n_agents,
                           a, r, info['agent_done'].copy())
        obs = no
        if d:
            obs = env.reset({'env_size': [100.,100.,60.], 'static_obs': 0,
                             'min_sep': 15., 'max_steps': 400, 'min_start_sep': 12.})
            buf.end_episode()
    buf.end_episode()
    print(f"[PASS] buffer size {len(buf)}")

    batch = buf.sample(16)
    assert batch is not None, "Buffer sample returned None"
    (b_obs, b_obs_next, b_prev_act, b_act, b_rew, b_done) = batch
    b_sz, seq_len = 16, algo_cfg['seq_len'] + algo_cfg['burn_in']
    bi = algo_cfg['burn_in']

    obs_all  = b_obs.reshape(b_sz * seq_len, n_agents, -1)
    act_all  = b_act.reshape(b_sz * seq_len, n_agents, -1)
    nobs_all = b_obs_next.reshape(b_sz * seq_len, n_agents, -1)
    nact_all = act_all.clone()

    done_mask = ~torch.cat([torch.zeros(b_sz,1,n_agents,dtype=torch.bool),
                            b_done[:,:-1,:]], dim=1)
    burn_mask = torch.arange(seq_len).view(1,-1,1) >= bi
    agent_mask = (burn_mask | (b_done & done_mask)) & done_mask

    # 3. Independent critics logic check
    for p in agents[0].critic.parameters(): p.requires_grad = True

    c_loss_total = torch.zeros(1)
    for i, ag in enumerate(agents):
        ag.critic_optimizer.zero_grad()
        cl_i, q_learn = ag.compute_critic_loss(
            obs_all, act_all, nobs_all, nact_all,
            b_rew[:, :, i], b_done[:, :, i], agent_mask[:, :, i]
        )
        cl_i.backward()
        ag.critic_optimizer.step()
        c_loss_total = c_loss_total + cl_i
        
    assert torch.isfinite(c_loss_total), "Critic loss is not finite"
    print(f"[PASS] independent critic avg loss {(c_loss_total/n_agents).item():.4f}")

    # 4. One actor update
    for ag in agents:
        for p in ag.critic.parameters(): p.requires_grad = False
        
    shared_opt.zero_grad()
    for ag in agents: ag.actor_optimizer.zero_grad()
    
    prev_act_all = b_prev_act.reshape(b_sz * seq_len, n_agents, -1)
    al = sum(ag.compute_actor_loss(obs_all, act_all, prev_act_all, agent_mask[:,:,ag.agent_id])
             for ag in agents) / n_agents
    al.backward()
    shared_opt.step()
    for ag in agents: 
        ag.actor_optimizer.step()
        for p in ag.critic.parameters(): p.requires_grad = True
        
    assert torch.isfinite(al), "Actor loss is not finite"
    print(f"[PASS] actor loss {al.item():.4f}")

    print("\n✅  All checks passed — safe to start a paid run.")

if __name__ == "__main__":
    main()
