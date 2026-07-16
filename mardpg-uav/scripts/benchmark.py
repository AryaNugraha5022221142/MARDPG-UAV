import argparse, time, yaml, torch, numpy as np
from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.algorithm.replay_buffer import SequenceReplayBuffer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['environment']
    algo_cfg = cfg['algorithm']
    device = args.device
    env = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    obs_dim = env.obs_dim
    act_dim = env.action_dim
    seq = algo_cfg['seq_len'] + algo_cfg['burn_in']
    bi = algo_cfg['burn_in']
    b_sz = algo_cfg['batch_size']
    agents = [MARDPGAgent(agent_id=i, n_agents=n_agents, obs_dim=obs_dim, action_dim=act_dim, huber_beta=algo_cfg.get('huber_beta', 10.0), device=device) for i in range(n_agents)]
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    shared_opt = torch.optim.Adam(agents[0].shared_extractor.parameters(), lr=0.001)
    buf = SequenceReplayBuffer(capacity=100000, seq_len=seq, n_agents=n_agents, obs_dim=obs_dim, action_dim=act_dim)
    stage_cfg = {'env_size': [100.0, 100.0, 60.0], 'static_obs': 0, 'min_sep': 15.0, 'max_steps': 400, 'min_start_sep': 12.0}
    obs = env.reset(stage_cfg)
    (env_time, update_time) = (0.0, 0.0)
    (ep_count, total_steps) = (0, 0)
    t_start = time.time()
    for ep in range(args.episodes):
        for ag in agents:
            ag.reset_hidden(1)
        prev_actions = np.zeros((n_agents, act_dim), dtype=np.float32)
        for _ in range(stage_cfg['max_steps']):
            t0 = time.perf_counter()
            actions = env.action_space.sample()
            (next_obs, rewards, done, info) = env.step(actions)
            env_time += time.perf_counter() - t0
            buf.add_transition(obs.copy(), prev_actions.copy(), actions, rewards, next_obs.copy(), info['agent_done'].copy())
            total_steps += 1
            obs = next_obs
            prev_actions = np.array(actions, dtype=np.float32)
            if done:
                break
        buf.end_episode()
        ep_count += 1
        if len(buf) >= b_sz and ep % 5 == 0:
            batch = buf.sample(b_sz)
            if batch is not None:
                t0 = time.perf_counter()
                (b_obs, b_nobs, b_prev_act, b_act, b_rew, b_done, b_pad) = [b.to(device) for b in batch]
                obs_all = b_obs.reshape(b_sz * seq, n_agents, -1)
                nobs_all = b_nobs.reshape(b_sz * seq, n_agents, -1)
                act_all = b_act.reshape(b_sz * seq, n_agents, -1)
                with torch.no_grad():
                    nf = agents[0].actor_target.shared(b_nobs.permute(0, 2, 1, 3).reshape(b_sz * n_agents * seq, obs_dim)).view(b_sz, n_agents, seq, -1)
                    tgt = []
                    for (i, ag) in enumerate(agents):
                        xn = torch.cat([nf[:, i], b_act[:, :, i, :]], dim=-1)
                        (hn, _) = ag.actor_target.lstm(xn, None)
                        tgt.append(ag.actor_target.tanh(ag.actor_target.fc_out(hn)) * ag.actor_target.action_bound)
                nact_all = torch.stack([t.reshape(b_sz * seq, -1) for t in tgt], dim=1)
                burn_mask = torch.arange(seq, device=device).view(1, -1, 1) >= bi
                alive_prev = ~torch.cat([torch.zeros(b_sz, 1, n_agents, dtype=torch.bool, device=device), b_done[:, :-1, :]], dim=1)
                amask = burn_mask & alive_prev & (~b_pad).unsqueeze(-1)
                for (i, ag) in enumerate(agents):
                    ag.critic_optimizer.zero_grad()
                    (cl_i, _, valid_steps) = ag.compute_critic_loss(obs_all, act_all, nobs_all, nact_all, b_rew[:, :, i], b_done[:, :, i], amask[:, :, i])
                    if valid_steps > 0:
                        cl_i.backward()
                        ag.critic_optimizer.step()
                for ag in agents:
                    for p in ag.critic.parameters():
                        p.requires_grad = False
                shared_opt.zero_grad()
                for ag in agents:
                    ag.actor_optimizer.zero_grad()
                prev_act_all = b_prev_act.reshape(b_sz * seq, n_agents, -1)
                actor_results = [ag.compute_actor_loss(obs_all, act_all, prev_act_all, amask[:, :, ag.agent_id]) for ag in agents]
                actor_losses = [r[0] for r in actor_results if r[1] > 0]
                if actor_losses:
                    (sum(actor_losses) / len(actor_losses)).backward()
                    shared_opt.step()
                    for ag in agents:
                        ag.actor_optimizer.step()
                for ag in agents:
                    for p in ag.critic.parameters():
                        p.requires_grad = True
                if device == 'cuda':
                    torch.cuda.synchronize()
                update_time += time.perf_counter() - t0
        obs = env.reset(stage_cfg)
    total_time = time.time() - t_start
    eps_per_sec = ep_count / total_time
    bottleneck = "ENV (GPU won't help much — raise grad_steps_per_update and/or parallelise envs)" if env_time > update_time else 'UPDATE (GPU will help)'
if __name__ == '__main__':
    main()
