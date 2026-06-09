"""
Measures env-step time vs. update time to decide whether a GPU is worthwhile.
Run on BOTH your CPU instance and the T4 before committing to a long paid run.

Usage:
    python benchmark.py --episodes 200
"""
import argparse, time, yaml, torch, numpy as np, copy
from mardpg_uav.environment.uav_env     import MultiUAVEnv
from mardpg_uav.algorithm.mardpg        import MARDPGAgent
from mardpg_uav.algorithm.replay_buffer import SequenceReplayBuffer
from mardpg_uav.networks.critic          import SharedCentralCritic

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--device',   default='cpu')
    args = ap.parse_args()

    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    env_cfg  = cfg['environment']
    algo_cfg = cfg['algorithm']
    net_cfg  = cfg['network']
    device   = args.device

    env      = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    obs_dim  = 32; act_dim = env.action_dim

    agents = []
    for i in range(n_agents):
        ag = MARDPGAgent(agent_id=i, n_agents=n_agents,
                         obs_dim=obs_dim, action_dim=act_dim, device=device)
        agents.append(ag)
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])

    shared_critic = SharedCentralCritic(n_agents, obs_dim, act_dim).to(device)
    shared_critic_target = copy.deepcopy(shared_critic).to(device)
    for ag in agents:
        ag.critic = shared_critic
        ag.critic_target = shared_critic_target

    shared_opt = torch.optim.Adam(agents[0].shared_extractor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(shared_critic.parameters(), lr=1e-3)

    buf = SequenceReplayBuffer(capacity=100_000,
                               seq_len=algo_cfg['seq_len']+algo_cfg['burn_in'],
                               n_agents=n_agents, obs_dim=obs_dim, action_dim=act_dim)

    stage_cfg = {'env_size': [100.,100.,60.], 'static_obs': 0,
                 'min_sep': 15., 'max_steps': 400, 'min_start_sep': 12.}
    obs = env.reset(stage_cfg)

    env_time    = 0.0
    update_time = 0.0
    ep_count    = 0
    total_steps = 0
    t_start     = time.time()

    print(f"Benchmarking {args.episodes} episodes on {device}…")

    for ep in range(args.episodes):
        for ag in agents:
            ag.reset_hidden(1)
        prev_actions = [np.zeros(act_dim, dtype=np.float32)] * n_agents

        for _ in range(stage_cfg['max_steps']):
            t0 = time.perf_counter()
            actions = env.action_space.sample()
            next_obs, rewards, done, info = env.step(actions)
            env_time += time.perf_counter() - t0

            buf.add_transition(obs.copy(), prev_actions,
                               actions, rewards, info['agent_done'].copy())
            total_steps += 1
            obs = next_obs
            prev_actions = list(actions)
            if done:
                break

        buf.end_episode()
        ep_count += 1

        # One update every ~100 steps
        if len(buf) >= algo_cfg['batch_size'] and ep % 5 == 0:
            batch = buf.sample(algo_cfg['batch_size'])
            if batch is not None:
                t0 = time.perf_counter()
                (b_obs, b_obs_next, b_prev_act, b_act, b_rew, b_done) = [
                    b.to(device) for b in batch]
                b_sz = algo_cfg['batch_size']
                seq  = algo_cfg['seq_len'] + algo_cfg['burn_in']
                bi   = algo_cfg['burn_in']

                obs_all  = b_obs.reshape(b_sz*seq, n_agents, -1)
                act_all  = b_act.reshape(b_sz*seq, n_agents, -1)
                nobs_all = b_obs_next.reshape(b_sz*seq, n_agents, -1)

                with torch.no_grad():
                    nf = agents[0].actor_target.shared(
                        b_obs_next.permute(0,2,1,3).reshape(b_sz*n_agents*seq, obs_dim)
                    ).view(b_sz, n_agents, seq, -1)
                    tgt = []
                    for i, ag in enumerate(agents):
                        xn = torch.cat([nf[:,i], b_act[:,:,i,:]], dim=-1)
                        hn,_ = ag.actor_target.lstm(xn, None)
                        tgt.append(ag.actor_target.tanh(
                            ag.actor_target.fc_out(hn)) * ag.actor_target.action_bound)
                nact_all = torch.stack([t.reshape(b_sz*seq,-1) for t in tgt], dim=1)

                shared_critic.train()
                h_cur = shared_critic.trunk(obs_all, act_all, seq)
                with torch.no_grad():
                    h_nxt = shared_critic_target.trunk(nobs_all, nact_all, seq)
                c_loss = sum(
                    (((shared_critic.q_from_trunk(h_cur,i).view(b_sz,seq)[:,bi:] -
                       (b_rew[:,:,i][:,bi:] + 0.99 *
                        shared_critic_target.q_from_trunk(h_nxt,i).view(b_sz,seq)[:,bi:] *
                        (~b_done[:,:,i][:,bi:]))).detach()) ** 2).mean()
                    for i in range(n_agents)) / n_agents
                critic_opt.zero_grad(); c_loss.backward(); critic_opt.step()

                for p in shared_critic.parameters(): p.requires_grad = False
                shared_opt.zero_grad()
                for ag in agents: ag.actor_optimizer.zero_grad()
                prev_act_all = b_prev_act.reshape(b_sz*seq, n_agents, -1)
                done_mask = ~torch.cat([torch.zeros(b_sz,1,n_agents,dtype=torch.bool,device=device),
                                        b_done[:,:-1,:]], dim=1)
                burn_mask = torch.arange(seq,device=device).view(1,-1,1) >= bi
                amask = (burn_mask|(b_done&done_mask))&done_mask
                al = sum(ag.compute_actor_loss(obs_all,act_all,prev_act_all,amask[:,:,ag.agent_id])
                         for ag in agents) / n_agents
                al.backward()
                shared_opt.step()
                for ag in agents: ag.actor_optimizer.step()
                for p in shared_critic.parameters(): p.requires_grad = True
                update_time += time.perf_counter() - t0

        obs = env.reset(stage_cfg)

    total_time = time.time() - t_start
    eps_per_sec = ep_count / total_time

    print(f"\n{'='*50}")
    print(f"Device          : {device}")
    print(f"Episodes        : {ep_count}")
    print(f"Total time      : {total_time:.1f}s")
    print(f"Throughput      : {eps_per_sec:.3f} ep/s")
    print(f"Env-step time   : {env_time:.2f}s  ({100*env_time/total_time:.1f}% of total)")
    print(f"Update time     : {update_time:.2f}s  ({100*update_time/total_time:.1f}% of total)")
    print(f"Other (IO etc.) : {total_time-env_time-update_time:.2f}s")
    print(f"\nBottleneck: {'ENV (GPU won\\'t help much)' if env_time > update_time else 'UPDATE (GPU will help)'}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
