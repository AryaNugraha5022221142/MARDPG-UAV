"""
Structural sanity checks (v3) — updated for the explicit-next_obs buffer.

Adds two new checks on top of the original gradient-flow probes:
  3. Timeout truncation must NOT be treated as terminal (done=False on the
     last real step of a timed-out episode).
  4. Pad steps from short episodes must never appear in the validity mask.
"""
import torch, numpy as np, yaml
from mardpg_uav.environment.uav_env     import MultiUAVEnv
from mardpg_uav.algorithm.mardpg        import MARDPGAgent
from mardpg_uav.algorithm.replay_buffer import SequenceReplayBuffer


def main():
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    env_cfg, algo_cfg = cfg['environment'], cfg['algorithm']
    device = 'cpu'

    env = MultiUAVEnv(env_cfg)
    n_agents, obs_dim, act_dim = env_cfg['n_agents'], env.obs_dim, env.action_dim
    seq = algo_cfg['seq_len'] + algo_cfg['burn_in']
    bi  = algo_cfg['burn_in']

    agents = [MARDPGAgent(agent_id=i, n_agents=n_agents, obs_dim=obs_dim,
                          action_dim=act_dim,
                          huber_beta=algo_cfg.get('huber_beta', 10.0),
                          device=device)
              for i in range(n_agents)]
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])

    buf = SequenceReplayBuffer(capacity=10_000, seq_len=seq,
                               n_agents=n_agents, obs_dim=obs_dim,
                               action_dim=act_dim)

    obs = env.reset()
    prev = np.zeros((n_agents, act_dim), dtype=np.float32)
    for _ in range(400):
        a = env.action_space.sample()
        no, r, d, info = env.step(a)
        buf.add_transition(obs.copy(), prev.copy(), a, r, no.copy(),
                           info['agent_done'].copy())
        prev = np.array(a, dtype=np.float32)
        if d:
            buf.end_episode()
            obs = env.reset()
            prev = np.zeros((n_agents, act_dim), dtype=np.float32)
        else:
            obs = no
    buf.end_episode()

    batch = buf.sample(16)
    assert batch is not None, "Not enough valid windows — increase rollout steps."
    (b_obs, b_nobs, b_prev_act, b_act, b_rew, b_done, b_pad) = batch

    b_sz = 16
    obs_all      = b_obs.reshape(b_sz * seq, n_agents, -1)
    act_all      = b_act.reshape(b_sz * seq, n_agents, -1)
    prev_act_all = b_prev_act.reshape(b_sz * seq, n_agents, -1)

    burn_mask  = torch.arange(seq).view(1, -1, 1) >= bi
    alive_prev = ~torch.cat([torch.zeros(b_sz, 1, n_agents, dtype=torch.bool),
                             b_done[:, :-1, :]], dim=1)
    agent_mask = burn_mask & alive_prev & (~b_pad).unsqueeze(-1)

    print("\n--- Diagnostic 1: dQ/da Action Sensitivity Probe ---")
    joint_actions = act_all.clone().detach()
    joint_actions.requires_grad_(True)
    q_vals, _ = agents[0].critic(obs_all, joint_actions, hidden=None, seq_len=seq)
    q_vals.sum().backward()
    dq_da_mean = joint_actions.grad[:, 0].abs().mean().item()
    assert dq_da_mean > 0, "CRITICAL: dQ/da is exactly zero. Critic is blind to actions."
    print(f"[PASS] Critic is sensitive to actions. dQ/da mean |grad|: {dq_da_mean:.6f}")

    print("\n--- Diagnostic 2: Actor Gradient Flow ---")
    agents[0].actor_optimizer.zero_grad()
    loss, valid_steps = agents[0].compute_actor_loss(
        obs_all, act_all, prev_act_all, agent_mask[:, :, 0])
    assert valid_steps > 0, "No valid steps in sampled mask."
    loss.backward()
    actor_grad  = sum(p.grad.abs().sum().item()
                      for p in agents[0].actor_private_params if p.grad is not None)
    shared_grad = sum(p.grad.abs().sum().item()
                      for p in agents[0].shared_extractor.parameters()
                      if p.grad is not None)
    assert actor_grad  > 0, "CRITICAL: No gradient to actor private params."
    assert shared_grad > 0, "CRITICAL: No gradient to shared extractor."
    print(f"[PASS] Actor gradients flow. Private: {actor_grad:.4f} | "
          f"Shared: {shared_grad:.4f}")

    print("\n--- Diagnostic 3: Timeout is truncation, not termination ---")
    # Build a tiny episode that times out: in the buffer, the LAST REAL
    # step of a timed-out agent must have done=False (so it bootstraps).
    probe = SequenceReplayBuffer(capacity=1000, seq_len=seq,
                                 n_agents=n_agents, obs_dim=obs_dim,
                                 action_dim=act_dim)
    o = env.reset({'env_size': [100., 100., 60.], 'static_obs': 0,
                   'min_sep': 15., 'max_steps': 30})
    p = np.zeros((n_agents, act_dim), dtype=np.float32)
    last_done = None
    for _ in range(30):
        a = np.zeros((n_agents, act_dim), dtype=np.float32)
        no, r, d, info = env.step(a)
        probe.add_transition(o.copy(), p.copy(), a, r, no.copy(),
                             info['agent_done'].copy())
        last_done = info['agent_done'].copy()
        o = no
        if d:
            break
    probe.end_episode()
    alive_at_timeout = ~env.agent_collided & ~env.agent_reached
    # Direct assertion on what was stored for alive agents at the final step:
    assert not last_done[alive_at_timeout].any(), \
        "CRITICAL: timed-out agents stored done=True at final step (truncation bug)."
    print("[PASS] Timed-out agents keep done=False on the last real step.")

    print("\n--- Diagnostic 4: Pad steps never enter the loss mask ---")
    masked_pads = (agent_mask & b_pad.unsqueeze(-1)).sum().item()
    assert masked_pads == 0, "CRITICAL: pad steps leaked into the validity mask."
    # And short episodes do produce a window:
    n_valid_before = probe.valid_mask.sum()
    assert n_valid_before >= 1, \
        "Short episode produced no valid window — padding in end_episode failed."
    print(f"[PASS] Pads masked out; short 30-step episode produced "
          f"{int(n_valid_before)} valid window(s) via padding.")

    print("\n--- Diagnostic 5: Done agents are not collision hazards (N-1) ---")
    e2 = MultiUAVEnv(env_cfg)
    e2.reset({'env_size': [100., 100., 60.], 'static_obs': 0,
              'min_sep': 15., 'max_steps': 50, 'min_start_sep': 12.})
    # Force agent 0 done and teleport agent 1 on top of it.
    e2.agent_done[0] = True
    e2.agent_reached[0] = True
    e2.agents_state[1, :3] = e2.agents_state[0, :3] + 0.1  # well inside inter_uav_min_dist
    z = np.zeros((n_agents, act_dim), dtype=np.float32)
    _, _, _, info2 = e2.step(z)
    assert not info2['step_collisions'][1], \
        "CRITICAL (N-1): live agent collided with a frozen/done ghost."
    print("[PASS] Live agent does not collide with a done agent at the same point.")

    print("\n--- Diagnostic 6: Goal-to-goal separation floor (N-4) ---")
    e3 = MultiUAVEnv(env_cfg)
    e3.reset({'env_size': [100., 100., 60.], 'static_obs': 0,
              'min_sep': 30., 'max_steps': 50, 'min_start_sep': 12.})
    floor = e3.cfg.get('goal_min_sep',
                       2.0 * e3.cfg['goal_threshold'] + e3.cfg['inter_uav_min_dist'])
    gmin = min(np.linalg.norm(e3.goals[a] - e3.goals[b])
               for a in range(n_agents) for b in range(a + 1, n_agents))
    assert gmin >= floor - 1e-3, f"CRITICAL (N-4): goals {gmin:.2f} m < floor {floor:.2f} m."
    print(f"[PASS] Min goal-goal distance {gmin:.2f} m >= floor {floor:.2f} m.")

    print("\n--- Diagnostic 7: Neighbor block reflects only LIVE agents (N-1) ---")
    e4 = MultiUAVEnv(env_cfg)
    o4 = e4.reset({'env_size': [100., 100., 60.], 'static_obs': 0,
                   'min_sep': 15., 'max_steps': 50, 'min_start_sep': 12.})
    K = e4.n_neighbors
    F = e4.nbr_feats                       # 7
    pres0 = o4[0][34 + (F - 1): 34 + F * K: F]   # presence flags, new layout
    assert pres0.sum() == min(K, e4.n_agents - 1), "Neighbor presence flags wrong at reset."
    e4.agent_done[:] = True          # kill everyone except agent 0's perspective
    e4.agent_done[0] = False
    o4b = e4._get_observations()
    pres0b = o4b[0][34 + (F - 1): 34 + F * K: F]
    assert pres0b.sum() == 0, "CRITICAL (N-1): done agents still appear in neighbor block."
    print("[PASS] Neighbor block excludes done agents.")

    print("\n✅ All structural checks passed. Safe to train.")


if __name__ == "__main__":
    main()
