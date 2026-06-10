import torch, numpy as np, yaml
from mardpg_uav.environment.uav_env   import MultiUAVEnv
from mardpg_uav.algorithm.mardpg      import MARDPGAgent
from mardpg_uav.algorithm.replay_buffer import SequenceReplayBuffer

def main():
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    env_cfg, algo_cfg = cfg['environment'], cfg['algorithm']
    device = 'cpu'

    env = MultiUAVEnv(env_cfg)
    n_agents, obs_dim, act_dim = env_cfg['n_agents'], 32, env.action_dim

    agents = [MARDPGAgent(agent_id=i, n_agents=n_agents, obs_dim=obs_dim, action_dim=act_dim, device=device) for i in range(n_agents)]
    for i in range(1, n_agents): agents[i].share_parameters(agents[0])

    buf = SequenceReplayBuffer(capacity=10_00, seq_len=algo_cfg['seq_len'] + algo_cfg['burn_in'], n_agents=n_agents, obs_dim=obs_dim, action_dim=act_dim)
    obs = env.reset()
    
    for _ in range(200):
        a = env.action_space.sample()
        no, r, d, info = env.step(a)
        buf.add_transition(obs.copy(), [np.zeros(act_dim)]*n_agents, a, r, info['agent_done'].copy())
        obs = env.reset() if d else no

    buf.end_episode()
    batch = buf.sample(16)
    (b_obs, b_obs_next, b_prev_act, b_act, b_rew, b_done) = batch
    
    b_sz, seq_len = 16, algo_cfg['seq_len'] + algo_cfg['burn_in']
    obs_all = b_obs.reshape(b_sz * seq_len, n_agents, -1)
    act_all = b_act.reshape(b_sz * seq_len, n_agents, -1)
    prev_act_all = b_prev_act.reshape(b_sz * seq_len, n_agents, -1)
    
    done_mask = ~torch.cat([torch.zeros(b_sz,1,n_agents,dtype=torch.bool), b_done[:,:-1,:]], dim=1)
    agent_mask = ((torch.arange(seq_len).view(1,-1,1) >= algo_cfg['burn_in']) | (b_done & done_mask)) & done_mask

    print("\n--- Diagnostic 1: dQ/da Action Sensitivity Probe ---")
    joint_actions = act_all.clone().detach()
    joint_actions.requires_grad_(True)
    q_vals, _ = agents[0].critic(obs_all, joint_actions, hidden=None, seq_len=seq_len)
    q_vals.sum().backward()
    
    dq_da_mean = joint_actions.grad[:, 0].abs().mean().item()
    assert dq_da_mean > 0, "CRITICAL: dQ/da is exactly zero. Critic is blind to actions."
    print(f"[PASS] Critic is sensitive to actions. dQ/da mean absolute gradient: {dq_da_mean:.6f}")

    print("\n--- Diagnostic 2: Actor Gradient Flow ---")
    agents[0].actor_optimizer.zero_grad()
    loss, valid_steps = agents[0].compute_actor_loss(obs_all, act_all, prev_act_all, agent_mask[:,:,0])
    loss.backward()

    actor_grad = sum(p.grad.abs().sum().item() for p in agents[0].actor_private_params if p.grad is not None)
    shared_grad = sum(p.grad.abs().sum().item() for p in agents[0].shared_extractor.parameters() if p.grad is not None)
    
    assert actor_grad > 0, "CRITICAL: No gradient flowing to actor private params."
    assert shared_grad > 0, "CRITICAL: No gradient flowing to shared extractor."
    print(f"[PASS] Actor gradients flow. Private sum: {actor_grad:.4f} | Shared sum: {shared_grad:.4f}")
    print("\n✅ All structural checks passed. Safe to train.")

if __name__ == "__main__":
    main()
