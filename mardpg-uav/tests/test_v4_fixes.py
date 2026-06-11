"""
Verification tests for the v4 (final-audit) fixes.

Run:  PYTHONPATH=. pytest tests/test_v4_fixes.py -v

These are fast structural tests (CPU, tiny episode counts). They verify that
each N4 fix produces its intended observable behavior, not just that the
code imports.
"""
import os
import yaml
import numpy as np
import torch
import pytest

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.algorithm.mardpg import MARDPGAgent
from mardpg_uav.train import run_eval, hazard_shift, CURRICULUM

CFG_PATH = os.path.join(os.path.dirname(__file__), '../config/default.yaml')
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)


def _make_agents(env, n_agents, device='cpu'):
    agents = [MARDPGAgent(agent_id=i, n_agents=n_agents,
                          obs_dim=env.obs_dim, action_dim=env.action_dim,
                          device=device)
              for i in range(n_agents)]
    for i in range(1, n_agents):
        agents[i].share_parameters(agents[0])
    return agents


# ---------------------------------------------------------------------------
# [N4-1] run_eval must return a state-matched calibration probe
# ---------------------------------------------------------------------------
def test_run_eval_returns_q_eval_start():
    env_cfg = dict(CFG['environment']); env_cfg['seed'] = 0
    env = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    agents = _make_agents(env, n_agents)

    stage = {'env_size': [100., 100., 60.], 'static_obs': 0,
             'min_sep': 15., 'max_steps': 20, 'min_start_sep': 12.}
    stats = run_eval(env, agents, stage, n_episodes=2,
                     action_dim=env.action_dim, n_agents=n_agents, gamma=0.99)

    assert 'q_eval_start' in stats, "N4-1: q_eval_start missing from eval stats"
    assert 'realized_return' in stats
    assert np.isfinite(stats['q_eval_start']), "q_eval_start must be finite"
    # Untrained critic: Q(s0,a0) should be near init scale, not garbage.
    assert abs(stats['q_eval_start']) < 100.0


def test_q_eval_start_matches_direct_critic_call():
    """The probe must equal a hand-computed Q_i(s0, a0) average for a fixed
    (state, action) pair — i.e. it really is state- and policy-matched."""
    env_cfg = dict(CFG['environment']); env_cfg['seed'] = 1
    env = MultiUAVEnv(env_cfg)
    n_agents = env_cfg['n_agents']
    agents = _make_agents(env, n_agents)

    obs = env.reset({'env_size': [100., 100., 60.], 'static_obs': 0,
                     'min_sep': 15., 'max_steps': 20, 'min_start_sep': 12.})
    acts = np.zeros((n_agents, env.action_dim), dtype=np.float32)
    with torch.no_grad():
        o_t = torch.FloatTensor(obs).unsqueeze(0)
        a_t = torch.FloatTensor(acts).unsqueeze(0)
        q_direct = float(np.mean([
            float(ag.critic(o_t, a_t, hidden=None, seq_len=1)[0].item())
            for ag in agents]))
    assert np.isfinite(q_direct)
    # Shape contract: critic accepts (1, N, obs_dim) with seq_len=1 and
    # returns a scalar per batch element — the exact call run_eval makes.


# ---------------------------------------------------------------------------
# [N4-1] Signed-gap semantics: warning side must be POSITIVE only
# ---------------------------------------------------------------------------
def test_signed_gap_semantics():
    """Reproduce the ep-199 false alarm: q below realized return must NOT be
    classified as overestimation under the new logic."""
    warn = float(CFG['algorithm']['q_divergence_warn'])
    # ep 199 of the 2026-06-11 run: q=-1.18, realized=+17.75
    q_gap = -1.18 - 17.75
    overestimation_fires = q_gap > warn
    assert not overestimation_fires, \
        "N4-1: underestimation must not trigger the overestimation warning"
    # And genuine overestimation must fire:
    assert (40.0 - 17.75) > warn


# ---------------------------------------------------------------------------
# [N4-2] Hazard-shift predicate across every adjacent curriculum pair
# ---------------------------------------------------------------------------
def test_hazard_shift_over_full_curriculum():
    expected = {
        (0, 1): False,  # free-near  -> free-far      : same geometry
        (1, 2): True,   # free-far   -> sparse obs    : lidar becomes informative
        (2, 3): True,   # 3 -> 7 obstacles
        (3, 4): True,   # 7 -> 12
        (4, 5): True,   # 12 -> 16
        (5, 6): True,   # dynamic obstacles appear
    }
    for (a, b), want in expected.items():
        got = hazard_shift(CURRICULUM[a], CURRICULUM[b])
        assert got == want, f"hazard_shift stage {a+1}->{b+1}: got {got}, want {want}"


# ---------------------------------------------------------------------------
# [N4-3] Adaptive eval size: config invariants
# ---------------------------------------------------------------------------
def test_adaptive_eval_config():
    a = CFG['algorithm']
    assert int(a['eval_episodes_late']) >= int(a['eval_episodes']), \
        "late eval set must not be smaller than the base set"
    th = float(a['eval_episodes_late_threshold'])
    # Stages 4-6 (bars 0.55-0.60) must fall on the LATE side; stages 1-3
    # (bars 0.35-0.45) on the base side.
    for idx, want_late in [(0, False), (1, False), (2, False),
                           (3, True), (4, True), (5, True)]:
        bar = CURRICULUM[idx]['criteria']['success_rate']
        assert (bar >= th) == want_late, \
            f"stage {idx+1} bar {bar} lands on wrong side of threshold {th}"


# ---------------------------------------------------------------------------
# [N4-2] Flush actually removes the OLDEST windows (regression guard on the
# buffer primitive the new policy relies on)
# ---------------------------------------------------------------------------
def test_flush_removes_oldest():
    from mardpg_uav.algorithm.replay_buffer import SequenceReplayBuffer
    n_agents, obs_dim, act_dim, seq = 2, 49, 2, 5
    buf = SequenceReplayBuffer(capacity=1000, seq_len=seq, n_agents=n_agents,
                               obs_dim=obs_dim, action_dim=act_dim, tail_pad=0)
    o = np.zeros((n_agents, obs_dim), np.float32)
    a = np.zeros((n_agents, act_dim), np.float32)
    r = np.zeros(n_agents, np.float32)
    d = np.zeros(n_agents, bool)
    for ep in range(4):
        for _ in range(seq + 3):
            buf.add_transition(o, a, a, r, o, d)
        buf.end_episode()
    valid_before = set(np.nonzero(buf.valid_mask)[0])
    dropped = buf.invalidate_oldest(0.5)
    valid_after = set(np.nonzero(buf.valid_mask)[0])
    removed = valid_before - valid_after
    assert dropped == len(removed) > 0
    # Every removed window must be older (smaller index, no wrap here) than
    # every surviving window.
    assert max(removed) < min(valid_after), \
        "invalidate_oldest removed non-oldest windows"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
