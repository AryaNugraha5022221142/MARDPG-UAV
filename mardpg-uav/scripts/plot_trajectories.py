import argparse
import numpy as np
from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.eval_rollout import load_agents, make_learned_act_fn
from scripts.train import CURRICULUM

def min_pair_dist(env):
    active = ~env.agent_done
    if active.sum() < 2:
        return np.nan
    P = env.agents_state[active, :3]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())

def segments_cross_xy(p0, p1, q0, q1):
    """2-D segment intersection test on the x-y plane (ignores z/timing)."""

    def ccw(a, b, c):
        return (c - a) * (b - a) - (b - a) * (c - a)
    d1 = ccw(q0, q1, p0)
    d2 = ccw(q0, q1, p1)
    d3 = ccw(p0, p1, q0)
    d4 = ccw(p0, p1, q1)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--variant', default='mardpg', choices=['mardpg', 'maddpg', 'ind_rdpg', 'iddpg'])
    ap.add_argument('--stage', type=int, default=5, help='1-based curriculum stage')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='traj')
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f'matplotlib required: {e}')
    (agents, cfg) = load_agents(args.checkpoint, args.config, device=args.device, variant=args.variant)
    env = MultiUAVEnv(cfg['environment'])
    stage_cfg = CURRICULUM[args.stage - 1]
    (act_fn, on_start) = make_learned_act_fn(agents, env)
    for ep in range(args.episodes):
        es = args.seed + ep
        np.random.seed(es)
        env.scene_gen.rng = np.random.RandomState(es)
        env.rangefinder.rng = np.random.RandomState(es)
        env.reset(stage_cfg)
        on_start(env)
        starts = env.agents_state[:, :3].copy()
        goals = env.goals.copy()
        paths = [env.agents_state[:, :3].copy()]
        mpd_series = []
        for _t in range(stage_cfg['max_steps']):
            acts = act_fn()
            (_, _, done, _info) = env.step(acts)
            paths.append(env.agents_state[:, :3].copy())
            mpd_series.append(min_pair_dist(env))
            if done:
                break
        P = np.array(paths)
        n = env.n_agents
        crossings = sum((segments_cross_xy(starts[i, :2], goals[i, :2], starts[j, :2], goals[j, :2]) for i in range(n) for j in range(i + 1, n)))
        closest = np.nanmin(mpd_series) if mpd_series else np.nan
        (fig, (axL, axR)) = plt.subplots(1, 2, figsize=(13, 5.5))
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        for i in range(n):
            axL.plot(P[:, i, 0], P[:, i, 1], color=colors[i], lw=1.6, alpha=0.9)
            axL.plot(starts[i, 0], starts[i, 1], 'o', color=colors[i], ms=8)
            axL.plot(goals[i, 0], goals[i, 1], 'x', color=colors[i], ms=10, mew=2)
            axL.plot([starts[i, 0], goals[i, 0]], [starts[i, 1], goals[i, 1]], '--', color=colors[i], lw=0.7, alpha=0.4)
        es_sz = cfg['environment']['env_size']
        axL.set_xlim(0, es_sz)
        axL.set_ylim(0, es_sz)
        axL.set_aspect('equal')
        axL.set_title(f'ep {ep}: top-down paths | straight-line crossings={crossings}')
        axL.set_xlabel('x (m)')
        axL.set_ylabel('y (m)')
        axR.plot(mpd_series, color='k', lw=1.5)
        axR.axhline(3.0, color='orange', ls='--', lw=1, label='near-miss band (3 m)')
        axR.axhline(1.0, color='red', ls='--', lw=1, label='collision (1 m)')
        axR.set_ylim(0, max(8.0, np.nanmax(mpd_series) if mpd_series else 8.0))
        axR.set_title(f'min pairwise distance | closest={closest:.2f} m')
        axR.set_xlabel('step')
        axR.set_ylabel('min pair distance (m)')
        axR.legend(loc='upper right', fontsize=8)
        out_png = f'{args.out}_stage{args.stage}_ep{ep}.png'
        fig.tight_layout()
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        verdict = 'OK: encounters happening' if crossings > 0 and closest < 6.0 else 'WEAK: few/no encounters — raise conflict_frac or lower ring_frac'
if __name__ == '__main__':
    main()