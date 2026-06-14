#!/usr/bin/env python3
"""
Evaluate the APF baseline on a curriculum stage (default stage 7) over N
episodes, reporting the SAME metrics as the learned policies so the numbers are
directly comparable. APF doesn't learn, so this single eval run is all it needs.
Run multiple --seed values and aggregate the same way you do the learned runs.

Run from the repo root (so `mardpg_uav` is importable):
    python evaluate_apf.py --stage 7 --episodes 200 --seed 1
"""
import argparse
import numpy as np

from mardpg_uav.train import load_config, CURRICULUM
from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.utils.metrics import MetricsTracker
from mardpg_uav.apf import APFController


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--stage", type=int, default=7, help="1-based stage index")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    cfg = load_config(a.config)
    env_cfg = cfg["environment"]
    env_cfg["seed"] = a.seed
    np.random.seed(a.seed)

    env = MultiUAVEnv(env_cfg)
    env.action_space.seed(a.seed)
    stage_cfg = CURRICULUM[a.stage - 1]
    ctrl = APFController(env)
    m = MetricsTracker()

    for _ in range(a.episodes):
        env.reset(stage_cfg)
        ph = [env.agents_state[:, :3].copy()]
        info = {}
        ep_r, L = 0.0, 0
        for _t in range(stage_cfg["max_steps"]):
            acts = ctrl.act()
            _, r, done, info = env.step(acts)
            ep_r += float(sum(r))
            L += 1
            ph.append(env.agents_state[:, :3].copy())
            if done:
                break
        m.record_episode(length=L, info=info,
                         start_pos=[ph[0][i] for i in range(env.n_agents)],
                         goal_pos=[env.goals[i] for i in range(env.n_agents)],
                         path_history=ph, rewards=[ep_r])

    s = m.get_window_stats(a.episodes)
    print(f"APF | stage {a.stage} | seed {a.seed} | n={a.episodes} | "
          f"success {s['success_rate']:.2%} | "
          f"collision {s['collision_rate']:.2%} | "
          f"dyn_collision {s['dyn_collision_rate']:.2%} | "
          f"trapped {s['trapped_rate']:.2%} | "
          f"path_eff {s['path_efficiency']:.2f} | "
          f"inter_uav_safe {s['inter_uav_safe']:.2f}")


if __name__ == "__main__":
    main()
