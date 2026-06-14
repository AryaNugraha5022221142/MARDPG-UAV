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
from mardpg_uav.apf import APFController
from mardpg_uav.eval_rollout import run_eval, make_apf_act_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--stage", type=int, default=7, help="1-based stage index")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    cfg = load_config(a.config)
    env_cfg = cfg["environment"]

    env = MultiUAVEnv(env_cfg)
    stage_cfg = CURRICULUM[a.stage - 1]
    ctrl = APFController(env)
    act_fn = make_apf_act_fn(env, ctrl)

    s, _ = run_eval(env, stage_cfg, act_fn, n_episodes=a.episodes, base_seed=a.seed)

    print(f"APF | stage {a.stage} | seed {a.seed} | n={a.episodes} | "
          f"success {s['success_rate']:.2%} | "
          f"collision {s['collision_rate']:.2%} | "
          f"dyn_collision {s['dyn_collision_rate']:.2%} | "
          f"trapped {s['trapped_rate']:.2%} | "
          f"path_eff {s['path_efficiency']:.2f} | "
          f"inter_uav_safe {s['inter_uav_safe']:.2f}")


if __name__ == "__main__":
    main()
