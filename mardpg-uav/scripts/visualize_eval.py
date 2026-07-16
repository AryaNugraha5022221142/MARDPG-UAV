import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import argparse
import numpy as np

from mardpg_uav.environment.uav_env import MultiUAVEnv
from mardpg_uav.rendering import RenderConfig, select_backend, plot_summary
from mardpg_uav.rendering.media import generate_episode_media
from scripts.train import CURRICULUM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints')
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--stage', type=int, default=len(CURRICULUM),
                   help='Curriculum stage (1..N) to evaluate in.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='trajectory_eval.png')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--animate', action='store_true', help='Also write an MP4.')
    p.add_argument('--video-fps', type=int, default=20)
    args = p.parse_args()

    # Static figures only need Agg; --animate still works headless via Agg.
    select_backend('auto', want_interactive=False)

    from mardpg_uav.eval_rollout import load_agents, run_eval, make_learned_act_fn

    if not os.path.exists(args.config):
        fb = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
        if os.path.exists(fb):
            args.config = fb

    agents, cfg = load_agents(args.checkpoint, args.config, args.device)
    env_cfg = cfg['environment']
    env = MultiUAVEnv(env_cfg)

    stage_cfg = dict(CURRICULUM[max(0, min(args.stage, len(CURRICULUM)) - 1)])
    stage_cfg['max_steps'] = env_cfg.get('max_steps_per_episode',
                                         stage_cfg.get('max_steps', 1500))
    stage_name = stage_cfg.get('name', f'stage {args.stage}')

    act_fn, on_start = make_learned_act_fn(agents, env)
    _, m = run_eval(env, stage_cfg, act_fn, n_episodes=1, base_seed=args.seed,
                    on_episode_start=on_start, collect_paths=True)

    info = m.episodes['info']
    path = np.array(m.episodes['path_history'])
    rnd = dict(
        path=path,
        goals=env.goals.copy(),
        reached=info.get('reached', np.zeros(env.n_agents, bool)),
        collided=info.get('collisions', np.zeros(env.n_agents, bool)),
        dyn_path=info.get('dyn_path', None),
        dyn_r=info.get('dyn_r', []),
    )

    plot_summary(env, env_cfg, rnd, stage_name, args.out)
    print(f"[visualize_eval] wrote summary figure -> {args.out}")

    if args.animate:
        rcfg = RenderConfig(enable_render=True, record_video=True,
                            save_png=False, video_fps=args.video_fps,
                            output_directory=os.path.dirname(os.path.abspath(args.out)) or '.')
        tag = os.path.splitext(os.path.basename(args.out))[0]
        produced = generate_episode_media(env, env_cfg, rnd, rcfg, tag,
                                          stage_name, rcfg.output_directory)
        print(f"[visualize_eval] media -> {produced}")


if __name__ == '__main__':
    main()
