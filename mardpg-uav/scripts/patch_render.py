import re

with open('scripts/evaluate_multiagent.py', 'r') as f:
    content = f.read()

# Add --render arg
if '--render' not in content:
    content = re.sub(
        r"p.add_argument\('--video', action='store_true', help='Generate video/animation of episodes'\)",
        "p.add_argument('--video', action='store_true', help='Generate video/animation of episodes')\n    p.add_argument('--render', action='store_true', help='Render the environment in real time')",
        content
    )
    
    # Pass it to evaluate
    content = re.sub(
        r"a\.base_seed, a\.suite == 'quick', wandb_log=a\.wandb, video=a\.video\)",
        "a.base_seed, a.suite == 'quick', wandb_log=a.wandb, video=a.video, render_rt=a.render)",
        content
    )
    
    content = re.sub(
        r"def evaluate\(methods, config, episodes, device, outdir, base_seed, quick, wandb_log=False, video=False\):",
        "def evaluate(methods, config, episodes, device, outdir, base_seed, quick, wandb_log=False, video=False, render_rt=False):",
        content
    )

    # Pass render_rt to run_episode
    content = re.sub(
        r"ep = run_episode\(env, provider, stage_cfg, env_cfg, scene_seed, capture_render=capture\)",
        "ep = run_episode(env, provider, stage_cfg, env_cfg, scene_seed, capture_render=capture, render_rt=render_rt)",
        content
    )
    content = re.sub(
        r"best_ep = run_episode\(env, provider, stage_cfg, env_cfg, best_seed, capture_render=True\)",
        "best_ep = run_episode(env, provider, stage_cfg, env_cfg, best_seed, capture_render=True, render_rt=render_rt)",
        content
    )

with open('scripts/evaluate_multiagent.py', 'w') as f:
    f.write(content)
