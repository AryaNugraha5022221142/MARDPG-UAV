import re

with open('scripts/evaluate_multiagent.py', 'r') as f:
    content = f.read()

# Add render_rt to run_episode def
if 'render_rt=False' not in content.split('def run_episode(')[1].split('):')[0]:
    content = re.sub(
        r"def run_episode\(env, policy, stage_cfg, env_cfg, seed, capture_render=False\):",
        "def run_episode(env, policy, stage_cfg, env_cfg, seed, capture_render=False, render_rt=False):",
        content
    )

# Setup real-time plot
setup_code = """
    if render_rt:
        import matplotlib.pyplot as plt
        if not hasattr(env, '_rt_fig'):
            plt.ion()
            env._rt_fig = plt.figure()
            env._rt_ax = env._rt_fig.add_subplot(111, projection='3d')
            env._rt_scats = []
            for _ in range(n_agents):
                env._rt_scats.append(env._rt_ax.plot([], [], [], marker='o', ls='')[0])
            from visualize_eval import _draw_static_obstacles
            _draw_static_obstacles(env._rt_ax, env, max_z=env_cfg['env_size'][2])
        
        env._rt_ax.set_xlim(0, env_cfg['env_size'][0])
        env._rt_ax.set_ylim(0, env_cfg['env_size'][1])
        env._rt_ax.set_zlim(0, env_cfg['env_size'][2])
        for i in range(n_agents):
            env._rt_ax.scatter(*env.goals[i], marker='*', color='blue')
"""

if 'if render_rt:' not in content:
    content = re.sub(
        r"    cum_reward = np\.zeros\(n_agents\)",
        setup_code.lstrip('\n') + "\n    cum_reward = np.zeros(n_agents)",
        content
    )

loop_code = """
        if render_rt:
            import matplotlib.pyplot as plt
            pos = env.agents_state[:, :3]
            for i in range(n_agents):
                if not env.agent_done[i]:
                    env._rt_scats[i].set_data([pos[i, 0]], [pos[i, 1]])
                    env._rt_scats[i].set_3d_properties([pos[i, 2]])
            plt.pause(0.001)
"""

if 'pos = env.agents_state[:, :3]' not in content:
    content = re.sub(
        r"        if capture_render and dyn:",
        loop_code.lstrip('\n') + "        if capture_render and dyn:",
        content
    )

with open('scripts/evaluate_multiagent.py', 'w') as f:
    f.write(content)
