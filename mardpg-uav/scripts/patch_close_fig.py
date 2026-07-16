import re

with open('scripts/evaluate_multiagent.py', 'r') as f:
    content = f.read()

# Add plt.close() at the end of run_episode
close_code = """
    if render_rt:
        import matplotlib.pyplot as plt
        if hasattr(env, '_rt_fig'):
            plt.close(env._rt_fig)
            delattr(env, '_rt_fig')
"""

if 'plt.close(env._rt_fig)' not in content:
    content = re.sub(
        r"    return ep",
        close_code.lstrip('\n') + "    return ep",
        content
    )

with open('scripts/evaluate_multiagent.py', 'w') as f:
    f.write(content)
