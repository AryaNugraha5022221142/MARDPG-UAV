import re

with open('mardpg-uav/scripts/visualize_eval.py', 'r') as f:
    content = f.read()

replacement = """
def _draw_cylinder(ax, pos, r, h, color=OBSTACLE_GRAY, alpha=0.3):
    (z0, z1) = (pos[2] - h / 2.0, pos[2] + h / 2.0)
    theta = np.linspace(0, 2 * np.pi, 32)
    zc = np.linspace(z0, z1, 2)
    (tg, zg) = np.meshgrid(theta, zc)
    xg = pos[0] + r * np.cos(tg)
    yg = pos[1] + r * np.sin(tg)
    ax.plot_surface(xg, yg, zg, color=color, alpha=alpha, linewidth=0, shade=True)
"""

content = re.sub(
    r"def _draw_cylinder\(ax, pos, r, h, color=OBSTACLE_GRAY, alpha=0\.3\):.*?ax\.plot_surface\(xg, yg, zg, color=color, alpha=alpha, linewidth=0, shade=True\)",
    replacement.strip('\n'),
    content,
    flags=re.DOTALL
)

with open('mardpg-uav/scripts/visualize_eval.py', 'w') as f:
    f.write(content)
