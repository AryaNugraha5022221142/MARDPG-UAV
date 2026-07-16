import re

with open('scripts/visualize_eval.py', 'r') as f:
    content = f.read()

# Add plt.close(fig) after savefig
content = re.sub(
    r"    plt\.savefig\(out_path, dpi=200, bbox_inches='tight'\)",
    "    plt.savefig(out_path, dpi=200, bbox_inches='tight')\n    plt.close(fig)",
    content
)

# For animate
content = re.sub(
    r"        anim\.save\(out_path\.replace\('\.mp4', '\.gif'\), writer='pillow', dpi=90\)",
    "        anim.save(out_path.replace('.mp4', '.gif'), writer='pillow', dpi=90)\n    plt.close(fig)",
    content
)

with open('scripts/visualize_eval.py', 'w') as f:
    f.write(content)
