import re

filepath = 'mardpg-uav/mardpg_uav/eval_rollout.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace the load_agents function in eval_rollout.py with an import and a wrapper, or just redirect usages.
# We can just remove it and import load_agents_strict from scripts.evaluate_multiagent
# Actually, eval_rollout.py is part of mardpg_uav, importing from scripts is generally bad practice.
# But scripts import from mardpg_uav.
# We can move load_agents_strict to mardpg_uav/eval_rollout.py and have scripts import it!
pass
