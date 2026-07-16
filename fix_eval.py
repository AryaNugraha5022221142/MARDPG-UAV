with open("mardpg-uav/scripts/evaluate_multiagent.py", "r") as f:
    lines = f.readlines()

new_lines = []
live_block = []
in_live_block = False

for i, line in enumerate(lines):
    if "One live renderer for the whole run" in line:
        in_live_block = True
    
    if in_live_block:
        live_block.append(line)
        if "live = LiveRenderer(env, env_cfg)" in line:
            in_live_block = False
    else:
        new_lines.append(line)

# Now we need to insert live_block after the try/except block for env.reset
out_lines = []
inserted = False
for line in new_lines:
    out_lines.append(line)
    if "ensure it lives in environment/assignment.py (audit Fix A).\")" in line:
        out_lines.extend(live_block)
        inserted = True

with open("mardpg-uav/scripts/evaluate_multiagent.py", "w") as f:
    f.writelines(out_lines)

print("Inserted:", inserted)
