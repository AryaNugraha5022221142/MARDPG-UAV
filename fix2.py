with open("mardpg-uav/scripts/train.py", "r") as f:
    lines = f.readlines()
with open("mardpg-uav/scripts/train.py", "w") as f:
    for i, line in enumerate(lines):
        if "obs = env.reset(stage_cfg)" in line:
            if i < 300:
                f.write("        obs = env.reset(stage_cfg)\n")
            else:
                f.write("            obs = env.reset(stage_cfg)\n")
        else:
            f.write(line)
