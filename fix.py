with open("mardpg-uav/scripts/train.py", "r") as f:
    lines = f.readlines()
with open("mardpg-uav/scripts/train.py", "w") as f:
    for line in lines:
        if "print(\"Initializing" in line or "print(\"MultiUAVEnv" in line: continue
        if "obs = env.reset(stage_cfg)" in line and "    obs = env.reset" not in line:
            f.write("            obs = env.reset(stage_cfg)\n")
        else:
            f.write(line)
