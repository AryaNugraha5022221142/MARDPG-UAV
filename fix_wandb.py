with open("mardpg-uav/scripts/train.py", "r") as f:
    lines = f.readlines()

with open("mardpg-uav/scripts/train.py", "w") as f:
    for line in lines:
        if "a.no_wandb" in line:
            f.write(line.replace("not a.no_wandb", "False"))
        else:
            f.write(line)
