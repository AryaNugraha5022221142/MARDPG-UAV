with open("mardpg-uav/scripts/train.py", "r") as f:
    lines = f.readlines()

with open("mardpg-uav/scripts/train.py", "w") as f:
    skip = False
    for i, line in enumerate(lines):
        if "torch.cuda.manual_seed_all(seed)" in line:
            continue
        if "torch.backends.cudnn.deterministic =" in line:
            skip = True
            continue
        if skip and "algo_cfg.get('deterministic_cudnn'" in line:
            skip = False
            continue
            
        if "device = algo_cfg.get('device'," in line:
            f.write("        device = algo_cfg.get('device', 'cpu')\n")
            continue
        if "'cuda' if torch.cuda.is_available() else 'cpu')" in line:
            continue
            
        if "if device == 'cuda' and not torch.cuda.is_available():" in line:
            f.write("    if device != 'cpu' and torch.cuda.is_available():\n")
            f.write("        device = 'cuda'\n")
            f.write("        torch.cuda.manual_seed_all(seed)\n")
            f.write("        torch.backends.cudnn.deterministic = bool(algo_cfg.get('deterministic_cudnn', False))\n")
            f.write("    else:\n")
            continue
            
        f.write(line)
