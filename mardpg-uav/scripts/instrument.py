import ast

def insert_prints(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # We will just inject some prints directly.
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        sline = line.strip()
        if sline == "def train(config_path: str = \"config/default.yaml\",":
            pass
        elif sline.startswith("cfg      = load_config"):
            new_lines.append("    print('[DEBUG] load_config done', flush=True)")
        elif sline.startswith("logger = WandbLogger"):
            new_lines.append("    print('[DEBUG] WandbLogger done', flush=True)")
        elif sline.startswith("env        = MultiUAVEnv"):
            new_lines.append("    print('[DEBUG] MultiUAVEnv done', flush=True)")
        elif sline.startswith("noise   = GaussianNoise"):
            new_lines.append("    print('[DEBUG] GaussianNoise done', flush=True)")
        elif sline.startswith("cl = CurriculumManager"):
            new_lines.append("    print('[DEBUG] CurriculumManager done', flush=True)")
        elif sline == "try:":
            new_lines.append("    print('[DEBUG] try block started', flush=True)")
        elif sline.startswith("obs = env.reset"):
            new_lines.append("            print('[DEBUG] env.reset done', flush=True)")
            
    with open(filepath.replace(".py", "_dbg.py"), 'w') as f:
        f.write('\n'.join(new_lines))

insert_prints("mardpg-uav/scripts/train.py")
