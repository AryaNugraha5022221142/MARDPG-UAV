import os
import argparse
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--seeds", nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--variant", default="mardpg", choices=["mardpg", "maddpg", "iddpg"])
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()

    for seed in a.seeds:
        run_name = f"{a.condition}_seed{seed}"
        out_dir = f"runs/{run_name}"
        
        cmd = [
            "python", "-m", "mardpg_uav.train",
            "--seed", str(seed),
            "--out-dir", out_dir,
            "--run-name", run_name
        ]
        
        cmd += ["--variant", a.variant]
        if a.no_curriculum:
            cmd.append("--no-curriculum")
            
        if a.no_wandb:
            cmd.append("--no-wandb")
            
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
