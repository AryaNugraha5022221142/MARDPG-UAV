import re

with open('mardpg-uav/scripts/evaluate_multiagent.py', 'r') as f:
    content = f.read()

replacement = """
                                if os.path.exists(out_vid):
                                    log_dict[f"eval/best_traj_video/{name}_{cname}"] = wandb.Video(out_vid, format="mp4")
                                elif os.path.exists(out_vid.replace('.mp4', '.gif')):
                                    log_dict[f"eval/best_traj_video/{name}_{cname}"] = wandb.Video(out_vid.replace('.mp4', '.gif'), format="gif")
"""
content = re.sub(
    r'                                if os\.path\.exists\(out_vid\):\n                                    log_dict\[f"eval/best_traj_video/{name}_{cname}"\] = wandb\.Video\(out_vid, format="mp4"\)',
    replacement.strip('\n'),
    content
)

with open('mardpg-uav/scripts/evaluate_multiagent.py', 'w') as f:
    f.write(content)
