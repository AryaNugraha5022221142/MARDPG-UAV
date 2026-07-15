import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse

def plot_training_logs(csv_path=None):
    if csv_path is None:
        candidates = sorted(glob.glob('checkpoints/training_log_*.csv'))
        csv_path = candidates[0] if candidates else 'checkpoints/training_log.csv'
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
    (fig, (ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(10, 15))
    ax1.plot(df['episode'], df['avg_reward'], 'b-', marker='o', markersize=4)
    ax1.set_title('Average Reward over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax2.plot(df['episode'], df['success_rate'], 'g-', marker='^', label='Success Rate', markersize=4)
    ax2.plot(df['episode'], df['collision_rate'], 'r-', marker='v', label='Collision Rate', markersize=4)
    if 'trapped_rate' in df.columns:
        ax2.plot(df['episode'], df['trapped_rate'], 'y-', marker='x', label='Trapped Rate', markersize=4)
    ax2.set_title('Success, Collision, & Trapped Rates')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rate (0.0 to 1.0)')
    ax2.legend()
    ax2.grid(True)
    ax3.plot(df['episode'], df['avg_episode_length'], 'purple', marker='s', markersize=4)
    ax3.set_title('Average Episode Length (Steps)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.grid(True)
    plt.tight_layout()
    output_img = 'checkpoints/training_curves.png'
    plt.savefig(output_img, dpi=300)
    try:
        plt.show()
    except:
        pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='checkpoints/training_log.csv', help='Path to training_log.csv')
    args = parser.parse_args()
    plot_training_logs(args.csv)