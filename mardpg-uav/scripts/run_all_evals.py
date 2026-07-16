import argparse
import subprocess
import os

def run(cmd):
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run complete evaluation pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory (e.g. checkpoints/final)')
    parser.add_argument('--outdir', type=str, default='eval_results', help='Directory to save eval results')
    parser.add_argument('--stage', type=int, default=7, help='Curriculum stage for single-episode visualize/animate')
    parser.add_argument('--suite', type=str, choices=['full', 'quick'], default='full', help='Generalization suite size')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per config for generalization')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd1 = ['python', os.path.join(script_dir, 'evaluate_generalization.py'), '--checkpoint', args.checkpoint, '--outdir', args.outdir, '--suite', args.suite, '--episodes', str(args.episodes)]
    run(cmd1)
    cmd2 = ['python', os.path.join(script_dir, 'visualize_eval.py'), '--checkpoint', args.checkpoint, '--stage', str(args.stage), '--out', os.path.join(args.outdir, f'visualize_stage_{args.stage}.png')]
    run(cmd2)
    cmd3 = ['python', os.path.join(script_dir, 'visualize_eval.py'), '--checkpoint', args.checkpoint, '--stage', str(args.stage), '--animate', '--out', os.path.join(args.outdir, f'animate_stage_{args.stage}.png')]
    run(cmd3)
if __name__ == '__main__':
    main()