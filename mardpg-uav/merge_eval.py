"""
merge_eval.py — aggregate evaluation results computed on SEPARATE machines
without ever moving the checkpoints.

Why this exists
---------------
Checkpoints are scattered across VMs (and the replay buffers inside them are
~0.6 GB each). You should NOT move checkpoints. Instead:

  1. On EACH VM, run evaluate_multiagent.py over ONLY the checkpoints that live
     on that VM, using IDENTICAL --base-seed, --episodes, --suite, --config.
     Each run writes a local eval_episodes.csv. (Eval reads only the small
     actor weights; the 0.6 GB replay buffers are never opened.)
  2. Copy ONLY the eval_episodes.csv files to one place (KB-MB, not GB) — paste,
     scp, or a shared drive.
  3. Run this script to concatenate and re-aggregate them centrally.

Why the merge is statistically valid
-------------------------------------
  * scene_seed = base_seed + episode is identical on every VM  -> the SAME map
    is scored for episode e everywhere, so cross-method comparisons stay PAIRED
    even though they were computed on different machines.
  * the `seed` column holds the TRUE training seed (parsed from the checkpoint
    path by evaluate_multiagent._seed_label), so seeds never collide on merge.

INVARIANTS you must keep identical across all VMs (else the merge is invalid):
  --base-seed   --episodes   --suite   --config
  and the --method NAME / VARIANT strings for the same method.

Run this on any machine that has the repo on its PYTHONPATH (either VM is fine);
it imports the aggregation functions from evaluate_multiagent.

Usage
-----
  python merge_eval.py vmA/eval_episodes.csv vmB/eval_episodes.csv \
      --outdir eval_results_merged
  # globs work too:
  python merge_eval.py 'collected/*eval_episodes.csv' --outdir eval_results_merged
"""
import argparse
import glob
import os
import pandas as pd

# Aggregation lives in the eval script; reuse it so logic can't drift.
from evaluate_multiagent import (aggregate_per_seed, aggregate_across_seeds,
                                 aggregate_method_iqm, _print_variance_report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+',
                   help='eval_episodes.csv files (or globs) from each VM')
    p.add_argument('--outdir', default='eval_results_merged')
    a = p.parse_args()

    paths = []
    for token in a.inputs:
        hits = sorted(glob.glob(token))
        paths.extend(hits if hits else [token])
    paths = [pth for pth in paths if os.path.exists(pth)]
    if not paths:
        raise SystemExit("No input CSVs found.")
    print("Merging episode files:")
    for pth in paths:
        print(f"  {pth}")

    df = pd.concat([pd.read_csv(pth) for pth in paths], ignore_index=True)

    # Dedupe in case a (method,seed,config,episode) was scored on both machines.
    key = ['method', 'variant', 'seed', 'config_name', 'episode']
    n0 = len(df)
    df = df.drop_duplicates(subset=key, keep='first')
    if len(df) != n0:
        print(f"[merge] dropped {n0 - len(df)} duplicate rows on {key}")

    # Coverage sanity: every (method,seed,config) should have the same #episodes.
    cov = df.groupby(['method', 'seed', 'config_name'])['episode'].nunique()
    if cov.nunique() > 1:
        print("[WARN] uneven episode counts across (method,seed,config). "
              "Pairing may be incomplete — confirm every VM used the SAME "
              "--episodes / --base-seed / --suite / --config.")
        print(cov.to_string())

    # Report which seeds you actually have per method (catch a missing VM run).
    print("\nSeeds present per method:")
    for (method, variant), g in df.groupby(['method', 'variant']):
        print(f"  {method:10s} ({variant:8s}): seeds {sorted(g['seed'].unique())}")

    os.makedirs(a.outdir, exist_ok=True)
    df.to_csv(os.path.join(a.outdir, 'eval_episodes.csv'), index=False)
    df_seed = aggregate_per_seed(df)
    df_sum = aggregate_across_seeds(df_seed)
    df_iqm = aggregate_method_iqm(df_seed, regimes=('in_dist',))
    df_seed.to_csv(os.path.join(a.outdir, 'eval_per_seed.csv'), index=False)
    df_sum.to_csv(os.path.join(a.outdir, 'eval_summary.csv'), index=False)
    df_iqm.to_csv(os.path.join(a.outdir, 'eval_method_iqm.csv'), index=False)

    _print_variance_report(df_seed, df_sum)
    print(f"\nMerged outputs written to {a.outdir}/")
    print(f"Next: python analyze_factorial.py --in {a.outdir}/eval_per_seed.csv "
          f"--metric success_rate --regime in_dist --per-config --plot")


if __name__ == '__main__':
    main()
