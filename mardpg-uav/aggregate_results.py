#!/usr/bin/env python3
"""
aggregate_results.py — turn per-seed eval CSVs into a defensible variant
comparison: mean +/- 95% CI across seeds, curriculum-progress distribution, and
the recurrence / centralization deltas with a difference-of-means CI.

Depends on the eval CSV from IMPLEMENTATION_FIXES_ADDENDUM.md (Fix 6e). Reads
run directories named like:
    cl_{variant}_seed{seed}     (curriculum)
    nocl_{variant}_seed{seed}   (no-curriculum ablation)
each containing eval_log_*.csv.

Usage:
    python aggregate_results.py --root checkpoints
    python aggregate_results.py --root checkpoints --stage 5
    python aggregate_results.py --root checkpoints --mode cl --out summary

Only numpy is required. No pandas, no scipy.
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np

RUN_RE = re.compile(r'^(cl|nocl)_(mardpg|maddpg|ind_rdpg|iddpg)_seed(\d+)$')

# Metrics to summarise: (key, higher_is_better)
METRICS = [
    ('success_rate', True),
    ('mission_success_rate', True),
    ('collision_rate', False),
    ('uav_collision_rate', False),
    ('trapped_rate', False),
    ('path_efficiency', True),
    ('conflict_resolution_rate', True),
    ('min_pair_dist', None),     # context metric, no "better" direction
]

VARIANT_ORDER = ['mardpg', 'maddpg', 'ind_rdpg', 'iddpg']
VARIANT_LABEL = {'mardpg': 'MARDPG (R+C)', 'maddpg': 'MADDPG (FF+C)',
                 'ind_rdpg': 'Ind-RDPG (R+I)', 'iddpg': 'IDDPG (FF+I)'}

# two-tailed t critical at 0.975 by df (fallback when scipy absent)
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
      15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
      21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
      27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_crit(df):
    if df <= 0:
        return float('nan')
    if df in _T:
        return _T[df]
    return 1.96 if df > 30 else _T[min(_T, key=lambda k: abs(k - df))]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def read_eval_csv(run_dir):
    """Return list of row-dicts (floats), chronological, or [] if none."""
    files = sorted(glob.glob(os.path.join(run_dir, 'eval_log_*.csv')))
    rows = []
    for path in files:
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                rows.append({k: _f(v) for k, v in r.items()})
    rows.sort(key=lambda r: r.get('episode', 0.0))
    return rows


def per_stage_last(rows):
    """{stage:int -> last eval row at that stage}. Last row wins (most converged)."""
    out = {}
    for r in rows:
        st = int(r.get('stage', 0))
        out[st] = r          # chronological order => last assignment is latest
    return out


def collect(root):
    """runs[(mode,variant)][seed] = {'by_stage':{...}, 'max_stage':int, 'final':row}"""
    runs = defaultdict(dict)
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d):
            continue
        m = RUN_RE.match(entry)
        if not m:
            continue
        mode, variant, seed = m.group(1), m.group(2), int(m.group(3))
        rows = read_eval_csv(d)
        if not rows:
            print(f"[warn] no eval_log_*.csv in {entry}; skipping")
            continue
        by_stage = per_stage_last(rows)
        runs[(mode, variant)][seed] = {
            'by_stage': by_stage,
            'max_stage': max(by_stage) if by_stage else 0,
            'final': rows[-1],
        }
    return runs


def mean_ci(vals):
    vals = [v for v in vals if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan'), 0
    mean = float(np.mean(vals))
    if n == 1:
        return mean, float('nan'), 1
    sd = float(np.std(vals, ddof=1))
    se = sd / math.sqrt(n)
    return mean, t_crit(n - 1) * se, n


def diff_ci(a, b):
    """Welch CI for mean(a) - mean(b)."""
    a = [v for v in a if not math.isnan(v)]
    b = [v for v in b if not math.isnan(v)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        d = (np.mean(a) if a else float('nan')) - (np.mean(b) if b else float('nan'))
        return float(d), float('nan')
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float(np.mean(a) - np.mean(b)), 0.0
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(np.mean(a) - np.mean(b)), t_crit(int(df)) * se


def stage_values(runs_for_variant, stage, key):
    out = []
    for seed, info in sorted(runs_for_variant.items()):
        row = info['by_stage'].get(stage)
        if row is not None:
            out.append(row.get(key, float('nan')))
    return out


def fmt(mean, ci):
    if math.isnan(mean):
        return "   n/a   "
    if math.isnan(ci):
        return f"{mean:6.3f}    "
    return f"{mean:6.3f}±{ci:5.3f}"


def common_deepest_stage(runs, variants, mode):
    """Deepest stage reached by EVERY seed of EVERY listed variant (for fair compare)."""
    deepest = None
    for v in variants:
        group = runs.get((mode, v), {})
        if not group:
            return None
        vmin = min(info['max_stage'] for info in group.values())
        deepest = vmin if deepest is None else min(deepest, vmin)
    return deepest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='checkpoints')
    ap.add_argument('--mode', default='cl', choices=['cl', 'nocl'])
    ap.add_argument('--stage', type=int, default=None,
                    help='compare at this 1-based stage; default = deepest stage all seeds reached')
    ap.add_argument('--out', default='summary', help='basename for summary.md/.csv')
    args = ap.parse_args()

    runs = collect(args.root)
    if not runs:
        print("No runs found. Expected dirs like cl_mardpg_seed0/ with eval_log_*.csv")
        return

    present = [v for v in VARIANT_ORDER if (args.mode, v) in runs]
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Variant comparison ({args.mode})  root={args.root}")
    emit("")

    # ---- Curriculum progress ------------------------------------------------
    emit("## Curriculum progress (max stage reached, per seed)")
    emit("")
    emit("| variant | seeds | max-stage per seed | mean max-stage |")
    emit("|---|---|---|---|")
    for v in present:
        group = runs[(args.mode, v)]
        per = {s: info['max_stage'] for s, info in sorted(group.items())}
        ms = list(per.values())
        emit(f"| {VARIANT_LABEL[v]} | {len(per)} | "
             f"{', '.join(f'{s}:{st}' for s, st in per.items())} | "
             f"{np.mean(ms):.2f} |")
    emit("")

    # ---- Pick comparison stage ---------------------------------------------
    compare_variants = [v for v in ['mardpg', 'maddpg', 'ind_rdpg'] if v in present]
    stage = args.stage
    if stage is None:
        stage = common_deepest_stage(runs, compare_variants or present, args.mode)
        if stage is None or stage < 1:
            emit("No common stage reached by all compared variants; "
                 "pass --stage N to force a stage. Per-variant finals below use each "
                 "run's own final stage.")
            stage = None

    # ---- Per-variant metric table at the comparison stage -------------------
    if stage is not None:
        emit(f"## Metrics at stage {stage} (mean ± 95% CI across seeds)")
    else:
        emit("## Metrics at each run's FINAL stage (mixed stages — interpret with care)")
    emit("")
    header = "| metric | " + " | ".join(VARIANT_LABEL[v] for v in present) + " |"
    emit(header)
    emit("|" + "---|" * (len(present) + 1))

    summary_rows = []
    for key, _hib in METRICS:
        cells = []
        for v in present:
            group = runs[(args.mode, v)]
            if stage is not None:
                vals = stage_values(group, stage, key)
            else:
                vals = [info['final'].get(key, float('nan'))
                        for info in group.values()]
            mean, ci, n = mean_ci(vals)
            cells.append(fmt(mean, ci))
            summary_rows.append({'stage': stage if stage else 'final',
                                 'variant': v, 'metric': key,
                                 'mean': mean, 'ci95': ci, 'n': n})
        emit(f"| {key} | " + " | ".join(cells) + " |")
    emit("")

    # ---- Deltas: recurrence and centralization ------------------------------
    if stage is not None and 'mardpg' in present:
        emit(f"## Effect deltas at stage {stage} (difference of means, 95% CI)")
        emit("")
        emit("Recurrence = MARDPG − MADDPG (both centralized). "
             "Centralization = MARDPG − Ind-RDPG (both recurrent). "
             "CI excluding 0 ⇒ the effect is distinguishable at this seed count.")
        emit("")
        emit("| metric | recurrence (MARDPG−MADDPG) | centralization (MARDPG−Ind-RDPG) |")
        emit("|---|---|---|")
        g_mar = runs.get((args.mode, 'mardpg'), {})
        g_mad = runs.get((args.mode, 'maddpg'), {})
        g_ind = runs.get((args.mode, 'ind_rdpg'), {})
        for key, _hib in METRICS:
            def cell(gb):
                if not gb:
                    return "    n/a    "
                d, ci = diff_ci(stage_values(g_mar, stage, key),
                                stage_values(gb, stage, key))
                if math.isnan(d):
                    return "    n/a    "
                star = ""
                if not math.isnan(ci) and abs(d) > ci:
                    star = " *"
                cis = f"±{ci:5.3f}" if not math.isnan(ci) else "      "
                return f"{d:+6.3f}{cis}{star}"
            emit(f"| {key} | {cell(g_mad)} | {cell(g_ind)} |")
        emit("")
        emit("`*` = 95% CI of the difference excludes 0.")
        emit("")

    emit("> Reminder: with 5 seeds these CIs are wide. A non-significant delta "
         "means 'not yet distinguishable', not 'no effect'. Report exactly this.")

    # ---- Write files --------------------------------------------------------
    with open(f"{args.out}.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=['stage', 'variant', 'metric',
                                          'mean', 'ci95', 'n'])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
