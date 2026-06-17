"""
analyze_factorial.py — the 2x2 ablation, decomposed.

The four variants are a 2x2 factorial:

                 centralized critic   independent critic
    recurrent    mardpg               ind_rdpg
    feedforward  maddpg               iddpg

Two questions your thesis must answer separately:
  * RECURRENCE main effect   : does memory (LSTM vs FF) help under partial obs?
  * CENTRALIZATION main effect: does the centralized critic — the defining
                                feature of "MARDPG" over Ind-RDPG — help at all?
  * INTERACTION              : does centralization help MORE when recurrent?

Effects are computed SEED-WISE (paired by training seed across the four
variants), then averaged, with a stratified bootstrap CI over seeds. This is
the correct unit of analysis: a difference that is large relative to seed-to-
seed noise is real; one that is small relative to it is not.

For seed s, with R/F = recurrent/feedforward, C/I = centralized/independent:
    rec_eff(s)  = 0.5*(mardpg + ind_rdpg) - 0.5*(maddpg + iddpg)
    cen_eff(s)  = 0.5*(mardpg + maddpg)   - 0.5*(ind_rdpg + iddpg)
    inter(s)    = (mardpg - ind_rdpg) - (maddpg - iddpg)

Input : eval_per_seed.csv produced by evaluate_multiagent.py (v2).
Output: console table + factorial_effects.csv + (optional) a bar plot.

CAVEAT printed at the bottom: the recurrence effect is only interpretable if
the feedforward variants are CONFIRMED non-buggy (they must learn the
near-Markovian stage 1). A feedforward variant stuck at 0% due to a setup
defect inflates the recurrence effect for the wrong reason.

Usage
-----
  python analyze_factorial.py --in eval_results/eval_per_seed.csv \
      --metric success_rate --regime in_dist --plot
"""
import argparse
import numpy as np
import pandas as pd

_CELLS = {  # variant -> (recurrent, centralized)
    'mardpg':   (True,  True),
    'ind_rdpg': (True,  False),
    'maddpg':   (False, True),
    'iddpg':    (False, False),
}
_HIGHER_BETTER = {'success_rate': True, 'mission_success': True,
                  'conflict_resolution_rate': True, 'path_eff_reached': True,
                  'collision_rate': False, 'uav_collision_rate': False,
                  'near_miss_ratio': False, 'closest_approach_m': True}


def _bootstrap_ci(per_seed_effect, n_boot=10000, ci=0.95, seed=7):
    v = np.asarray([x for x in per_seed_effect if not np.isnan(x)], float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    if n == 1:
        return (float(v[0]), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return (float(v.mean()),
            float(np.percentile(boots, 100 * (1 - ci) / 2)),
            float(np.percentile(boots, 100 * (1 + ci) / 2)))


def _cell_table(df, metric, config_filter=None):
    """Return {variant: {seed: value}} for one metric, optionally one config
    (else averaged across configs per seed)."""
    sub = df.copy()
    if config_filter is not None:
        sub = sub[sub['config_name'] == config_filter]
    out = {}
    for variant in _CELLS:
        g = sub[sub['variant'] == variant]
        if g.empty:
            out[variant] = {}
            continue
        # average over configs within each seed (balanced design assumed)
        per_seed = g.groupby('seed')[metric].mean()
        out[variant] = {int(s): float(v) for s, v in per_seed.items()}
    return out


def _effects_for(df, metric, config_filter, regime_label):
    cells = _cell_table(df, metric, config_filter)
    common = set.intersection(*[set(cells[v].keys()) for v in _CELLS]) \
        if all(cells[v] for v in _CELLS) else set()
    if not common:
        present = {v: sorted(cells[v].keys()) for v in _CELLS}
        raise SystemExit(
            f"[{regime_label}/{config_filter or 'ALL'}] no training seed is "
            f"present in all four variants. Seeds per variant: {present}. "
            f"A balanced factorial needs the same seeds across all four cells.")
    seeds = sorted(common)
    rec, cen, inter = [], [], []
    for s in seeds:
        rc, ri = cells['mardpg'][s], cells['ind_rdpg'][s]
        fc, fi = cells['maddpg'][s], cells['iddpg'][s]
        rec.append(0.5 * (rc + ri) - 0.5 * (fc + fi))
        cen.append(0.5 * (rc + fc) - 0.5 * (ri + fi))
        inter.append((rc - ri) - (fc - fi))
    rows = []
    for label, vals in [('recurrence', rec), ('centralization', cen),
                        ('interaction', inter)]:
        m, lo, hi = _bootstrap_ci(vals)
        rows.append(dict(regime=regime_label, config=config_filter or 'ALL',
                         metric=metric, effect=label, n_seeds=len(seeds),
                         estimate=m, ci_lo=lo, ci_hi=hi,
                         per_seed=";".join(f"{x:+.3f}" for x in vals)))
    # also dump the four cell means for context
    for variant in _CELLS:
        cv = [cells[variant][s] for s in seeds]
        rows.append(dict(regime=regime_label, config=config_filter or 'ALL',
                         metric=metric, effect=f'cell_{variant}', n_seeds=len(seeds),
                         estimate=float(np.mean(cv)), ci_lo=np.nan, ci_hi=np.nan,
                         per_seed=";".join(f"{x:.3f}" for x in cv)))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True, help='eval_per_seed.csv')
    p.add_argument('--metric', default='success_rate')
    p.add_argument('--regime', default='in_dist',
                   help="'in_dist', 'ood', or 'all'")
    p.add_argument('--per-config', action='store_true',
                   help='also report each config separately (not just pooled)')
    p.add_argument('--out', default='factorial_effects.csv')
    p.add_argument('--plot', action='store_true')
    a = p.parse_args()

    df = pd.read_csv(a.infile)
    if a.regime != 'all':
        df = df[df['regime'] == a.regime]
    if df.empty:
        raise SystemExit(f"No rows for regime={a.regime} in {a.infile}")

    frames = [_effects_for(df, a.metric, None, a.regime)]
    if a.per_config:
        for cname in sorted(df['config_name'].unique()):
            try:
                frames.append(_effects_for(df, a.metric, cname, a.regime))
            except SystemExit as e:
                print(f"  (skipped {cname}: {e})")
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(a.out, index=False)

    hb = _HIGHER_BETTER.get(a.metric, True)
    arrow = "higher=better" if hb else "lower=better"
    print(f"\n2x2 FACTORIAL — metric={a.metric} ({arrow}), regime={a.regime}")
    print("=" * 74)
    eff = res[res['effect'].isin(['recurrence', 'centralization', 'interaction'])]
    cells = res[res['effect'].str.startswith('cell_')]
    print("Cell means (averaged over configs & seeds):")
    for _, r in cells[cells['config'] == 'ALL'].iterrows():
        print(f"   {r['effect']:16s} = {r['estimate']:.3f}")
    print("-" * 74)
    print(f"{'effect':16s} {'estimate':>9s} {'95% CI':>20s}  interpretation")
    print("-" * 74)
    for _, r in eff[eff['config'] == 'ALL'].iterrows():
        ci = (f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
              if not np.isnan(r['ci_lo']) else "  (need >=2 seeds)")
        crosses0 = (not np.isnan(r['ci_lo'])) and (r['ci_lo'] <= 0 <= r['ci_hi'])
        verdict = ("CI includes 0 -> no detectable effect" if crosses0
                   else "CI excludes 0 -> effect detected")
        print(f"{r['effect']:16s} {r['estimate']:+9.3f} {ci:>20s}  {verdict}")
    print("-" * 74)
    print("CAVEAT: the recurrence effect is interpretable only if the "
          "feedforward\nvariants (maddpg/iddpg) are CONFIRMED non-buggy — i.e. "
          "they learn the near-\nMarkovian stage 1. A FF variant stuck at 0% "
          "from a setup defect inflates this\neffect for the wrong reason. "
          "Verify before reporting.")
    print(f"\nWrote {a.out}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            e = eff[eff['config'] == 'ALL']
            labels = e['effect'].tolist()
            est = e['estimate'].values
            lo = e['ci_lo'].values
            hi = e['ci_hi'].values
            yerr = np.vstack([est - lo, hi - est])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, est, yerr=yerr, capsize=6, color=['#4477aa', '#ee6677', '#228833'])
            ax.axhline(0, color='k', lw=1)
            ax.set_ylabel(f'Δ {a.metric}')
            ax.set_title(f'2x2 factorial effects ({a.regime}, {arrow})')
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            out_png = a.out.replace('.csv', '.png')
            fig.savefig(out_png, dpi=200, bbox_inches='tight')
            print(f"Wrote {out_png}")
        except Exception as ex:
            print(f"[plot skipped] {ex}")


if __name__ == "__main__":
    main()
