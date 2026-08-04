"""Slide 8 candidate: mROI at elevated spend, faceted per seed -- r2=0.9 basis.

Mirrors `build_results_figures.py:plot_mroi_scaling()`'s exact two-panel
style (left: mROI value with 90% credible bands; right: % error against
truth, with a callout) but faceted into 3 columns, one per fitted seed
(1320, 7, 42), instead of a single draw -- so the "default degrades with
spend, informed stays flat" story is shown to replicate, not asserted from
one dataset. Sourced from `scratch_check_mroi_recovery_r90.py`'s
`all_seeds.csv` (mean/ci_lo/ci_hi per seed x variant x multiplier, already
computed from saved InferenceData -- no refit needed).

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_mroi_r90_slide8.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Presentation-only channel naming; TV/Display stay the data keys.
import channel_labels
DATA_CSV = os.path.join(
    HERE, 'fitted_models', 'scratch_check_mroi_recovery_r90', 'all_seeds.csv')
OUT_PATH = os.path.join(HERE, 'figures', 'slide8_mroi_r90_perseed.png')

TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'
TRUTH_DASH = (0, (6, 3))
SEEDS = [1320, 7, 42]
MULTS = [1.0, 2.0, 3.0, 5.0, 10.0]


def main() -> None:
  df = pd.read_csv(DATA_CSV)
  df = df[df.spend_multiplier.isin(MULTS)]

  fig, axes = plt.subplots(2, 3, figsize=(13.33, 7.6), sharex='col')
  for col, seed in enumerate(SEEDS):
    sub = df[df.seed == seed]
    ax_top, ax_bot = axes[0][col], axes[1][col]

    for variant, (label, color) in (
        ('truth', ('Truth (known DGP)', TRUTH)),
        ('default', ('Meridian default prior', DEFAULT)),
        ('ec_alpha_only', ('Reach-informed prior', INFORMED)),
    ):
      s = sub[sub.variant == variant].sort_values('spend_multiplier')
      is_truth = variant == 'truth'
      ax_top.plot(s.spend_multiplier, s['mean'], marker='o', color=color,
                  lw=2.6 if is_truth else 2.2, markersize=5,
                  label=label if col == 0 else None,
                  linestyle=TRUTH_DASH if is_truth else '-',
                  zorder=5 if is_truth else 3)
      if not is_truth:
        ax_top.fill_between(s.spend_multiplier, s.ci_lo, s.ci_hi,
                             color=color, alpha=0.18, lw=0)

    ax_top.set_xscale('log')
    ax_top.set_xticks(MULTS)
    ax_top.set_xticklabels([f'{int(m)}x' for m in MULTS])
    ax_top.set_title(f'seed {seed}', fontsize=12, color='#555555')
    if col == 0:
      ax_top.set_ylabel(f'{channel_labels.label("TV")} marginal ROI')

    for variant, color in (('default', DEFAULT), ('ec_alpha_only', INFORMED)):
      s = sub[sub.variant == variant].sort_values('spend_multiplier')
      ax_bot.plot(s.spend_multiplier, s.pct_error, 'o-', color=color, lw=2.0,
                  markersize=5)
    ax_bot.axhline(0, color=TRUTH, lw=1.1)
    ax_bot.set_xscale('log')
    ax_bot.set_xticks(MULTS)
    ax_bot.set_xticklabels([f'{int(m)}x' for m in MULTS])
    ax_bot.set_xlabel('Spend, as a multiple of today')
    if col == 0:
      ax_bot.set_ylabel('Error against the truth (%)')

    worst = sub[(sub.variant == 'default')
                & (sub.spend_multiplier == MULTS[-1])]['pct_error']
    if len(worst):
      err_worst = float(worst.iloc[0])
      ax_bot.annotate(
          f'{err_worst:+.0f}% at {int(MULTS[-1])}x',
          xy=(MULTS[-1], err_worst), xytext=(MULTS[-1] * 0.55, err_worst),
          fontsize=10, fontweight='bold', color=DEFAULT, ha='center',
          va='top' if err_worst < 0 else 'bottom')

  fig.legend(*axes[0][0].get_legend_handles_labels(),
             loc='upper center', ncol=3, fontsize=10, frameon=False,
             bbox_to_anchor=(0.5, 1.04))
  fig.suptitle(
      'What does the next dollar earn? — default degrades with spend, '
      'informed stays close (3 seeds, oracle_r2=0.9, alpha_m=0.3)',
      fontsize=13, fontweight='bold', y=1.10)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
  print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
  main()
