"""Slide 7 candidate: recovery box plot (like slide 10's style) at the deck's
finalized basis -- alpha_m=0.3, oracle_r2=0.9, 3 seeds (1320, 7, 42).

Slide 7 currently shows a table (truth / default / informed, for ec_m / roi_m
/ alpha_m, TV + Display) built from the canonical alpha_m=0.8, oracle_r2=0.80
realistic_baseline_2ch run. This renders the same comparison as a box plot --
mirroring `build_results_figures.py:plot_seed_stability()`'s exact style
(matplotlib boxplot, DEFAULT/INFORMED colors, jittered points, dashed zero
line) -- but using this session's settled basis, from
`scratch_ablation_r90.py`'s per_seed_all.csv.

TV only (that data doesn't have a Display channel comparison the way the
canonical run's table does).

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_recovery_boxplot_r90.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, 'figures', 'slide7_recovery_boxplot_r90.png')

TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'


def main() -> None:
  df = pd.read_csv(os.path.join(
      HERE, 'fitted_models', 'scratch_ablation_r90', 'per_seed_all.csv'
  )).sort_values('seed')

  panels = [
      ('ec_err', 'Saturation point (ec_m)\nerror %', True),
      ('roi_err', 'ROI\nerror %', False),
      ('alpha_err', 'Adstock (alpha_m)\nerror %', False),
  ]
  fig, axes = plt.subplots(1, 3, figsize=(13.33, 4.6))
  rng = np.random.default_rng(0)
  for ax, (key, label, structural) in zip(axes, panels):
    data = [df[f'default_{key}'].astype(float).values,
            df[f'exact_{key}'].astype(float).values]
    bp = ax.boxplot(data, tick_labels=['Default', 'Informed'], widths=0.55,
                     patch_artist=True, medianprops=dict(color='k', lw=2))
    for patch, color in zip(bp['boxes'], [DEFAULT, INFORMED]):
      patch.set_facecolor(color)
      patch.set_alpha(0.5)
    for i, values in enumerate(data):
      ax.scatter(np.full(len(values), i + 1) + rng.uniform(-.09, .09, len(values)),
                 values, color='k', s=17, zorder=3)
    ax.axhline(0, color=TRUTH, lw=1.1, ls='--')
    ax.set_title(label, fontsize=12, fontweight='bold' if structural else None)
    ax.set_ylabel('')
  axes[0].set_ylabel('Error against the truth (%)')
  fig.suptitle(
      f'TV channel, {len(df)} seeds, oracle_r2=0.9, alpha_m=0.3 -- saturation '
      'point recovers cleanly; ROI improves but stays biased; adstock '
      'unaffected',
      fontsize=13, fontweight='bold', y=1.02)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
  print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
  main()
