"""Slide 7 candidate, small multiples: one proper posterior per seed, unpooled.

Follow-up to `scratch_plot_recovery_boxplot_r90_fulldraws.py`, which pooled
draws across all 3 seeds into one box per variant -- a caveat there was that
pooling blends between-seed and within-fit (posterior) uncertainty into one
box, which is not a formally correct combined posterior. This avoids that
entirely: a 3 (parameter) x 3 (seed) grid, each cell showing that ONE seed's
own actual posterior draws (2 chains x 1000 keep = 2000 draws per box), so
every box is a real, individually-calibrated distribution -- nothing pooled
or averaged across seeds.

Same source as the pooled version: `scratch_ablation_r90.py`'s saved
`InferenceData` under `fitted_models/scratch_ablation_r90/models/`.

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_recovery_boxplot_r90_perseed.py
"""
from __future__ import annotations

import os
import warnings

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', message='trace group is not defined')

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, 'fitted_models', 'scratch_ablation_r90', 'models')
SUMMARY_CSV = os.path.join(HERE, 'fitted_models', 'scratch_ablation_r90', 'per_seed_all.csv')
OUT_PATH = os.path.join(HERE, 'figures', 'slide7_recovery_boxplot_r90_perseed.png')

TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'
SEEDS = [1320, 7, 42]
TRUE_EC_TV = 9.0
TRUE_ALPHA_TV = 0.3

_CACHE: dict[tuple[int, str], dict] = {}


def _draws(seed: int, variant: str) -> dict[str, np.ndarray]:
  key = (seed, variant)
  if key not in _CACHE:
    path = os.path.join(MODELS_DIR, f'seed_{seed}_{variant}_inference_data.nc')
    idata = az.from_netcdf(path)
    _CACHE[key] = {
        param: idata.posterior[param].sel(media_channel='TV').values.flatten()
        for param in ('ec_m', 'roi_m', 'alpha_m')
    }
  return _CACHE[key]


def main() -> None:
  summary = pd.read_csv(SUMMARY_CSV)
  true_roi_by_seed = dict(zip(summary['seed'], summary['true_roi_m']))

  panels = [
      ('ec_m', 'Saturation point (ec_m)', True),
      ('roi_m', 'ROI', False),
      ('alpha_m', 'Adstock (alpha_m)', False),
  ]

  fig, axes = plt.subplots(3, 3, figsize=(13.33, 10.5), sharex='col')
  for row, seed in enumerate(SEEDS):
    for col, (param, label, structural) in enumerate(panels):
      ax = axes[row][col]
      true_val = (TRUE_EC_TV if param == 'ec_m' else
                  TRUE_ALPHA_TV if param == 'alpha_m' else
                  true_roi_by_seed[seed])
      data = []
      for variant in ('default', 'ec_alpha_only'):
        draws = _draws(seed, variant)[param]
        data.append((draws / true_val - 1) * 100)

      bp = ax.boxplot(data, tick_labels=['Default', 'Informed'], widths=0.55,
                       patch_artist=True, medianprops=dict(color='k', lw=2),
                       showfliers=False)
      for patch, color in zip(bp['boxes'], [DEFAULT, INFORMED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
      ax.axhline(0, color=TRUTH, lw=1.0, ls='--')

      if row == 0:
        ax.set_title(f'{label}\nerror %', fontsize=11,
                      fontweight='bold' if structural else None)
      if col == 0:
        ax.set_ylabel(f'seed {seed}', fontsize=11, color='#555555')

  fig.suptitle(
      'TV channel, per-seed posteriors (2000 draws each, unpooled), '
      'oracle_r2=0.9, alpha_m=0.3',
      fontsize=13, fontweight='bold', y=1.01)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
  print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
  main()
