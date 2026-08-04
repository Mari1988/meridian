"""Diagnostic: does the informed prior's error track how wrong its anchor was?

The informed arm's anchors are perturbed ~25% per seed, so for any parameter
the posterior actually identifies from the prior, the fitted error should
follow the anchor error. Plotting that per parameter separates two very
different situations:

  * a tight positive line means the posterior is relaying the anchor -- the
    data is not identifying the parameter, it is reporting what it was told;
  * a shapeless cloud means the anchor is not what drives the error, so
    something else in the fit is.

Not a deck figure. Diagnostic only, to check whether the "the model reports
its prior" reading is supported per parameter rather than asserted from one
correlation.

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_anchor_vs_error.py
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import arviz as az
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import channel_labels
import r90_basis
from model_utils import PRIOR_VARIANTS

RUN_DIR = os.path.join(HERE, 'fitted_models', 'scratch_ablation_r90_pinned')
OUT_PATH = os.path.join(HERE, 'figures', 'scratch_anchor_vs_error.png')
DATA_PATH = os.path.join(RUN_DIR, 'anchor_vs_error.csv')

CHANNELS = ['TV', 'Display']
IDX = {'TV': 0, 'Display': 1}
INFORMED = 'ec_alpha_noisy'
POINT = '#2C7FB8'
LINE = '#C0392B'
GRID = '#DDDDDD'

# (anchor column, fitted-error column, panel title)
PANELS = [
    ('ec_pert_pct', 'ec_err_pct', 'Saturation (ec_m)'),
    ('ec_pert_pct', 'roi_err_pct', 'ROI  (vs the ec_m anchor)'),
    ('alpha_pert_pct', 'alpha_err_pct', 'Adstock (alpha_m)'),
]


def build_frame() -> pd.DataFrame:
  """Per seed x channel: how wrong each anchor was, and how wrong the fit was."""
  truths = pd.read_csv(os.path.join(RUN_DIR, 'true_params_by_seed.csv'))
  # Anchors depend only on rng(seed): ec_m/alpha_m are pinned across seeds, so
  # any seed's scenario supplies the same base for the perturbation.
  cfg, sim, _, _ = r90_basis.build_scenario(1320)

  rows = []
  for seed in r90_basis.SEEDS:
    prior = PRIOR_VARIANTS[INFORMED](sim, cfg, rng=np.random.default_rng(seed))
    ec_anchor = np.exp(prior.ec_m.loc.numpy())
    alpha_anchor = prior.alpha_m.loc.numpy()
    post = az.from_netcdf(
        os.path.join(RUN_DIR, 'models', f'seed_{seed}_{INFORMED}'
                     '_inference_data.nc')
    ).posterior
    for channel in CHANNELS:
      i = IDX[channel]
      t = truths[(truths.seed == seed) & (truths.channel == channel)].iloc[0]
      rows.append({
          'seed': seed,
          'channel': channel,
          'ec_pert_pct': (float(ec_anchor[i]) / t.true_ec_m - 1) * 100,
          'alpha_pert_pct': (
              (float(alpha_anchor[i]) / t.true_alpha_m - 1) * 100
          ),
          'ec_err_pct': (
              float(post['ec_m'].values[..., i].mean()) / t.true_ec_m - 1
          ) * 100,
          'roi_err_pct': (
              float(post['roi_m'].values[..., i].mean()) / t.true_roi_m - 1
          ) * 100,
          'alpha_err_pct': (
              float(post['alpha_m'].values[..., i].mean()) / t.true_alpha_m - 1
          ) * 100,
      })
  return pd.DataFrame(rows)


def main() -> int:
  df = build_frame()
  df.to_csv(DATA_PATH, index=False)

  fig, axes = plt.subplots(
      len(CHANNELS), len(PANELS), figsize=(13.0, 7.4), squeeze=False
  )
  for row, channel in enumerate(CHANNELS):
    sub = df[df.channel == channel]
    for col, (xcol, ycol, title) in enumerate(PANELS):
      ax = axes[row][col]
      x, y = sub[xcol].values, sub[ycol].values
      ax.axhline(0, color=GRID, lw=1, zorder=1)
      ax.axvline(0, color=GRID, lw=1, zorder=1)
      ax.scatter(x, y, s=55, color=POINT, alpha=0.85, zorder=3,
                 edgecolors='white', linewidths=0.8)
      for xi, yi, seed in zip(x, y, sub.seed.values):
        ax.annotate(str(seed), (xi, yi), fontsize=7, color='#666666',
                    xytext=(4, 3), textcoords='offset points', zorder=4)

      r = float(np.corrcoef(x, y)[0, 1])
      # Only draw the fit when there is a relationship worth reading; a line
      # through a cloud invites the eye to see one that is not there.
      if abs(r) >= 0.5:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=LINE, lw=1.8, zorder=2)
        verdict = 'tracks the anchor'
      else:
        verdict = 'no relationship'
      ax.set_title(f'{title}\nr = {r:+.2f} — {verdict}', fontsize=10,
                   fontweight='bold' if abs(r) >= 0.5 else None)
      ax.tick_params(labelsize=8)
      if row == len(CHANNELS) - 1:
        ax.set_xlabel(
            'anchor error vs truth (%)'
            + ('' if xcol == 'ec_pert_pct' else '  [alpha anchor]'),
            fontsize=9,
        )
      if col == 0:
        ax.set_ylabel(
            f'{channel_labels.label(channel)}\nfitted error vs truth (%)',
            fontsize=10, fontweight='bold',
        )

  fig.suptitle(
      'Informed prior: does the fitted error follow how wrong the anchor was? '
      '(10 seeds, pinned truths)',
      fontsize=13, fontweight='bold', y=1.0,
  )
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  print(f'Wrote {OUT_PATH}')
  print(f'Wrote {DATA_PATH}\n')
  for channel in CHANNELS:
    sub = df[df.channel == channel]
    for xcol, ycol, title in PANELS:
      r = sub[xcol].corr(sub[ycol])
      print(f'  {channel:8s} {title:28s} r = {r:+.3f}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
