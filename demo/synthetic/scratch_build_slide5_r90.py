"""Slide 5's DGP facts and time-series figure at the deck's final basis.

Slide 5 describes the datasets slides 7-8 are fitted to, so it must be built
from the same DGP they are: `r90_basis`, where every true parameter is pinned
across seeds. It previously carried its own copy of the config, which left
Channel-2's `alpha_m` drawn per seed and made slide 5's table disagree with
the rest of the deck.

Data generation only: no MCMC, no model fit. Rebuilding the scenario takes
~40s, so the series are cached to CSV and the figure re-renders from those in
under a second.

Writes:
  fitted_models/scratch_slide5_r90/channels.csv      per-channel true params
  fitted_models/scratch_slide5_r90/dataset.csv       dataset-level scalars
  fitted_models/scratch_slide5_r90/weekly_series.csv national weekly series
  figures/slide5_series_r90.png                      the slide 5 figure

Usage:
  .venv/bin/python demo/synthetic/scratch_build_slide5_r90.py          # build + plot
  .venv/bin/python demo/synthetic/scratch_build_slide5_r90.py --plot   # plot from cache

Not part of the tracked pipeline -- a standalone scratch script, matching
`scratch_ablation_r90.py`'s conventions (its `BASE_OVERRIDES` are copied here
verbatim so the two describe the same DGP).
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import channel_labels
import r90_basis

OUT_DIR = os.path.join(HERE, 'fitted_models', 'scratch_slide5_r90')
FIGURE_PATH = os.path.join(HERE, 'figures', 'slide5_series_r90.png')

SEED = 1320
# The DGP lives in `r90_basis` so slide 5 describes exactly the datasets
# slides 7-8 fit. It previously carried its own BASE_OVERRIDES copy, which
# left Channel-2's alpha_m unpinned and made slide 5's table disagree with
# the rest of the deck.
N_KNOTS = r90_basis.N_KNOTS
TARGET_ORACLE_R2 = r90_basis.TARGET_ORACLE_R2

TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'
DPI = 200


def export() -> None:
  """Builds the scenario and writes the three CSVs."""
  import realistic_baseline as rb

  cfg, sim, _, gt = r90_basis.build_scenario(SEED)
  os.makedirs(OUT_DIR, exist_ok=True)

  ec = sim.ec_m.numpy()
  alpha = sim.alpha_m.numpy()
  roi = np.asarray(gt['roi_m'])
  channels = list(cfg.channel_names)

  pd.DataFrame(
      {
          'channel': channels,
          'true_ec_m': [float(v) for v in ec],
          'true_roi_m': [float(v) for v in roi],
          'true_alpha_m': [float(v) for v in alpha],
          # 1/(1+ec_m) is the ceiling share captured at a median week, since media
          # is scaled by its own median. Not printed on the slide any more (the
          # column was cut), but slide 6 states the same quantity, so keeping it
          # here lets the two be cross-checked.
          'ceiling_frac': [1.0 / (1.0 + float(v)) for v in ec],
      }
  ).to_csv(os.path.join(OUT_DIR, 'channels.csv'), index=False)

  diag = rb.baseline_diagnostics(sim, N_KNOTS)
  kpi = sim.kpi_gt.numpy()
  pd.DataFrame(
      [
          {
              'seed': SEED,
              'n_geos': int(kpi.shape[0]),
              'n_times': int(kpi.shape[1]),
              'target_oracle_r2': TARGET_ORACLE_R2,
              'oracle_r2': float(diag['oracle_r2']),
              'noise_scale': float(diag['noise_scale']),
              'media_share_pct': float(diag['media_share_pct']),
              'max_lag': int(cfg.max_lag),
          }
      ]
  ).to_csv(os.path.join(OUT_DIR, 'dataset.csv'), index=False)

  # `mu_trend_t` / `mu_seasonal_t` are the DGP's own components, exactly as
  # slide 5's equation names them -- not a decomposition fitted after the fact.
  # They live on the per-capita `mu_t` scale, not the national sales scale.
  # Exported but NOT plotted: a trend+seasonality panel was tried on slide 5
  # and cut. Kept here so it can be re-plotted without a 40s re-export.
  series = {
      'date': sim.time_index,
      'sales': kpi.sum(axis=0),
      'trend': sim.mu_trend_t.numpy(),
      'seasonality': sim.mu_seasonal_t.numpy(),
  }
  impressions_tm = sim.impression_gtm.numpy().sum(axis=0)
  for i, channel in enumerate(channels):
    series[channel] = impressions_tm[:, i]
  pd.DataFrame(series).to_csv(
      os.path.join(OUT_DIR, 'weekly_series.csv'), index=False
  )
  print(f'wrote CSVs to {OUT_DIR}')


def plot() -> str:
  """Renders slide 5's figure: weekly sales, then each channel's delivery."""
  series = pd.read_csv(
      os.path.join(OUT_DIR, 'weekly_series.csv'), parse_dates=['date']
  )
  channels = [
      c
      for c in series.columns
      if c not in ('date', 'sales', 'trend', 'seasonality')
  ]

  # One panel per channel rather than both on one axis: the two channels'
  # impression volumes differ by roughly an order of magnitude, so a shared
  # y-axis would flatten one of them into the baseline.
  fig, axes = plt.subplots(
      1 + len(channels),
      1,
      figsize=(5.4, 5.6),
      sharex=True,
      gridspec_kw={'height_ratios': [1.5] + [1.0] * len(channels)},
  )

  ax = axes[0]
  ax.plot(series['date'], series['sales'], color=TRUTH, lw=1.2)
  ax.set_title(
      'Weekly simulated sales — national total', fontsize=11, fontweight='bold'
  )
  ax.set_ylabel('Sales', fontsize=9)

  for ax, channel, color in zip(axes[1:], channels, (DEFAULT, INFORMED)):
    ax.plot(series['date'], series[channel], color=color, lw=1.0)
    ax.set_title(
        f'{channel_labels.label(channel)} — weekly impressions',
        fontsize=10,
        fontweight='bold',
    )
    ax.set_ylabel('Impressions', fontsize=9)

  for ax in axes:
    ax.tick_params(labelsize=8)
    ax.margins(x=0.01)
  fig.autofmt_xdate()
  fig.tight_layout(h_pad=0.9)

  os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
  fig.savefig(FIGURE_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  print(f'wrote {FIGURE_PATH}')
  return FIGURE_PATH


def main() -> int:
  if '--plot' not in sys.argv:
    export()
  plot()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
