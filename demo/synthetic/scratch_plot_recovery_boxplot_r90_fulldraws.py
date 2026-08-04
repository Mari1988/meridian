"""Slide 7, built from full posterior draws instead of 3 seed means.

`scratch_plot_recovery_boxplot_r90.py` built its box plot from 3 points per
variant (one posterior MEAN per seed) -- thin boxes that don't show
within-fit posterior uncertainty at all, just seed-to-seed spread of a single
summary statistic. This instead loads the saved `InferenceData` (`.nc` files
under `scratch_ablation_r90/models/`, persisted specifically so this kind of
follow-up wouldn't need a refit) and pools the raw MCMC draws across all 3
seeds x 2 chains x 1000 keep = 6000 draws per variant per parameter, so the
box's IQR/whiskers reflect actual posterior spread (including each seed's own
90% HDI), not just the 3 seeds' point estimates.

Both channels are shown, one row each. The point of the second row is the
*contrast in severity*: Channel-2's true `ec_m` (~1.3) sits inside the default
prior's high-density region, so the default is pulled only slightly, while
Channel-1's (9.0) sits far outside it and is pulled hard. Same prior, same
mechanism, dose-dependent in how far the truth is from it.

All three columns are plotted as % error against each draw's own truth.
An earlier version put `alpha_m` in percentage points because Channel-2's true
carryover was drawn per seed and landed at 0.006 in one of them, where a ratio
error explodes. The pinned basis (`r90_basis`) fixes it at 0.15, and
Channel-1's at 0.30, so that hazard is gone and consistent units across the
grid are worth more.

One measurement note:

  * Pooling draws across seeds treats between-seed and within-fit (posterior)
    uncertainty as exchangeable, which they are not -- this is a richer visual
    than 3 points, not a formally correct combined posterior. Reasonable for
    "how much does full uncertainty vary" at a glance; not a substitute for a
    proper hierarchical treatment.

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_recovery_boxplot_r90_fulldraws.py
"""

from __future__ import annotations

import os
import textwrap

import arviz as az
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Presentation-only channel naming; TV/Display stay the data keys.
import channel_labels
import r90_basis

RUN_DIR = os.path.join(HERE, 'fitted_models', 'scratch_ablation_r90_pinned')
MODELS_DIR = os.path.join(RUN_DIR, 'models')
TRUTHS_CSV = os.path.join(RUN_DIR, 'true_params_by_seed.csv')
OUT_PATH = os.path.join(
    HERE, 'figures', 'slide7_recovery_boxplot_r90_fulldraws.png'
)
STATS_CSV = os.path.join(RUN_DIR, 'pooled_box_stats.csv')

TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'
SEEDS = r90_basis.SEEDS
CHANNELS = ['TV', 'Display']
# The ACHIEVABLE informed prior (both anchors perturbed ~25% per seed), not
# the oracle `ec_alpha_only` that was centred on the exact truth.
INFORMED_VARIANT = 'ec_alpha_noisy'

# param -> (column heading, is-the-structural-result, absolute-not-percent)
PANELS = [
    ('ec_m', 'Saturation point (ec_m)', True, False),
    ('roi_m', 'ROI', False, False),
    ('alpha_m', 'Adstock (alpha_m)', False, False),
]


def pooled_error(
    param: str,
    variant: str,
    channel: str,
    truths: pd.DataFrame,
    absolute: bool = False,
) -> np.ndarray:
  """All draws (chains x keep) for one channel, pooled across the 3 seeds.

  Returns % error against that seed's own truth, or absolute error in the
  parameter's own units when `absolute` is set.
  """
  pooled = []
  for seed in SEEDS:
    true_val = float(
        truths[(truths.seed == seed) & (truths.channel == channel)][
            f'true_{param}'
        ].iloc[0]
    )
    path = os.path.join(MODELS_DIR, f'seed_{seed}_{variant}_inference_data.nc')
    draws = (
        az.from_netcdf(path)
        .posterior[param]
        .sel(media_channel=channel)
        .values.flatten()
    )
    pooled.append(
        draws - true_val if absolute else (draws / true_val - 1) * 100
    )
  return np.concatenate(pooled)


def main() -> None:
  truths = pd.read_csv(TRUTHS_CSV)

  # `sharey` per column is load-bearing, not cosmetic: the argument of this
  # figure is the CONTRAST down each column (the same default prior missing
  # Channel-1 badly and Channel-2 barely). On independent y-scales the two
  # boxes look comparably wrong, which is the opposite of the finding.
  fig, axes = plt.subplots(
      len(CHANNELS),
      len(PANELS),
      # Sized to the slide's actual picture budget (12.1 x 4.3in, so ~2.8:1
      # once the wrapped title and footnote are added). Taller than this and
      # `_picture_fitted` caps the height and shrinks the width, leaving the
      # figure floating in white space.
      figsize=(13.33, 4.4),
      squeeze=False,
      sharey='col',
  )
  n_draws = None
  # Every box's summary is written alongside the figure so the slide text can
  # quote it without reloading six `.nc` files at deck-build time -- and so a
  # quoted number always traces to the box the audience is looking at.
  stats = []
  for row, channel in enumerate(CHANNELS):
    for col, (param, heading, structural, absolute) in enumerate(PANELS):
      ax = axes[row][col]
      data = [
          pooled_error(param, 'default', channel, truths, absolute),
          pooled_error(param, INFORMED_VARIANT, channel, truths, absolute),
      ]
      n_draws = len(data[0])
      for variant, values in zip(('default', INFORMED_VARIANT), data):
        stats.append(
            {
                'channel': channel,
                'param': param,
                'variant': variant,
                'unit': 'points' if absolute else 'pct',
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'n_draws': len(values),
            }
        )
      bp = ax.boxplot(
          data,
          tick_labels=['Default', 'Informed'],
          widths=0.55,
          patch_artist=True,
          medianprops=dict(color='k', lw=2),
          showfliers=False,  # 20k draws -- fliers would be solid ink
      )
      for patch, color in zip(bp['boxes'], [DEFAULT, INFORMED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
      ax.axhline(0, color=TRUTH, lw=1.1, ls='--')
      if row == 0:
        ax.set_title(
            heading, fontsize=12, fontweight='bold' if structural else None
        )
      if col == 0:
        # Row label doubles as the y-axis label so the grid needs no legend.
        ax.set_ylabel(
            f'{channel_labels.label(channel)}\nerror vs truth (%)',
            fontsize=11,
            fontweight='bold',
        )
      ax.tick_params(labelsize=9)

  fig.suptitle(
      textwrap.fill(
          f'Both channels, posterior draws pooled across {len(SEEDS)} seeds, '
          'oracle_r2=0.9 — the default misses badly where the truth sits far '
          'from its prior, mildly where it does not',
          width=88,
      ),
      fontsize=13,
      fontweight='bold',
      y=1.005,
  )
  # Saying that zero is per-seed matters even though the truths are pinned:
  # each draw is still scored against its own dataset's truth, so the zero
  # line means the same thing everywhere. Wrapped, because `bbox_inches=
  # 'tight'` expands the canvas to fit the longest line of text -- unwrapped,
  # this one caption stretched the figure to ~30in and squashed the panels.
  fig.text(
      0.5,
      -0.03,
      textwrap.fill(
          f'n={n_draws} draws per box ({len(SEEDS)} seeds × 2 chains × 1000). '
          f'Every true parameter is identical in all {len(SEEDS)} datasets — '
          'only the noise differs. Box = interquartile range, whiskers to the '
          'most extreme non-outlier draw. “Informed” anchors are ~25% off, '
          'redrawn per seed.',
          width=132,
      ),
      ha='center',
      va='top',
      fontsize=9,
      color='#888888',
  )
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
  print(f'Wrote {OUT_PATH}')

  stats_df = pd.DataFrame(stats)
  stats_df.to_csv(STATS_CSV, index=False)
  print(f'Wrote {STATS_CSV}')
  print(
      stats_df.pivot(
          index=['channel', 'param', 'unit'],
          columns='variant',
          values='median',
      )
      .round(1)
      .to_string()
  )


if __name__ == '__main__':
  main()
