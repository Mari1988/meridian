"""Slide 8 at the pinned basis: pooled mROI curves for both channels.

Replaces the per-seed small multiples. Two panels:

  LEFT   pooled mROI for Channel-1 with a 50% band -- the three lines fan
         apart as spend scales.
  RIGHT  the same for Channel-2, where they coincide.

Channel-2 is the control, and the pair is the argument: the same default
prior extrapolated to the same 10x stays within 6% on Channel-2 and collapses
to -58% on Channel-1. So the failure is not "the model degrades when
extrapolating" -- it is extrapolating from a misplaced saturation point.

The per-multiplier medians are no longer drawn as a table here; the slide's
own text reads them from `mroi_both_channels.csv`.

Why absolute mROI units are legal here: with the truths pinned
(`r90_basis`), true mROI varies only ~0.4-3% across the ten seeds, so draws
can be pooled without indexing. On the old unpinned basis true ROI moved
6.60 -> 10.88 between seeds and the equivalent figure had to be rebased to
an index of 100.

Two honesty notes carried into the footnote:

  * The 50% band pools draws across seeds, so it mixes within-fit posterior
    uncertainty with between-seed variation. It is "where half the draws sit
    across ten datasets", not a calibrated 50% credible interval for one
    model. The claim being made is about where the bands sit relative to the
    truth, not about their width.
  * Any coverage claim belongs at 90%, quoted in the slide text rather than
    read off this band -- a narrower band excludes the truth more readily,
    which would make the claim look stronger for the wrong reason.

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_mroi_r90_slide8_pinned.py
  .venv/bin/python demo/synthetic/scratch_plot_mroi_r90_slide8_pinned.py \
      --run-dir=demo/synthetic/fitted_models/scratch_ablation_r90_alpha08 \
      --out=demo/synthetic/figures/r90_alpha08_mroi.png

The run directory is a parameter so other r90 arms can reuse this figure; the
deck's own slide 8 still renders from the default (pinned) directory. The
footnote's per-channel mROI range is COMPUTED from whichever run is loaded --
it used to be typed in, which would have silently mis-stated any other arm.
"""

from __future__ import annotations

import os
import sys
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import channel_labels
import r90_basis

DEFAULT_RUN_DIR = os.path.join(
    HERE, 'fitted_models', 'scratch_ablation_r90_pinned'
)
DEFAULT_OUT = os.path.join(HERE, 'figures', 'slide8_mroi_r90_pinned.png')

TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'
TRUTH_DASH = (0, (6, 3))
CURVE_CHANNELS = ['TV', 'Display']

# The informed arm's variant NAME is what appears in the CSV, and it differs
# per arm (`ec_alpha_noisy` for the deck, `ec_centred25_alpha_u05` for the
# exact-anchor ablation, ...). Keyed off a flag rather than hardcoded: a
# mismatch would silently drop the informed line from the figure while still
# rendering truth and default, which looks like a result rather than a bug.
INFORMED_VARIANT = next(
    (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--informed=')),
    'ec_alpha_noisy',
)
INFORMED_LABEL = next(
    (a.split('=', 1)[1] for a in sys.argv[1:]
     if a.startswith('--informed-label=')),
    'Reach-informed prior',
)
STYLE = {
    'truth': ('Truth (known DGP)', TRUTH, 2.4),
    'default': ('Meridian default prior', DEFAULT, 2.2),
    INFORMED_VARIANT: (INFORMED_LABEL, INFORMED, 2.2),
}

# `--band=90` widens the shaded region from the interquartile range to the
# 5th-95th percentiles. Default stays 50 so the deck's figure is unchanged.
#
# WHAT THIS BAND IS, PRECISELY: the per-seed quantiles are AVERAGED across
# seeds, so it is the *typical single model's* credible interval, not the
# interval of all pooled draws. The latter would be wider (it would absorb
# between-seed variation) and cannot be computed from these CSVs, which store
# quantiles rather than draws. Label it as the average per-model interval --
# calling it "90% of pooled draws" would overstate its width.
BAND = int(next(
    (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--band=')),
    '50',
))
QLO, QHI = ('q05', 'q95') if BAND == 90 else ('q25', 'q75')


def _curve_panel(ax, df, channel: str, show_legend: bool) -> None:
  """Pooled mROI for one channel, with a 50% band.

  Independent y-axes per channel on purpose: Channel-2 is already past its
  knee, so its marginal return collapses from 3.3 to 0.2 while Channel-1's
  falls from 5.9 to 1.6. A shared axis would squash Channel-2 flat and hide
  the very thing this panel is here to show -- that its three lines coincide.
  """
  sub = df[df.channel == channel]
  for variant, (label, color, lw) in STYLE.items():
    s = (
        sub[sub.variant == variant]
        .groupby('spend_multiplier')
        .agg(mean=('mean', 'mean'), lo=(QLO, 'mean'), hi=(QHI, 'mean'))
        .reset_index()
    )
    if s.empty:
      continue
    is_truth = variant == 'truth'
    ax.plot(
        s.spend_multiplier, s['mean'], marker='o', markersize=5, color=color,
        lw=lw, label=label, zorder=5 if is_truth else 3,
        linestyle=TRUTH_DASH if is_truth else '-',
    )
    if not is_truth:
      ax.fill_between(
          s.spend_multiplier, s.lo, s.hi, color=color, alpha=0.18, lw=0
      )
  ax.set_xscale('log')
  ax.set_xticks(sorted(sub.spend_multiplier.unique()))
  ax.set_xticklabels(
      [f'{int(m)}×' for m in sorted(sub.spend_multiplier.unique())]
  )
  ax.set_xlabel('Spend, as a multiple of today')
  ax.set_ylabel(f'{channel_labels.label(channel)} marginal ROI')
  ax.set_title(
      f'{channel_labels.label(channel)} — what the next dollar earns',
      fontsize=12, fontweight='bold',
  )
  if show_legend:
    ax.legend(fontsize=8, loc='upper right')


def _true_range(df, channel: str) -> str:
  """'3.3 to 0.2' — the truth's own mROI at the lowest and highest spend."""
  s = (
      df[(df.channel == channel) & (df.variant == 'truth')]
      .groupby('spend_multiplier')['mean']
      .mean()
      .sort_index()
  )
  return f'{s.iloc[0]:.1f} to {s.iloc[-1]:.1f}'


def main() -> int:
  run_dir = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--run-dir=')),
      DEFAULT_RUN_DIR,
  )
  out_path = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--out=')),
      DEFAULT_OUT,
  )
  df = pd.read_csv(os.path.join(run_dir, 'mroi_both_channels.csv'))
  # Fail loudly on a name mismatch: a missing informed line renders as a
  # clean two-line figure that looks like a result rather than a bug.
  missing = {'truth', 'default', INFORMED_VARIANT} - set(df.variant.unique())
  assert not missing, (
      f'{run_dir} has no rows for {sorted(missing)}; it holds '
      f'{sorted(df.variant.unique())}. Pass --informed=<variant>.'
  )
  fig, axes = plt.subplots(1, len(CURVE_CHANNELS), figsize=(12.6, 4.6))
  for i, channel in enumerate(CURVE_CHANNELS):
    _curve_panel(axes[i], df, channel, show_legend=(i == 0))

  fig.text(
      0.5, -0.04,
      textwrap.fill(
          # Seed count from the DATA, not from r90_basis.SEEDS: that constant
          # is 10, and this figure is now also used for 50-draw runs.
          f'{df.seed.nunique()} seeds, all with IDENTICAL true parameters '
          '— only the noise differs. Shaded band = the '
          f'{"5th–95th" if BAND == 90 else "25th–75th"} posterior percentiles '
          'AVERAGED across seeds, i.e. the interval a typical single model '
          'reports — not the spread of all draws pooled, which would be wider. '
          'Y-axes are '
          'per channel: Channel-2 sits past its knee, so its marginal return '
          f'falls from {_true_range(df, "Display")} while Channel-1’s falls '
          f'from {_true_range(df, "TV")}.',
          width=132,
      ),
      ha='center', va='top', fontsize=9, color='#888888',
  )
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  print(f'Wrote {out_path}')

  med = (
      df[df.variant != 'truth']
      .groupby(['channel', 'variant', 'spend_multiplier'])['pct_error']
      .median()
      .round(1)
  )
  print(med.to_string())
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
