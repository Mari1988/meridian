"""Slide-7 style recovery grid: both channels x three metrics, one figure.

`scratch_plot_wellspec_recovery.py` renders one 1x3 figure per channel; this
is the same content as a single 2x3 grid (rows = channels), which is how the
deck actually presents it. Parameterised by run directory and basis so it
serves any r90 arm rather than being copied per arm.

Channel-1 comes from the run's own per-seed CSVs; Channel-2 from
`per_seed_all_ch2.csv` (`scratch_extract_r90_wellspec_ch2.py`), because every
ablation script records index 0 only.

Y-AXIS: computed from the data, not fixed. The wellspec figure hard-codes
+-80 and CLAUDE.md warns against narrowing it, because at that arm's spread a
+-60 axis clips nine of ten default `ec_m` points. The alpha08 arm runs wider
still (default `roi_m` to +115%), so a fixed +-80 would clip there instead.
Both channels share one axis so the pair reads against each other, and the
assertion below fails loudly rather than cropping a point out of view.

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_recovery_grid_r90.py \
      --run-dir=demo/synthetic/fitted_models/scratch_ablation_r90_alpha08 \
      --basis=alpha08
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import channel_labels
import r90_basis

INK = '#1A1A1A'
MUTED = '#5A5A5A'
ACCENT = '#C0392B'
BLUE = '#2C7FB8'

METRICS = [
    ('ec_err', 'Saturation  ec_m'),
    ('roi_err', 'ROI  roi_m'),
    ('alpha_err', 'Adstock  alpha_m'),
]


def _ylim(all_values: list[np.ndarray]) -> int:
  """Symmetric limit that contains every point, rounded up to a clean step."""
  peak = max(float(np.abs(v).max()) for v in all_values)
  return int(np.ceil((peak * 1.08) / 20.0) * 20)


def build(run_dir: str, basis: r90_basis.Basis, out_path: str) -> str:
  ch1 = pd.read_csv(os.path.join(run_dir, 'per_seed_all.csv'))
  ch2 = pd.read_csv(os.path.join(run_dir, 'per_seed_all_ch2.csv'))
  rows = [('TV', ch1), ('Display', ch2)]

  series = {}
  for channel, df in rows:
    for metric, _ in METRICS:
      series[(channel, metric)] = [
          df[f'default_{metric}'].values,
          df[f'informed_{metric}'].values,
      ]

  ylim = _ylim(list(np.concatenate(v) for v in series.values()))

  fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4))
  fig.subplots_adjust(left=0.065, right=0.985, top=0.822, bottom=0.075,
                      wspace=0.17, hspace=0.34)

  for r, (channel, _) in enumerate(rows):
    label = channel_labels.label(channel)
    for c, (metric, title) in enumerate(METRICS):
      ax = axes[r][c]
      vals = series[(channel, metric)]
      positions = [0.0, 1.0]
      colors = [ACCENT, BLUE]

      bp = ax.boxplot(
          vals, positions=positions, widths=0.56, patch_artist=True,
          medianprops=dict(color=INK, lw=1.6),
          whiskerprops=dict(color=MUTED, lw=1.0),
          capprops=dict(color=MUTED, lw=1.0),
          flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none', alpha=0.6),
      )
      for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)

      rng = np.random.default_rng(0)
      for pos, v, color in zip(positions, vals, colors):
        ax.scatter(pos + rng.uniform(-0.15, 0.15, len(v)), v, s=20,
                   color=color, alpha=0.75, zorder=3, linewidths=0)

      ax.axhline(0, color=INK, lw=1.1, zorder=1)
      ax.set_xticks(positions)
      # Labels are MEDIANS, matching the line the box draws. The informed
      # errors are skewed on several panels, so a mean label would contradict
      # the figure.
      ax.set_xticklabels(
          [f'default\n{np.median(vals[0]):+.1f}%',
           f'informed\n{np.median(vals[1]):+.1f}%'],
          fontsize=10.5, color=INK,
      )
      ax.set_title(f'{title}  ({label})', fontsize=12, color=INK, pad=8)
      ax.set_xlim(-0.62, 1.62)
      ax.set_ylim(-ylim, ylim)
      # Built outward from 0 so the zero reference line always carries a tick
      # and a grid line; `arange(-ylim, ...)` skips it whenever the step does
      # not divide ylim evenly.
      step = 20 if ylim <= 100 else 40
      ticks = np.arange(step, ylim + 1, step)
      ax.set_yticks(np.concatenate([-ticks[::-1], [0], ticks]))
      ax.tick_params(axis='y', labelsize=9, colors=MUTED)
      for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
      for s in ('left', 'bottom'):
        ax.spines[s].set_color('#CCCCCC')
      ax.grid(axis='y', color='#EEEEEE', lw=0.8)
      ax.set_axisbelow(True)

      clipped = int((np.abs(np.concatenate(vals)) > ylim).sum())
      assert not clipped, (
          f'{channel}/{metric}: {clipped} point(s) outside the y-axis -- '
          'widen it rather than shipping a figure that hides them'
      )

    axes[r][0].set_ylabel('% error vs. that seed’s own truth', fontsize=10,
                          color=MUTED)

  true_alpha = basis.true_alpha['TV']
  fig.text(
      0.065, 0.945,
      'Does the model recover the truth it was built from — '
      'even when perturbed ~25%?',
      ha='left', fontsize=16.5, color=INK, fontweight='bold',
  )
  fig.text(
      0.065, 0.903,
      f'{len(r90_basis.SEEDS)} draws, identical truths, only the noise '
      'realization varies — each scored against its own seed’s truth. '
      'Labels are medians.',
      ha='left', fontsize=9.5, color=MUTED,
  )
  fig.text(
      0.065, 0.878,
      f'Channel-1 true adstock α = {true_alpha}; default fits at Meridian’s '
      'out-of-the-box max_lag = 8, informed at the DGP’s 13.',
      ha='left', fontsize=9.5, color=MUTED,
  )

  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=190, facecolor='white')
  plt.close(fig)
  return out_path


def main() -> int:
  run_dir = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--run-dir=')),
      os.path.join(HERE, 'fitted_models', 'scratch_ablation_r90_alpha08'),
  )
  basis = r90_basis.BASES[
      next(
          (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--basis=')),
          'alpha08',
      )
  ]
  out_path = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--out=')),
      os.path.join(HERE, 'figures', f'r90_{basis.name}_recovery_grid.png'),
  )

  tagged = pd.read_csv(os.path.join(run_dir, 'per_seed_all.csv')).get('basis')
  found = str(tagged.iloc[0]) if tagged is not None else 'alpha03'
  assert found == basis.name, (
      f'{run_dir} holds basis {found!r}, not {basis.name!r}'
  )

  print(build(run_dir, basis, out_path))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
