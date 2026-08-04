"""Slide-7 equivalent, well-specified baseline, one figure per channel.

Both channels render on the SAME +-80 y-axis so the pair can be read against
each other: Channel-1 is the under-reached channel the study is about,
Channel-2 the well-reached control.

Channel-1 comes from the ablation's own per-seed CSVs; Channel-2 from
`per_seed_all_ch2.csv`, extracted from the saved `InferenceData` because the
ablation script records index 0 only.
"""

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

FIGURE_DIR = os.path.join(HERE, 'figures')
DEFAULT_RUN = os.path.join(HERE, 'fitted_models', 'scratch_ablation_r90_wellspec')

# `--run-dir=` / `--out-prefix=` so other arms reuse this figure instead of
# forking a near-identical copy -- the same reason `r90_basis` exists. Defaults
# reproduce the original wellspec figures byte-for-byte.
RUN = next(
    (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--run-dir=')),
    DEFAULT_RUN,
)
OUT_PREFIX = next(
    (a.split('=', 1)[1] for a in sys.argv[1:]
     if a.startswith('--out-prefix=')),
    'wellspec_slide7',
)
# Appended to the subtitle. The arm's prior is NOT visible in the data files,
# so a run whose "informed" column means something different must say so on
# the figure itself.
NOTE = next(
    (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--note=')),
    '',
)
# Panels only, no figure title/subtitle -- for embedding under a slide's own
# title. The deck currently carries hand-cropped copies of the full figure,
# which is how a stale header survives a rebuild.
NO_HEADER = '--no-header' in sys.argv

INK = '#1A1A1A'
MUTED = '#5A5A5A'
ACCENT = '#C0392B'
BLUE = '#2C7FB8'
YLIM = 80

CHANNELS = [
    ('TV', os.path.join(RUN, 'per_seed_all.csv'),
     'the under-reached channel', f'{OUT_PREFIX}_ch1.png'),
    ('Display', os.path.join(RUN, 'per_seed_all_ch2.csv'),
     'the well-reached control', f'{OUT_PREFIX}_ch2.png'),
]


def build(channel_key, csv_path, gloss, outname):
  df = pd.read_csv(csv_path)
  label = channel_labels.label(channel_key)
  panels = [
      ('ec_err', f'Saturation  ec_m  ({label})'),
      ('roi_err', f'ROI  roi_m  ({label})'),
      ('alpha_err', f'Adstock  alpha_m  ({label})'),
  ]

  fig, axes = plt.subplots(
      1, 3, figsize=(13.2, 4.1 if NO_HEADER else 4.9))
  fig.subplots_adjust(left=0.06, right=0.985,
                      top=0.94 if NO_HEADER else 0.78,
                      bottom=0.16 if NO_HEADER else 0.14, wspace=0.16)

  for ax, (metric, title) in zip(axes, panels):
    series = [df[f'default_{metric}'].values, df[f'informed_{metric}'].values]
    positions = [0.0, 1.0]
    colors = [ACCENT, BLUE]

    bp = ax.boxplot(series, positions=positions, widths=0.56,
                    patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6),
                    whiskerprops=dict(color=MUTED, lw=1.0),
                    capprops=dict(color=MUTED, lw=1.0),
                    flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none',
                                    alpha=0.6))
    for patch, c in zip(bp['boxes'], colors):
      patch.set_facecolor(c)
      patch.set_alpha(0.20)
      patch.set_edgecolor(c)
      patch.set_linewidth(1.4)

    # Marker size, opacity and jitter all scale with the number of draws.
    # The settings that read well at 10 points (s=20, alpha=0.75) turn into a
    # solid band at 50, hiding the very distribution the strip is there to
    # show. Interpolating rather than hardcoding keeps both run sizes legible
    # from the same script.
    n = len(series[0])
    size = float(np.interp(n, [10, 50], [20, 9]))
    opacity = float(np.interp(n, [10, 50], [0.75, 0.38]))
    # Jitter stays inside the box's own half-width (0.28) so the strip reads
    # as belonging to that box rather than drifting between the two.
    spread = float(np.interp(n, [10, 50], [0.15, 0.24]))
    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, series, colors):
      ax.scatter(pos + rng.uniform(-spread, spread, len(vals)), vals,
                 s=size, color=c, alpha=opacity, zorder=3, linewidths=0)

    ax.axhline(0, color=INK, lw=1.1, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f'default\n{np.median(series[0]):+.1f}%',
         f'informed\n{np.median(series[1]):+.1f}%'],
        fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=12, color=INK, pad=8)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(-YLIM, YLIM)
    ax.set_yticks(np.arange(-YLIM, YLIM + 1, 20))
    ax.tick_params(axis='y', labelsize=9, colors=MUTED)
    for s in ('top', 'right'):
      ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
      ax.spines[s].set_color('#CCCCCC')
    ax.grid(axis='y', color='#EEEEEE', lw=0.8)
    ax.set_axisbelow(True)

  axes[0].set_ylabel('% error vs. that seed’s own truth', fontsize=10,
                     color=MUTED)

  if not NO_HEADER:
    fig.text(0.06, 0.945,
             f'Does the model recover the truth it was built from?  '
             f'{label} — {gloss}',
             ha='left', fontsize=15.5, color=INK, fontweight='bold')
    # Draw count comes from the data, never a literal: this said "10 draws"
    # while the run behind it grew to 50, which is the kind of caption that
    # outlives the number it describes.
    fig.text(0.06, 0.895,
             f'{len(df)} draws, identical truths, only the noise realization '
             'varies — each scored against its own seed’s truth. Baseline is '
             'one the fitted spline represents exactly. Labels are medians.',
             ha='left', fontsize=9.5, color=MUTED)
    # Its OWN line, not appended: the line above already fills the canvas at
    # this width, so appending silently ran the note off the right edge.
    if NOTE:
      fig.text(0.06, 0.855, NOTE, ha='left', fontsize=9.5, color=ACCENT)

  path = os.path.join(FIGURE_DIR, outname)
  fig.savefig(path, dpi=190, facecolor='white')
  plt.close(fig)
  return path


if __name__ == '__main__':
  for args in CHANNELS:
    print(build(*args))
