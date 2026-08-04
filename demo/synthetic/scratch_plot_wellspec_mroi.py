"""Slide-8 equivalent: mROI as spend scales, realistic vs well-specified.

Lines are the median across ten datasets of each fit's posterior-mean mROI;
bands are the 10th-90th percentile of those per-seed point estimates, i.e.
BETWEEN-dataset spread, not a credible interval. Coverage printed underneath
is the honest within-fit number: the share of the ten datasets whose own 90%
credible interval contains the truth at that multiplier.
"""

import glob
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
OUT = os.path.join(FIGURE_DIR, 'wellspec_slide8.png')

INK = '#1A1A1A'
MUTED = '#5A5A5A'
ACCENT = '#C0392B'
BLUE = '#2C7FB8'
MULTS = [1.0, 2.0, 3.0, 5.0, 10.0]
CH1 = channel_labels.label('TV')


def load(dirname):
  files = [f for f in sorted(glob.glob(
      os.path.join(HERE, 'fitted_models', dirname, 'mroi_*.csv')))
      if 'both' not in f]
  return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


RUNS = [
    ('realistic baseline', load('scratch_ablation_r90_pinned')),
    ('well-specified baseline', load('scratch_ablation_r90_wellspec')),
]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
fig.subplots_adjust(left=0.155, right=0.985, top=0.75, bottom=0.26, wspace=0.08)

x = np.arange(len(MULTS))
for ax, (label, d) in zip(axes, RUNS):
  truth = [d[(d.variant == 'truth') & (d.spend_multiplier == m)]
           .true_mroi.median() for m in MULTS]
  ax.plot(x, truth, color=INK, lw=2.2, ls='--', marker='o', ms=6,
          label='true mROI', zorder=5)

  for variant, color, name in [('default', ACCENT, 'default prior'),
                               ('ec_alpha_noisy', BLUE, 'informed prior')]:
    sub = d[d.variant == variant]
    med, lo, hi = [], [], []
    for m in MULTS:
      v = sub[sub.spend_multiplier == m]['mean'].values
      med.append(np.median(v))
      lo.append(np.percentile(v, 10))
      hi.append(np.percentile(v, 90))
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, med, color=color, lw=2.4, marker='o', ms=6, label=name,
            zorder=4)

  for i, m in enumerate(MULTS):
    for variant, color, dy in [('default', ACCENT, -30),
                               ('ec_alpha_noisy', BLUE, -46)]:
      cov = sub_cov = d[(d.variant == variant)
                        & (d.spend_multiplier == m)].truth_in_ci.mean()
      ax.annotate(f'{int(round(cov * 10))}/10', xy=(i, 0),
                  xycoords=('data', 'axes fraction'),
                  xytext=(0, dy), textcoords='offset points',
                  ha='center', va='top', fontsize=9.5, color=color)

  if ax is axes[0]:
    for name, color, dy in [('default', ACCENT, -30),
                            ('informed', BLUE, -46)]:
      ax.annotate(f'{name}  ', xy=(0, 0),
                  xycoords=('axes fraction', 'axes fraction'),
                  xytext=(-8, dy), textcoords='offset points',
                  ha='right', va='top', fontsize=9, color=color)
    ax.annotate('90% interval contains truth', xy=(0, 0),
                xycoords=('axes fraction', 'axes fraction'),
                xytext=(-8, -12), textcoords='offset points',
                ha='right', va='top', fontsize=9, color=MUTED,
                style='italic')

  ax.set_xticks(x)
  ax.set_xticklabels([f'{int(m)}x' for m in MULTS], fontsize=11, color=INK)
  ax.set_xlabel('spend, as a multiple of today’s', fontsize=10, color=MUTED,
                labelpad=42)
  ax.set_title(label, fontsize=12.5, color=INK, pad=8)
  ax.axhline(0, color='#DDDDDD', lw=1.0)
  ax.set_xlim(-0.35, len(MULTS) - 0.65)
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  for s in ('left', 'bottom'):
    ax.spines[s].set_color('#CCCCCC')
  ax.grid(axis='y', color='#EEEEEE', lw=0.8)
  ax.set_axisbelow(True)
  ax.tick_params(axis='y', labelsize=9, colors=MUTED)

axes[0].set_ylabel('marginal ROI', fontsize=10, color=MUTED)
axes[0].legend(frameon=False, fontsize=10, loc='upper right')

fig.text(0.035, 0.945,
         'So ask the model the question you actually bought it for',
         ha='left', fontsize=15.5, color=INK, fontweight='bold')
fig.text(0.035, 0.895,
         f'“Should I spend more here?” — marginal return for {CH1} at spend '
         'above today’s, across 10 datasets. Bands are the 10th–90th '
         'percentile of per-dataset estimates.',
         ha='left', fontsize=9.5, color=MUTED)

fig.savefig(OUT, dpi=190, facecolor='white')
print(OUT)
