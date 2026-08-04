"""Renders the figures for the ARF deck's results slides.

Fast: everything is read from CSVs written by the notebooks and
`export_curve_data.py`, or (for the weekly-sales plot) rebuilt from the
seed-1320 scenario directly -- data generation only, no MCMC fit, so it's
still fast. Nothing here fits a model, so the figures can be restyled and
re-rendered freely without a 5-minute refit.

Figures:
  results_execution_vs_curve.png  slide 6  -- where execution sits on the true curve

Only the slide 6 figure is still wired into the deck. The other four below were
built for the canonical alpha_m=0.8/oracle_r2=0.80 basis and have since been
superseded: slides 5, 7, 8 and 9 moved to the r90 basis (alpha_m=0.3,
oracle_r2=0.9) and render from `scratch_build_slide5_r90.py` and the three
`scratch_plot_*_r90_*.py` scripts, and the seed-stability slide was cut. They
are left here because they are cheap and are the only rendering of the
canonical run.

  results_weekly_sales.png        superseded by slide5_series_r90.png
  results_mroi_scaling.png        superseded by slide8_mroi_r90_perseed.png
  results_response_curves.png     superseded by slide9_response_curve_r90_pooled.png
  results_seed_stability.png      unused -- its slide was cut

Usage:
  .venv/bin/python demo/synthetic/build_results_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import channel_labels
import prior_plots
import results_facts

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
DPI = 200
FIGSIZE = (13.33, 5.2)

# Shared with the section 1 figures via `prior_plots.apply_slide_style()`.
TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'
STYLE = {
    'truth': ('Truth (known DGP)', TRUTH, 2.6),
    'default': ('Meridian default prior', DEFAULT, 2.2),
    'ec_alpha_only': ('Reach-informed prior', INFORMED, 2.2),
}
# The informed prior tracks the truth so closely that a solid truth line
# disappears underneath it. Dashing it lets both read at once.
TRUTH_DASH = (0, (6, 3))


def _save(fig, name: str) -> str:
  os.makedirs(FIGURE_DIR, exist_ok=True)
  path = os.path.join(FIGURE_DIR, name)
  fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
  plt.close(fig)
  return path


def plot_weekly_sales() -> str:
  """Slide 5 -- weekly simulated sales, national total (sum over 20 geos).

  This is the DGP's actual output, not a diagnostic: the same `sim` object
  `plot_execution_vs_curve` reuses for slide 6, so it's bit-identical to what
  the equation on slide 5 describes and what the rest of the deck fits to.
  """
  from export_curve_data import build_scenario

  _, sim, _, _ = _cached_scenario(build_scenario)
  weekly_sales = sim.kpi_gt.numpy().sum(axis=0)
  dates = sim.time_index

  fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.6))
  ax.plot(dates, weekly_sales, color=TRUTH, lw=1.3)
  ax.set_title('Weekly simulated sales — national total', fontsize=12,
               fontweight='bold')
  ax.set_ylabel('Sales (simulated units)')
  fig.autofmt_xdate()
  fig.tight_layout()
  return _save(fig, 'results_weekly_sales.png')


def plot_execution_vs_curve() -> str:
  """Slide 6 -- the true Hill curve with observed execution scattered on it.

  The x-axis is in multiples of the channel's own median non-zero per-capita
  execution, deliberately the same denomination as section 1's saturation
  slide: x = 1.0 is "what you already run". That is what makes TV's position
  legible without any statistics -- its execution sits in the near-linear part
  of its own curve, while Display's sits around the knee.
  """
  import realistic_baseline as rb  # noqa: F401  (import cost only if needed)
  from export_curve_data import build_scenario

  cfg, sim, _, _ = _cached_scenario(build_scenario)
  channels = list(cfg.channel_names)
  facts = results_facts.dgp_facts()
  ceiling = {'TV': facts['ceiling_frac_tv'],
             'Display': facts['ceiling_frac_display']}

  x_all = sim.transformed_ipc_gtm.numpy()
  ec_all = sim.ec_m.numpy()
  # A shared x-axis wide enough to contain TV's half-saturation point is what
  # makes the contrast land: TV's delivery sits far to the left of its own
  # ec_m, Display's straddles its own.
  x_max = float(max(ec_all)) * 1.25

  fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)
  for ax, (i, channel) in zip(axes, enumerate(channels)):
    ec = float(ec_all[i])
    x = x_all[:, :, i]
    x = x[x > 0]
    lo, hi = np.quantile(x, 0.01), np.quantile(x, 0.99)

    grid = np.linspace(0, x_max, 500)
    ax.plot(grid, grid / (grid + ec), color='#BBBBBB', lw=2.4, zorder=2,
            label='True response curve')
    # The dots lie exactly on the curve by construction, so plotting them alone
    # just hides them under the line. Drawing the observed arc in colour on top
    # of a grey full curve is the honest version of the same idea: it shows
    # which part of its own curve the channel has actually explored.
    arc = grid[(grid >= lo) & (grid <= hi)]
    ax.plot(arc, arc / (arc + ec), color=DEFAULT, lw=5.0, alpha=0.9, zorder=3,
            solid_capstyle='round', label='Where the delivery actually sits')
    rng = np.random.default_rng(0)
    sample = rng.choice(x, size=min(2000, x.size), replace=False)
    ax.scatter(sample, sample / (sample + ec), s=14, alpha=0.05,
               color='#7B241C', edgecolors='none', zorder=4)

    ax.axvline(ec, color=TRUTH, lw=1.6, ls='--', zorder=5)
    ax.annotate(f'half-saturation\n(ec_m = {ec:.2f})', xy=(ec, 0.86),
                xytext=(ec + x_max * 0.02, 0.86), fontsize=10, color=TRUTH,
                fontweight='bold', va='top')
    ax.axvline(1.0, color=INFORMED, lw=1.4, ls=':', zorder=5,
               label='A median week (x = 1.0)' if i == 0 else None)

    ax.set_title(
        f'{channel_labels.label(channel)} — {ceiling[channel]} of its ceiling '
        'effect today', fontsize=13, fontweight='bold')
    ax.set_xlabel('Media, in multiples of this channel’s own median week')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, x_max)
    if i == 0:
      ax.set_ylabel('Share of maximum possible effect')
      ax.legend(loc='upper center', fontsize=9)
  fig.tight_layout()
  return _save(fig, 'results_execution_vs_curve.png')


_SCENARIO = {}


def _cached_scenario(builder):
  """Builds the scenario once per process (it takes ~20s)."""
  if 'v' not in _SCENARIO:
    _SCENARIO['v'] = builder()
  return _SCENARIO['v']


def _band_plot(ax, df, channel, value_label):
  for variant, (label, color, lw) in STYLE.items():
    sub = df[(df.variant == variant) & (df.channel == channel)]
    if sub.empty:
      continue
    sub = sub.sort_values('spend_multiplier')
    is_truth = variant == 'truth'
    ax.plot(sub.spend_multiplier, sub['mean'], 'o-', color=color, lw=lw,
            label=label, markersize=5,
            zorder=5 if is_truth else 3,
            linestyle=TRUTH_DASH if is_truth else '-')
    if not np.allclose(sub.ci_lo, sub['mean']):  # truth has no interval
      ax.fill_between(sub.spend_multiplier, sub.ci_lo, sub.ci_hi,
                      color=color, alpha=0.15, lw=0, zorder=2)
  ax.axvline(1.0, color='#999999', lw=1.0, ls=':', zorder=1)
  ax.set_xlabel('Spend, as a multiple of today')
  ax.set_ylabel(value_label)
  ax.legend(fontsize=9)


def plot_mroi_scaling() -> str:
  """Slide 8 -- marginal ROI as spend scales, with credible bands."""
  df = results_facts.mroi_sweep()
  facts = results_facts.mroi_facts()

  fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)
  _band_plot(ax_left, df, 'TV', 'TV marginal ROI')
  ax_left.set_title('What does the next dollar earn?', fontsize=13,
                    fontweight='bold')
  ax_left.set_xscale('log')
  ax_left.set_xticks([1, 2, 3, 5, 10])
  ax_left.set_xticklabels(['1x', '2x', '3x', '5x', '10x'])

  truth = df[(df.variant == 'truth') & (df.channel == 'TV')].set_index(
      'spend_multiplier')['mean']
  for variant, (label, color, lw) in STYLE.items():
    if variant == 'truth':
      continue
    sub = df[(df.variant == variant) & (df.channel == 'TV')].set_index(
        'spend_multiplier')
    err = (sub['mean'] / truth - 1) * 100
    ax_right.plot(err.index, err.values, 'o-', color=color, lw=lw, label=label,
                  markersize=5)
  ax_right.axhline(0, color=TRUTH, lw=1.2)
  ax_right.set_xscale('log')
  ax_right.set_xticks([1, 2, 3, 5, 10])
  ax_right.set_xticklabels(['1x', '2x', '3x', '5x', '10x'])
  ax_right.set_xlabel('Spend, as a multiple of today')
  ax_right.set_ylabel('Error against the truth (%)')
  ax_right.set_title('The error is smallest exactly where your data is',
                     fontsize=13, fontweight='bold')
  ax_right.legend(fontsize=9)

  err3 = facts.get('m3x_default_err_pct')
  if err3 is not None:
    ax_right.annotate(f'{err3:+.0f}% at 3x', xy=(3, err3),
                      xytext=(3.3, err3 * 0.75), fontsize=11,
                      fontweight='bold', color=DEFAULT)
  fig.tight_layout()
  return _save(fig, 'results_mroi_scaling.png')


def plot_response_curves() -> str:
  """Slide 9 -- the shape error, and where the two curves cross.

  Note the title deliberately does NOT claim the curves pass through the same
  point. On this DGP they do not: the default prior's curve sits well above the
  truth at today's spend (the same overstatement the ROI row shows) and well
  below it further out. The curves cross, so the error changes sign -- which is
  a sharper story than "they agree today", and the honest one here.
  """
  df = results_facts.response_curves()
  fig, ax = plt.subplots(1, 1, figsize=(13.33, 5.0))
  _band_plot(ax, df, 'TV', 'TV incremental revenue')
  ax.set_title(
      'The default prior overstates what TV earns today — and understates how '
      'much room it has left',
      fontsize=14, fontweight='bold')

  # Mark the crossover: below it the default flatters TV, above it the default
  # tells you to stop spending.
  wide = df[df.channel == 'TV'].pivot(
      index='spend_multiplier', columns='variant', values='mean')
  gap = wide['default'] - wide['truth']
  sign_change = np.where(np.diff(np.sign(gap.values)) != 0)[0]
  if sign_change.size:
    k = sign_change[0]
    x0, x1 = gap.index[k], gap.index[k + 1]
    g0, g1 = gap.values[k], gap.values[k + 1]
    cross = x0 + (x1 - x0) * abs(g0) / (abs(g0) + abs(g1))
    ax.axvline(cross, color='#666666', lw=1.2, ls='-.', zorder=1)
    ax.annotate(f'curves cross at {cross:.1f}× today’s spend',
                xy=(cross, ax.get_ylim()[1] * 0.14),
                xytext=(cross + 0.25, ax.get_ylim()[1] * 0.10),
                fontsize=10, color='#444444')
  ax.annotate('today’s spend', xy=(1.0, ax.get_ylim()[1] * 0.90),
              xytext=(1.12, ax.get_ylim()[1] * 0.90), fontsize=10,
              color='#666666')
  fig.tight_layout()
  return _save(fig, 'results_response_curves.png')


def plot_seed_stability() -> str:
  """Slide 10 -- the same comparison across ten simulated datasets."""
  per_seed = results_facts.per_seed()
  panels = [
      ('ec_err', 'Saturation point (ec_m)\nerror %', True),
      ('roi_err', 'ROI\nerror %', False),
      ('alpha_err', 'Adstock (alpha_m)\nerror %', False),
  ]
  fig, axes = plt.subplots(1, 3, figsize=(13.33, 4.6))
  rng = np.random.default_rng(0)
  for ax, (key, label, structural) in zip(axes, panels):
    data = [per_seed[f'default_{key}'].astype(float).values,
            per_seed[f'ec_alpha_only_{key}'].astype(float).values]
    bp = ax.boxplot(data, tick_labels=['Default', 'Informed'], widths=0.55,
                    patch_artist=True, medianprops=dict(color='k', lw=2))
    for patch, color in zip(bp['boxes'], [DEFAULT, INFORMED]):
      patch.set_facecolor(color)
      patch.set_alpha(0.5)
    for i, values in enumerate(data):
      ax.scatter(np.full(len(values), i + 1) + rng.uniform(-.09, .09,
                                                           len(values)),
                 values, color='k', s=17, zorder=3)
    ax.axhline(0, color=TRUTH, lw=1.1, ls='--')
    ax.set_title(label, fontsize=12, fontweight='bold' if structural else None)
    ax.set_ylabel('')
  axes[0].set_ylabel('Error against the truth (%)')
  fig.suptitle(
      f'Across {len(per_seed)} independently simulated datasets — '
      'the saturation result is structural; ROI is not',
      fontsize=13, fontweight='bold', y=1.02)
  fig.tight_layout()
  return _save(fig, 'results_seed_stability.png')


def build_all() -> list[str]:
  prior_plots.apply_slide_style()
  paths = [
      plot_weekly_sales(),
      plot_execution_vs_curve(),
      plot_mroi_scaling(),
      plot_response_curves(),
      plot_seed_stability(),
  ]
  return paths


def main() -> None:
  for path in build_all():
    print(f'Wrote {path}')


if __name__ == '__main__':
  main()
