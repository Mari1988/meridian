# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slide figure: the default `ec_m` prior vs. a planning-derived one.

The section-1 saturation slide shows that Meridian's default `ec_m` prior is
centered on the status quo and is identical for every channel. This figure
answers the obvious follow-up -- *what would you use instead?* -- by building
the alternative out of two inputs a media team already has, and showing that
the answer lands nowhere near the default.

The construction, for one channel:

    ec_m = audience_convention x effective_frequency
           -----------------------------------------
              current_reach x current_frequency

  * `audience_convention` -- the share of the addressable audience whose
    coverage marks half-saturation. Uniform(0.30, 0.70): deliberately a wide,
    frankly-stated judgement rather than a point claim.
  * `effective_frequency` -- NOT invented here. This is Meridian's own
    `ec_rf` prior, `Shift(0.1)(LogNormal(0.7, 0.4))` (median 2.11, 90%
    interval ~[1.1, 4.0]), lifted straight from `prior_distribution.py`.
    Meridian applies Hill to *raw, unscaled* frequency for R&F channels
    (`equations.py`, `hill_transformer.forward(frequency)`), so `ec_rf` is
    denominated in exposures per reached person -- the same units this term
    needs. Using it means the frequency half of the derivation is Google's
    number, not ours, and needs no external literature to defend.
  * the denominator is the channel's current execution, which is what puts
    the result into Meridian's `ec_m` units (multiples of a median non-zero
    per-capita week). It is measured delivery, not an assumption.

Note the asymmetry the figure exists to make visible: the default prior is
channel-agnostic, so it is the *same* curve whatever the channel's audience
or delivery. The planning-derived prior is not -- it is built from this
channel's reach and frequency, which is the entire point.

CAVEAT, and it must be checked before this reaches a slide: the denominator
below uses `current_reach x current_frequency` as an analytic stand-in for
the channel's median non-zero per-capita week. It ignores flighting and
seasonality, and it is self-consistent with (not independent of) the implied
effective frequency quoted elsewhere in the study. Confirm it against a probe
build of the simulator before printing any of these numbers.

Scratch/exploratory: not wired into `build_section1_figures.py` or asserted
by `prior_plots_check.py`.

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_informed_prior_band.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
  sys.path.insert(0, HERE)

import channel_labels
import prior_plots
import r90_basis


OUT_PATH = os.path.join(HERE, 'figures', 'scratch_informed_prior_band.png')
FIGSIZE = (13.33, 5.6)
DPI = 200

# Both channels, in the order they appear on the slide. Channel-1 is the
# under-reached, under-frequency'd case the study is about; Channel-2 is the
# well-delivered control, and including it is what shows this construction
# tracks delivery rather than simply shifting every channel to the right.
CHANNELS = ('TV', 'Display')

# The one judgement call in the derivation, stated as a range rather than a
# point. Everything else is either Meridian's own prior or measured delivery.
CONVENTION_LOW = 0.30
CONVENTION_HIGH = 0.70

# `prior_distribution.py` -- `ec_rf`/`ec_orf`, Shift(0.1)(LogNormal(0.7, 0.4)).
EC_RF_LOC = 0.7
EC_RF_SCALE = 0.4
EC_RF_SHIFT = 0.1

INFORMED_COLOR = '#e6820e'
N_DRAWS = 400_000
SEED = 0


def _channel_execution(channel: str) -> tuple[float, float]:
  """Current reach fraction and mean weekly frequency, from `r90_basis`.

  Read from `r90_basis.BASE_OVERRIDES` rather than restated here: a local copy
  of that config is exactly how the moving-estimand bug survived, and the same
  reasoning applies to any script that needs these numbers.
  """
  overrides = r90_basis.BASE_OVERRIDES
  reach = float(overrides['current_reach_frac'][channel])
  freq_low, freq_high = overrides['frequency_range'][channel]
  return reach, 0.5 * (float(freq_low) + float(freq_high))


def _ec_rf_draws(rng: np.random.Generator, size: int) -> np.ndarray:
  """Draws from Meridian's default `ec_rf` prior (effective frequency)."""
  return EC_RF_SHIFT + rng.lognormal(EC_RF_LOC, EC_RF_SCALE, size=size)


def informed_ec_draws(channel: str) -> tuple[np.ndarray, dict]:
  """Planning-derived `ec_m` draws for one channel, plus the facts quoted."""
  reach, frequency = _channel_execution(channel)
  denominator = reach * frequency

  rng = np.random.default_rng(SEED)
  convention = rng.uniform(CONVENTION_LOW, CONVENTION_HIGH, size=N_DRAWS)
  effective_frequency = _ec_rf_draws(rng, N_DRAWS)
  ec_draws = convention * effective_frequency / denominator

  default = prior_plots._ec_m_prior()  # pylint: disable=protected-access
  facts = {
      'reach': reach,
      'current_frequency': frequency,
      'denominator': denominator,
      'ec_rf_median': float(np.median(effective_frequency)),
      'informed_median': float(np.median(ec_draws)),
      'informed_q05': float(np.percentile(ec_draws, 5)),
      'informed_q95': float(np.percentile(ec_draws, 95)),
      'default_median': float(default.median()),
      'default_q99': float(default.ppf(0.99)),
      # How much of the planning-derived prior the default considers
      # plausible at all -- the honest overlap, not a rhetorical zero.
      'frac_below_default_q99': float(
          np.mean(ec_draws < default.ppf(0.99))
      ),
      # The business translation, at slope 1 (Meridian fixes it there):
      # hill(1) = 1 / (1 + ec_m) is the share of the ceiling captured today.
      'ceiling_today_default': float(np.median(1.0 / (1.0 + default.rvs(
          N_DRAWS, random_state=SEED
      )))),
      'ceiling_today_informed': float(np.median(1.0 / (1.0 + ec_draws))),
  }
  return ec_draws, facts


def _log_density(draws: np.ndarray, grid: np.ndarray) -> np.ndarray:
  """Density of `draws` with respect to log(x), evaluated on `grid`.

  Densities here are plotted per unit *log*, not per unit x, because the
  x-axis is logarithmic and the three priors sit decades apart. Plotting
  ordinary p(x) would make the narrowest-in-x prior tower over the others --
  Channel-2's planning-derived prior is Channel-1's divided by 10, so its
  p(x) peak is 10x taller for no reason the audience should read into. On a
  log axis the honest comparison is area-per-decade, which is what this is.
  """
  kde = stats.gaussian_kde(np.log(draws))
  return kde(np.log(grid))


def plot(ax_density, ax_bars, channels: tuple[str, ...] = CHANNELS) -> dict:
  """Left: the default prior against one planning-derived prior per channel.

  Right: the same disagreement as the number a planner actually reads. Both
  panels exist to make one contrast visible -- the default is a single curve
  whatever the channel, while the planning-derived priors separate because
  the channels' delivery separates. Channel-2 landing on top of the default
  is the control: this construction is not a blanket shift to the right.
  """
  default = prior_plots._ec_m_prior()  # pylint: disable=protected-access
  facts = {ch: informed_ec_draws(ch)[1] for ch in channels}
  draws = {ch: informed_ec_draws(ch)[0] for ch in channels}

  # --- Left: every prior on a shared log axis. ---
  grid = np.logspace(np.log10(0.1), np.log10(30.0), 600)
  default_pdf = default.pdf(grid) * grid  # per unit log, as above.

  ax_density.plot(grid, default_pdf, color=prior_plots.PRIOR_COLOR, lw=2.5)
  ax_density.fill_between(
      grid, default_pdf, color=prior_plots.PRIOR_COLOR, alpha=0.22
  )
  peaks = [default_pdf.max()]
  for ch, style in zip(channels, ('-', '--')):
    pdf = _log_density(draws[ch], grid)
    peaks.append(pdf.max())
    ax_density.plot(grid, pdf, color=INFORMED_COLOR, lw=2.5, ls=style)
    ax_density.fill_between(grid, pdf, color=INFORMED_COLOR, alpha=0.16)
    # Delivery figures stay off this panel -- they are on the right panel's
    # axis labels, and repeating them here crowded the curves they annotate.
    ax_density.text(
        facts[ch]['informed_median'],
        pdf.max() * 1.10,
        channel_labels.label(ch),
        color=INFORMED_COLOR,
        fontsize=11,
        ha='center',
        fontweight='bold',
    )

  top = max(peaks)
  ax_density.axvline(1.0, color=prior_plots.TODAY_COLOR, lw=2, ls='--')
  ax_density.annotate(
      'what you\nrun today',
      xy=(1.0, top * 0.86),
      xytext=(0.13, top * 0.92),
      color=prior_plots.TODAY_COLOR,
      fontsize=10,
      arrowprops=dict(
          arrowstyle='->', color=prior_plots.TODAY_COLOR, lw=1.4
      ),
  )
  ax_density.text(
      0.235,
      default_pdf.max() * 0.42,
      'Meridian default\none curve for\nevery channel',
      color=prior_plots.PRIOR_COLOR,
      fontsize=10,
      ha='center',
      fontweight='bold',
  )
  ax_density.set_xscale('log')
  ax_density.set_xlim(0.1, 30)
  ax_density.set_ylim(0, top * 1.42)
  ax_density.set_xticks([0.1, 0.3, 1, 3, 10, 30])
  ax_density.set_xticklabels(['0.1', '0.3', '1', '3', '10', '30'])
  ax_density.set_yticklabels([])
  ax_density.set_xlabel(
      'Half-saturation point\n(multiples of your current execution)'
  )
  ax_density.set_ylabel('Prior density')
  ax_density.set_title('Where does half-saturation sit?', fontsize=11)

  # --- Right: what each prior says you have already captured. ---
  positions = np.arange(len(channels))
  width = 0.34
  series = (
      ('ceiling_today_default', 'Meridian default', prior_plots.PRIOR_COLOR),
      ('ceiling_today_informed', 'Planning-derived', INFORMED_COLOR),
  )
  for offset, (key, name, color) in zip((-width / 2, width / 2), series):
    values = [facts[ch][key] for ch in channels]
    ax_bars.bar(
        positions + offset, values, width, color=color, label=name, alpha=0.9
    )
    for pos, value in zip(positions + offset, values):
      ax_bars.text(
          pos,
          value + 0.025,
          f'{value:.0%}',
          ha='center',
          fontsize=12,
          fontweight='bold',
          color=color,
      )
  ax_bars.set_xticks(positions)
  ax_bars.set_xticklabels([
      f'{channel_labels.label(ch)}\n({facts[ch]["reach"]:.0%} reach)'
      for ch in channels
  ])
  ax_bars.set_ylabel('Fraction of ceiling effect captured today')
  ax_bars.set_ylim(0, 1)
  ax_bars.set_title(
      'The default cannot tell these channels apart', fontsize=11
  )
  ax_bars.legend(loc='upper left', fontsize=9, framealpha=0.9)
  ax_bars.grid(axis='x', visible=False)

  return facts


def main() -> None:
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  prior_plots.apply_slide_style()
  fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)
  facts = plot(ax_left, ax_right)
  fig.tight_layout()
  fig.savefig(OUT_PATH, dpi=DPI, bbox_inches='tight', facecolor='white')
  plt.close(fig)

  first = facts[CHANNELS[0]]
  print(f'Wrote: {OUT_PATH}\n')
  print('effective frequency (ec_rf) median %.3f/wk' % first['ec_rf_median'])
  print('Meridian default ec_m: median %.2f, 99th pctile %.2f  (identical '
        'for every channel)' % (first['default_median'], first['default_q99']))
  print()
  for channel in CHANNELS:
    f = facts[channel]
    print(f'{channel_labels.label(channel)} ({channel}): reach '
          f'{f["reach"]:.0%}, frequency {f["current_frequency"]:.2f}/wk '
          f'-> denominator {f["denominator"]:.3f}')
    print('  planning-derived ec_m: median %.2f, 90%% interval [%.2f, %.2f]'
          % (f['informed_median'], f['informed_q05'], f['informed_q95']))
    print('  below the default\'s 99th pctile: %.2f%%'
          % (100 * f['frac_below_default_q99']))
    print('  ceiling captured today -- default %.0f%%, planning-derived %.0f%%'
          % (100 * f['ceiling_today_default'],
             100 * f['ceiling_today_informed']))


if __name__ == '__main__':
  main()
