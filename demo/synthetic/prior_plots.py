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

"""Figures for section 1 of the ARF Analytics Council talk.

Section 1 asks "what does an MMM assume before it sees your data?" and answers
it with three of Meridian's out-of-the-box defaults:

  * `ec_m` (half-saturation)  -- a prior that is centered on the status quo.
  * `slope_m` (Hill slope)    -- not estimated at all; fixed at 1.0.
  * `alpha_m` (adstock decay) -- genuinely flat, no complaint there; but the
                                 binding assumption (`max_lag`) is not a prior
                                 in the first place.

Nothing here fits a model or runs MCMC -- every number is either analytic or a
draw from a prior, so the whole module renders in seconds. Hill curves are
evaluated through `model_utils.hill_value`, which wraps Meridian's own
`adstock_hill.HillTransformer`, so the curves on the slides are drawn by the
same code the model uses rather than a reimplementation.

Sources for the default values quoted here:
  * `meridian/model/prior_distribution.py` (`ec_m`, `slope_m`, `alpha_m`)
  * `meridian/model/spec.py` (`max_lag`)
  * https://developers.google.com/meridian/docs/advanced-modeling/default-prior-distributions
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from model_utils import hill_value


# --- Meridian defaults, mirrored from the source of truth. ---------------
# `prior_distribution.py:383-391` -- TruncatedNormal(0.8, 0.8, 0.1, 10).
EC_M_LOC = 0.8
EC_M_SCALE = 0.8
EC_M_LOW = 0.1
EC_M_HIGH = 10.0
# `prior_distribution.py:419-423` -- Deterministic(1.0).
SLOPE_M_FIXED = 1.0
# `prior_distribution.py:355-362` -- Uniform(0.0, 1.0).
ALPHA_M_LOW = 0.0
ALPHA_M_HIGH = 1.0
# `spec.py:237` -- a hard truncation, not a prior.
MAX_LAG = 8

# Consistent with the response-curve overlay already used in
# `meridian_tv_underreach_case_study.ipynb`.
TODAY_COLOR = '#c0392b'
PRIOR_COLOR = '#2c7fb8'
ACCENT_COLORS = ('#2c7fb8', '#e6820e', '#3b8c3b')

_N_PRIOR_DRAWS = 200_000
_SEED = 0


def _ec_m_prior() -> stats.rv_continuous:
  """Meridian's default `ec_m` prior as a frozen scipy distribution."""
  a = (EC_M_LOW - EC_M_LOC) / EC_M_SCALE
  b = (EC_M_HIGH - EC_M_LOC) / EC_M_SCALE
  return stats.truncnorm(a, b, loc=EC_M_LOC, scale=EC_M_SCALE)


def _adstock_weights(alpha: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
  """Normalized geometric adstock weights, matching Meridian's `_adstock`.

  `adstock_hill._adstock` hardcodes `normalize=True`, so the transform is a
  weighted *average* over the window rather than a weighted sum -- adstocked
  media stays on the same scale as raw media regardless of `alpha`.

  Args:
    alpha: 1-D array of decay rates.
    max_lag: Window is `max_lag + 1` periods, per `spec.ModelSpec.max_lag`.

  Returns:
    Array of shape `(len(alpha), max_lag + 1)`, each row summing to 1.
  """
  lags = np.arange(max_lag + 1)
  weights = np.asarray(alpha)[:, np.newaxis] ** lags[np.newaxis, :]
  return weights / weights.sum(axis=1, keepdims=True)


def half_life(alpha: np.ndarray) -> np.ndarray:
  """Weeks for adstock weight to decay by half: alpha = 0.5^(1/half_life)."""
  return np.log(0.5) / np.log(alpha)


def default_prior_facts() -> dict[str, float]:
  """Numbers quoted on the section 1 slides, computed from the defaults.

  Returning them from one place keeps the slide text and the charts from
  drifting apart -- `build_section1_figures.py` prints these and the speaker
  notes quote them.
  """
  ec = _ec_m_prior()

  # `ec_m` is denominated in multiples of the channel's own median non-zero
  # per-capita execution, so x=1.0 is "what you already run".
  ec_draws = ec.rvs(_N_PRIOR_DRAWS, random_state=_SEED)
  ceiling_frac_today = 1.0 / (1.0 + ec_draws)  # hill(1) with slope=1.

  return {
      # Slide 2 -- saturation.
      'ec_m_median': float(ec.median()),
      'ec_m_q05': float(ec.ppf(0.05)),
      'ec_m_q95': float(ec.ppf(0.95)),
      'p_past_half_saturation_today': float(ec.cdf(1.0)),
      'p_ec_within_50pct_of_today': float(ec.cdf(1.5) - ec.cdf(0.5)),
      'p_over_third_of_ceiling_today': float(
          np.mean(ceiling_frac_today > 1 / 3)
      ),
      'median_ceiling_frac_today': float(np.median(ceiling_frac_today)),
      # Slide 4 -- adstock.
      'carryover_lost_beyond_max_lag_at_alpha_090': float(0.9 ** (MAX_LAG + 1)),
  }


def plot_ec_prior_and_curves(ax_density, ax_curves) -> None:
  """Slide 2 -- the default half-saturation prior, and what it implies.

  Left: the prior density, with the x-axis labelled in the units that make it
  interpretable. Right: saturation curves drawn from that prior, so the room
  sees the business claim rather than a density.
  """
  ec = _ec_m_prior()
  facts = default_prior_facts()

  # --- Left: the prior itself. ---
  grid = np.linspace(EC_M_LOW, 4.0, 500)
  ax_density.plot(grid, ec.pdf(grid), color=PRIOR_COLOR, lw=2.5)
  ax_density.fill_between(
      grid,
      ec.pdf(grid),
      where=(grid >= 0.5) & (grid <= 1.5),
      color=PRIOR_COLOR,
      alpha=0.25,
  )
  ax_density.axvline(1.0, color=TODAY_COLOR, lw=2, ls='--')
  ax_density.annotate(
      'what you\nrun today',
      xy=(1.0, ax_density.get_ylim()[1] * 0.92),
      xytext=(1.75, ec.pdf(grid).max() * 0.88),
      color=TODAY_COLOR,
      fontsize=10,
      arrowprops=dict(arrowstyle='->', color=TODAY_COLOR, lw=1.4),
  )
  ax_density.text(
      0.65,
      ec.pdf(grid).max() * 0.30,
      f'{facts["p_ec_within_50pct_of_today"]:.0%}\nwithin ±50%\nof today',
      color=PRIOR_COLOR,
      fontsize=10,
      ha='center',
      fontweight='bold',
  )
  ax_density.set_xlabel(
      'Half-saturation point\n(multiples of your current execution)'
  )
  ax_density.set_ylabel('Prior density')
  ax_density.set_title(
      'Meridian default: ec_m ~ TruncatedNormal(0.8, 0.8, [0.1, 10])',
      fontsize=11,
  )
  ax_density.set_xlim(0, 4.0)

  # --- Right: what those draws mean as response curves, as a fan chart. ---
  x = np.linspace(0.01, 3.0, 200)
  draws = ec.rvs(5000, random_state=_SEED)
  curves = hill_value(x, draws, slope=np.full_like(draws, SLOPE_M_FIXED))
  for lo, hi, alpha in ((10, 90, 0.18), (25, 75, 0.30)):
    ax_curves.fill_between(
        x,
        np.percentile(curves, lo, axis=1),
        np.percentile(curves, hi, axis=1),
        color=PRIOR_COLOR,
        alpha=alpha,
        label=f'{lo}-{hi}th percentile',
    )
  median_curve = hill_value(
      x, np.array([ec.median()]), slope=np.array([SLOPE_M_FIXED])
  )[:, 0]
  ax_curves.plot(
      x, median_curve, color=PRIOR_COLOR, lw=3, label='Prior median channel'
  )
  ax_curves.axvline(1.0, color=TODAY_COLOR, lw=2, ls='--', label='Today')
  ax_curves.axhline(0.5, color='grey', lw=1, ls=':')
  ax_curves.plot([1.0], [0.5], marker='o', color=TODAY_COLOR, ms=10, zorder=5)
  ax_curves.set_xlabel('Multiples of your current execution')
  ax_curves.set_ylabel('Fraction of ceiling effect')
  ax_curves.set_title('Implied response curve', fontsize=11)
  ax_curves.set_ylim(0, 1)
  ax_curves.set_xlim(0, 3)
  ax_curves.legend(loc='lower right', fontsize=8, framealpha=0.9)


def plot_slope_shapes(ax_hill, ax_marginal) -> None:
  """Slide 3 -- the Hill slope is fixed at 1.0 and never estimated.

  Left: what the fixed slope can and cannot represent. Right: the consequence
  for marginal return, which is what a budget optimizer actually reads.
  """
  x = np.linspace(0.01, 3.0, 400)
  ec = 1.0
  slopes = (1.0, 2.0, 3.0)

  for slope, color in zip(slopes, ACCENT_COLORS):
    curve = hill_value(x, np.array([ec]), slope=np.array([slope]))[:, 0]
    is_default = slope == SLOPE_M_FIXED
    # Meridian does *let* you override `slope_m` -- it warns against it
    # (`prior_distribution.py:844-856`). The claim is about the default.
    label = (
        f"slope = {slope:.0f}  (Meridian's default)"
        if is_default
        else f'slope = {slope:.0f}'
    )
    ax_hill.plot(
        x,
        curve,
        color=color,
        lw=3 if is_default else 2,
        ls='-' if is_default else '--',
        label=label,
    )
    marginal = np.gradient(curve, x)
    ax_marginal.plot(
        x,
        marginal,
        color=color,
        lw=3 if is_default else 2,
        ls='-' if is_default else '--',
        label=f'slope = {slope:.0f}',
    )
    peak = x[int(np.argmax(marginal))]
    ax_marginal.plot([peak], [marginal.max()], marker='o', color=color, ms=7)

  ax_hill.axvline(1.0, color=TODAY_COLOR, lw=2, ls=':', label='Today')
  ax_hill.set_xlabel('Media execution (multiples of today)')
  ax_hill.set_ylabel('Hill output')
  ax_hill.set_title('Response curve under varying slope', fontsize=11)
  ax_hill.set_ylim(0, 1)
  ax_hill.legend(loc='lower right', fontsize=9, framealpha=0.9)

  ax_marginal.set_xlabel('Media execution (multiples of today)')
  ax_marginal.set_ylabel('Marginal return')
  ax_marginal.set_title('Marginal return under varying slope', fontsize=11)
  ax_marginal.legend(loc='upper right', fontsize=9, framealpha=0.9)


def plot_adstock_prior_and_decay(ax_prior, ax_decay) -> None:
  """Slide 4 -- adstock: the prior is genuinely flat; `max_lag` is the catch.

  Left: `alpha_m`'s flat prior itself, so the room sees there is no hidden
  skew here (unlike `ec_m`). Right: what different decay rates imply in
  half-life terms, plus the hard `max_lag` truncation those curves run into.
  """
  facts = default_prior_facts()

  # --- Left: the flat alpha_m prior. ---
  height = 1.0 / (ALPHA_M_HIGH - ALPHA_M_LOW)
  ax_prior.plot(
      [ALPHA_M_LOW, ALPHA_M_HIGH], [height, height], color=PRIOR_COLOR, lw=2.5
  )
  ax_prior.fill_between(
      [ALPHA_M_LOW, ALPHA_M_HIGH], [height, height],
      color=PRIOR_COLOR, alpha=0.25,
  )
  ax_prior.set_ylim(0, height * 1.3)
  ax_prior.set_xlim(-0.05, 1.05)
  ax_prior.set_xlabel('Decay rate (alpha_m)')
  ax_prior.set_ylabel('Prior density')
  ax_prior.set_title('Meridian default: alpha_m ~ Uniform(0, 1)', fontsize=11)

  # --- Right: decay shapes by half-life, and what max_lag truncates. ---
  lags = np.arange(MAX_LAG + 1)
  for alpha, color in zip((0.3, 0.6, 0.9), ACCENT_COLORS):
    weights = _adstock_weights(np.array([alpha]))[0]
    hl = float(half_life(np.array([alpha]))[0])
    ax_decay.plot(
        lags, weights, marker='o', color=color, lw=2,
        label=f'alpha = {alpha}  (half-life {hl:.1f} wk)',
    )
    if alpha == 0.9:
      ax_decay.annotate(
          f'at alpha=0.9, '
          f'{facts["carryover_lost_beyond_max_lag_at_alpha_090"]:.0%} of true '
          'carryover\nlies beyond week 8 and is discarded',
          xy=(MAX_LAG, weights[-1]),
          xytext=(3.0, weights.max() * 0.72),
          fontsize=9,
          color=color,
          arrowprops=dict(arrowstyle='->', color=color, lw=1.4),
      )
  ax_decay.axvline(
      MAX_LAG, color=TODAY_COLOR, lw=2, ls='--', label='max_lag = 8 (hard cut)'
  )
  ax_decay.set_xlabel('Weeks after the impression')
  ax_decay.set_ylabel('Share of the effect')
  ax_decay.set_title(
      'max_lag is not a prior -- it is a hard truncation', fontsize=11
  )
  ax_decay.legend(fontsize=9, framealpha=0.9)


def apply_slide_style() -> None:
  """Matplotlib defaults tuned for projected slides."""
  plt.rcParams.update(
      {
          'figure.facecolor': 'white',
          'axes.facecolor': 'white',
          'axes.grid': True,
          'grid.alpha': 0.25,
          'axes.spines.top': False,
          'axes.spines.right': False,
          'font.size': 11,
          'axes.titlesize': 12,
          'axes.labelsize': 11,
          'legend.frameon': True,
      }
  )
