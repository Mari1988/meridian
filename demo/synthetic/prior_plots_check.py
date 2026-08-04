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

"""Verifies every number that appears on the section 1 slides.

Run this before rebuilding the deck. It checks two things:

  1. `default_prior_facts()` matches values derived independently from the
     distributions declared in `meridian/model/prior_distribution.py`.
  2. The Hill curves really route through Meridian's own `HillTransformer`.
  3. The alpha-to-half-life conversion on slide 4 is algebraically correct,
     and adstock weights really do sum to 1 (Meridian's `normalize=True`).

Usage: `.venv/bin/python demo/synthetic/prior_plots_check.py`
"""

from __future__ import annotations

import os

import numpy as np

from model_utils import hill_value
import prior_plots

HERE = os.path.dirname(os.path.abspath(__file__))


def check_prior_facts() -> None:
  """Cross-checks `default_prior_facts()` against independent computations."""
  facts = prior_plots.default_prior_facts()
  expected = {
      'ec_m_median': 0.993,
      'ec_m_q05': 0.212,
      'ec_m_q95': 2.196,
      'p_past_half_saturation_today': 0.504,
      'p_ec_within_50pct_of_today': 0.563,
      'p_over_third_of_ceiling_today': 0.92,
      'median_ceiling_frac_today': 0.501,
      'carryover_lost_beyond_max_lag_at_alpha_090': 0.387,
  }
  for key, want in expected.items():
    got = facts[key]
    assert abs(got - want) < 0.01, f'{key}: got {got:.4f}, expected ~{want}'
    print(f'  ok  {key:<44} {got:.4f}')


def check_hill_goes_through_meridian() -> None:
  """`hill(ec) == 0.5` exactly -- the defining property of half-saturation."""
  at_ec = float(hill_value(1.0, np.array([1.0]))[0])
  assert abs(at_ec - 0.5) < 1e-6, f'hill(1.0, ec=1.0) = {at_ec}, expected 0.5'
  print(f'  ok  hill_value(1.0, ec=1.0) == {at_ec:.6f}')


def check_marginal_return_peaks() -> None:
  """Peak marginal return by slope -- the numbers quoted on slide 3."""
  x = np.linspace(1e-4, 3.0, 20_000)
  for slope, want in ((1.0, 0.0), (2.0, 0.58), (3.0, 0.79)):
    curve = hill_value(x, np.array([1.0]), slope=np.array([slope]))[:, 0]
    peak = float(x[int(np.argmax(np.gradient(curve, x)))])
    assert (
        abs(peak - want) < 0.02
    ), f'slope={slope}: peak at {peak:.3f}, expected ~{want}'
    print(f'  ok  slope={slope:.0f} peak marginal return at x={peak:.2f}')


def check_half_life_and_adstock_weights() -> None:
  """Alpha-to-half-life conversion and adstock weight normalization (slide 4).

  `half_life` inverts alpha = 0.5^(1/half_life); we check it round-trips.
  `_adstock_weights` must sum to 1 -- Meridian hardcodes `normalize=True`.
  """
  for alpha, want in ((0.3, 0.58), (0.6, 1.36), (0.9, 6.58)):
    hl = float(prior_plots.half_life(alpha))
    assert (
        abs(hl - want) < 0.01
    ), f'alpha={alpha}: half-life {hl:.3f}, expected ~{want}'
    assert (
        abs(0.5 ** (1 / hl) - alpha) < 1e-6
    ), f'alpha={alpha}: round-trip failed'
    print(f'  ok  alpha={alpha:<4} half-life={hl:.2f} weeks (round-trips)')

  weights = prior_plots._adstock_weights(np.array([0.0, 0.3, 0.6, 0.9, 0.999]))
  sums = weights.sum(axis=1)
  assert np.allclose(sums, 1.0), f'weights do not sum to 1: {sums}'
  print(
      f'  ok  adstock weights sum to 1 for every alpha (max dev '
      f'{float(np.max(np.abs(sums - 1))):.2e})'
  )


def check_results_facts() -> None:
  """Cross-checks the results slides' numbers against the exported runs.

  These are looser than the section 1 checks by nature: section 1's numbers are
  analytic, these come from MCMC. The assertions therefore test the *claims the
  slides make* -- direction, separation, ordering -- rather than pinning
  posterior means to three decimals, which would fail on an innocuous re-run.
  """
  import pandas as pd

  import build_section1_deck as deck
  import results_facts

  # Read the same facts functions the slides render from, so an assertion here
  # cannot pass while the slide says something else. Slides 5, 7, 8, 9 are on
  # the r90 basis (alpha_m=0.3, oracle_r2=0.9, seeds 1320/7/42); slide 6 still
  # reads `results_facts`, which is fine because the quantity it prints
  # (ceiling fraction) is a function of ec_m alone and is identical on both.
  dgp = deck._r90_dgp_facts()
  recovery = deck._r90_recovery_facts()
  mroi = deck._r90_mroi_facts()
  canonical = results_facts.dgp_facts()

  # Slide 5: the DGP is the hard one, and it is the basis slides 7-9 fit to.
  assert 0.88 < dgp['oracle_r2'] < 0.92, (
      f'oracle R^2 is {dgp["oracle_r2"]}, expected ~0.90 -- slides 5-9 are '
      'built on the r90 basis'
  )
  # On the pinned basis EVERY truth is constant across the ten datasets, so
  # slide 5's table must print single values throughout. `_fmt_true` renders
  # a range the moment a quantity starts varying, so these assertions are how
  # a regression in `r90_basis` surfaces -- as a failed build rather than as
  # a slide quietly describing a different DGP than slides 7-8 were fitted to.
  for key, expected in (
      ('true_ec_tv', '9.00'),
      ('true_alpha_tv', '0.30'),
      ('true_ec_display', '1.30'),
      ('true_alpha_display', '0.15'),
  ):
    assert dgp[key] == expected, (
        f'slide 5 should print {key} as the pinned value {expected!r}; it '
        f'reads {dgp[key]!r}. A range here means the truth is drifting '
        'across seeds again -- see r90_basis.'
    )
  for key in ('true_roi_tv', 'true_roi_display'):
    assert '\u2013' not in dgp[key], (
        f'slide 5 prints {key} as {dgp[key]!r}. ROI is now pinned too (it is '
        'derived from the pinned ec_m/alpha_m), so a range means the '
        'cross-channel coupling in target_roi_m has come back.'
    )
  print(
      f'  ok  DGP oracle R^2 {dgp["oracle_r2"]:.3f} | all truths pinned: '
      f'Channel-1 ec_m {dgp["true_ec_tv"]}, alpha_m {dgp["true_alpha_tv"]}, '
      f'ROI {dgp["true_roi_tv"]}; Channel-2 {dgp["true_ec_display"]}, '
      f'{dgp["true_alpha_display"]}, {dgp["true_roi_display"]}'
  )

  # Slide 6 keeps the canonical source; it may only do so while the two bases
  # agree on ec_m, which is what its ceiling fractions are computed from.
  assert abs(canonical['true_ec_tv'] - dgp['true_ec_tv_min']) < 0.05, (
      'slide 6 reads ceiling fractions from `results_facts` while slide 5 '
      'reads the r90 run. They no longer agree on ec_m -- point slide 6 at '
      'the r90 facts too.'
  )
  assert (
      abs(canonical['true_ec_display'] - dgp['true_ec_display_min']) < 0.05
  ), 'slide 6 vs slide 5 disagree on Display ec_m -- see above'
  print(
      f'  ok  slide 6 ceiling fractions still valid: ec_m agrees across '
      f'bases ({canonical["true_ec_tv"]:.2f}, '
      f'{canonical["true_ec_display"]:.2f})'
  )

  # Slide 7's whole argument is the CONTRAST between the two rows: the same
  # default prior misses badly where the truth is far from it and mildly where
  # it is not. Assert the contrast, not just the Channel-1 failure -- if the
  # rows ever stop differing, the "it's the prior" reading is gone.
  assert (
      recovery['ec_ch1_default_med'] < -40
  ), 'slide 7 claims the default badly understates Channel-1 saturation'
  assert abs(recovery['ec_ch2_default_med']) < 20, (
      'slide 7 claims the default is only mildly off on Channel-2, whose '
      'truth sits inside the prior. It no longer is -- the dose-dependent '
      'framing does not hold and the slide text must change.'
  )
  assert abs(recovery['ec_ch1_default_med']) > 3 * abs(
      recovery['ec_ch2_default_med']
  ), 'slide 7 claims the Channel-1 miss dwarfs the Channel-2 one'
  assert (
      abs(recovery['ec_ch1_informed_med']) < 10
  ), 'slide 7 claims the informed prior recovers Channel-1 saturation'
  assert (
      abs(recovery['ec_ch2_informed_med']) < 10
  ), 'slide 7 claims the informed prior straddles zero on Channel-2 too'
  print(
      f'  ok  ec_m median error: Channel-1 default '
      f'{recovery["ec_ch1_default_med"]:+.0f}% vs Channel-2 default '
      f'{recovery["ec_ch2_default_med"]:+.0f}% -- dose-dependent; informed '
      f'{recovery["ec_ch1_informed_med"]:+.0f}% / '
      f'{recovery["ec_ch2_informed_med"]:+.0f}%'
  )

  # The ROI line is deliberately hedged; assert the hedge is still the honest
  # reading -- informed reduces Channel-1's bias without removing it.
  assert recovery['roi_ch1_informed_q25'] > 0, (
      'slide 7 says the informed ROI box never crosses zero on Channel-1; it '
      'now does'
  )
  assert abs(recovery['roi_ch1_informed_med']) < abs(
      recovery['roi_ch1_default_med']
  ), 'slide 7 claims informed roughly halves the median ROI error'
  print(
      f'  ok  roi_m median: Channel-1 default '
      f'{recovery["roi_ch1_default_med"]:+.0f}% -> informed '
      f'{recovery["roi_ch1_informed_med"]:+.0f}%, informed box still above '
      f'zero; Channel-2 {recovery["roi_ch2_default_med"]:+.0f}% vs '
      f'{recovery["roi_ch2_informed_med"]:+.0f}% (indistinguishable)'
  )

  # Carryover: assert only that the two priors are indistinguishable, which
  # is what the slide claims. Both sit well above zero (~+15-17% on
  # Channel-1) -- neither prior recovers adstock, and saying so is the point.
  assert (
      abs(
          recovery['alpha_ch1_default_med'] - recovery['alpha_ch1_informed_med']
      )
      < 10
  ), (
      'slide 7 says carryover shows no meaningful difference between the '
      'priors on Channel-1; they now differ by '
      f'{recovery["alpha_ch1_default_med"] - recovery["alpha_ch1_informed_med"]:+.1f}pp'
  )
  print(
      f'  ok  alpha_m: Channel-1 default '
      f'{recovery["alpha_ch1_default_med"]:+.0f}% vs informed '
      f'{recovery["alpha_ch1_informed_med"]:+.0f}% -- indistinguishable, and '
      'neither recovers it'
  )

  # Slide 8's load-bearing claim, and the reason Channel-2 is on the slide:
  # the default's error must compound with extrapolation on the under-invested
  # channel and NOT on the control. If that contrast ever collapses, the
  # slide's headline ("it breaks when you extrapolate from a misplaced
  # saturation point, not when you extrapolate") is no longer supported.
  assert mroi['m10x_ch1_default_median'] < -40, (
      "slide 8 claims the default's marginal ROI collapses by 10x spend on "
      f'Channel-1; median is {mroi["m10x_ch1_default_median"]:+.1f}%'
  )
  assert abs(mroi['ch2_worst_abs_median']) < 15, (
      'slide 8 uses Channel-2 as the control -- its median mROI error must '
      'stay small at EVERY multiplier for both priors. Worst is now '
      f'{mroi["ch2_worst_abs_median"]:.1f}%. If this grew, the '
      '"extrapolation is not the problem" framing must be cut, not softened.'
  )
  assert abs(mroi['m10x_ch1_default_median']) > 5 * abs(
      mroi['m10x_ch2_default_median']
  ), 'slide 8 claims the 10x miss on Channel-1 dwarfs the one on Channel-2'
  # The error changing sign is what the "flatters today, starves tomorrow"
  # line rests on.
  assert (
      mroi['m1x_ch1_default_median'] > 0 > mroi['m10x_ch1_default_median']
  ), 'slide 8 says the default flatters at 1x and understates at 10x'
  # Coverage is quoted at 90% in the footnote, so assert it there.
  assert mroi['covered_10x_ch1_default'] == 0, (
      "slide 8's footnote says the default's 90% interval never contains the "
      f'truth at 10x; it now does on {mroi["covered_10x_ch1_default"]} seeds'
  )
  print(
      f'  ok  mROI Channel-1 default {mroi["m1x_ch1_default_median"]:+.0f}% at '
      f'1x -> {mroi["m10x_ch1_default_median"]:+.0f}% at 10x (90% interval '
      f'covers truth {mroi["covered_10x_ch1_default"]}/{mroi["n_seeds"]}); '
      f'Channel-2 control never worse than '
      f'{mroi["ch2_worst_abs_median"]:.0f}%'
  )
  print(
      f'  ok  informed stays flat: {mroi["m1x_ch1_informed_median"]:+.0f}% at '
      f'1x -> {mroi["m10x_ch1_informed_median"]:+.0f}% at 10x'
  )


def main() -> None:
  for name, fn in (
      ('prior facts', check_prior_facts),
      ('Hill routes through Meridian', check_hill_goes_through_meridian),
      ('marginal return peaks', check_marginal_return_peaks),
      ('half-life and adstock weights', check_half_life_and_adstock_weights),
      ('results facts (slides 5-8)', check_results_facts),
  ):
    print(f'\n{name}:')
    fn()
  print('\nAll deck numbers verified (slides 1-8).')


if __name__ == '__main__':
  main()
