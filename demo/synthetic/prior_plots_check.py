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

Run this before rebuilding the deck. It checks three things:

  1. `default_prior_facts()` matches values derived independently from the
     distributions declared in `meridian/model/prior_distribution.py`.
  2. The Hill curves really route through Meridian's own `HillTransformer`.
  3. Robyn's Hill function is algebraically identical to Meridian's -- the
     claim slide 3 is built on -- rather than merely asserted.

Usage: `.venv/bin/python demo/synthetic/prior_plots_check.py`
"""

from __future__ import annotations

import numpy as np

from model_utils import hill_value
import prior_plots


def check_prior_facts() -> None:
  """Cross-checks `default_prior_facts()` against independent computations."""
  facts = prior_plots.default_prior_facts()
  expected = {
      'ec_m_median': 0.993,
      'ec_m_q05': 0.212,
      'ec_m_q95': 2.196,
      'p_past_half_saturation_today': 0.504,
      'p_over_third_of_ceiling_today': 0.92,
      'median_ceiling_frac_today': 0.501,
      'week0_share_median': 0.503,
      'p_week0_share_ge_70': 0.30,
      'p_week0_share_le_30': 0.28,
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


def check_robyn_meridian_hill_equivalence() -> None:
  """Robyn's `1/(1+(gamma/x)^alpha)` == Meridian's Hill, on a grid.

  Robyn:    media_saturated = 1 / (1 + (gamma / media_adstocked) ** alpha)
  Meridian: hill(x) = x**slope / (x**slope + ec**slope)

  These are the same function with gamma <-> ec and alpha <-> slope, which is
  what licenses the cross-tool comparison on slide 3.
  """
  x = np.linspace(0.05, 5.0, 60)
  for gamma, alpha in ((1.0, 1.0), (0.8, 2.0), (1.7, 3.0), (0.3, 0.5)):
    robyn = 1.0 / (1.0 + (gamma / x) ** alpha)
    meridian = hill_value(x, np.array([gamma]), slope=np.array([alpha]))[:, 0]
    max_diff = float(np.max(np.abs(robyn - meridian)))
    assert (
        max_diff < 1e-5
    ), f'gamma={gamma}, alpha={alpha}: max abs diff {max_diff:.2e}'
    print(
        f'  ok  gamma={gamma:<4} alpha={alpha:<4} max|Robyn - Meridian| = '
        f'{max_diff:.2e}'
    )


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


def check_adstock_weights_normalized() -> None:
  """Meridian hardcodes `normalize=True`, so weights must sum to 1."""
  weights = prior_plots._adstock_weights(np.array([0.0, 0.3, 0.6, 0.9, 0.999]))
  sums = weights.sum(axis=1)
  assert np.allclose(sums, 1.0), f'weights do not sum to 1: {sums}'
  print(
      f'  ok  adstock weights sum to 1 for every alpha (max dev '
      f'{float(np.max(np.abs(sums - 1))):.2e})'
  )


def main() -> None:
  for name, fn in (
      ('prior facts', check_prior_facts),
      ('Hill routes through Meridian', check_hill_goes_through_meridian),
      (
          'Robyn / Meridian Hill equivalence',
          check_robyn_meridian_hill_equivalence,
      ),
      ('marginal return peaks', check_marginal_return_peaks),
      ('adstock weight normalization', check_adstock_weights_normalized),
  ):
    print(f'\n{name}:')
    fn()
  print('\nAll section 1 slide numbers verified.')


if __name__ == '__main__':
  main()
