"""The deck's r90 basis, in one place: a seed sweep that varies ONLY noise.

Every r90 script used to carry its own copy of `BASE_OVERRIDES` and its own
two-pass `ec_m` solve. That duplication is what let the following bug live
unnoticed, so the basis now lives here and the scripts import it.

THE BUG THIS MODULE EXISTS TO PREVENT
-------------------------------------
A seed sweep is supposed to answer one question: does the *randomness in the
data* change the conclusion? That requires the true parameters -- the estimand
-- to be identical in every draw, with only the noise realization moving.

The earlier basis pinned only Channel-1's `ec_m` (9.0) and `alpha_m` (0.3).
Channel-2's `alpha_m` was drawn from `Uniform(0, 0.3)`, landing at 0.156,
0.006 and 0.033 in seeds 1320, 7 and 42. That looks harmless -- it is the
other channel, and only its carryover. It is not, because of this line in
`data_simulator.simulate_adstock_hill_params()`:

    target_roi_m = base * (ec_m / ec_gmean) ** roi_ec_elasticity
                        * (alpha_m / alpha_gmean) ** roi_alpha_elasticity

Both geometric means are taken ACROSS CHANNELS. So Channel-2's drawn
`alpha_m` moves `alpha_gmean` (0.216 / 0.041 / 0.100) and therefore moves
**Channel-1's** true ROI: 6.60 / 10.88 / 8.35 across the three seeds. The ROI
calibration in `calibrate_channel_effects()` is exact -- it hits
`target_roi_m` to the last digit -- so the drift was never estimation noise.
The target itself was moving.

That gave the sweep two sources of variation at once, and made its ROI spread
uninterpretable: it mixed "the same estimator on a different noise draw" with
"the estimator at a different true ROI", which are different questions.

WHAT IS PINNED HERE
-------------------
All four structural truths, for both channels, so `target_roi_m` is constant
and (by the exactness above) so is every realized `roi_m`:

  * `ec_m`    -- solved per channel via `saturation_frequency` (below).
  * `alpha_m` -- degenerate `(x, x)` ranges for both channels.
  * `slope_m` -- already degenerate `(1.0, 1.0)` in the simulator's defaults.
  * `roi_m`   -- follows from the three above; asserted, not assumed.

`build_scenario()` asserts every one of them on every call. If a future edit
un-pins something, the run fails rather than quietly reintroducing the second
source of variation.

WHAT STILL VARIES WITH THE SEED
-------------------------------
Exactly what should: the media execution draw, the baseline AR(1) shock, the
residual noise, and the geo intercepts. `oracle_r2` is re-calibrated per draw
to 0.9, so signal-to-noise is held at the same target too.

CHANGING THE TRUTHS: USE A `Basis`, NEVER A SECOND COPY OF THIS MODULE
---------------------------------------------------------------------
A study sometimes needs the same sweep at a *different* pinned truth -- e.g.
Channel-1 at `alpha_m=0.8` instead of 0.3, to make `max_lag` bite. Copying
this module and editing a constant would recreate exactly the duplication the
docstring above exists to prevent. Instead every truth-carrying value hangs
off a `Basis` (below), `build_scenario(..., basis=...)` takes one, and the
module-level `TRUE_EC` / `TRUE_ALPHA` / `BASE_OVERRIDES` names remain bound to
`ALPHA03` so existing callers are untouched.

Note that `alpha_m` is NOT a free knob: `target_roi_m` normalises it by a
geometric mean taken ACROSS channels, so moving one channel's `alpha_m` moves
BOTH channels' true `roi_m`. `expected_true_roi()` derives that from the
simulator's own formula per basis, and `build_scenario` asserts it, so the
sweep stays clean -- but two bases are two different estimands, and ROI error
LEVELS are not comparable between them.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
  sys.path.insert(0, HERE)

# The same ten draws as the canonical `realistic_baseline_2ch_seeds` sweep, so
# this basis is comparable to it draw-for-draw rather than using a fresh
# arbitrary list. The first three are the seeds the 3-seed version of this
# study used, which keeps its results a strict subset.
SEEDS = [1320, 7, 42, 8, 99, 101, 555, 2024, 12345, 31337]

# A 50-draw extension for arms that need a tighter read on the SPREAD rather
# than the median. Ten draws pin a median well enough but leave the tails of
# a wide distribution (the informed arm's `ec_m` runs -53% to +43%) resting on
# one or two points each.
#
# `SEEDS` stays the first ten entries, so a 50-seed run is a strict superset
# of every 10-seed result and the two can be compared draw-for-draw rather
# than only in aggregate. The extra forty came from
# `np.random.default_rng(90210).integers(1, 100000)`, de-duplicated against
# the original ten; they are written out as a literal so the list cannot move
# if that helper is ever re-run differently.
SEEDS_EXTRA_40 = [
    21877, 61225, 90033, 93612, 28214, 99768, 938, 52463, 96275, 88982,
    36452, 33080, 16847, 6928, 32325, 2559, 44329, 15510, 32226, 71064,
    62256, 84275, 89066, 38578, 80694, 42975, 41172, 28767, 3573, 86071,
    77175, 49024, 2016, 93750, 54351, 35532, 78122, 55463, 57719, 61686,
]
SEEDS_50 = SEEDS + SEEDS_EXTRA_40
assert len(set(SEEDS_50)) == 50, 'seed collision -- draws would be duplicated'

N_KNOTS = 8
TARGET_ORACLE_R2 = 0.9


@dataclasses.dataclass(frozen=True)
class Basis:
  """One set of pinned structural truths. `name` tags every output row."""

  name: str
  true_ec: dict[str, float]
  true_alpha: dict[str, float]


# The deck's basis. Channel-2's values are the centres of the ranges they used
# to be drawn from (`ec_m` ran 1.29-1.32; `alpha_m ~ Uniform(0, 0.3)` has mean
# 0.15), so this basis sits where the old one averaged rather than moving the
# study somewhere new.
ALPHA03 = Basis(
    name='alpha03',
    true_ec={'TV': 9.0, 'Display': 1.3},
    true_alpha={'TV': 0.3, 'Display': 0.15},
)

# Channel-1 at heavy carryover, everything else identical. At `alpha_m=0.3`
# the `default` arm's out-of-the-box `max_lag=8` window loses essentially
# nothing (0.3**8 ~ 1e-4, 100.0% of the DGP's 13-week adstock mass sits in
# lags 0-8); at 0.8 it loses a real 9.4%. So this basis is what makes the
# documented `max_lag` split (default 8 vs informed 13) a lever with
# something to measure rather than a formality.
#
# It is a DIFFERENT ESTIMAND, not a re-run: raising TV's `alpha_m` raises the
# cross-channel `alpha_gmean` (0.2121 -> 0.3464), which moves true `roi_m` for
# both channels (TV 6.6405 -> 7.6930, Display 6.0237 -> 5.1995). Compare ROI
# direction and spread against `ALPHA03`, never levels.
ALPHA08 = Basis(
    name='alpha08',
    true_ec=dict(ALPHA03.true_ec),
    true_alpha=dict(ALPHA03.true_alpha, TV=0.8),
)

BASES = {b.name: b for b in (ALPHA03, ALPHA08)}
DEFAULT_BASIS = ALPHA03

# Back-compat aliases: every existing caller reads these and means `ALPHA03`.
TRUE_EC = ALPHA03.true_ec
TRUE_ALPHA = ALPHA03.true_alpha

# `simulate_adstock_hill_params()` derives `ec_m` linearly in
# `saturation_frequency` (both `ec_impressions` and `median_m` are computed
# from quantities that do not depend on it), and `saturation_frequency`
# leaves media execution itself untouched. So one probe build is enough to
# rescale each channel onto its target exactly.
DEFAULT_SATURATION_FREQUENCY = 4.0


def base_overrides(basis: Basis = DEFAULT_BASIS) -> dict:
  """`SimulationConfig` overrides for one basis. The only DGP definition."""
  return {
      'n_imp_channels': 2,
      'channel_names': ['TV', 'Display'],
      'target_audience_pop_frac': {'TV': 0.60, 'Display': 0.50},
      'current_reach_frac': {'TV': 0.10, 'Display': 0.5},
      'frequency_range': {'TV': (1, 2), 'Display': (1, 5)},
      'target_roi': {'TV': 8.0, 'Display': 5.0},
      'max_lag': 13,
      'n_times': 156,
      'roi_ec_elasticity': -0.3,
      'roi_alpha_elasticity': 0.3,
      # Degenerate ranges: this is what pins `alpha_m` exactly.
      'adstock_retention_range': {
          channel: (alpha, alpha)
          for channel, alpha in basis.true_alpha.items()
      },
      'baseline_scale': 0.065,
  }


BASE_OVERRIDES = base_overrides(ALPHA03)

RF_SOURCE_MAP = {'TV': 'Channel3'}
PLAIN_SOURCE_MAP = {'Display': 'Channel2'}

EC_TOL = 0.05
ALPHA_TOL = 1e-6
# `roi_m` is pinned only as tightly as the float path that produces it; this
# is far below any difference that would matter to a conclusion.
ROI_TOL = 1e-3


def real_df() -> pd.DataFrame:
  """The Meridian demo dataset the scenarios are scaled onto, cached."""
  cache = os.path.join(HERE, '.cache_geo_media_rf.csv')
  if not os.path.exists(cache):
    pd.read_csv(
        'https://raw.githubusercontent.com/google/meridian/refs/heads/main/'
        'meridian/data/simulated_data/csv/geo_media_rf.csv'
    ).to_csv(cache, index=False)
  return pd.read_csv(cache)


def expected_true_roi(basis: Basis = DEFAULT_BASIS) -> dict[str, float]:
  """The `roi_m` the pinned truths imply, from the simulator's own formula.

  Derived rather than measured, so `build_scenario`'s assertion is an
  independent check on the DGP and not a tautology against whatever it
  happened to produce.

  Both geometric means below are taken ACROSS channels, which is why a basis
  that moves one channel's `alpha_m` moves every channel's true `roi_m`.
  """
  overrides = base_overrides(basis)
  channels = overrides['channel_names']
  ec = np.array([basis.true_ec[c] for c in channels])
  alpha = np.array([basis.true_alpha[c] for c in channels])
  base = np.array([overrides['target_roi'][c] for c in channels])
  ec_gmean = np.exp(np.mean(np.log(ec)))
  alpha_gmean = np.exp(np.mean(np.log(alpha)))
  multiplier = (ec / ec_gmean) ** overrides['roi_ec_elasticity'] * (
      alpha / alpha_gmean
  ) ** overrides['roi_alpha_elasticity']
  return dict(zip(channels, base * multiplier))


def build_scenario(
    seed: int,
    target_oracle_r2: float | None = None,
    realism=None,
    basis: Basis = DEFAULT_BASIS,
):
  """Builds one draw at the pinned basis. Returns `(cfg, sim, data, gt)`.

  Data generation only -- no MCMC. Raises if any pinned truth misses, which
  is the whole point: a silent miss would put the study back to varying two
  things at once.

  Args:
    seed: The draw. Only the noise realization varies with it; every
      structural truth is pinned and asserted below.
    target_oracle_r2: Overrides `TARGET_ORACLE_R2` for this draw.
    realism: A `realistic_baseline.RealisticBaselineConfig`, or `None` for
      the realistic default. Threaded through here rather than reconstructed
      in caller scripts so a baseline ablation cannot quietly diverge from
      the pinned truths -- the same reason `BASE_OVERRIDES` lives in this
      module. `ec_m`/`alpha_m`/`roi_m` do not depend on the baseline, so the
      assertions below hold for any `realism` and will catch it if that ever
      stops being true.
    basis: Which pinned truths to build. Defaults to `ALPHA03`, the deck's.
      Every truth used below is read off this object, so a caller cannot fit
      one basis while checking against another's numbers.
  """
  from data_simulator import SimulationConfig
  import realistic_baseline as rb

  r2 = TARGET_ORACLE_R2 if target_oracle_r2 is None else target_oracle_r2
  df = real_df()
  overrides_base = base_overrides(basis)
  channels = overrides_base['channel_names']

  def build(saturation_frequency=None, oracle_r2=r2):
    overrides = dict(overrides_base, seed_num=seed)
    if saturation_frequency is not None:
      overrides['saturation_frequency'] = saturation_frequency
    cfg = SimulationConfig.from_dict(overrides)
    sim, data, gt = rb.build_real_augmented_realistic(
        cfg,
        df,
        rf_source_map=RF_SOURCE_MAP,
        plain_source_map=PLAIN_SOURCE_MAP,
        realism=realism,
        target_oracle_r2=oracle_r2,
        n_knots=N_KNOTS,
    )
    return cfg, sim, data, gt

  # Probe build at the default frequency, purely to read this draw's natural
  # `ec_m`; `oracle_r2=None` skips the noise calibration, which is the slow
  # part and is irrelevant to the solve.
  _, probe, _, _ = build(oracle_r2=None)
  probe_ec = probe.ec_m.numpy()
  saturation_frequency = {
      channel: (
          DEFAULT_SATURATION_FREQUENCY
          * basis.true_ec[channel]
          / float(probe_ec[i])
      )
      for i, channel in enumerate(channels)
  }

  cfg, sim, data, gt = build(saturation_frequency)

  ec, alpha = sim.ec_m.numpy(), sim.alpha_m.numpy()
  roi = np.asarray(gt['roi_m'])
  expected_roi = expected_true_roi(basis)
  for i, channel in enumerate(channels):
    assert abs(float(ec[i]) - basis.true_ec[channel]) < EC_TOL, (
        f'seed {seed}: {channel} ec_m is {float(ec[i]):.4f}, expected '
        f'{basis.true_ec[channel]} -- the ec_m solve missed, so this draw '
        'does not share the estimand with the others'
    )
    assert abs(float(alpha[i]) - basis.true_alpha[channel]) < ALPHA_TOL, (
        f'seed {seed}: {channel} alpha_m is {float(alpha[i]):.6f}, expected '
        f'{basis.true_alpha[channel]} -- adstock_retention_range is not pinned'
    )
    assert abs(float(roi[i]) - expected_roi[channel]) < ROI_TOL, (
        f'seed {seed} ({basis.name}): {channel} roi_m is {float(roi[i]):.6f}, '
        f'expected '
        f'{expected_roi[channel]:.6f}. True ROI is drifting across seeds -- '
        "see this module's docstring; the sweep is only a seed sweep while "
        'this holds.'
    )
  return cfg, sim, data, gt
