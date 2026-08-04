"""Fits the two-channel realistic-baseline scenario and exports curve data.

Produces the inputs for the ARF deck's marginal-ROI and response-curve slides:
posterior response curves and marginal ROI at elevated spend, **with credible
intervals**, for `default` vs `ec_alpha_only` against the known DGP truth.

The credible intervals are the point. `final/roi-vs-mroi-metric-selection.ipynb`
computed its mROI sweep from posterior *means*, and flagged in its own Section 5
that the talk's framing -- "confidently wrong, not merely uncertain" -- is a claim
about intervals that posterior means cannot support. This script computes mROI
per posterior draw instead, so the `default`-vs-truth gap can be shown as
separated bands rather than differing lines.

Configuration is the settled one from `realistic-baseline-noise-2ch.ipynb`:
two channels, non-spline baseline with AR(1) + cross-geo-correlated noise
calibrated to oracle R^2 = 0.80, `baseline_scale=0.065` (which holds media at
~31% of KPI once Social is dropped), seed 1320, TV's true `ec_m` pinned at 9.0.

`default` fits at Meridian's real out-of-the-box `max_lag` (8); `ec_alpha_only`
(informed) fits at the DGP's own window (`config.max_lag`, 13) -- max_lag is
treated as informed the same way the `ec_m`/`alpha_m` priors are, not held
fixed across variants. See `MAX_LAG_BY_VARIANT` and `model_utils.
build_model_spec`'s `max_lag` argument.

Slow (two MCMC fits, ~5 minutes). Run once; the figure builder reads the CSVs.
Inference data is persisted alongside so the curves can be recomputed without
refitting.

Usage:
  .venv/bin/python demo/synthetic/export_curve_data.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meridian.analysis import analyzer
from meridian.model import adstock_hill
from meridian.model import model

from data_simulator import SimulationConfig
from model_utils import build_model_spec
from model_utils import MERIDIAN_DEFAULT_MAX_LAG
import realistic_baseline as rb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'fitted_models', 'realistic_baseline_2ch')

SEED = 1320
TARGET_R2 = 0.80
N_KNOTS = 8
TRUE_EC_TV = 9.0
VARIANTS = ['default', 'ec_alpha_only']
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
CONFIDENCE_LEVEL = 0.90

# `default` gets Meridian's real out-of-the-box max_lag (8) -- what a
# practitioner who never touches it actually gets. `ec_alpha_only` (the
# informed variant) already anchors ec_m/alpha_m on the true DGP values, so it
# also gets the wider window (`config.max_lag`, 13) an advertiser who knows
# the channel's carryover would choose -- max_lag is informed the same way
# alpha_m's prior is, not an independent variable.
MAX_LAG_BY_VARIANT = {'default': MERIDIAN_DEFAULT_MAX_LAG}

# Response curves want a smooth sweep; the mROI slide quotes round multiples.
CURVE_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
MROI_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0, 10.0]

BASE_OVERRIDES = {
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
    'adstock_retention_range': {'TV': (0.8, 0.8), 'Display': (0.0, 0.3)},
    'baseline_scale': 0.065,
    'seed_num': SEED,
}


def _real_df() -> pd.DataFrame:
  cache = os.path.join(HERE, '.cache_geo_media_rf.csv')
  if not os.path.exists(cache):
    pd.read_csv(
        'https://raw.githubusercontent.com/google/meridian/refs/heads/main/'
        'meridian/data/simulated_data/csv/geo_media_rf.csv'
    ).to_csv(cache, index=False)
  return pd.read_csv(cache)


def build_scenario():
  """The `ec9` scenario with TV's true `ec_m` solved to exactly 9.0."""
  real_df = _real_df()

  def build(saturation_frequency_tv=None, target_r2=TARGET_R2):
    overrides = dict(BASE_OVERRIDES)
    if saturation_frequency_tv is not None:
      overrides['saturation_frequency'] = {
          'TV': saturation_frequency_tv, 'Display': 4.0}
    cfg = SimulationConfig.from_dict(overrides)
    return rb.build_real_augmented_realistic(
        cfg, real_df,
        rf_source_map={'TV': 'Channel3'},
        plain_source_map={'Display': 'Channel2'},
        target_oracle_r2=target_r2, n_knots=N_KNOTS,
    )

  # `ec_m` is linear in `saturation_frequency`, so one uncalibrated build
  # calibrates the solve.
  sim_base, _, _ = build(target_r2=None)
  sat_freq = 4.0 * TRUE_EC_TV / float(sim_base.ec_m.numpy()[0])

  cfg = SimulationConfig.from_dict(
      dict(BASE_OVERRIDES, saturation_frequency={
          'TV': sat_freq, 'Display': 4.0}))
  sim, data, gt = rb.build_real_augmented_realistic(
      cfg, real_df,
      rf_source_map={'TV': 'Channel3'},
      plain_source_map={'Display': 'Channel2'},
      target_oracle_r2=TARGET_R2, n_knots=N_KNOTS,
  )
  assert abs(float(sim.ec_m.numpy()[0]) - TRUE_EC_TV) < 0.05, 'ec_m solve missed'
  assert abs(float(sim.alpha_m.numpy()[0]) - 0.8) < 1e-6, 'TV alpha != 0.8'
  return cfg, sim, data, gt


def true_outcome_ratio(cfg, sim, channel_idx: int, multiplier: float) -> float:
  """Relative incremental outcome at `multiplier` x historical delivery.

  Routed through Meridian's own `AdstockTransformer`/`HillTransformer` using the
  DGP's true parameters, so the truth series is generated by the same code the
  model fits -- not a hand-rolled Hill formula. Returned as a ratio to 1x, which
  is all that is needed: the arbitrary outcome scale cancels when it is anchored
  to the known true ROI.
  """
  x = sim.transformed_ipc_gtm.numpy()[:, :, channel_idx:channel_idx + 1]
  ec = float(sim.ec_m.numpy()[channel_idx])
  alpha = float(sim.alpha_m.numpy()[channel_idx])
  beta_g = sim.beta_gm.numpy()[:, channel_idx]

  def outcome(mult: float) -> float:
    m = tf.constant(x * mult, dtype=tf.float32)
    adstocked = adstock_hill.AdstockTransformer(
        alpha=tf.constant([alpha], tf.float32),
        max_lag=cfg.max_lag, n_times_output=cfg.n_times).forward(m)
    hilled = adstock_hill.HillTransformer(
        ec=tf.constant([ec], tf.float32),
        slope=tf.constant([1.0], tf.float32)).forward(adstocked).numpy()[:, :, 0]
    return float((hilled * beta_g[:, None]).sum())

  return outcome(multiplier) / outcome(1.0)


def fit(data, sim, cfg, variant):
  spec = build_model_spec(variant, sim, cfg, media_prior_type='roi',
                          knots=N_KNOTS,
                          max_lag=MAX_LAG_BY_VARIANT.get(variant))
  mmm = model.Meridian(input_data=data, model_spec=spec)
  mmm.sample_prior(500)
  mmm.sample_posterior(**MCMC_KWARGS)
  return mmm


def mroi_draws(az, multiplier: float, spend) -> np.ndarray:
  """Marginal ROI posterior draws at `multiplier` x historical spend.

  Mirrors how `Analyzer.response_curves` scales spend internally -- via
  `incremental_outcome(scaling_factor1=...)` -- but keeps the draws instead of
  collapsing them, so the result carries a credible interval. A 1% bump matches
  `marginal_roi`'s own `incremental_increase` default.
  """
  base = az.incremental_outcome(
      use_posterior=True, scaling_factor0=0.0, scaling_factor1=multiplier,
      inverse_transform_outcome=True, include_non_paid_channels=False)
  bumped = az.incremental_outcome(
      use_posterior=True, scaling_factor0=0.0, scaling_factor1=multiplier * 1.01,
      inverse_transform_outcome=True, include_non_paid_channels=False)
  return (np.asarray(bumped) - np.asarray(base)) / (0.01 * multiplier * spend)


def _ci(draws: np.ndarray, axis=(0, 1)) -> tuple[float, float, float]:
  lo = (1 - CONFIDENCE_LEVEL) / 2
  return (float(np.mean(draws, axis=axis)),
          float(np.quantile(draws, lo, axis=axis)),
          float(np.quantile(draws, 1 - lo, axis=axis)))


def main() -> None:
  os.makedirs(OUT_DIR, exist_ok=True)
  t0 = time.time()

  cfg, sim, data, gt = build_scenario()
  channels = list(cfg.channel_names)
  true_roi = np.asarray(gt['roi_m'])
  # Total historical spend per channel, the denominator for both metrics.
  spend = sim.cost_gtm.numpy().sum(axis=(0, 1))
  print(f'scenario built | true ec_m {np.round(sim.ec_m.numpy(), 3)} '
        f'| true roi_m {np.round(true_roi, 3)}', flush=True)

  curve_rows, mroi_rows = [], []

  # --- truth ------------------------------------------------------------
  for i, ch in enumerate(channels):
    for mult in CURVE_MULTIPLIERS:
      ratio = true_outcome_ratio(cfg, sim, i, mult)
      value = true_roi[i] * spend[i] * ratio  # R(m) = ROI(1) * S * R(m)/R(1)
      curve_rows.append({'variant': 'truth', 'channel': ch,
                         'spend_multiplier': mult, 'mean': value,
                         'ci_lo': value, 'ci_hi': value})
    for mult in MROI_MULTIPLIERS:
      # mROI(m) = ROI(1) * [R(1.01m) - R(m)] / (0.01 * m * R(1)).
      d = (true_outcome_ratio(cfg, sim, i, mult * 1.01)
           - true_outcome_ratio(cfg, sim, i, mult))
      value = true_roi[i] * d / (0.01 * mult)
      mroi_rows.append({'variant': 'truth', 'channel': ch,
                        'spend_multiplier': mult, 'mean': value,
                        'ci_lo': value, 'ci_hi': value})

  # --- fitted variants --------------------------------------------------
  for variant in VARIANTS:
    t = time.time()
    mmm = fit(data, sim, cfg, variant)
    mmm.inference_data.to_netcdf(
        os.path.join(OUT_DIR, f'{variant}_inference_data.nc'))
    az = analyzer.Analyzer(mmm)
    print(f'{variant}: fitted in {time.time() - t:.0f}s', flush=True)

    curves = az.response_curves(spend_multipliers=CURVE_MULTIPLIERS,
                                confidence_level=CONFIDENCE_LEVEL)
    inc = curves.incremental_outcome
    for i, ch in enumerate(channels):
      for mult in CURVE_MULTIPLIERS:
        sel = inc.sel(channel=ch, spend_multiplier=mult)
        curve_rows.append({
            'variant': variant, 'channel': ch, 'spend_multiplier': mult,
            'mean': float(sel.sel(metric='mean')),
            'ci_lo': float(sel.sel(metric='ci_lo')),
            'ci_hi': float(sel.sel(metric='ci_hi')),
        })

    for mult in MROI_MULTIPLIERS:
      draws = mroi_draws(az, mult, spend)
      for i, ch in enumerate(channels):
        mean, lo, hi = _ci(draws[..., i])
        mroi_rows.append({'variant': variant, 'channel': ch,
                          'spend_multiplier': mult, 'mean': mean,
                          'ci_lo': lo, 'ci_hi': hi})
    print(f'{variant}: curves + mROI exported', flush=True)

  pd.DataFrame(curve_rows).to_csv(
      os.path.join(OUT_DIR, 'response_curves.csv'), index=False)
  mroi = pd.DataFrame(mroi_rows)
  mroi.to_csv(os.path.join(OUT_DIR, 'mroi_sweep.csv'), index=False)

  tv = mroi.query('channel == "TV"').pivot(
      index='spend_multiplier', columns='variant', values='mean')
  print(f'\nTV marginal ROI by spend multiple:\n{tv.round(3)}')
  print(f'\ndone in {time.time() - t0:.0f}s -> {OUT_DIR}')


if __name__ == '__main__':
  main()
