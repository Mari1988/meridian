"""Does the mROI-at-elevated-spend story hold up where raw roi_m didn't?

This study's landed notebook (`final/roi-vs-mroi-metric-selection.ipynb`)
established that roi_m at current spend is the LEAST discriminating metric --
default's overstatement and understated elasticity partly cancel there -- and
that marginal ROI (mROI) at elevated spend is the sharp discriminator instead.
Everything in this session's ablation thread (`scratch_ablation_exact_ec.py`,
`scratch_ablation_low_roi.py`, `scratch_ablation_low_noise.py`) has been
checking plain roi_m at current (1x) spend, where even the exact-ec_m informed
variant carries a persistent ~+8 to +25% bias. This checks whether that same
informed variant looks better on mROI at elevated spend, using the same
per-posterior-draw mROI methodology as `export_curve_data.py` (not posterior
means -- that script's own point is that means can't support a "confidently
wrong" claim, only draw-level credible intervals can).

Same 3 seeds as `scratch_ablation_low_noise.py` (1320, 7, 42), but at the
STANDARD realistic setting (oracle_r2=0.80, target_roi TV=8.0/Display=5.0,
alpha_m=0.3) -- i.e. bit-identical data to `scratch_ablation_exact_ec.py`, so
the roi_m-at-1x numbers already on record (seed 1320: informed +15.2%, seed 7:
+22.6%, seed 42: +11.1%) are the reference point "does mROI look better than
this."

Computes, per seed x variant (default, ec_alpha_only) x spend multiplier in
{1, 2, 3, 5}, for TV: posterior mean mROI, 90% credible interval, true mROI
(via Meridian's own AdstockTransformer/HillTransformer on the DGP truth, not a
hand-rolled formula), % error of the mean, and whether the truth falls inside
the credible interval (the "confidently wrong" check).

Usage:
  .venv/bin/python demo/synthetic/scratch_check_mroi_recovery.py <seed> <out_dir>
  .venv/bin/python demo/synthetic/scratch_check_mroi_recovery.py --report <out_dir>

Not part of the tracked pipeline -- a standalone scratch script, self-contained.
"""
from __future__ import annotations

import glob
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from meridian.analysis import analyzer
from meridian.model import adstock_hill
from meridian.model import model

from data_simulator import SimulationConfig
from model_utils import build_model_spec
from model_utils import MERIDIAN_DEFAULT_MAX_LAG
import realistic_baseline as rb

SEEDS = [1320, 7, 42]
TARGET_R2 = 0.80
N_KNOTS = 8
TRUE_EC_TV = 9.0
TRUE_ALPHA_TV = 0.3
VARIANTS = ['default', 'ec_alpha_only']
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
CONFIDENCE_LEVEL = 0.90
MROI_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0]

# Same treatment as export_curve_data.py: default gets Meridian's real
# out-of-the-box max_lag (8); the informed variant gets the DGP's own window
# (config.max_lag, 13), since max_lag is informed the same way ec_m/alpha_m
# are, not held fixed across variants.
MAX_LAG_BY_VARIANT = {'default': MERIDIAN_DEFAULT_MAX_LAG}

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
    'adstock_retention_range': {'TV': (TRUE_ALPHA_TV, TRUE_ALPHA_TV), 'Display': (0.0, 0.3)},
    'baseline_scale': 0.065,
}


def _real_df(study_dir: str) -> pd.DataFrame:
  cache = os.path.join(study_dir, '.cache_geo_media_rf.csv')
  if not os.path.exists(cache):
    pd.read_csv(
        'https://raw.githubusercontent.com/google/meridian/refs/heads/main/'
        'meridian/data/simulated_data/csv/geo_media_rf.csv'
    ).to_csv(cache, index=False)
  return pd.read_csv(cache)


def build_scenario(seed: int, study_dir: str):
  real_df = _real_df(study_dir)

  def build(saturation_frequency_tv=None, r2=TARGET_R2):
    overrides = dict(BASE_OVERRIDES, seed_num=seed)
    if saturation_frequency_tv is not None:
      overrides['saturation_frequency'] = {
          'TV': saturation_frequency_tv, 'Display': 4.0}
    cfg = SimulationConfig.from_dict(overrides)
    sim, data, gt = rb.build_real_augmented_realistic(
        cfg, real_df,
        rf_source_map={'TV': 'Channel3'},
        plain_source_map={'Display': 'Channel2'},
        target_oracle_r2=r2, n_knots=N_KNOTS,
    )
    return cfg, sim, data, gt

  _, sim_base, _, _ = build(r2=None)
  ec_base = float(sim_base.ec_m.numpy()[0])
  sat_freq = 4.0 * TRUE_EC_TV / ec_base

  cfg, sim, data, gt = build(sat_freq)
  assert abs(float(sim.ec_m.numpy()[0]) - TRUE_EC_TV) < 0.05, (
      f'seed {seed}: ec_m solve missed ({float(sim.ec_m.numpy()[0])})')
  assert abs(float(sim.alpha_m.numpy()[0]) - TRUE_ALPHA_TV) < 1e-6, (
      f'seed {seed}: TV alpha wrong')
  return cfg, sim, data, gt


def true_outcome_ratio(cfg, sim, channel_idx: int, multiplier: float) -> float:
  """Relative incremental outcome at `multiplier` x historical delivery.

  Routed through Meridian's own AdstockTransformer/HillTransformer using the
  DGP's true parameters -- mirrors export_curve_data.py exactly.
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


def true_mroi(cfg, sim, channel_idx: int, true_roi: float, multiplier: float) -> float:
  """True marginal ROI at `multiplier` x historical spend, truth-side formula.

  mROI(m) = ROI(1) * [R(1.01m) - R(m)] / (0.01 * m), matching
  export_curve_data.py's truth computation.
  """
  d = (true_outcome_ratio(cfg, sim, channel_idx, multiplier * 1.01)
       - true_outcome_ratio(cfg, sim, channel_idx, multiplier))
  return true_roi * d / (0.01 * multiplier)


def mroi_draws(az, multiplier: float, spend) -> np.ndarray:
  """Marginal ROI posterior draws at `multiplier` x historical spend.

  Mirrors export_curve_data.py's `mroi_draws` exactly -- per-draw, not
  posterior-mean, so credible intervals are meaningful.
  """
  base = az.incremental_outcome(
      use_posterior=True, scaling_factor0=0.0, scaling_factor1=multiplier,
      inverse_transform_outcome=True, include_non_paid_channels=False)
  bumped = az.incremental_outcome(
      use_posterior=True, scaling_factor0=0.0, scaling_factor1=multiplier * 1.01,
      inverse_transform_outcome=True, include_non_paid_channels=False)
  return (np.asarray(bumped) - np.asarray(base)) / (0.01 * multiplier * spend)


def _ci(draws: np.ndarray) -> tuple[float, float, float]:
  lo = (1 - CONFIDENCE_LEVEL) / 2
  return (float(np.mean(draws)), float(np.quantile(draws, lo)),
          float(np.quantile(draws, 1 - lo)))


def run_seed(seed: int, study_dir: str) -> pd.DataFrame:
  cfg, sim, data, gt = build_scenario(seed, study_dir)
  true_roi = float(np.asarray(gt['roi_m'])[0])  # TV is channel 0
  spend = sim.cost_gtm.numpy().sum(axis=(0, 1))

  rows = []
  for mult in MROI_MULTIPLIERS:
    rows.append({
        'seed': seed, 'variant': 'truth', 'spend_multiplier': mult,
        'mean': true_mroi(cfg, sim, 0, true_roi, mult),
        'ci_lo': np.nan, 'ci_hi': np.nan,
    })

  for variant in VARIANTS:
    spec = build_model_spec(variant, sim, cfg, media_prior_type='roi',
                             knots=N_KNOTS, max_lag=MAX_LAG_BY_VARIANT.get(variant))
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)
    az = analyzer.Analyzer(mmm)

    for mult in MROI_MULTIPLIERS:
      draws = mroi_draws(az, mult, spend)
      mean, lo, hi = _ci(draws[..., 0])  # TV is channel 0
      rows.append({
          'seed': seed, 'variant': variant, 'spend_multiplier': mult,
          'mean': mean, 'ci_lo': lo, 'ci_hi': hi,
      })
    del mmm, az

  df = pd.DataFrame(rows)
  truth = df[df.variant == 'truth'].set_index('spend_multiplier')['mean']
  df['true_mroi'] = df['spend_multiplier'].map(truth)
  df['pct_error'] = np.where(
      df.variant == 'truth', 0.0,
      (df['mean'] / df['true_mroi'] - 1) * 100)
  df['truth_in_ci'] = np.where(
      df.variant == 'truth', np.nan,
      (df['true_mroi'] >= df['ci_lo']) & (df['true_mroi'] <= df['ci_hi']))
  return df


def main_report(out_dir: str) -> None:
  files = sorted(glob.glob(os.path.join(out_dir, 'seed_*.csv')))
  if not files:
    print(f'no seed_*.csv files found in {out_dir}')
    return
  df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
  df.to_csv(os.path.join(out_dir, 'all_seeds.csv'), index=False)
  pd.set_option('display.width', 200)

  for mult in MROI_MULTIPLIERS:
    print(f'\n=== TV mROI @ {mult}x spend ===')
    sub = df[df.spend_multiplier == mult]
    piv = sub.pivot(index='seed', columns='variant', values='mean')
    err = sub[sub.variant != 'truth'].pivot(index='seed', columns='variant', values='pct_error')
    ci_in = sub[sub.variant != 'truth'].pivot(index='seed', columns='variant', values='truth_in_ci')
    print(piv.round(3))
    print('% error vs truth:')
    print(err.round(1))
    print('truth inside 90% CI:')
    print(ci_in)

  print('\n=== Summary: median |% error| across seeds, by multiplier ===')
  summary = (df[df.variant != 'truth']
             .groupby(['spend_multiplier', 'variant'])['pct_error']
             .agg(lambda s: s.abs().median())
             .unstack('variant'))
  print(summary.round(1))

  print('\n=== Summary: CI coverage rate (truth inside CI), by multiplier ===')
  cov = (df[df.variant != 'truth']
         .groupby(['spend_multiplier', 'variant'])['truth_in_ci']
         .mean().unstack('variant'))
  print(cov.round(2))


def main() -> int:
  if sys.argv[1:2] == ['--report']:
    main_report(sys.argv[2])
    return 0

  seed = int(sys.argv[1])
  out_dir = sys.argv[2]
  study_dir = HERE
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f'seed_{seed}.csv')

  if os.path.exists(out_path):
    print(f'seed {seed}: already done, skipping', flush=True)
    return 0

  t0 = time.time()
  df = run_seed(seed, study_dir)
  df.to_csv(out_path, index=False)
  print(f'seed {seed} done in {time.time() - t0:.0f}s', flush=True)
  tv = df.pivot(index='spend_multiplier', columns='variant', values='mean')
  print(tv.round(3))
  err = df[df.variant != 'truth'].pivot(index='spend_multiplier', columns='variant', values='pct_error')
  print('% error vs truth:')
  print(err.round(1))
  return 0


if __name__ == '__main__':
  sys.exit(main())
