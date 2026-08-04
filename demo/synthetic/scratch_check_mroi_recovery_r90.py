"""mROI-at-elevated-spend at an intermediate noise level (oracle_r2=0.9).

Interpolates between `scratch_check_mroi_recovery.py` (oracle_r2=0.80: default
collapses from +7.7% to -36.2% median error across 1x-5x spend, informed
stays flat ~14-15%) and `scratch_check_mroi_recovery_lownoise.py` (oracle_r2
target 0.99, achieved ~0.984-0.986: default -20.7% to -32.7% at 5x, informed
SHRINKS to 5.0% median with 90% CI coverage reaching 3/3 seeds at 5x). This
checks the middle point, oracle_r2=0.9, to see whether the improvement from
noise reduction is roughly linear in oracle_r2 or concentrated near the
near-noiseless end.

Same 3 seeds (1320, 7, 42), target_roi TV=8.0/Display=5.0, alpha_m=0.3.

Computes, per seed x variant (default, ec_alpha_noisy) x spend multiplier in
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
import r90_basis
import realistic_baseline as rb

SEEDS = r90_basis.SEEDS
TARGET_R2 = r90_basis.TARGET_ORACLE_R2
N_KNOTS = r90_basis.N_KNOTS
# The ACHIEVABLE informed prior: both anchors perturbed ~25% before use. See
# `model_utils.PRIOR_VARIANTS['ec_alpha_noisy']` -- centring on the exact
# truth would be an oracle no advertiser can supply.
VARIANTS = ['default', 'ec_alpha_noisy']
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
CONFIDENCE_LEVEL = 0.90
MROI_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0, 10.0]

# Same treatment as export_curve_data.py: default gets Meridian's real
# out-of-the-box max_lag (8); the informed variant gets the DGP's own window
# (config.max_lag, 13), since max_lag is informed the same way ec_m/alpha_m
# are, not held fixed across variants.
MAX_LAG_BY_VARIANT = {'default': MERIDIAN_DEFAULT_MAX_LAG}

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


def run_seed(seed: int, study_dir: str, out_dir: str) -> pd.DataFrame:
  cfg, sim, data, gt = r90_basis.build_scenario(seed)
  true_roi = float(np.asarray(gt['roi_m'])[0])  # TV is channel 0
  spend = sim.cost_gtm.numpy().sum(axis=(0, 1))

  models_dir = os.path.join(out_dir, 'models')
  os.makedirs(models_dir, exist_ok=True)

  rows = []
  for mult in MROI_MULTIPLIERS:
    rows.append({
        'seed': seed, 'variant': 'truth', 'spend_multiplier': mult,
        'mean': true_mroi(cfg, sim, 0, true_roi, mult),
        'ci_lo': np.nan, 'ci_hi': np.nan,
    })

  for variant in VARIANTS:
    spec = build_model_spec(variant, sim, cfg,
                             rng=np.random.default_rng(seed),
                             media_prior_type='roi',
                             knots=N_KNOTS,
                             max_lag=MAX_LAG_BY_VARIANT.get(variant))
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
    mmm.inference_data.to_netcdf(
        os.path.join(models_dir, f'seed_{seed}_{variant}_inference_data.nc'))
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
  df = run_seed(seed, study_dir, out_dir)
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
