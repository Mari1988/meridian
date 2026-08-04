"""HDI coverage check: does the 90% credible interval actually contain truth?

Follow-up to `scratch_check_roi_calculation.py`, which confirmed for one seed
that (a) the posterior-mean calculation used throughout this session's
ablations matches both `arviz.summary()` and `Analyzer.roi()` exactly, and
(b) the true roi_m fell OUTSIDE the 90% HDI for that seed -- i.e. the ~+15%
bias isn't posterior noise the model is appropriately uncertain about, it's a
confident miss. Every ablation script in this session (`scratch_ablation_
exact_ec.py` etc.) only ever reported the point-estimate % error; this
extends that with `arviz.summary(hdi_prob=0.9)` coverage across all 10 seeds
and both variants, for all three parameters (ec_m, alpha_m, roi_m).

Same 10 seeds and standard config as `scratch_ablation_exact_ec.py`
(oracle_r2=0.80, target_roi TV=8.0/Display=5.0, alpha_m=0.3) -- bit-identical
data, so this is a direct extension of that run, not a new scenario. Fits
both 'default' and 'ec_alpha_only' (exact ec_m + informed alpha_m) per seed.

Usage:
  # one seed:
  .venv/bin/python demo/synthetic/scratch_check_hdi_coverage.py <seed> <out_dir>

  # all ten, from repo root:
  OUT=demo/synthetic/fitted_models/scratch_check_hdi_coverage
  mkdir -p "$OUT"
  for seed in 1320 7 42 101 555 2024 8 99 12345 31337; do
    .venv/bin/python demo/synthetic/scratch_check_hdi_coverage.py "$seed" "$OUT"
  done

  # then assemble + report:
  .venv/bin/python demo/synthetic/scratch_check_hdi_coverage.py --report "$OUT"

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

import arviz as az
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from meridian.model import model

from data_simulator import SimulationConfig
from model_utils import build_model_spec
import realistic_baseline as rb

SEEDS = [1320, 7, 42, 101, 555, 2024, 8, 99, 12345, 31337]
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
N_KNOTS = 8
TRUE_EC_TV = 9.0
TRUE_ALPHA_TV = 0.3
VARIANTS = ['default', 'ec_alpha_only']
HDI_PROB = 0.9

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


def run_seed(seed: int, study_dir: str) -> pd.DataFrame:
  real_df = _real_df(study_dir)

  def build(saturation_frequency_tv=None, r2=0.80):
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

  true_roi = float(np.asarray(gt['roi_m'])[0])
  truth = {'ec_m': TRUE_EC_TV, 'alpha_m': TRUE_ALPHA_TV, 'roi_m': true_roi}

  rows = []
  for variant in VARIANTS:
    spec = build_model_spec(variant, sim, cfg, media_prior_type='roi',
                             knots=N_KNOTS, max_lag=None)
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)

    summary = az.summary(mmm.inference_data, hdi_prob=HDI_PROB,
                          var_names=['ec_m', 'alpha_m', 'roi_m'])
    for param in ('ec_m', 'alpha_m', 'roi_m'):
      row_label = f'{param}[TV]'
      s = summary.loc[row_label]
      true_val = truth[param]
      mean = float(s['mean'])
      hdi_lo = float(s['hdi_5%'])
      hdi_hi = float(s['hdi_95%'])
      rows.append({
          'seed': seed, 'variant': variant, 'param': param,
          'true_value': true_val, 'mean': mean,
          'pct_error': (mean / true_val - 1) * 100,
          'hdi_lo': hdi_lo, 'hdi_hi': hdi_hi,
          'hdi_width': hdi_hi - hdi_lo,
          'truth_in_hdi': bool(hdi_lo <= true_val <= hdi_hi),
      })
    del mmm

  return pd.DataFrame(rows)


def main_report(out_dir: str) -> None:
  files = sorted(glob.glob(os.path.join(out_dir, 'seed_*.csv')))
  if not files:
    print(f'no seed_*.csv files found in {out_dir}')
    return
  df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
  df.to_csv(os.path.join(out_dir, 'all_seeds.csv'), index=False)
  pd.set_option('display.width', 200)

  print(f'{df["seed"].nunique()}/{len(SEEDS)} seeds completed\n')

  for param in ('ec_m', 'alpha_m', 'roi_m'):
    print(f'=== {param} ===')
    sub = df[df.param == param]
    piv_err = sub.pivot(index='seed', columns='variant', values='pct_error')
    piv_cov = sub.pivot(index='seed', columns='variant', values='truth_in_hdi')
    print('% error (posterior mean vs. truth):')
    print(piv_err.round(1))
    print('truth inside 90% HDI:')
    print(piv_cov)
    print()

  print('=== Summary: 90% HDI coverage rate across all seeds ===')
  cov_summary = (df.groupby(['param', 'variant'])['truth_in_hdi']
                 .mean().unstack('variant'))
  print(cov_summary.round(2))
  print()
  print('=== Summary: median |% error| across all seeds ===')
  err_summary = (df.groupby(['param', 'variant'])['pct_error']
                 .agg(lambda s: s.abs().median()).unstack('variant'))
  print(err_summary.round(1))
  print()
  print('=== Summary: median HDI width across all seeds ===')
  width_summary = (df.groupby(['param', 'variant'])['hdi_width']
                   .median().unstack('variant'))
  print(width_summary.round(3))


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
  print(df[['variant', 'param', 'true_value', 'mean', 'pct_error',
            'hdi_lo', 'hdi_hi', 'truth_in_hdi']].to_string(index=False))
  return 0


if __name__ == '__main__':
  sys.exit(main())
