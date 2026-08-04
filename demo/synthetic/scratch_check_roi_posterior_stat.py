"""Diagnostic: does roi_m's posterior MEAN overstate its MEDIAN?

Follow-up to `scratch_ablation_exact_ec.py`, which reported roi_m error using
the posterior mean and found a persistent ~+15-25% bias even with ec_m fixed
to the exact truth. `roi_m`'s prior is `LogNormal(0.2, 0.9)` -- a right-skewed
shape. If the posterior inherits that skew, the mean will systematically sit
above the median/mode, and reporting mean vs. median could explain a
meaningful chunk of the apparent bias -- a reporting-statistic artifact, not
a real recovery failure. This is the cheap thing to rule in/out before
touching the DGP.

Fits ONE seed (1320) with the 'ec_alpha_only' variant (exact ec_m + informed
alpha_m) at the ORIGINAL, non-confounded high-ROI setting (target_roi
TV=8.0, Display=5.0) -- identical config to `scratch_ablation_exact_ec.py`,
so its recorded ec_m/roi_m mean-based errors (ec -0.5%, roi +15.2% for this
seed) are directly comparable here. Reports posterior mean, median, and skew
(mean - median) for both roi_m and ec_m.

Usage:
  .venv/bin/python demo/synthetic/scratch_check_roi_posterior_stat.py [seed]

Not part of the tracked pipeline -- a standalone scratch script, self-contained.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from meridian.model import model

from data_simulator import SimulationConfig
from model_utils import build_model_spec
import realistic_baseline as rb

MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
N_KNOTS = 8
TRUE_EC_TV = 9.0
TRUE_ALPHA_TV = 0.3

BASE_OVERRIDES = {
    'n_imp_channels': 2,
    'channel_names': ['TV', 'Display'],
    'target_audience_pop_frac': {'TV': 0.60, 'Display': 0.50},
    'current_reach_frac': {'TV': 0.10, 'Display': 0.5},
    'frequency_range': {'TV': (1, 2), 'Display': (1, 5)},
    'target_roi': {'TV': 8.0, 'Display': 5.0},  # original, non-confounded setting
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


def main() -> int:
  seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1320
  study_dir = HERE
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
  assert abs(float(sim.ec_m.numpy()[0]) - TRUE_EC_TV) < 0.05
  assert abs(float(sim.alpha_m.numpy()[0]) - TRUE_ALPHA_TV) < 1e-6

  true_roi = float(np.asarray(gt['roi_m'])[0])

  spec = build_model_spec('ec_alpha_only', sim, cfg, media_prior_type='roi',
                           knots=N_KNOTS, max_lag=None)
  mmm = model.Meridian(input_data=data, model_spec=spec)
  mmm.sample_prior(500)
  mmm.sample_posterior(**MCMC_KWARGS)
  post = mmm.inference_data.posterior

  def report(name, draws, true_val):
    mean = float(np.mean(draws))
    median = float(np.median(draws))
    skew = mean - median
    mean_err = (mean / true_val - 1) * 100
    median_err = (median / true_val - 1) * 100
    print(f'{name}: true={true_val:.4f}')
    print(f'  posterior mean   = {mean:.4f}  (err {mean_err:+.1f}%)')
    print(f'  posterior median = {median:.4f}  (err {median_err:+.1f}%)')
    print(f'  mean - median (skew) = {skew:+.4f}')
    print(f'  mean_err - median_err = {mean_err - median_err:+.1f} pts')
    print()

  roi_draws = post['roi_m'].values[..., 0].flatten()
  ec_draws = post['ec_m'].values[..., 0].flatten()
  alpha_draws = post['alpha_m'].values[..., 0].flatten()

  print(f'=== seed {seed}, target_roi TV=8.0/Display=5.0, ec_alpha_only variant ===\n')
  report('roi_m', roi_draws, true_roi)
  report('ec_m', ec_draws, TRUE_EC_TV)
  report('alpha_m', alpha_draws, TRUE_ALPHA_TV)
  return 0


if __name__ == '__main__':
  sys.exit(main())
