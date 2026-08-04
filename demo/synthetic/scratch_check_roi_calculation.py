"""Verifies the roi_m calculation method used throughout this session's ablations.

Two things to check, prompted by a question about whether the earlier ROI
comparisons were computed correctly:

1. Does `mmm.inference_data.posterior['roi_m'].values[..., 0].mean()` (what
   every ablation script in this session used) match what
   `arviz.summary(mmm.inference_data, hdi_prob=0.9)` reports as the mean for
   `roi_m[0]`? These should be identical -- both are the posterior mean of the
   exact same MCMC draws -- but this confirms it rather than assuming it, and
   surfaces the 90% HDI, which the earlier ablations never reported (only the
   point estimate).

2. Does the raw `roi_m` POSTERIOR PARAMETER (the model's internal
   parameterization, used to derive beta_m from roi_m/ec_m/alpha_m/slope_m
   under `media_prior_type='roi'`) match `analyzer.Analyzer.roi()` (the
   analysis-level metric: incremental_outcome from zeroing the channel's
   spend, divided by total spend)? These are two different code paths in
   Meridian and are not guaranteed to agree exactly -- if they diverge
   meaningfully, the earlier ablations were reading the wrong "ROI" for what
   a practitioner would actually report.

Runs ONE seed (1320), ec_alpha_only variant, at the standard realistic
setting (oracle_r2=0.80, target_roi TV=8.0/Display=5.0, alpha_m=0.3) --
identical config to scratch_ablation_exact_ec.py, so its recorded number
(roi +15.2%) is the reference point.

Usage:
  .venv/bin/python demo/synthetic/scratch_check_roi_calculation.py [seed]
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import arviz as az
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from meridian.analysis import analyzer
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

  print(f'=== seed {seed}, ec_alpha_only, true roi_m (TV) = {true_roi:.4f} ===\n')

  # --- Check 1: raw-array .mean() vs. arviz.summary()'s reported mean ------
  post = mmm.inference_data.posterior
  raw_mean = float(post['roi_m'].values[..., 0].mean())
  raw_err = (raw_mean / true_roi - 1) * 100
  print(f'[1] What every ablation script computed:')
  print(f'    post["roi_m"].values[..., 0].mean() = {raw_mean:.4f}  (err {raw_err:+.1f}%)\n')

  summary = az.summary(mmm.inference_data, hdi_prob=0.9, var_names=['roi_m'])
  print('    arviz.summary(mmm.inference_data, hdi_prob=0.9, var_names=["roi_m"]):')
  print(summary.to_string())
  print()
  az_mean = float(summary.iloc[0]['mean'])
  az_hdi_lo = float(summary.iloc[0]['hdi_5%'])
  az_hdi_hi = float(summary.iloc[0]['hdi_95%'])
  print(f'    az.summary mean = {az_mean:.4f}   (matches raw .mean(): {np.isclose(raw_mean, az_mean, atol=1e-3)})')
  print(f'    90% HDI = [{az_hdi_lo:.4f}, {az_hdi_hi:.4f}]')
  print(f'    true value {true_roi:.4f} inside 90% HDI: {az_hdi_lo <= true_roi <= az_hdi_hi}\n')

  # --- Check 2: raw roi_m parameter vs. Analyzer.roi() (incremental_outcome/spend) ---
  azr = analyzer.Analyzer(mmm)
  roi_analysis = azr.roi(use_posterior=True).numpy()  # (chains, draws, channels)
  roi_analysis_tv = roi_analysis[..., 0]
  analysis_mean = float(roi_analysis_tv.mean())
  analysis_err = (analysis_mean / true_roi - 1) * 100
  print(f'[2] Analyzer.roi() (incremental_outcome / spend), TV channel:')
  print(f'    mean = {analysis_mean:.4f}  (err {analysis_err:+.1f}%)')
  print(f'    vs. raw roi_m parameter mean = {raw_mean:.4f}  (err {raw_err:+.1f}%)')
  print(f'    difference = {analysis_mean - raw_mean:+.4f}  ({analysis_err - raw_err:+.1f} pts)')
  hdi_lo_a = float(np.quantile(roi_analysis_tv, 0.05))
  hdi_hi_a = float(np.quantile(roi_analysis_tv, 0.95))
  print(f'    90% CI (quantile) = [{hdi_lo_a:.4f}, {hdi_hi_a:.4f}]')
  print(f'    true value inside Analyzer.roi() 90% CI: {hdi_lo_a <= true_roi <= hdi_hi_a}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
