"""Ablation: does the persistent roi_m bias scale with the ROI level?

Follow-up to `scratch_ablation_exact_ec.py`, which showed that forcing ec_m to
the exact truth barely changes roi_m's error (~+21% median either way) --
falsifying the hypothesis that residual ec_m error, propagated through the
ec_m<->roi_m coupling under `media_prior_type='roi'`, was the main driver.
That leaves DGP noise / baseline misspecification (oracle R^2 = 0.80, not
1.0) as the leading candidate, established by elimination rather than direct
proof.

This script asks a different, orthogonal question: is the ~20-25% bias a
fixed PERCENTAGE regardless of the true ROI level, or does it scale/shrink at
a lower ROI? Same 10 seeds, same DGP shape, but `target_roi` lowered from
{'TV': 8.0, 'Display': 5.0} to {'TV': 3.0, 'Display': 2.0} (a similar
TV:Display ratio, ~1.5-1.6x, so this mainly rescales the absolute ROI level
rather than changing the channels' relative economics). Fits both 'default'
and the exact-ec_m informed variant ('ec_alpha_only') per seed, mirroring
scratch_seed_and_perturbation_sweep.py + scratch_ablation_exact_ec.py's
combined design so results are directly comparable to those two runs'
per_seed_all.csv.

Runs one seed per subprocess (same memory-safety pattern as the other
scratch scripts in this study). Writes one CSV row per seed; safe to re-run.

Usage:
  # one seed:
  .venv/bin/python demo/synthetic/scratch_ablation_low_roi.py <seed> <out_dir>

  # all ten, from repo root:
  OUT=demo/synthetic/fitted_models/scratch_ablation_low_roi
  mkdir -p "$OUT"
  for seed in 1320 7 42 101 555 2024 8 99 12345 31337; do
    echo "=== seed $seed ==="
    .venv/bin/python demo/synthetic/scratch_ablation_low_roi.py "$seed" "$OUT"
  done

  # then assemble + report (also joins the original target_roi=8/5 run for a
  # side-by-side default / exact-informed comparison across ROI levels):
  .venv/bin/python demo/synthetic/scratch_ablation_low_roi.py --report "$OUT" \
      demo/synthetic/fitted_models/scratch_ablation_exact_ec \
      demo/synthetic/fitted_models/scratch_seed_and_perturbation_sweep

Not part of the tracked pipeline -- a standalone scratch script, self-contained
(no dependency on any other conversation/session state).
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
TRUE_ALPHA_TV = 0.3  # must match the other scratch scripts in this study

BASE_OVERRIDES = {
    'n_imp_channels': 2,
    'channel_names': ['TV', 'Display'],
    'target_audience_pop_frac': {'TV': 0.60, 'Display': 0.50},
    'current_reach_frac': {'TV': 0.10, 'Display': 0.5},
    'frequency_range': {'TV': (1, 2), 'Display': (1, 5)},
    'target_roi': {'TV': 3.0, 'Display': 2.0},  # lowered from 8.0 / 5.0
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


def run_seed(seed: int, target_r2: float, study_dir: str) -> dict:
  real_df = _real_df(study_dir)

  def build(saturation_frequency_tv=None, r2=target_r2):
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

  diag = rb.baseline_diagnostics(sim, N_KNOTS)
  true_roi = float(np.asarray(gt['roi_m'])[0])
  row = {
      'seed': seed,
      'oracle_r2': round(diag['oracle_r2'], 4),
      'media_share_pct': round(diag['media_share_pct'], 1),
      'true_roi_m': round(true_roi, 3),
  }

  def fit(variant):
    spec = build_model_spec(variant, sim, cfg, media_prior_type='roi',
                             knots=N_KNOTS, max_lag=None)
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)
    return mmm

  mmm = fit('default')
  post = mmm.inference_data.posterior
  row['default_ec_err'] = round((float(post['ec_m'].values[..., 0].mean()) / TRUE_EC_TV - 1) * 100, 1)
  row['default_roi_err'] = round((float(post['roi_m'].values[..., 0].mean()) / true_roi - 1) * 100, 1)
  row['default_alpha_err'] = round((float(post['alpha_m'].values[..., 0].mean()) / TRUE_ALPHA_TV - 1) * 100, 1)
  del mmm

  mmm = fit('ec_alpha_only')  # exact ec_m + informed alpha_m
  post = mmm.inference_data.posterior
  row['exact_ec_err'] = round((float(post['ec_m'].values[..., 0].mean()) / TRUE_EC_TV - 1) * 100, 1)
  row['exact_roi_err'] = round((float(post['roi_m'].values[..., 0].mean()) / true_roi - 1) * 100, 1)
  row['exact_alpha_err'] = round((float(post['alpha_m'].values[..., 0].mean()) / TRUE_ALPHA_TV - 1) * 100, 1)
  del mmm

  return row


def main_report(out_dir: str, hi_roi_ablation_dir: str | None, hi_roi_sweep_dir: str | None) -> None:
  files = sorted(glob.glob(os.path.join(out_dir, 'per_seed_*.csv')))
  if not files:
    print(f'no per_seed_*.csv files found in {out_dir}')
    return
  df = (pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        .sort_values('seed').reset_index(drop=True))
  df.to_csv(os.path.join(out_dir, 'per_seed_all.csv'), index=False)

  pd.set_option('display.width', 200)
  print(f'{len(df)}/{len(SEEDS)} seeds completed -- target_roi TV=3.0, Display=2.0\n')
  print(df[['seed', 'true_roi_m', 'media_share_pct', 'default_roi_err', 'exact_roi_err',
            'default_ec_err', 'exact_ec_err']].to_string(index=False))
  print()
  for metric in ('roi_err', 'ec_err', 'alpha_err'):
    d, e = df[f'default_{metric}'], df[f'exact_{metric}']
    print(f'{metric:10s} default medianAbs={d.abs().median():6.1f} median={d.median():+6.1f}  '
          f'exact-ec medianAbs={e.abs().median():6.1f} median={e.median():+6.1f}')

  if hi_roi_ablation_dir:
    hi_path = os.path.join(hi_roi_ablation_dir, 'per_seed_all.csv')
    if os.path.exists(hi_path):
      hi = pd.read_csv(hi_path)[['seed', 'exact_roi_err', 'exact_ec_err', 'exact_alpha_err']].rename(
          columns={'exact_roi_err': 'hi_roi_exact_roi_err',
                   'exact_ec_err': 'hi_roi_exact_ec_err',
                   'exact_alpha_err': 'hi_roi_exact_alpha_err'})
      merged = df.merge(hi, on='seed', how='left')
      if hi_roi_sweep_dir:
        sweep_path = os.path.join(hi_roi_sweep_dir, 'per_seed_all.csv')
        if os.path.exists(sweep_path):
          sweep = pd.read_csv(sweep_path)[['seed', 'default_roi_err', 'true_roi_m']].rename(
              columns={'default_roi_err': 'hi_roi_default_roi_err',
                       'true_roi_m': 'hi_roi_true_roi_m'})
          merged = merged.merge(sweep, on='seed', how='left')
      print('\n--- ROI-level comparison: target_roi 3/2 (this run) vs 8/5 (original) ---')
      cols = [c for c in ['seed', 'true_roi_m', 'exact_roi_err', 'hi_roi_true_roi_m',
                          'hi_roi_exact_roi_err', 'hi_roi_default_roi_err', 'default_roi_err']
              if c in merged.columns]
      print(merged[cols].to_string(index=False))
      print()
      print(f"low-roi (3/2)  exact-ec roi_err: medianAbs={merged['exact_roi_err'].abs().median():.1f}  median={merged['exact_roi_err'].median():+.1f}")
      if 'hi_roi_exact_roi_err' in merged.columns:
        print(f"hi-roi  (8/5)  exact-ec roi_err: medianAbs={merged['hi_roi_exact_roi_err'].abs().median():.1f}  median={merged['hi_roi_exact_roi_err'].median():+.1f}")
      print(f"low-roi (3/2)  default   roi_err: medianAbs={merged['default_roi_err'].abs().median():.1f}  median={merged['default_roi_err'].median():+.1f}")
      if 'hi_roi_default_roi_err' in merged.columns:
        print(f"hi-roi  (8/5)  default   roi_err: medianAbs={merged['hi_roi_default_roi_err'].abs().median():.1f}  median={merged['hi_roi_default_roi_err'].median():+.1f}")


def main() -> int:
  if sys.argv[1:2] == ['--report']:
    out_dir = sys.argv[2]
    hi_roi_ablation_dir = sys.argv[3] if len(sys.argv) > 3 else None
    hi_roi_sweep_dir = sys.argv[4] if len(sys.argv) > 4 else None
    main_report(out_dir, hi_roi_ablation_dir, hi_roi_sweep_dir)
    return 0

  seed = int(sys.argv[1])
  out_dir = sys.argv[2]
  target_r2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
  study_dir = HERE
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f'per_seed_{seed}.csv')

  if os.path.exists(out_path):
    print(f'seed {seed}: already done, skipping', flush=True)
    return 0

  t0 = time.time()
  row = run_seed(seed, target_r2, study_dir)
  pd.DataFrame([row]).to_csv(out_path, index=False)
  print(
      f"seed {seed} done in {time.time() - t0:.0f}s | true_roi={row['true_roi_m']:.2f} | "
      f"default ec {row['default_ec_err']:+.1f}% roi {row['default_roi_err']:+.1f}% | "
      f"exact-ec ec {row['exact_ec_err']:+.1f}% roi {row['exact_roi_err']:+.1f}%",
      flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
