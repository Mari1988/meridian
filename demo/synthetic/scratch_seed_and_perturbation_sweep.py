"""Two-factor sweep: DGP-seed variation x ec_m-prior perturbation variation.

For each of 10 DGP seeds (matching the study's standard SEEDS list):
  - Build the DGP/data once for that seed. TV true ec_m=9.0, alpha_m=0.3
    (continuing the alpha=0.3 investigation from this study's other scratch
    scripts -- change TRUE_ALPHA_TV below to revisit alpha=0.8 instead).
  - Fit 'default' once against that seed's data.
  - Fit an informed hybrid variant once against the SAME data: ec_m prior is
    `build_reach_based_ec_prior(..., audience_noise_scale=0.25)` (an
    advertiser's imperfect audience/reach-based estimate), alpha_m prior is
    `build_alpha_prior` (informed, same as `ec_alpha_only`). The random ec_m
    perturbation is constrained to +/-25% via rejection sampling: redraw with
    a different (still reproducible) sub-seed until it lands in that range,
    rather than accepting whatever LogNormal(0, 0.25) happens to produce
    (which occasionally exceeds 60%+, as seen in the single-seed check this
    sweep follows up on).

This isolates the paired question: "per dataset, does a *plausible-magnitude*
(<=25%) ec_m misspecification still leave the informed variant ahead of
default?" -- across 10 independent datasets, not just one.

Runs one seed per subprocess (matching seed_sweep_worker.py's memory-safety
pattern -- many MCMC fits in one Python process risks memory exhaustion).
Writes one CSV row per seed; safe to re-run (skips seeds already done).

Usage:
  # one seed:
  .venv/bin/python demo/synthetic/scratch_seed_and_perturbation_sweep.py <seed> <out_dir>

  # all ten, from repo root:
  OUT=demo/synthetic/fitted_models/scratch_seed_and_perturbation_sweep
  mkdir -p "$OUT"
  for seed in 1320 7 42 101 555 2024 8 99 12345 31337; do
    echo "=== seed $seed ==="
    .venv/bin/python demo/synthetic/scratch_seed_and_perturbation_sweep.py "$seed" "$OUT"
  done
  # then assemble + report:
  .venv/bin/python demo/synthetic/scratch_seed_and_perturbation_sweep.py --report "$OUT"

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
from meridian.model import prior_distribution

from data_simulator import SimulationConfig
import model_utils
from model_utils import build_alpha_prior
from model_utils import build_model_spec
from model_utils import build_reach_based_ec_prior
from model_utils import MERIDIAN_DEFAULT_MAX_LAG
import realistic_baseline as rb

model_utils.PRIOR_VARIANTS['ec_noisy_and_alpha'] = (
    lambda sim, config, rng=None: prior_distribution.PriorDistribution(
        ec_m=build_reach_based_ec_prior(sim, config, audience_noise_scale=0.25, rng=rng),
        alpha_m=build_alpha_prior(sim, config),
    )
)

SEEDS = [1320, 7, 42, 101, 555, 2024, 8, 99, 12345, 31337]
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
N_KNOTS = 8
TRUE_EC_TV = 9.0
TRUE_ALPHA_TV = 0.3  # set to 0.8 to instead match the main published config
MAX_PERTURB_PCT = 25.0
MAX_REJECTION_TRIES = 200
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


def _bounded_perturbation_rng(seed: int) -> tuple[np.random.Generator, float, int]:
  """Rejection-samples a sub-seed whose ec_m perturbation is within +/-25%.

  Returns (rng, assumed_off_pct, n_tries). `rng` is a fresh generator at the
  accepted sub-seed -- passing it to `build_reach_based_ec_prior` reproduces
  exactly the perturbation this function computed, since it's the first (and
  only) draw taken from that generator.
  """
  for attempt in range(MAX_REJECTION_TRIES):
    subseed = seed * 1000 + attempt  # deterministic, distinct per (seed, attempt)
    candidate = np.random.default_rng(subseed).lognormal(mean=0.0, sigma=0.25)
    off_pct = (candidate - 1.0) * 100
    if abs(off_pct) <= MAX_PERTURB_PCT:
      return np.random.default_rng(subseed), off_pct, attempt + 1
  raise RuntimeError(f'seed {seed}: no perturbation within +/-{MAX_PERTURB_PCT}% '
                     f'found in {MAX_REJECTION_TRIES} tries')


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
  assert diag['media_share_pct'] > 20.0, (
      f"seed {seed}: media share {diag['media_share_pct']:.1f}% is weak-signal territory")

  true_roi = float(np.asarray(gt['roi_m'])[0])
  row = {
      'seed': seed,
      'oracle_r2': round(diag['oracle_r2'], 4),
      'media_share_pct': round(diag['media_share_pct'], 1),
      'clipped_frac_pct': round(diag['clipped_frac_pct'], 2),
      'true_roi_m': round(true_roi, 3),
  }

  def fit(variant, rng=None):
    spec = build_model_spec(variant, sim, cfg, rng=rng, media_prior_type='roi',
                            knots=N_KNOTS, max_lag=MAX_LAG_BY_VARIANT.get(variant))
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)
    return mmm

  # --- default: one fit, no perturbation involved. ---
  mmm = fit('default')
  post = mmm.inference_data.posterior
  row['default_ec_err'] = round((float(post['ec_m'].values[..., 0].mean()) / TRUE_EC_TV - 1) * 100, 1)
  row['default_roi_err'] = round((float(post['roi_m'].values[..., 0].mean()) / true_roi - 1) * 100, 1)
  row['default_alpha_err'] = round((float(post['alpha_m'].values[..., 0].mean()) / TRUE_ALPHA_TV - 1) * 100, 1)
  del mmm

  # --- informed: one fit, ec_m prior perturbed but bounded to +/-25%. ---
  rng, assumed_off_pct, n_tries = _bounded_perturbation_rng(seed)
  row['assumed_off_pct'] = round(assumed_off_pct, 1)
  row['perturbation_tries'] = n_tries
  mmm = fit('ec_noisy_and_alpha', rng=rng)
  post = mmm.inference_data.posterior
  row['informed_ec_err'] = round((float(post['ec_m'].values[..., 0].mean()) / TRUE_EC_TV - 1) * 100, 1)
  row['informed_roi_err'] = round((float(post['roi_m'].values[..., 0].mean()) / true_roi - 1) * 100, 1)
  row['informed_alpha_err'] = round((float(post['alpha_m'].values[..., 0].mean()) / TRUE_ALPHA_TV - 1) * 100, 1)
  del mmm

  return row


def main_report(out_dir: str) -> None:
  files = sorted(glob.glob(os.path.join(out_dir, 'per_seed_*.csv')))
  if not files:
    print(f'no per_seed_*.csv files found in {out_dir}')
    return
  df = (pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        .sort_values('seed').reset_index(drop=True))
  df.to_csv(os.path.join(out_dir, 'per_seed_all.csv'), index=False)
  pd.set_option('display.width', 160)
  print(f'{len(df)}/{len(SEEDS)} seeds completed\n')
  print(df.to_string(index=False))
  print()
  for metric in ('ec_err', 'roi_err', 'alpha_err'):
    d, i = df[f'default_{metric}'], df[f'informed_{metric}']
    wins = int((i.abs() < d.abs()).sum())
    print(f'{metric:10s} default medianAbs={d.abs().median():6.1f}  '
          f'informed medianAbs={i.abs().median():6.1f}  '
          f'informed wins {wins}/{len(df)}')


def main() -> int:
  if sys.argv[1:2] == ['--report']:
    main_report(sys.argv[2])
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
      f"seed {seed} done in {time.time() - t0:.0f}s | "
      f"default ec {row['default_ec_err']:+.1f}% roi {row['default_roi_err']:+.1f}% "
      f"alpha {row['default_alpha_err']:+.1f}% | "
      f"informed (perturbed {row['assumed_off_pct']:+.1f}%, "
      f"{row['perturbation_tries']} tries) ec {row['informed_ec_err']:+.1f}% "
      f"roi {row['informed_roi_err']:+.1f}% alpha {row['informed_alpha_err']:+.1f}%",
      flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
