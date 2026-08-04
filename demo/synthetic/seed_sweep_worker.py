"""One simulator draw of the two-channel realistic-baseline sweep, per process.

Why a worker script rather than a loop inside the notebook: forty Meridian fits
in a single kernel exhausts memory and the kernel dies (TensorFlow retains graph
state across `sample_posterior` calls -- the repeated `tf.function` retracing
warnings are the visible symptom). Two fits per process is the size already
known to run cleanly, so each seed gets its own process and writes one CSV row.

Usage:
    python seed_sweep_worker.py <seed> <out_dir> [target_r2]

Writes `<out_dir>/per_seed_<seed>.csv`. Exits non-zero on failure, leaving no
partial file, so a driver can retry or skip that draw without losing the others.
Re-running with an existing output file is a no-op, which makes the sweep
restartable.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meridian.model import model

from data_simulator import SimulationConfig
from model_utils import build_model_spec
from model_utils import MERIDIAN_DEFAULT_MAX_LAG
import realistic_baseline as rb

VARIANTS = ['default', 'ec_alpha_only']
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
N_KNOTS = 8
TRUE_EC_TV = 9.0

# `default` gets Meridian's real out-of-the-box max_lag (8); `ec_alpha_only`
# (informed) gets the DGP's own wider window (`config.max_lag`, 13) -- treated
# as informed the same way its ec_m/alpha_m priors are. See
# `export_curve_data.py`'s `MAX_LAG_BY_VARIANT` for the same split.
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
    'adstock_retention_range': {'TV': (0.8, 0.8), 'Display': (0.0, 0.3)},
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
  """One build and two fits for one draw. Returns one summary row."""
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

  # `ec_m` is linear in `saturation_frequency`, so one uncalibrated build per
  # seed pins TV's true `ec_m` to exactly 9.0 *for this draw*.
  _, sim_base, _, _ = build(r2=None)
  ec_base = float(sim_base.ec_m.numpy()[0])
  sat_freq = 4.0 * TRUE_EC_TV / ec_base

  cfg, sim, data, gt = build(sat_freq)
  assert abs(float(sim.ec_m.numpy()[0]) - TRUE_EC_TV) < 0.05, (
      f'ec_m solve missed ({float(sim.ec_m.numpy()[0])})')
  assert abs(float(sim.alpha_m.numpy()[0]) - 0.8) < 1e-6, 'TV alpha != 0.8'

  diag = rb.baseline_diagnostics(sim, N_KNOTS)
  # Media share varies with the draw (`baseline_scale` is fixed at 0.065, the
  # rest is the seed). That variation is part of what the sweep is measuring,
  # so it is recorded rather than rejected -- an earlier two-sided 28-34% band
  # threw away seeds for being *above* it, which would have biased the sample
  # toward lower-signal draws. Only a genuinely weak-signal draw is fatal: the
  # outline records ~13.7% as the level that degrades both prior variants, so
  # anything near it says nothing about priors and must not enter the range.
  assert diag['media_share_pct'] > 20.0, (
      f"media share {diag['media_share_pct']:.1f}% is weak-signal territory; "
      'this draw cannot speak to prior quality')

  row = {
      'seed': seed,
      'oracle_r2': round(diag['oracle_r2'], 4),
      'media_share_pct': round(diag['media_share_pct'], 1),
      'clipped_frac_pct': round(diag['clipped_frac_pct'], 2),
      'true_roi_m': round(float(np.asarray(gt['roi_m'])[0]), 3),
  }

  for variant in VARIANTS:
    spec = build_model_spec(variant, sim, cfg, media_prior_type='roi',
                            knots=N_KNOTS,
                            max_lag=MAX_LAG_BY_VARIANT.get(variant))
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)
    post = mmm.inference_data.posterior
    ec = float(post['ec_m'].values[..., 0].mean())
    roi = float(post['roi_m'].values[..., 0].mean())
    alpha = float(post['alpha_m'].values[..., 0].mean())
    true_roi = float(np.asarray(gt['roi_m'])[0])
    row[f'{variant}_ec_m'] = round(ec, 3)
    row[f'{variant}_ec_err'] = round((ec / TRUE_EC_TV - 1) * 100, 1)
    row[f'{variant}_roi_m'] = round(roi, 3)
    row[f'{variant}_roi_err'] = round((roi / true_roi - 1) * 100, 1)
    row[f'{variant}_alpha_err'] = round((alpha / 0.8 - 1) * 100, 1)
    del mmm

  return row


def main() -> int:
  seed = int(sys.argv[1])
  out_dir = sys.argv[2]
  target_r2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
  study_dir = os.path.dirname(os.path.abspath(__file__))
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f'per_seed_{seed}.csv')

  if os.path.exists(out_path):
    print(f'seed {seed}: already done, skipping', flush=True)
    return 0

  t0 = time.time()
  row = run_seed(seed, target_r2, study_dir)
  # Write only on success, so a failed draw leaves no partial file to mistake
  # for a result.
  pd.DataFrame([row]).to_csv(out_path, index=False)
  print(
      f"seed {seed} done in {time.time() - t0:.0f}s | "
      f"default ec {row['default_ec_err']:+.1f}% roi {row['default_roi_err']:+.1f}% "
      f"alpha {row['default_alpha_err']:+.1f}% | "
      f"informed ec {row['ec_alpha_only_ec_err']:+.1f}% "
      f"roi {row['ec_alpha_only_roi_err']:+.1f}% "
      f"alpha {row['ec_alpha_only_alpha_err']:+.1f}%",
      flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
