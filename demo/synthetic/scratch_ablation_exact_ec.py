"""Ablation: does an *exact* ec_m prior (no perturbation) erase roi_m bias?

Follow-up to `scratch_seed_and_perturbation_sweep.py`'s open question: that
script's 'informed' variant used a bounded +/-25% ec_m perturbation (a
stylized imperfect-advertiser-estimate), which confounds two possible sources
of the remaining roi_m bias it showed (median +18.5% across 10 seeds):

  (a) residual ec_m error propagating into roi_m through the ec_m<->roi_m
      coupling under `media_prior_type='roi'` -- `beta_m` is *derived* from
      `roi_m`, `ec_m`, `alpha_m` (see `roi_m`'s field docstring in
      `meridian/model/prior_distribution.py`), so any leftover ec_m error
      still gets compensated by a biased roi_m to keep fitting the data, and
  (b) DGP noise / baseline misspecification (oracle R^2 = 0.80, not 1.0;
      persistent, cross-geo-correlated residual) aliasing with media and
      getting misattributed regardless of ec_m accuracy.

This isolates (a) by fitting `PRIOR_VARIANTS['ec_alpha_only']` -- the *exact*
true ec_m (`audience_noise_scale=0.0`, `build_reach_based_ec_prior`'s default)
plus the same informed alpha_m prior -- on the SAME 10 seeds' data as the
original sweep (identical `BASE_OVERRIDES`/`build()` logic, so seed N
reproduces bit-identical data to that script's seed N run). It does not
refit 'default' or the perturbed-informed variant; those already exist in
`scratch_seed_and_perturbation_sweep.py`'s `per_seed_all.csv`. Only the new
"exact ec_m" fit runs here.

If exact ec_m's roi_m error collapses toward 0%, (a) is the dominant driver.
Whatever bias remains under exact ec_m is attributable to (b) plus any
residual ec_m<->roi_m identifiability slack the sampler exploits even with a
tight/exact prior.

Runs one seed per subprocess (same memory-safety pattern as
`scratch_seed_and_perturbation_sweep.py`: many MCMC fits in one process risks
memory exhaustion). Writes one CSV row per seed; safe to re-run (skips seeds
already done).

Usage:
  # one seed:
  .venv/bin/python demo/synthetic/scratch_ablation_exact_ec.py <seed> <out_dir>

  # all ten, from repo root:
  OUT=demo/synthetic/fitted_models/scratch_ablation_exact_ec
  mkdir -p "$OUT"
  for seed in 1320 7 42 101 555 2024 8 99 12345 31337; do
    echo "=== seed $seed ==="
    .venv/bin/python demo/synthetic/scratch_ablation_exact_ec.py "$seed" "$OUT"
  done

  # then assemble + report (joins against the original sweep's per_seed_all.csv
  # for the default / perturbed-informed / exact-informed 3-way comparison):
  .venv/bin/python demo/synthetic/scratch_ablation_exact_ec.py --report "$OUT" \
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
from model_utils import MERIDIAN_DEFAULT_MAX_LAG
import realistic_baseline as rb

SEEDS = [1320, 7, 42, 101, 555, 2024, 8, 99, 12345, 31337]
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
N_KNOTS = 8
TRUE_EC_TV = 9.0
TRUE_ALPHA_TV = 0.3  # must match scratch_seed_and_perturbation_sweep.py

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

  # Identical DGP-reconstruction path to scratch_seed_and_perturbation_sweep.py
  # so seed N reproduces bit-identical data to that script's seed N run.
  _, sim_base, _, _ = build(r2=None)
  ec_base = float(sim_base.ec_m.numpy()[0])
  sat_freq = 4.0 * TRUE_EC_TV / ec_base

  cfg, sim, data, gt = build(sat_freq)
  assert abs(float(sim.ec_m.numpy()[0]) - TRUE_EC_TV) < 0.05, (
      f'seed {seed}: ec_m solve missed ({float(sim.ec_m.numpy()[0])})')
  assert abs(float(sim.alpha_m.numpy()[0]) - TRUE_ALPHA_TV) < 1e-6, (
      f'seed {seed}: TV alpha wrong')

  true_roi = float(np.asarray(gt['roi_m'])[0])
  row = {'seed': seed, 'true_roi_m': round(true_roi, 3)}

  # 'ec_alpha_only': exact true ec_m (audience_noise_scale=0.0, the default)
  # + informed alpha_m -- same max_lag treatment as the perturbed-informed
  # variant (not in MAX_LAG_BY_VARIANT, so build_model_spec falls back to
  # config.max_lag, the DGP's own window).
  spec = build_model_spec(
      'ec_alpha_only', sim, cfg, media_prior_type='roi',
      knots=N_KNOTS, max_lag=None)
  mmm = model.Meridian(input_data=data, model_spec=spec)
  mmm.sample_prior(500)
  mmm.sample_posterior(**MCMC_KWARGS)
  post = mmm.inference_data.posterior
  row['exact_ec_err'] = round((float(post['ec_m'].values[..., 0].mean()) / TRUE_EC_TV - 1) * 100, 1)
  row['exact_roi_err'] = round((float(post['roi_m'].values[..., 0].mean()) / true_roi - 1) * 100, 1)
  row['exact_alpha_err'] = round((float(post['alpha_m'].values[..., 0].mean()) / TRUE_ALPHA_TV - 1) * 100, 1)
  del mmm

  return row


def main_report(out_dir: str, orig_sweep_dir: str | None) -> None:
  files = sorted(glob.glob(os.path.join(out_dir, 'per_seed_*.csv')))
  if not files:
    print(f'no per_seed_*.csv files found in {out_dir}')
    return
  df = (pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        .sort_values('seed').reset_index(drop=True))
  df.to_csv(os.path.join(out_dir, 'per_seed_all.csv'), index=False)

  if orig_sweep_dir:
    orig_path = os.path.join(orig_sweep_dir, 'per_seed_all.csv')
    if os.path.exists(orig_path):
      orig = pd.read_csv(orig_path)[
          ['seed', 'default_roi_err', 'default_ec_err', 'default_alpha_err',
           'informed_roi_err', 'informed_ec_err', 'informed_alpha_err',
           'assumed_off_pct']
      ].rename(columns={
          'informed_roi_err': 'perturbed_informed_roi_err',
          'informed_ec_err': 'perturbed_informed_ec_err',
          'informed_alpha_err': 'perturbed_informed_alpha_err',
      })
      df = df.merge(orig, on='seed', how='left')

  pd.set_option('display.width', 200)
  print(f'{len(df)}/{len(SEEDS)} seeds completed\n')
  cols = ['seed', 'assumed_off_pct', 'default_roi_err',
          'perturbed_informed_roi_err', 'exact_roi_err',
          'default_ec_err', 'perturbed_informed_ec_err', 'exact_ec_err']
  cols = [c for c in cols if c in df.columns]
  print(df[cols].to_string(index=False))
  print()

  for metric, variants in (
      ('roi_err', ['default_roi_err', 'perturbed_informed_roi_err', 'exact_roi_err']),
      ('ec_err', ['default_ec_err', 'perturbed_informed_ec_err', 'exact_ec_err']),
      ('alpha_err', ['default_alpha_err', 'perturbed_informed_alpha_err', 'exact_alpha_err']),
  ):
    present = [v for v in variants if v in df.columns]
    if not present:
      continue
    print(f'--- {metric} ---')
    for v in present:
      d = df[v]
      print(f'  {v:28s} medianAbs={d.abs().median():6.1f}  median={d.median():+6.1f}  '
            f'range=[{d.min():+.1f}, {d.max():+.1f}]')
    print()


def main() -> int:
  if sys.argv[1:2] == ['--report']:
    out_dir = sys.argv[2]
    orig_sweep_dir = sys.argv[3] if len(sys.argv) > 3 else None
    main_report(out_dir, orig_sweep_dir)
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
      f"exact ec_m: ec {row['exact_ec_err']:+.1f}% roi {row['exact_roi_err']:+.1f}% "
      f"alpha {row['exact_alpha_err']:+.1f}%",
      flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
