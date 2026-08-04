"""Informed prior with an EXACT `ec_m` anchor: what survives a perfect anchor?

`ec_alpha_noisy` -- the deck's informed arm -- moves two things at once. It
perturbs the anchor ~25% AND states 25% uncertainty, so its per-dataset spread
(-53% to +43% on Channel-1's `ec_m`) cannot be attributed to either. This arm
sets the anchor error to exactly zero and changes nothing else:

    ec_m    ~ LogNormal(log(true ec_m), 0.26926)      # sqrt(0.10^2 + 0.25^2)
    alpha_m ~ Uniform(0, 0.5)                          # both channels, a BOUND

The `ec_m` scale is byte-identical to `ec_alpha_noisy`'s, so the one and only
difference from the deck's informed arm is where the prior is centred. Whatever
spread remains is not anchor error.

TWO THINGS THIS IS NOT
----------------------
1. NOT an achievable prior, and NOT a headline result. An exact anchor is an
   oracle -- the same objection that retired `ec_alpha_only` from the deck.
   Read this as a DECOMPOSITION of `ec_alpha_noisy`, never as evidence that an
   informed prior recovers `ec_m`.
2. NOT an `alpha_m` anchor. `build_alpha_range_prior` never reads
   `sim.alpha_m`; `Uniform(0, 0.5)` is a plausible bound a practitioner could
   assert without knowing the answer. So the `alpha_m` panel here measures
   "bounded support vs Meridian's unbounded `Uniform(0, 1)`", which is a
   different claim from the `ec_m` panels and must be quoted as such.

WHY THE BOUND IS 0.5 AND NOT 0.3
--------------------------------
Channel-1's true `alpha_m` is pinned at 0.30. A `Uniform(0, 0.3)` would put
its upper boundary exactly ON the truth: the posterior mean in the existing
well-specified run lands at or above 0.30 in 6 of 10 seeds (median 0.316), so
the prior would assign zero density where the posterior actually sits, censor
it, and force a strictly negative bias in every seed. That bias would be an
artifact of the bound, not a property of the estimator. `(0, 0.5)` leaves both
truths interior (0.30, 0.15), is still half of Meridian's default support, and
its mean (0.25) is neither channel's truth.

BOTH CHANNELS, ONE PASS
-----------------------
Unlike `scratch_ablation_r90.py` / `_wellspec.py`, which record `[..., 0]` and
need `scratch_extract_r90_wellspec_ch2.py` plus
`scratch_mroi_both_channels_r90.py` afterwards (~20 min of scenario rebuilds),
this records recovery AND mROI for every channel inside the fitting loop, off
the same posteriors. No post-pass, and the two figures cannot disagree.

Usage:
  .venv/bin/python demo/synthetic/scratch_ablation_r90_centred.py \
      --realism=wellspec --all
  .venv/bin/python demo/synthetic/scratch_ablation_r90_centred.py \
      --realism=wellspec --report

`--realism` is REQUIRED: the two baselines are different runs with different
conclusions, and defaulting one of them would be how the wrong arm gets quoted.
Each lands in its own directory.

Not part of the tracked pipeline -- a standalone scratch script.
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

from meridian.analysis import analyzer
from meridian.model import model

from model_utils import build_model_spec
from model_utils import MERIDIAN_DEFAULT_MAX_LAG
import r90_basis
import realistic_baseline as rb
import scratch_check_mroi_recovery_r90 as mroi_mod

INFORMED_DEFAULT = 'ec_centred25_alpha_u05'

# Same sampler as every other r90 arm, so the runs differ in the prior and the
# baseline and in nothing else.
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
N_KNOTS = r90_basis.N_KNOTS
MAX_LAG_BY_VARIANT = {'default': MERIDIAN_DEFAULT_MAX_LAG}
MROI_MULTIPLIERS = mroi_mod.MROI_MULTIPLIERS
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# The well-specified baseline, identical to `scratch_ablation_r90_wellspec`'s:
# `mu_t` becomes a linear trend the 8-knot spline spans exactly, and
# `calibrate_to_oracle_r2` scales the geo-idiosyncratic residual up to hold
# oracle R^2 at 0.9. Total noise is preserved; only its character moves from
# partly-national to entirely per-geo.
REALISM = {
    'wellspec': rb.RealisticBaselineConfig(mu_ar1_sd=0.0,
                                           seasonal_amplitude=0.0),
    'realistic': None,
}


def _anchor(prior, param: str) -> float:
  """Channel-1's prior centre for `param`, in the parameter's OWN units.

  `ec_m`'s prior is LogNormal, so its `loc` is on the LOG scale; `alpha_m`'s
  is Uniform here, which has no `loc` at all. Both cases are handled so this
  can be called blindly on either arm.
  """
  dist = getattr(prior, param, None)
  loc = getattr(dist, 'loc', None)
  if loc is None:
    return np.nan
  value = float(np.asarray(loc).flat[0])
  if type(dist).__name__.startswith('LogNormal'):
    value = float(np.exp(value))  # the median, comparable to the truth
  return round(value, 4)


def run_seed(seed: int, out_dir: str, realism_key: str,
             informed_variant: str) -> dict:
  """Fits both arms on one draw and records BOTH channels from each."""
  cfg, sim, data, gt = r90_basis.build_scenario(
      seed, realism=REALISM[realism_key])
  basis = r90_basis.DEFAULT_BASIS

  diag = rb.baseline_diagnostics(sim, N_KNOTS)
  # Measured off the rebuilt scenario rather than copied from `r90_basis`'s
  # constants: `build_scenario` already asserts they agree, so measuring keeps
  # the error columns from being self-consistent by construction.
  ec_m, alpha_m = sim.ec_m.numpy(), sim.alpha_m.numpy()
  roi_m = np.asarray(gt['roi_m'])
  channels = list(cfg.channel_names)
  spend = sim.cost_gtm.numpy().sum(axis=(0, 1))

  row = {
      'seed': seed,
      'basis': basis.name,
      'realism': realism_key,
      'informed_variant': informed_variant,
      'target_oracle_r2': r90_basis.TARGET_ORACLE_R2,
      'achieved_oracle_r2': round(diag['oracle_r2'], 4),
      'noise_scale': round(diag['noise_scale'], 5),
      'media_share_pct': round(diag['media_share_pct'], 1),
      'true_alpha_m': round(float(alpha_m[0]), 4),
      'true_roi_m': round(float(roi_m[0]), 3),
  }
  row_ch2 = {'seed': seed, 'basis': basis.name, 'realism': realism_key}
  row_truths = [
      {
          'seed': seed,
          'basis': basis.name,
          'channel': channel,
          'true_ec_m': float(ec_m[i]),
          'true_roi_m': float(roi_m[i]),
          'true_alpha_m': float(alpha_m[i]),
      }
      for i, channel in enumerate(channels)
  ]

  mroi_rows = [
      {
          'seed': seed,
          'channel': channel,
          'variant': 'truth',
          'spend_multiplier': mult,
          'mean': mroi_mod.true_mroi(cfg, sim, i, float(roi_m[i]), mult),
      }
      for i, channel in enumerate(channels)
      for mult in MROI_MULTIPLIERS
  ]

  models_dir = os.path.join(out_dir, 'models')
  os.makedirs(models_dir, exist_ok=True)

  def record(variant: str, prefix: str) -> None:
    """Fits one arm once; recovery and mROI both come off that same fit."""
    t0 = time.time()
    spec = build_model_spec(
        variant, sim, cfg,
        rng=np.random.default_rng(seed),
        media_prior_type='roi',
        knots=N_KNOTS,
        max_lag=MAX_LAG_BY_VARIANT.get(variant))
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)

    for param in ('ec_m', 'alpha_m'):
      row[f'{prefix}_{param}_anchor'] = _anchor(mmm.model_spec.prior, param)

    # THE POINT OF THIS ARM: the informed `ec_m` anchor must sit exactly on
    # the truth. If a future edit reintroduces a perturbation, this run stops
    # being a decomposition of `ec_alpha_noisy` and silently becomes a second
    # copy of it. Relative tolerance -- these are float32 values.
    if prefix == 'informed':
      anchor = row[f'{prefix}_ec_m_anchor']
      assert abs(anchor / float(ec_m[0]) - 1) < 1e-3, (
          f'seed {seed}: informed ec_m anchor is {anchor}, expected the truth '
          f'{float(ec_m[0]):.4f} -- the anchor is being perturbed, so this is '
          'no longer the exact-anchor arm')

    post = mmm.inference_data.posterior
    for target, idx in ((row, 0), (row_ch2, 1)):
      for param, key, true_vec in (
          ('ec_m', 'ec', ec_m),
          ('roi_m', 'roi', roi_m),
          ('alpha_m', 'alpha', alpha_m),
      ):
        fitted = float(post[param].values[..., idx].mean())
        truth = float(true_vec[idx])
        target[f'{prefix}_{key}_err'] = round((fitted / truth - 1) * 100, 1)
        # `alpha_m` is a retention rate and Channel-2's truth is small, so
        # carry percentage POINTS too -- a ratio against 0.15 overstates.
        if param == 'alpha_m':
          target[f'{prefix}_alpha_pp'] = round((fitted - truth) * 100, 2)

    az_ = analyzer.Analyzer(mmm)
    for mult in MROI_MULTIPLIERS:
      draws = np.asarray(mroi_mod.mroi_draws(az_, mult, spend))
      for i, channel in enumerate(channels):
        d = draws[..., i].flatten()
        truth = mroi_mod.true_mroi(cfg, sim, i, float(roi_m[i]), mult)
        entry = {
            'seed': seed,
            'channel': channel,
            'variant': variant,
            'spend_multiplier': mult,
            'mean': float(d.mean()),
            'true_mroi': truth,
            'pct_error': (float(d.mean()) / truth - 1) * 100,
        }
        for q in QUANTILES:
          entry[f'q{int(q * 100):02d}'] = float(np.quantile(d, q))
        # The 90% interval is q05/q95 -- quoted rather than the 50% band the
        # figure draws, since a narrower band excludes the truth more readily.
        entry['truth_in_ci'] = float(entry['q05'] <= truth <= entry['q95'])
        mroi_rows.append(entry)

    print(f'  seed {seed} {prefix:<9} '
          f'ec {row[f"{prefix}_ec_err"]:+7.1f}%  '
          f'roi {row[f"{prefix}_roi_err"]:+7.1f}%  '
          f'alpha {row[f"{prefix}_alpha_err"]:+7.1f}%  '
          f'| ch2 ec {row_ch2[f"{prefix}_ec_err"]:+7.1f}%  '
          f'({time.time() - t0:.0f}s)', flush=True)

    mmm.inference_data.to_netcdf(
        os.path.join(models_dir, f'seed_{seed}_{variant}_inference_data.nc'))
    del mmm, az_

  record('default', 'default')
  record(informed_variant, 'informed')

  os.makedirs(out_dir, exist_ok=True)
  pd.DataFrame([row]).to_csv(
      os.path.join(out_dir, f'per_seed_{seed}.csv'), index=False)
  # Named `ch2_per_seed_*` rather than `per_seed_*_ch2`: the report globs
  # `per_seed_*.csv`, which the latter would match and silently pool the two
  # channels into one table.
  pd.DataFrame([row_ch2]).to_csv(
      os.path.join(out_dir, f'ch2_per_seed_{seed}.csv'), index=False)
  pd.DataFrame(mroi_rows).to_csv(
      os.path.join(out_dir, f'mroi_{seed}.csv'), index=False)
  pd.DataFrame(row_truths).to_csv(
      os.path.join(out_dir, f'true_params_{seed}.csv'), index=False)
  return row


def report(out_dir: str) -> None:
  """Assembles the aggregates the figure scripts read, and checks the pins."""
  files = sorted(
      f for f in glob.glob(os.path.join(out_dir, 'per_seed_*.csv'))
      if not os.path.basename(f).startswith('per_seed_all'))
  if not files:
    print(f'no per_seed_*.csv in {out_dir}')
    return
  df = (pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        .sort_values('seed').reset_index(drop=True))
  df.to_csv(os.path.join(out_dir, 'per_seed_all.csv'), index=False)

  ch2_files = sorted(glob.glob(os.path.join(out_dir, 'ch2_per_seed_*.csv')))
  ch2 = (pd.concat([pd.read_csv(f) for f in ch2_files], ignore_index=True)
         .sort_values('seed').reset_index(drop=True))
  ch2.to_csv(os.path.join(out_dir, 'per_seed_all_ch2.csv'), index=False)

  mroi = pd.concat(
      [pd.read_csv(os.path.join(out_dir, f'mroi_{s}.csv')) for s in df['seed']],
      ignore_index=True)
  mroi.to_csv(os.path.join(out_dir, 'mroi_both_channels.csv'), index=False)

  truths = pd.concat(
      [pd.read_csv(os.path.join(out_dir, f'true_params_{s}.csv'))
       for s in df['seed']], ignore_index=True).sort_values(['channel', 'seed'])
  truths.to_csv(os.path.join(out_dir, 'true_params_by_seed.csv'), index=False)

  # Relative tolerance, not exact equality: float32 values recomputed per draw
  # differ in the last bit or two even when perfectly pinned (float32 eps
  # ~1.2e-7). The drift this basis was built to remove was ~5e-1.
  PIN_RTOL = 1e-5
  drift = {
      f'{channel}.{col}': rel
      for channel, group in truths.groupby('channel')
      for col in ('true_ec_m', 'true_roi_m', 'true_alpha_m')
      if (rel := (group[col].max() - group[col].min())
          / abs(group[col].mean())) > PIN_RTOL
  }
  assert not drift, (
      f'a truth varies across seeds beyond float noise: {drift}\n'
      'See r90_basis.py -- this is not a clean seed sweep.')

  informed = str(df['informed_variant'].iloc[0])
  print(f'\n{len(df)} seeds | realism={df["realism"].iloc[0]} | '
        f'informed={informed}')
  print(f'truths pinned across all seeds; only the noise realization varies\n')
  print(truths.groupby('channel').first().drop(columns='seed').to_string())

  anchors = df['informed_ec_m_anchor']
  print(f'\ninformed ec_m anchor (Channel-1): {anchors.min():.4f} to '
        f'{anchors.max():.4f} -- exact, unperturbed\n')

  for name, table in (('CHANNEL-1', df), ('CHANNEL-2', ch2)):
    print(f'--- {name} ---')
    for metric in ('ec_err', 'roi_err', 'alpha_err'):
      d, i = f'default_{metric}', f'informed_{metric}'
      wins = int((table[i].abs() < table[d].abs()).sum())
      print(f'  {metric:<10} default median {table[d].median():+7.1f}%  '
            f'[{table[d].min():+.1f}, {table[d].max():+.1f}]   '
            f'informed median {table[i].median():+7.1f}%  '
            f'[{table[i].min():+.1f}, {table[i].max():+.1f}]   '
            f'informed wins {wins}/{len(table)}')
    for arm in ('default', 'informed'):
      print(f'  alpha_m {arm:<9} median {table[f"{arm}_alpha_pp"].median():+.2f}'
            ' percentage points')
  print()

  cov = (mroi[mroi.variant != 'truth']
         .groupby(['channel', 'variant', 'spend_multiplier'])
         .agg(median_pct_err=('pct_error', 'median'),
              coverage=('truth_in_ci', 'sum')))
  print('mROI: median % error and 90% interval coverage (of 10 seeds)')
  print(cov.round(1).to_string())


def main() -> int:
  realism_key = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--realism=')),
      None)
  if realism_key not in REALISM:
    print(f'--realism= is required; choose from {sorted(REALISM)}')
    return 2
  informed_variant = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--informed=')),
      INFORMED_DEFAULT)
  out_dir = os.path.join(
      HERE, 'fitted_models', f'scratch_ablation_r90_centred_{realism_key}')

  if '--report' in sys.argv:
    report(out_dir)
    return 0

  seeds = r90_basis.SEEDS
  for arg in sys.argv[1:]:
    if arg.startswith('--seeds='):
      seeds = [int(s) for s in arg.split('=', 1)[1].split(',')]

  os.makedirs(out_dir, exist_ok=True)
  print(f'exact-anchor ablation | realism={realism_key} | '
        f'informed={informed_variant} | {len(seeds)} seeds: {seeds}')
  t0 = time.time()
  for seed in seeds:
    if os.path.exists(os.path.join(out_dir, f'per_seed_{seed}.csv')):
      print(f'seed {seed}: already done, skipping', flush=True)
      continue
    run_seed(seed, out_dir, realism_key, informed_variant)
  print(f'\ntotal {time.time() - t0:.0f}s')
  report(out_dir)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
