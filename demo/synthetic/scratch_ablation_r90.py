"""ec_m/roi_m/alpha_m point recovery at oracle_r2=0.9 -- the deck's final basis.

This session settled on alpha_m=0.3, oracle_r2=0.9, 3 seeds (1320, 7, 42) as
the basis for the deck (confirmed after comparing 0.80, 0.9, and 0.99 via the
mROI-at-elevated-spend checks -- 0.9 is a deliberate middle ground, not the
near-noiseless 0.99 extreme). `scratch_ablation_low_noise.py` computed this
exact ec_m/roi_m/alpha_m point-recovery comparison at oracle_r2 target 0.99;
this is the same comparison at 0.9, for the recovery box plot replacing
slide 7's table (`scratch_plot_recovery_boxplot_r99.py` -- to be re-pointed
at this script's output).

Fits both 'default' and 'ec_alpha_only' (exact ec_m + informed alpha_m) per
seed, target_roi TV=8.0/Display=5.0 (the original, non-confounded setting --
scratch_ablation_low_roi.py showed lowering target_roi also lowers
media_share_pct, a confound this avoids).

Usage:
  # one seed:
  .venv/bin/python demo/synthetic/scratch_ablation_r90.py <seed> <out_dir> [oracle_r2] [--basis=NAME]

  # 3 seeds, from repo root:
  OUT=demo/synthetic/fitted_models/scratch_ablation_r90
  mkdir -p "$OUT"
  for seed in 1320 7 42; do
    .venv/bin/python demo/synthetic/scratch_ablation_r90.py "$seed" "$OUT"
  done

  # then assemble + report:
  .venv/bin/python demo/synthetic/scratch_ablation_r90.py --report "$OUT"

`--basis=NAME` selects one of `r90_basis.BASES` (default `alpha03`, the
deck's). `--basis=alpha08` reruns this identical comparison with Channel-1's
true `alpha_m` at 0.8 instead of 0.3, which is what gives the `max_lag` split
below something to measure. ALWAYS give a non-default basis its own
`out_dir` -- the per-seed CSVs are keyed by seed alone, so two bases sharing a
directory would silently interleave. The basis name is recorded in every row
so a mixed directory is at least detectable after the fact.

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from meridian.analysis import analyzer
from meridian.model import model

from data_simulator import SimulationConfig
from model_utils import build_model_spec
from model_utils import MERIDIAN_DEFAULT_MAX_LAG
import r90_basis
import scratch_check_mroi_recovery_r90 as mroi_mod
import realistic_baseline as rb

MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
# The DGP itself lives in `r90_basis` -- constants, the two-pass ec_m solve,
# and the assertions that keep every seed on the same estimand. Do not
# reintroduce a local BASE_OVERRIDES here; a divergent second copy is what
# let true ROI drift across seeds unnoticed.
N_KNOTS = r90_basis.N_KNOTS
DEFAULT_ORACLE_R2 = r90_basis.TARGET_ORACLE_R2

# `max_lag` is informed the same way `ec_m`/`alpha_m` are, not held fixed
# across variants: `default` fits at Meridian's real out-of-the-box window
# (8), the informed variant at the DGP's own (`config.max_lag`, 13). This
# script previously passed `max_lag=None` for BOTH, handing the default the
# true window -- a favour to it, and an inconsistency with the mROI sweep,
# which has always used the split. Adstock recovery tracks `max_lag` rather
# than the prior, so this materially changes the default's `alpha_m` column.
MAX_LAG_BY_VARIANT = {'default': MERIDIAN_DEFAULT_MAX_LAG}

# The mROI sweep is computed from the SAME two fits rather than by a second
# script re-fitting identical models -- which is only sound now that the
# `max_lag` treatment above matches it.
MROI_MULTIPLIERS = mroi_mod.MROI_MULTIPLIERS

def _anchor(prior, param: str) -> float:
  """Channel-1's prior centre for `param`, in the parameter's OWN units.

  Recorded per seed because `build_alpha_prior` CLIPS its perturbed centre
  into `[1e-4, 0.99]`. At a true `alpha_m` of 0.3 that ceiling is
  unreachable; at 0.8 a ~25% upward perturbation hits it, so the anchor error
  stops being symmetric and any anchor-vs-error reading has to use the anchor
  actually handed to the sampler rather than the nominal perturbation.

  Two traps this exists to avoid:
    * `ec_m`'s informed prior is LogNormal, so its `loc` is on the LOG scale
      (2.158 means a median of 8.65, not 2.158) while `alpha_m`'s
      TruncatedNormal `loc` is already in natural units. Writing both raw
      into one CSV would silently mix scales.
    * Meridian's own defaults are scalar, not per-channel (`ec_m` is
      `TruncatedNormal(0.8, 0.8)` for every channel at once), so `loc[0]`
      raises on a 0-d array; `alpha_m`'s default `Uniform(0, 1)` has no
      `loc` at all.
  """
  dist = getattr(prior, param, None)
  loc = getattr(dist, 'loc', None)
  if loc is None:
    return np.nan
  value = float(np.asarray(loc).flat[0])
  if type(dist).__name__.startswith('LogNormal'):
    value = float(np.exp(value))  # the median, comparable to the truth
  return round(value, 4)


def run_seed(
    seed: int,
    target_r2: float,
    study_dir: str,
    out_dir: str,
    basis: r90_basis.Basis = r90_basis.DEFAULT_BASIS,
) -> dict:
  # Scenario construction lives in `r90_basis` so every r90 script draws from
  # the same pinned truths; see that module's docstring for why. It asserts
  # ec_m/alpha_m/roi_m land on their targets for BOTH channels, which is what
  # makes this a seed sweep rather than a sweep over two things at once.
  cfg, sim, data, gt = r90_basis.build_scenario(seed, target_r2, basis=basis)
  row_truths: list[dict] = []

  diag = rb.baseline_diagnostics(sim, N_KNOTS)
  # MEASURED off the rebuilt scenario, not copied from `r90_basis`'s
  # constants: `build_scenario` already asserts they agree, so recording the
  # measured values keeps the error columns below from being self-consistent
  # by construction if the DGP ever drifts.
  ec_m, alpha_m = sim.ec_m.numpy(), sim.alpha_m.numpy()
  roi_m = np.asarray(gt['roi_m'])
  true_ec, true_alpha, true_roi = (
      float(ec_m[0]), float(alpha_m[0]), float(roi_m[0]))
  for i, channel in enumerate(cfg.channel_names):
    row_truths.append({
        'seed': seed,
        'basis': basis.name,
        'channel': channel,
        'true_ec_m': float(ec_m[i]),
        'true_roi_m': float(roi_m[i]),
        'true_alpha_m': float(alpha_m[i]),
    })
  row = {
      'seed': seed,
      'basis': basis.name,
      'true_alpha_m': round(true_alpha, 4),
      'target_oracle_r2': target_r2,
      'achieved_oracle_r2': round(diag['oracle_r2'], 4),
      'noise_scale': round(diag['noise_scale'], 5),
      'media_share_pct': round(diag['media_share_pct'], 1),
      'true_roi_m': round(true_roi, 3),
  }

  def fit(variant):
    # A seeded rng so the informed variant's prior mis-specification is
    # reproducible: `ec_alpha_noisy` perturbs both anchors, and re-running
    # must reproduce the same "how wrong the media team was" draw.
    spec = build_model_spec(variant, sim, cfg,
                             rng=np.random.default_rng(seed),
                             media_prior_type='roi',
                             knots=N_KNOTS,
                             max_lag=MAX_LAG_BY_VARIANT.get(variant))
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)
    return mmm

  models_dir = os.path.join(out_dir, 'models')
  os.makedirs(models_dir, exist_ok=True)
  spend = sim.cost_gtm.numpy().sum(axis=(0, 1))
  mroi_rows = [
      {
          'seed': seed,
          'basis': basis.name,
          'variant': 'truth',
          'spend_multiplier': mult,
          'mean': mroi_mod.true_mroi(cfg, sim, 0, true_roi, mult),
          'ci_lo': np.nan,
          'ci_hi': np.nan,
          'true_mroi': mroi_mod.true_mroi(cfg, sim, 0, true_roi, mult),
          'pct_error': 0.0,
          'truth_in_ci': np.nan,
      }
      for mult in MROI_MULTIPLIERS
  ]

  def record(variant: str, prefix: str) -> None:
    """Fits one arm once, then takes BOTH analyses off that same model."""
    mmm = fit(variant)

    # The anchors the informed arm was actually handed, recorded because
    # `build_alpha_prior` CLIPS its perturbed centre into [1e-4, 0.99]. At a
    # true alpha_m of 0.3 that ceiling is unreachable; at 0.8 a ~25% upward
    # perturbation hits it, so the anchor error is no longer symmetric and
    # the anchor-vs-error relationship must be read off what was used, not
    # off the nominal perturbation.
    for param in ('ec_m', 'alpha_m'):
      row[f'{prefix}_{param}_anchor'] = _anchor(mmm.model_spec.prior, param)

    post = mmm.inference_data.posterior
    row[f'{prefix}_ec_err'] = round(
        (float(post['ec_m'].values[..., 0].mean()) / true_ec - 1) * 100, 1)
    row[f'{prefix}_roi_err'] = round(
        (float(post['roi_m'].values[..., 0].mean()) / true_roi - 1) * 100, 1)
    row[f'{prefix}_alpha_err'] = round(
        (float(post['alpha_m'].values[..., 0].mean()) / true_alpha - 1) * 100, 1)

    az_ = analyzer.Analyzer(mmm)
    for mult in MROI_MULTIPLIERS:
      draws = mroi_mod.mroi_draws(az_, mult, spend)
      mean, lo, hi = mroi_mod._ci(draws[..., 0])  # TV is channel 0
      truth = mroi_mod.true_mroi(cfg, sim, 0, true_roi, mult)
      mroi_rows.append({
          'seed': seed,
          'basis': basis.name,
          'variant': variant,
          'spend_multiplier': mult,
          'mean': mean,
          'ci_lo': lo,
          'ci_hi': hi,
          'true_mroi': truth,
          'pct_error': (mean / truth - 1) * 100,
          'truth_in_ci': float(lo <= truth <= hi),
      })

    mmm.inference_data.to_netcdf(
        os.path.join(models_dir, f'seed_{seed}_{variant}_inference_data.nc'))
    del mmm, az_

  record('default', 'default')
  # The ACHIEVABLE informed prior, not the oracle one: both anchors are
  # perturbed ~25% before use, because no advertiser knows the true ec_m or
  # adstock benchmark exactly. `ec_alpha_only` (exact) is deliberately not
  # run -- a prior centred on the truth is unbiased by construction, so
  # beating the default with it proves less than it appears to.
  record('ec_alpha_noisy', 'informed')

  pd.DataFrame(mroi_rows).to_csv(
      os.path.join(out_dir, f'mroi_{seed}.csv'), index=False)
  pd.DataFrame(row_truths).to_csv(
      os.path.join(out_dir, f'true_params_{seed}.csv'), index=False)
  return row


def main_report(out_dir: str, hi_noise_ablation_dir: str | None) -> None:
  # Exclude the aggregate this function itself writes: 'per_seed_*.csv' also
  # matches 'per_seed_all.csv', which silently doubles every row on a re-run.
  files = sorted(
      f
      for f in glob.glob(os.path.join(out_dir, 'per_seed_*.csv'))
      if not os.path.basename(f).startswith('per_seed_all')
  )
  if not files:
    print(f'no per_seed_*.csv files found in {out_dir}')
    return
  df = (pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        .sort_values('seed').reset_index(drop=True))
  df.to_csv(os.path.join(out_dir, 'per_seed_all.csv'), index=False)

  # Assemble the measured truths and prove the estimand is identical in every
  # draw. `build_scenario` asserts this per seed; only here can it be checked
  # ACROSS seeds, which is the property that makes this a seed sweep at all.
  mroi_files = sorted(glob.glob(os.path.join(out_dir, 'mroi_*.csv')))
  if mroi_files:
    mroi = pd.concat(
        [pd.read_csv(f) for f in mroi_files], ignore_index=True
    ).sort_values(['seed', 'variant', 'spend_multiplier'])
    mroi.to_csv(os.path.join(out_dir, 'all_seeds.csv'), index=False)

  # Build the file list from the seeds actually present, NOT by globbing
  # 'true_params_*.csv' -- that pattern also matches the aggregate written
  # just below, which double-counts every seed on any re-run.
  truths = pd.concat(
      [
          pd.read_csv(os.path.join(out_dir, f'true_params_{seed}.csv'))
          for seed in df['seed']
      ],
      ignore_index=True,
  ).sort_values(['channel', 'seed'])
  truths.to_csv(os.path.join(out_dir, 'true_params_by_seed.csv'), index=False)

  # Compare on a RELATIVE tolerance, not exact equality: these are float32
  # values recomputed per draw, so they differ in the last bit or two even
  # when perfectly pinned. float32 eps is ~1.2e-7; 1e-5 leaves headroom for
  # accumulated rounding while still catching real drift by orders of
  # magnitude (the drift this basis was built to remove was ~5e-1).
  PIN_RTOL = 1e-5
  drift = {}
  for channel, group in truths.groupby('channel'):
    for col in ('true_ec_m', 'true_roi_m', 'true_alpha_m'):
      rel = (group[col].max() - group[col].min()) / abs(group[col].mean())
      if rel > PIN_RTOL:
        drift[f'{channel}.{col}'] = rel
  assert not drift, (
      'a truth varies across seeds beyond float noise -- NOT a clean seed '
      f'sweep: {drift}\nSee r90_basis.py.'
  )
  worst = max(
      (group[col].max() - group[col].min()) / abs(group[col].mean())
      for _, group in truths.groupby('channel')
      for col in ('true_ec_m', 'true_roi_m', 'true_alpha_m')
  )
  print(
      f'truths PINNED across all {truths.seed.nunique()} seeds to float32 '
      f'precision (worst relative spread {worst:.1e}); only the noise '
      'realization varies:'
  )
  print(
      truths.groupby('channel')
      .first()
      .drop(columns='seed')
      .to_string()
  )
  print()

  pd.set_option('display.width', 200)
  basis_name = str(df['basis'].iloc[0]) if 'basis' in df else 'alpha03 (untagged)'
  assert 'basis' not in df or df['basis'].nunique() == 1, (
      f'this directory mixes bases {sorted(df["basis"].unique())} -- those are '
      'different estimands and must not be pooled'
  )
  print(
      f'{len(df)} seeds completed -- basis={basis_name} '
      f'target_oracle_r2={df["target_oracle_r2"].iloc[0]}\n'
  )
  cols = ['seed', 'achieved_oracle_r2', 'noise_scale', 'media_share_pct',
          'default_ec_err', 'default_roi_err', 'default_alpha_err',
          'informed_ec_err', 'informed_roi_err', 'informed_alpha_err']
  print(df[[c for c in cols if c in df]].to_string(index=False))
  print()
  for metric in ('ec_err', 'roi_err', 'alpha_err'):
    d, i = f'default_{metric}', f'informed_{metric}'
    if d in df and i in df:
      wins = int((df[i].abs() < df[d].abs()).sum())
      print(
          f'{metric:<10} default median {df[d].median():+7.1f}%  '
          f'[{df[d].min():+.1f}, {df[d].max():+.1f}]   '
          f'informed median {df[i].median():+7.1f}%  '
          f'[{df[i].min():+.1f}, {df[i].max():+.1f}]   '
          f'informed wins {wins}/{len(df)}'
      )
  print()

  if hi_noise_ablation_dir:
    hi_path = os.path.join(hi_noise_ablation_dir, 'per_seed_all.csv')
    if os.path.exists(hi_path):
      hi = pd.read_csv(hi_path)[['seed', 'informed_ec_err', 'informed_roi_err']].rename(
          columns={'informed_ec_err': 'hi_noise_informed_ec_err',
                   'informed_roi_err': 'hi_noise_informed_roi_err'})
      merged = df.merge(hi, on='seed', how='inner')
      if len(merged):
        print('--- low-noise (this run) vs original oracle_r2=0.80 ablation ---')
        print(merged[['seed', 'informed_roi_err', 'hi_noise_informed_roi_err']].to_string(index=False))
        print()
        print(f"low-noise  informed roi_err: medianAbs={merged['informed_roi_err'].abs().median():.1f}  median={merged['informed_roi_err'].median():+.1f}")
        print(f"orig-noise informed roi_err: medianAbs={merged['hi_noise_informed_roi_err'].abs().median():.1f}  median={merged['hi_noise_informed_roi_err'].median():+.1f}")


def main() -> int:
  argv = [a for a in sys.argv[1:] if not a.startswith('--basis=')]
  basis_arg = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--basis=')),
      r90_basis.DEFAULT_BASIS.name,
  )
  if basis_arg not in r90_basis.BASES:
    print(f'unknown --basis={basis_arg}; choose from {sorted(r90_basis.BASES)}')
    return 2
  basis = r90_basis.BASES[basis_arg]

  if argv[0:1] == ['--report']:
    out_dir = argv[1]
    hi_noise_ablation_dir = argv[2] if len(argv) > 2 else None
    main_report(out_dir, hi_noise_ablation_dir)
    return 0

  seed = int(argv[0])
  out_dir = argv[1]
  target_r2 = float(argv[2]) if len(argv) > 2 else DEFAULT_ORACLE_R2
  study_dir = HERE
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f'per_seed_{seed}.csv')

  if os.path.exists(out_path):
    print(f'seed {seed}: already done, skipping', flush=True)
    return 0

  # Per-seed filenames carry no basis, so a directory holding two bases would
  # pool incomparable estimands in `--report` with nothing to flag it. Refuse
  # up front rather than after twenty minutes of sampling.
  existing = sorted(
      f
      for f in glob.glob(os.path.join(out_dir, 'per_seed_*.csv'))
      if not os.path.basename(f).startswith('per_seed_all')
  )
  if existing:
    prior_basis = pd.read_csv(existing[0]).get('basis')
    prior_name = (
        str(prior_basis.iloc[0]) if prior_basis is not None
        else r90_basis.DEFAULT_BASIS.name
    )
    if prior_name != basis.name:
      print(
          f'{out_dir} already holds basis {prior_name!r}; refusing to write '
          f'{basis.name!r} into it. Use a separate out_dir per basis.'
      )
      return 2

  t0 = time.time()
  row = run_seed(seed, target_r2, study_dir, out_dir, basis=basis)
  pd.DataFrame([row]).to_csv(out_path, index=False)
  print(
      f"seed {seed} [{basis.name}] done in {time.time() - t0:.0f}s | "
      f"achieved_r2={row['achieved_oracle_r2']:.4f} "
      f"noise_scale={row['noise_scale']:.4f} | "
      f"default ec {row['default_ec_err']:+.1f}% roi {row['default_roi_err']:+.1f}% | "
      f"informed ec {row['informed_ec_err']:+.1f}% roi {row['informed_roi_err']:+.1f}%",
      flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
