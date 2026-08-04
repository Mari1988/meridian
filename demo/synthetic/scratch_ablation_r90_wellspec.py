"""Baseline ablation: is the residual ROI bias the prior's fault or the DGP's?

`scratch_ablation_r90.py` on the realistic baseline leaves BOTH arms with an
upward ROI bias on Channel-1 that the informed prior does not remove:

    default  +40.3%  =  +21.9% (saturation-driven)  +  18.4% (remainder)
    informed +20.6%  =   +1.2% (saturation-driven)  +  19.3% (remainder)

The remainders are the same size and correlate across seeds at +0.99, i.e.
they are a property of the data draw rather than of the prior. The suspected
cause is baseline misspecification: `mu_t` carries a national AR(1) shock and
a seasonal term the fitted spline cannot represent, Channel-1's national
media correlates with the unrepresentable part (+0.176 at seed 1320, against
Channel-2's +0.004), and `beta_m >= 0` rectifies what would otherwise be a
sign-random confound into a systematic overstatement.

This script removes that confound from BOTH arms by handing the model a
baseline it can actually represent -- trend only, which an 8-knot spline
spans exactly -- while `calibrate_to_oracle_r2` scales the geo-idiosyncratic
residual up to hold oracle R^2 at the same 0.9. Total noise is unchanged;
only its character moves from partly-national to entirely per-geo, and
per-geo noise is what cross-geo averaging can dispose of.

PREDICTIONS, recorded before the run so the check is real either way:
  * `roi_m`: the remainder collapses. default -> ~+22%, informed -> ~+1%.
  * `ec_m`: essentially UNCHANGED (default ~-69%, informed ~-4%). Mechanism 1
    is prior-driven, not baseline-driven. If `ec_m` recovery also improves
    materially, then part of the saturation failure was baseline confounding
    too and the study's thesis needs revising rather than confirming.

This is a SECOND reported arm, not a replacement for the headline run: it
gives up the DGP's cross-geo correlation, which is one of the properties that
makes the realistic baseline realistic. The pair is the result -- the prior
effect surviving both a realistic and a well-specified baseline is a stronger
claim than either alone.

Usage:
  .venv/bin/python demo/synthetic/scratch_ablation_r90_wellspec.py           # 3 seeds
  .venv/bin/python demo/synthetic/scratch_ablation_r90_wellspec.py --all     # 10 seeds
  .venv/bin/python demo/synthetic/scratch_ablation_r90_wellspec.py --report  # re-print
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
import scratch_check_mroi_recovery_r90 as mroi_mod
import realistic_baseline as rb


# The informed arm, and where its output lands. `--bounded25` swaps the
# unbounded lognormal anchor error for one that is off by EXACTLY +-25%, which
# is what supports a claim of the form "even if you are off by 25%, recovery
# holds"; the default `ec_alpha_noisy` cannot support it, since half its draws
# land outside +-25%. Each variant gets its own directory: the per-seed CSVs
# are keyed by seed alone, so sharing one would interleave them silently.
INFORMED_VARIANT = (
    'ec_alpha_bounded25' if '--bounded25' in sys.argv else 'ec_alpha_noisy'
)
OUT_DIR = os.path.join(
    HERE,
    'fitted_models',
    'scratch_ablation_r90_wellspec_b25'
    if INFORMED_VARIANT == 'ec_alpha_bounded25'
    else 'scratch_ablation_r90_wellspec',
)

# Same sampler settings as `scratch_ablation_r90.py`, so the two runs differ
# in the baseline and nothing else.
MCMC_KWARGS = dict(n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
N_KNOTS = r90_basis.N_KNOTS
MAX_LAG_BY_VARIANT = {'default': MERIDIAN_DEFAULT_MAX_LAG}

# The whole ablation, in two fields. `mu_t` becomes trend-only: a linear drift
# an 8-knot spline represents exactly (measured unrepresentable variance from
# the trend term: 0.0000). Everything else -- the geo-idiosyncratic AR(1)
# residual, its persistence, its fat tails -- is left at the realistic
# defaults, and the noise LEVEL is re-calibrated to the same oracle R^2.
WELLSPEC = rb.RealisticBaselineConfig(
    mu_ar1_sd=0.0,
    seasonal_amplitude=0.0,
)

SEEDS_SMOKE = r90_basis.SEEDS[:3]  # 1320, 7, 42


def run_seed(seed: int) -> dict:
  """Fits both arms on one well-specified-baseline draw."""
  cfg, sim, data, gt = r90_basis.build_scenario(seed, realism=WELLSPEC)

  diag = rb.baseline_diagnostics(sim, N_KNOTS)
  ec_m, alpha_m = sim.ec_m.numpy(), sim.alpha_m.numpy()
  roi_m = np.asarray(gt['roi_m'])
  true_ec, true_alpha, true_roi = (
      float(ec_m[0]), float(alpha_m[0]), float(roi_m[0]))

  row = {
      'seed': seed,
      'achieved_oracle_r2': round(diag['oracle_r2'], 4),
      'noise_scale': round(diag['noise_scale'], 5),
      'media_share_pct': round(diag['media_share_pct'], 1),
      'true_roi_m': round(true_roi, 3),
  }

  def fit(variant):
    # Same seeded rng as the headline run, so the informed arm's prior
    # mis-specification is the SAME "how wrong the media team was" draw.
    # Otherwise this ablation would move the anchor and the baseline at once.
    spec = build_model_spec(
        variant, sim, cfg,
        rng=np.random.default_rng(seed),
        media_prior_type='roi',
        knots=N_KNOTS,
        max_lag=MAX_LAG_BY_VARIANT.get(variant))
    mmm = model.Meridian(input_data=data, model_spec=spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(**MCMC_KWARGS)
    return mmm

  models_dir = os.path.join(OUT_DIR, 'models')
  os.makedirs(models_dir, exist_ok=True)
  spend = sim.cost_gtm.numpy().sum(axis=(0, 1))
  mroi_rows = [
      {
          'seed': seed,
          'variant': 'truth',
          'spend_multiplier': mult,
          'mean': mroi_mod.true_mroi(cfg, sim, 0, true_roi, mult),
          'ci_lo': np.nan,
          'ci_hi': np.nan,
          'true_mroi': mroi_mod.true_mroi(cfg, sim, 0, true_roi, mult),
          'pct_error': 0.0,
          'truth_in_ci': np.nan,
      }
      for mult in mroi_mod.MROI_MULTIPLIERS
  ]

  def record(variant: str, prefix: str) -> None:
    t0 = time.time()
    mmm = fit(variant)
    post = mmm.inference_data.posterior
    row[f'{prefix}_ec_err'] = round(
        (float(post['ec_m'].values[..., 0].mean()) / true_ec - 1) * 100, 1)
    row[f'{prefix}_roi_err'] = round(
        (float(post['roi_m'].values[..., 0].mean()) / true_roi - 1) * 100, 1)
    row[f'{prefix}_alpha_err'] = round(
        (float(post['alpha_m'].values[..., 0].mean()) / true_alpha - 1) * 100,
        1)
    print(f'  seed {seed} {prefix:<9} ec {row[f"{prefix}_ec_err"]:+7.1f}%  '
          f'roi {row[f"{prefix}_roi_err"]:+7.1f}%  '
          f'alpha {row[f"{prefix}_alpha_err"]:+7.1f}%  '
          f'({time.time() - t0:.0f}s)', flush=True)

    # Slide 8's panels and slide 7's full-draw boxes come off this same fit,
    # not a second one: refitting an identical model to compute mROI would
    # double the cost and risk the two figures disagreeing.
    az_ = analyzer.Analyzer(mmm)
    for mult in mroi_mod.MROI_MULTIPLIERS:
      draws = mroi_mod.mroi_draws(az_, mult, spend)
      mean, lo, hi = mroi_mod._ci(draws[..., 0])  # Channel-1 is index 0.
      truth = mroi_mod.true_mroi(cfg, sim, 0, true_roi, mult)
      mroi_rows.append({
          'seed': seed,
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
  record(INFORMED_VARIANT, 'informed')
  row['informed_variant'] = INFORMED_VARIANT

  os.makedirs(OUT_DIR, exist_ok=True)
  pd.DataFrame([row]).to_csv(
      os.path.join(OUT_DIR, f'per_seed_{seed}.csv'), index=False)
  pd.DataFrame(mroi_rows).to_csv(
      os.path.join(OUT_DIR, f'mroi_{seed}.csv'), index=False)
  return row


def report() -> None:
  """Prints the ablation against the realistic-baseline run it ablates."""
  files = sorted(
      f for f in glob.glob(os.path.join(OUT_DIR, 'per_seed_*.csv'))
      # Must exclude EVERY per_seed_all* aggregate, not just the exact
      # 'per_seed_all.csv': 'per_seed_all_ch2.csv' (written by
      # scratch_extract_r90_wellspec_ch2.py) also matches 'per_seed_*.csv',
      # and folding those Channel-2 rows into this Channel-1 table silently
      # pooled two channels -- 50 seeds reported as 60, with the summary
      # medians contaminated.
      if not os.path.basename(f).startswith('per_seed_all'))
  if not files:
    print(f'no per_seed_*.csv in {OUT_DIR}')
    return
  new = (pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
         .sort_values('seed').reset_index(drop=True))
  new.to_csv(os.path.join(OUT_DIR, 'per_seed_all.csv'), index=False)

  base_path = os.path.join(
      HERE, 'fitted_models', 'scratch_ablation_r90_pinned', 'per_seed_all.csv')
  old = pd.read_csv(base_path)
  old = old[old.seed.isin(new.seed)].sort_values('seed').reset_index(drop=True)

  print(f'\nwell-specified baseline, {len(new)} seed(s): '
        f'{sorted(new.seed.tolist())}')
  print('same seeds on the realistic baseline shown for comparison\n')
  hdr = '{:<10} {:>10} {:>12} {:>10}'
  print(hdr.format('metric', 'realistic', 'well-spec', 'change'))
  for metric in ['roi_err', 'ec_err', 'alpha_err']:
    for arm in ['default', 'informed']:
      c = f'{arm}_{metric}'
      o, n = old[c].mean(), new[c].mean()
      print(hdr.format(f'{arm[:4]} {metric.split("_")[0]}',
                       f'{o:+.1f}%', f'{n:+.1f}%', f'{n - o:+.1f}'))
  print('\nper seed (well-specified):')
  print(new.to_string(index=False))


def main() -> None:
  if '--report' in sys.argv:
    report()
    return
  seeds = r90_basis.SEEDS if '--all' in sys.argv else SEEDS_SMOKE
  # `--seeds 8,99,...` fits only those draws. `report()` globs every
  # `per_seed_*.csv` in `OUT_DIR`, so topping up an existing 3-seed run still
  # reports all ten rather than only the new ones.
  for arg in sys.argv[1:]:
    if arg.startswith('--seeds='):
      seeds = [int(s) for s in arg.split('=', 1)[1].split(',')]
  print(f'well-specified-baseline ablation over {len(seeds)} seed(s): {seeds}')
  t0 = time.time()
  for seed in seeds:
    run_seed(seed)
  print(f'\ntotal {time.time() - t0:.0f}s')
  report()


if __name__ == '__main__':
  main()
