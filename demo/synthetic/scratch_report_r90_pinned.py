"""Summarises the pinned-basis run for review. Reads only; changes nothing.

Reports the three things worth checking before any of this reaches a slide:

  1. that the estimand really is identical in every draw (the whole point of
     the re-run -- see `r90_basis`);
  2. per-seed point recovery for both arms, plus how each seed's informed
     prior was actually mis-specified, since that is now a per-seed draw;
  3. pooled posterior box statistics for both channels, and the mROI sweep.

Usage:
  .venv/bin/python demo/synthetic/scratch_report_r90_pinned.py [run_dir]
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

import r90_basis

DEFAULT_RUN_DIR = os.path.join(
    HERE, 'fitted_models', 'scratch_ablation_r90_pinned'
)
CHANNELS = ['TV', 'Display']
VARIANTS = [('default', 'default'), ('ec_alpha_noisy', 'informed')]
PARAMS = ['ec_m', 'roi_m', 'alpha_m']


def _pooled(run_dir, param, variant, channel, truths, seeds, absolute=False):
  """Every draw for one channel, pooled across seeds, vs each seed's truth."""
  out = []
  for seed in seeds:
    true_val = float(
        truths[(truths.seed == seed) & (truths.channel == channel)][
            f'true_{param}'
        ].iloc[0]
    )
    path = os.path.join(
        run_dir, 'models', f'seed_{seed}_{variant}_inference_data.nc'
    )
    draws = (
        az.from_netcdf(path)
        .posterior[param]
        .sel(media_channel=channel)
        .values.flatten()
    )
    out.append(draws - true_val if absolute else (draws / true_val - 1) * 100)
  return np.concatenate(out)


def main() -> int:
  run_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_DIR
  pd.set_option('display.width', 220)

  per_seed = pd.read_csv(os.path.join(run_dir, 'per_seed_all.csv'))
  truths = pd.read_csv(os.path.join(run_dir, 'true_params_by_seed.csv'))
  seeds = sorted(per_seed.seed.unique().tolist())

  print('=' * 78)
  print(f'PINNED-BASIS RUN — {len(seeds)} seeds — {run_dir}')
  print('=' * 78)

  print('\n1. IS THE ESTIMAND IDENTICAL IN EVERY DRAW?\n')
  spread = truths.groupby('channel')[
      ['true_ec_m', 'true_roi_m', 'true_alpha_m']
  ].agg(['min', 'max', 'nunique'])
  print(spread.to_string())
  clean = (
      truths.groupby('channel')[
          ['true_ec_m', 'true_roi_m', 'true_alpha_m']
      ].nunique()
      == 1
  ).all().all()
  print(f'\n  => {"PINNED - clean seed sweep" if clean else "*** VARIES ***"}')

  print('\n2. DGP CONDITIONS PER SEED (these SHOULD vary — it is the noise)\n')
  print(
      per_seed[
          ['seed', 'achieved_oracle_r2', 'noise_scale', 'media_share_pct']
      ].to_string(index=False)
  )

  print('\n3. POINT RECOVERY PER SEED (Channel-1 / TV, posterior means)\n')
  cols = ['seed'] + [
      f'{v}_{p}_err' for v in ('default', 'informed') for p in
      ('ec', 'roi', 'alpha')
  ]
  print(per_seed[cols].to_string(index=False))
  print('\n  medians:')
  print(per_seed[cols[1:]].median().round(1).to_string())
  print('\n  informed better than default, by |error|, out of '
        f'{len(per_seed)} seeds:')
  for p in ('ec', 'roi', 'alpha'):
    wins = int(
        (
            per_seed[f'informed_{p}_err'].abs()
            < per_seed[f'default_{p}_err'].abs()
        ).sum()
    )
    print(f'    {p:6s} {wins}/{len(per_seed)}')

  print('\n4. POOLED POSTERIOR BOXES, BOTH CHANNELS\n')
  rows = []
  for channel in CHANNELS:
    for param in PARAMS:
      absolute = param == 'alpha_m'
      for variant, label in VARIANTS:
        vals = _pooled(
            run_dir, param, variant, channel, truths, seeds, absolute
        )
        rows.append({
            'channel': channel,
            'param': param,
            'unit': 'points' if absolute else 'pct',
            'variant': label,
            'median': round(float(np.median(vals)), 2),
            'q25': round(float(np.percentile(vals, 25)), 2),
            'q75': round(float(np.percentile(vals, 75)), 2),
        })
  box = pd.DataFrame(rows)
  print(box.to_string(index=False))

  print('\n5. mROI AT ELEVATED SPEND (Channel-1)\n')
  mroi = pd.read_csv(os.path.join(run_dir, 'all_seeds.csv'))
  fitted = mroi[mroi.variant != 'truth']
  summary = (
      fitted.groupby(['variant', 'spend_multiplier'])
      .agg(
          median_pct_err=('pct_error', 'median'),
          min_pct_err=('pct_error', 'min'),
          max_pct_err=('pct_error', 'max'),
          seeds_covering_truth=('truth_in_ci', 'sum'),
          n=('truth_in_ci', 'size'),
      )
      .round(1)
  )
  print(summary.to_string())
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
