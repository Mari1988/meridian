"""Marginal ROI for BOTH channels at the pinned basis, without refitting.

The pinned run recorded mROI for Channel-1 only (`channel_idx=0`). Slide 8
now wants both channels, and the fitted posteriors already contain everything
needed: `Meridian.__init__` accepts `inference_data` directly, so each saved
`.nc` can be reattached to a freshly built scenario and handed to `Analyzer`.
No MCMC is re-run -- the cost is rebuilding each seed's scenario (~2 min) to
recover `cfg`/`sim`/`data`/spend and the true curve.

Writes per-draw posterior quantiles rather than just means, so the figure can
draw whatever interval it wants (50% HDI, 90%, ...) without coming back here.

A note on pooling: with the truths pinned, true mROI now varies by only
~0.4-3% across seeds (residual media-execution differences), so seeds can be
pooled in ABSOLUTE units. On the old unpinned basis this was impossible and
the response-curve figure had to index everything to 100.

Writes:
  <run_dir>/mroi_both_channels.csv

Usage:
  .venv/bin/python demo/synthetic/scratch_mroi_both_channels_r90.py
  .venv/bin/python demo/synthetic/scratch_mroi_both_channels_r90.py \
      --run-dir=demo/synthetic/fitted_models/scratch_ablation_r90_alpha08 \
      --basis=alpha08

`--basis` MUST match the basis the posteriors in `--run-dir` were fitted on:
the scenario rebuilt here supplies the truth every % error is measured
against, so a mismatch would silently score one estimand's fits against
another's truth. Checked against the run's own `per_seed_all.csv` below.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import arviz as az_lib
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

DEFAULT_RUN_DIR = os.path.join(
    HERE, 'fitted_models', 'scratch_ablation_r90_pinned'
)

VARIANTS = ['default', 'ec_alpha_noisy']
MAX_LAG_BY_VARIANT = {'default': MERIDIAN_DEFAULT_MAX_LAG}
MULTIPLIERS = mroi_mod.MROI_MULTIPLIERS
# Quantiles stored so the figure can pick its own interval later.
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# The scenario rebuilt here supplies the DATA the saved posteriors are
# reattached to and the TRUTH every % error is measured against, so it must be
# built on the same baseline the posteriors were fitted on. Defaulting to the
# realistic baseline silently paired well-specified fits with realistic data
# and produced plausible, wrong numbers -- hence an explicit flag.
REALISM = {
    'realistic': None,
    'wellspec': rb.RealisticBaselineConfig(mu_ar1_sd=0.0,
                                           seasonal_amplitude=0.0),
}


def run_seed(seed: int, run_dir: str, basis: r90_basis.Basis,
             realism_key: str) -> list[dict]:
  cfg, sim, data, gt = r90_basis.build_scenario(
      seed, realism=REALISM[realism_key], basis=basis)
  spend = sim.cost_gtm.numpy().sum(axis=(0, 1))
  channels = list(cfg.channel_names)
  true_roi = np.asarray(gt['roi_m'])

  rows = []
  for i, channel in enumerate(channels):
    for mult in MULTIPLIERS:
      rows.append({
          'seed': seed,
          'channel': channel,
          'variant': 'truth',
          'spend_multiplier': mult,
          'mean': mroi_mod.true_mroi(cfg, sim, i, float(true_roi[i]), mult),
      })

  for variant in VARIANTS:
    spec = build_model_spec(
        variant,
        sim,
        cfg,
        rng=np.random.default_rng(seed),
        media_prior_type='roi',
        knots=r90_basis.N_KNOTS,
        max_lag=MAX_LAG_BY_VARIANT.get(variant),
    )
    idata = az_lib.from_netcdf(
        os.path.join(
            run_dir, 'models', f'seed_{seed}_{variant}_inference_data.nc'
        )
    )
    # Reattach the saved posterior instead of sampling a new one.
    mmm = model.Meridian(
        input_data=data, model_spec=spec, inference_data=idata
    )
    az_obj = analyzer.Analyzer(mmm)
    for mult in MULTIPLIERS:
      draws = mroi_mod.mroi_draws(az_obj, mult, spend)
      for i, channel in enumerate(channels):
        d = np.asarray(draws)[..., i].flatten()
        row = {
            'seed': seed,
            'channel': channel,
            'variant': variant,
            'spend_multiplier': mult,
            'mean': float(d.mean()),
        }
        for q in QUANTILES:
          row[f'q{int(q * 100):02d}'] = float(np.quantile(d, q))
        rows.append(row)
    del mmm, az_obj
  return rows


def df_seeds(run_dir: str) -> list[int]:
  """Every seed this run actually fitted, read off its per-seed CSVs."""
  path = os.path.join(run_dir, 'per_seed_all.csv')
  assert os.path.exists(path), (
      f'{path} missing -- run the ablation\'s --report first so the seed list '
      'is known')
  return sorted(pd.read_csv(path)['seed'].astype(int).tolist())


def main() -> int:
  run_dir = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--run-dir=')),
      DEFAULT_RUN_DIR,
  )
  basis_name = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--basis=')),
      r90_basis.DEFAULT_BASIS.name,
  )
  basis = r90_basis.BASES[basis_name]
  out_path = os.path.join(run_dir, 'mroi_both_channels.csv')

  # The truth every % error below is measured against comes from the scenario
  # rebuilt here, not from the saved fits -- so scoring one basis's posteriors
  # against another's truth would produce plausible, wrong numbers with
  # nothing to flag them. The run tags each row with its basis; use it.
  per_seed = os.path.join(run_dir, 'per_seed_all.csv')
  if os.path.exists(per_seed):
    tagged = pd.read_csv(per_seed).get('basis')
    found = str(tagged.iloc[0]) if tagged is not None else 'alpha03'
    assert found == basis.name, (
        f'{run_dir} holds basis {found!r} but --basis={basis.name!r} was '
        'given; the truths would not match the fits'
    )

  realism_key = next(
      (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--realism=')),
      None,
  )
  if realism_key not in REALISM:
    print(f'--realism= is required; choose from {sorted(REALISM)}')
    return 2

  # Seeds come from the run's OWN per-seed CSVs, not from a constant: this
  # directory may hold 10 draws or 50, and hardcoding r90_basis.SEEDS silently
  # covered only the first ten of a 50-draw run.
  seeds = [int(s) for s in df_seeds(run_dir)]
  for arg in sys.argv[1:]:
    if arg.startswith('--seeds='):
      seeds = [int(s) for s in arg.split('=', 1)[1].split(',')]

  # One file per seed, in a SUBDIRECTORY so nothing here can be caught by the
  # `per_seed_*` / `mroi_*` globs the ablation scripts run over `run_dir`.
  part_dir = os.path.join(run_dir, f'mroi_both_{realism_key}')
  os.makedirs(part_dir, exist_ok=True)

  if '--assemble' not in sys.argv:
    for i, seed in enumerate(seeds, start=1):
      part = os.path.join(part_dir, f'{seed}.csv')
      if os.path.exists(part):
        print(f'--- seed {seed} ({i}/{len(seeds)}) already done', flush=True)
        continue
      print(f'--- seed {seed} ({i}/{len(seeds)}) [{basis.name}/{realism_key}]',
            flush=True)
      pd.DataFrame(run_seed(seed, run_dir, basis, realism_key)).to_csv(
          part, index=False)

  parts = sorted(glob.glob(os.path.join(part_dir, '*.csv')))
  if not parts:
    print(f'nothing to assemble in {part_dir}')
    return 1
  df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
  print(f'assembling {len(parts)} seeds from {part_dir}')
  df.to_csv(out_path, index=False)

  # Attach each fitted row's true value and % error for convenience.
  truth = (
      df[df.variant == 'truth']
      .set_index(['seed', 'channel', 'spend_multiplier'])['mean']
      .rename('true_mroi')
  )
  out = df[df.variant != 'truth'].join(
      truth, on=['seed', 'channel', 'spend_multiplier']
  )
  out['pct_error'] = (out['mean'] / out['true_mroi'] - 1) * 100
  pd.concat([df[df.variant == 'truth'], out]).to_csv(out_path, index=False)

  print(f'\nwrote {out_path}')
  print(
      out.groupby(['channel', 'variant', 'spend_multiplier'])['pct_error']
      .median()
      .round(1)
      .to_string()
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
