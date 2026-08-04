"""Channel-2 recovery from an r90 ablation's saved posteriors.

Any r90 ablation's per-seed CSVs record Channel-1 only (`[..., 0]`), so this
reads the saved `InferenceData` instead of refitting. Truths come from
`r90_basis`, which pins them and asserts them on every scenario build.

Usage:
  .venv/bin/python demo/synthetic/scratch_extract_r90_wellspec_ch2.py
  .venv/bin/python demo/synthetic/scratch_extract_r90_wellspec_ch2.py \
      --run-dir=demo/synthetic/fitted_models/scratch_ablation_r90_alpha08 \
      --basis=alpha08

`--basis` must match the basis the posteriors were fitted on -- it supplies
the truths the errors are measured against. Asserted against the run's own
`per_seed_all.csv` tag below.
"""

import glob
import os
import sys

import arviz as az
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r90_basis

RUN = next(
    (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--run-dir=')),
    os.path.join(HERE, 'fitted_models', 'scratch_ablation_r90_wellspec'),
)
BASIS = r90_basis.BASES[
    next(
        (a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--basis=')),
        r90_basis.DEFAULT_BASIS.name,
    )
]
OUT = os.path.join(RUN, 'per_seed_all_ch2.csv')

_per_seed = os.path.join(RUN, 'per_seed_all.csv')
if os.path.exists(_per_seed):
  _tag = pd.read_csv(_per_seed).get('basis')
  _found = str(_tag.iloc[0]) if _tag is not None else 'alpha03'
  assert _found == BASIS.name, (
      f'{RUN} holds basis {_found!r} but --basis={BASIS.name!r} was given'
  )

CH = 1  # Channel-2 / Display.
TRUE = {
    'ec': BASIS.true_ec['Display'],
    'alpha': BASIS.true_alpha['Display'],
    'roi': float(r90_basis.expected_true_roi(BASIS)['Display']),
}
ARMS = {'default': 'default', 'ec_alpha_noisy': 'informed'}

rows = {}
for path in sorted(glob.glob(os.path.join(RUN, 'models', '*.nc'))):
  name = os.path.basename(path)
  seed = int(name.split('_')[1])
  variant = name.replace(f'seed_{seed}_', '').replace(
      '_inference_data.nc', '')
  prefix = ARMS[variant]
  post = az.from_netcdf(path).posterior
  row = rows.setdefault(seed, {'seed': seed})
  for param, key in [('ec_m', 'ec'), ('roi_m', 'roi'), ('alpha_m', 'alpha')]:
    fitted = float(post[param].values[..., CH].mean())
    row[f'{prefix}_{key}_err'] = round((fitted / TRUE[key] - 1) * 100, 1)
    # `alpha_m` is a retention rate, so also carry the percentage-POINT
    # difference: CLAUDE.md's warning about ratio comparisons on Channel-2
    # applies whenever the denominator is small.
    if param == 'alpha_m':
      row[f'{prefix}_alpha_pp'] = round((fitted - TRUE[key]) * 100, 2)

df = pd.DataFrame(list(rows.values())).sort_values('seed')
df.to_csv(OUT, index=False)
print(f'true: ec {TRUE["ec"]}, alpha {TRUE["alpha"]}, roi {TRUE["roi"]:.3f}')
print(df.to_string(index=False))
print()
for arm in ['default', 'informed']:
  print('{:<9} ec {:+6.1f}%  roi {:+6.1f}%  alpha {:+6.1f}% ({:+.2f} pp)'.format(
      arm,
      df[f'{arm}_ec_err'].median(), df[f'{arm}_roi_err'].median(),
      df[f'{arm}_alpha_err'].median(), df[f'{arm}_alpha_pp'].median()))
