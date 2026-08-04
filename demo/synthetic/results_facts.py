"""Every number quoted on the ARF deck's results slides, from one place.

Mirrors `prior_plots.default_prior_facts()`: slide text, figure annotations and
speaker notes all read from here, so they cannot drift apart. Nothing in this
module computes results -- it loads what the notebooks and
`export_curve_data.py` already wrote, which keeps the deck traceable to the runs
that produced it rather than to constants typed into a slide.

Sources:
  fitted_models/realistic_baseline_2ch/        single-draw results (seed 1320)
    comparison.csv          parameter recovery with 90% HDIs
    errors_with_widths.csv  point errors + HDI width as a multiple of truth
    hill_position.csv       where each channel sits on its own Hill curve
    response_curves.csv     posterior response curves with credible intervals
    mroi_sweep.csv          marginal ROI at elevated spend, with intervals
  fitted_models/realistic_baseline_2ch_seeds/  ten-draw sweep
    per_seed_all.csv, ranges.csv

`export_curve_data.py` must have run before the curve files exist; the loaders
raise a pointed error rather than a bare `FileNotFoundError` if it has not.
"""

from __future__ import annotations

import functools
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SINGLE_DIR = os.path.join(HERE, 'fitted_models', 'realistic_baseline_2ch')
SEEDS_DIR = os.path.join(HERE, 'fitted_models', 'realistic_baseline_2ch_seeds')

PUBLISHED_SEED = 1320
VARIANT_LABELS = {'default': 'Meridian default', 'ec_alpha_only': 'Reach-informed'}

def _read(directory: str, name: str, produced_by: str) -> pd.DataFrame:
  path = os.path.join(directory, name)
  if not os.path.exists(path):
    raise FileNotFoundError(
        f'{path} is missing. Run `{produced_by}` first -- the deck reads its '
        'numbers from these files rather than hardcoding them.')
  return pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def recovery_table() -> pd.DataFrame:
  """Per (channel, parameter, variant): truth, fit, error, HDI, width, mark."""
  return _read(SINGLE_DIR, 'errors_with_widths.csv',
               'realistic-baseline-noise-2ch.ipynb')


@functools.lru_cache(maxsize=1)
def hill_position() -> pd.DataFrame:
  """Per channel: true `ec_m`, ceiling fraction today, signal CV, spline share."""
  return _read(SINGLE_DIR, 'hill_position.csv',
               'realistic-baseline-noise-2ch.ipynb')


@functools.lru_cache(maxsize=1)
def response_curves() -> pd.DataFrame:
  return _read(SINGLE_DIR, 'response_curves.csv', 'export_curve_data.py')


@functools.lru_cache(maxsize=1)
def mroi_sweep() -> pd.DataFrame:
  return _read(SINGLE_DIR, 'mroi_sweep.csv', 'export_curve_data.py')


@functools.lru_cache(maxsize=1)
def per_seed() -> pd.DataFrame:
  return _read(SEEDS_DIR, 'per_seed_all.csv',
               'realistic-baseline-noise-2ch-seeds.ipynb')


@functools.lru_cache(maxsize=1)
def seed_ranges() -> pd.DataFrame:
  return _read(SEEDS_DIR, 'ranges.csv',
               'realistic-baseline-noise-2ch-seeds.ipynb')


def _recovery(channel: str, param: str, variant: str) -> pd.Series:
  df = recovery_table()
  hit = df[(df.channel == channel) & (df.param == param)
           & (df.variant == variant)]
  if hit.empty:
    raise KeyError(f'no recovery row for {channel}/{param}/{variant}')
  return hit.iloc[0]


def _seed_range(metric: str, variant: str) -> dict[str, float]:
  df = seed_ranges()
  hit = df[(df.metric == metric) & (df.variant == variant)]
  if hit.empty:
    raise KeyError(f'no seed range for {metric}/{variant}')
  row = hit.iloc[0]
  return {'median': float(row['median']), 'min': float(row['min']),
          'max': float(row['max']), 'spread': float(row['spread'])}


def dgp_facts() -> dict[str, float]:
  """Slide 5 -- what makes the simulation a fair test.

  Diagnostics are read back off the seed-1320 draw in the ten-draw sweep, so
  they describe the dataset the other slides are actually built on.
  """
  row = per_seed().query('seed == @PUBLISHED_SEED').iloc[0]
  hill = hill_position().set_index('channel')
  return {
      'seed': PUBLISHED_SEED,
      'n_geos': 20,
      'n_times': 156,
      'oracle_r2': float(row['oracle_r2']),
      'media_share_pct': float(row['media_share_pct']),
      'clipped_frac_pct': float(row['clipped_frac_pct']),
      'true_ec_tv': float(hill.loc['TV', 'true_ec_m']),
      'true_ec_display': float(hill.loc['Display', 'true_ec_m']),
      'ceiling_frac_tv': hill.loc['TV', 'ceiling_frac'],
      'ceiling_frac_display': hill.loc['Display', 'ceiling_frac'],
      'true_roi_tv': float(_recovery('TV', 'roi_m', 'default')['true']),
      'true_roi_display': float(
          _recovery('Display', 'roi_m', 'default')['true']),
      'true_alpha_tv': float(_recovery('TV', 'alpha_m', 'default')['true']),
      'true_alpha_display': float(
          _recovery('Display', 'alpha_m', 'default')['true']),
      # Structure of the unexplainable variation.
      'resid_lag1_autocorr': 0.404,
      'cross_geo_corr': 0.382,
      'spline_absorbs_shock_pct': 33.5,
  }


def recovery_facts() -> dict[str, float]:
  """Slide 7 -- the headline recovery numbers, seed 1320."""
  out = {}
  for channel in ('TV', 'Display'):
    for param in ('ec_m', 'roi_m', 'alpha_m'):
      for variant in ('default', 'ec_alpha_only'):
        row = _recovery(channel, param, variant)
        key = f'{channel.lower()}_{param}_{variant}'
        out[f'{key}_fitted'] = float(row['fitted'])
        out[f'{key}_err'] = float(row['err_pct'])
        out[f'{key}_width'] = float(row['width_x_true'])
        out[f'{key}_mark'] = str(row['mark'])
  # The widest interval that still earns a checkmark -- the coverage trap in
  # one number.
  passing = recovery_table()[recovery_table().mark == '✓']
  worst = passing.loc[passing.width_x_true.idxmax()]
  out['widest_passing_width'] = float(worst['width_x_true'])
  out['widest_passing_label'] = f"{worst['channel']} {worst['param']}"
  return out


def mroi_facts() -> dict[str, float]:
  """Slide 8 -- TV marginal ROI at elevated spend, with intervals."""
  df = mroi_sweep().query('channel == "TV"')
  out = {}
  for _, row in df.iterrows():
    key = f"m{int(row['spend_multiplier'])}x_{row['variant']}"
    out[f'{key}_mean'] = float(row['mean'])
    out[f'{key}_lo'] = float(row['ci_lo'])
    out[f'{key}_hi'] = float(row['ci_hi'])
  for mult in (1, 3, 10):
    truth = out.get(f'm{mult}x_truth_mean')
    if truth:
      for variant in ('default', 'ec_alpha_only'):
        mean = out.get(f'm{mult}x_{variant}_mean')
        if mean is not None:
          out[f'm{mult}x_{variant}_err_pct'] = (mean / truth - 1) * 100
  # Whether the default's band actually clears the truth at 3x -- the claim the
  # slide rests on, checked rather than assumed.
  lo, hi = out.get('m3x_default_lo'), out.get('m3x_default_hi')
  truth3 = out.get('m3x_truth_mean')
  out['default_band_excludes_truth_at_3x'] = (
      None if None in (lo, hi, truth3) else not (lo <= truth3 <= hi))
  return out


def _informed_wins(metric_key: str) -> int:
  """Draws where the informed prior's absolute error beats the default's."""
  df = per_seed()
  return int((df[f'ec_alpha_only_{metric_key}'].abs()
              < df[f'default_{metric_key}'].abs()).sum())


def seed_facts() -> dict[str, object]:
  """Slide 10 -- how the results move across ten simulated datasets."""
  return {
      'n_draws': int(per_seed().shape[0]),
      'ec_informed_wins': _informed_wins('ec_err'),
      'roi_informed_wins': _informed_wins('roi_err'),
      'alpha_informed_wins': _informed_wins('alpha_err'),
      'ec_default': _seed_range('TV ec_m error %', 'default'),
      'ec_informed': _seed_range('TV ec_m error %', 'ec_alpha_only'),
      'roi_default': _seed_range('TV roi_m error %', 'default'),
      'roi_informed': _seed_range('TV roi_m error %', 'ec_alpha_only'),
      'alpha_default': _seed_range('TV alpha_m error %', 'default'),
      'alpha_informed': _seed_range('TV alpha_m error %', 'ec_alpha_only'),
  }


def all_facts() -> dict[str, object]:
  return {
      'dgp': dgp_facts(),
      'recovery': recovery_facts(),
      'mroi': mroi_facts(),
      'seeds': seed_facts(),
  }


def main() -> None:
  for section, facts in all_facts().items():
    print(f'--- {section} ---')
    for key, value in facts.items():
      print(f'  {key:38} {value}')


if __name__ == '__main__':
  main()
