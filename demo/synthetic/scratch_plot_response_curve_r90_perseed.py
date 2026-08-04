"""Response curve per seed (not pooled) -- companion to the slide 9 figure.

Same computation as `scratch_plot_response_curve_r90_pooled.py`
(reconstructs `Meridian`/`Analyzer` from the saved `InferenceData`, no
refit; indexes each seed's incremental-outcome draws to 100 = that seed's
own true value at 1x spend), but keeps each seed separate instead of
pooling -- 3 columns, mirroring `scratch_plot_mroi_r90_slide8.py`'s
small-multiples layout, so the "curves cross" story can be checked
individually per seed before trusting the pooled version.

Usage:
  .venv/bin/python demo/synthetic/scratch_plot_response_curve_r90_perseed.py
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(
    HERE, 'fitted_models', 'scratch_check_mroi_recovery_r90', 'models')
OUT_PATH = os.path.join(
    HERE, 'figures', 'slide9_response_curve_r90_perseed.png')

TRUTH = '#1A1A1A'
DEFAULT = '#C0392B'
INFORMED = '#2C7FB8'
TRUTH_DASH = (0, (6, 3))
SEEDS = [1320, 7, 42]
TRUE_EC_TV = 9.0
TRUE_ALPHA_TV = 0.3
TARGET_R2 = 0.9
CURVE_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
CONFIDENCE_LEVEL = 0.90

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


def _real_df() -> pd.DataFrame:
  cache = os.path.join(HERE, '.cache_geo_media_rf.csv')
  if not os.path.exists(cache):
    pd.read_csv(
        'https://raw.githubusercontent.com/google/meridian/refs/heads/main/'
        'meridian/data/simulated_data/csv/geo_media_rf.csv'
    ).to_csv(cache, index=False)
  return pd.read_csv(cache)


def build_scenario(seed: int):
  from data_simulator import SimulationConfig
  import realistic_baseline as rb
  real_df = _real_df()

  def build(saturation_frequency_tv=None, r2=TARGET_R2):
    overrides = dict(BASE_OVERRIDES, seed_num=seed)
    if saturation_frequency_tv is not None:
      overrides['saturation_frequency'] = {
          'TV': saturation_frequency_tv, 'Display': 4.0}
    cfg = SimulationConfig.from_dict(overrides)
    sim, data, gt = rb.build_real_augmented_realistic(
        cfg, real_df,
        rf_source_map={'TV': 'Channel3'},
        plain_source_map={'Display': 'Channel2'},
        target_oracle_r2=r2, n_knots=8,
    )
    return cfg, sim, data, gt

  _, sim_base, _, _ = build(r2=None)
  sat_freq = 4.0 * TRUE_EC_TV / float(sim_base.ec_m.numpy()[0])
  cfg, sim, data, gt = build(sat_freq)
  assert abs(float(sim.ec_m.numpy()[0]) - TRUE_EC_TV) < 0.05
  assert abs(float(sim.alpha_m.numpy()[0]) - TRUE_ALPHA_TV) < 1e-6
  return cfg, sim, data, gt


def true_outcome_ratio(cfg, sim, channel_idx: int, multiplier: float) -> float:
  from meridian.model import adstock_hill
  x = sim.transformed_ipc_gtm.numpy()[:, :, channel_idx:channel_idx + 1]
  ec = float(sim.ec_m.numpy()[channel_idx])
  alpha = float(sim.alpha_m.numpy()[channel_idx])
  beta_g = sim.beta_gm.numpy()[:, channel_idx]

  def outcome(mult: float) -> float:
    m = tf.constant(x * mult, dtype=tf.float32)
    adstocked = adstock_hill.AdstockTransformer(
        alpha=tf.constant([alpha], tf.float32),
        max_lag=cfg.max_lag, n_times_output=cfg.n_times).forward(m)
    hilled = adstock_hill.HillTransformer(
        ec=tf.constant([ec], tf.float32),
        slope=tf.constant([1.0], tf.float32)).forward(adstocked).numpy()[:, :, 0]
    return float((hilled * beta_g[:, None]).sum())

  return outcome(multiplier) / outcome(1.0)


def main() -> None:
  from meridian.analysis import analyzer
  from meridian.model import model
  from model_utils import build_model_spec
  from model_utils import MERIDIAN_DEFAULT_MAX_LAG

  max_lag_by_variant = {'default': MERIDIAN_DEFAULT_MAX_LAG}
  lo_q, hi_q = (1 - CONFIDENCE_LEVEL) / 2, 1 - (1 - CONFIDENCE_LEVEL) / 2

  rows = []
  for seed in SEEDS:
    cfg, sim, data, gt = build_scenario(seed)
    true_roi = float(np.asarray(gt['roi_m'])[0])
    spend = sim.cost_gtm.numpy().sum(axis=(0, 1))[0]

    ratio_at = {m: true_outcome_ratio(cfg, sim, 0, m) for m in CURVE_MULTIPLIERS}
    true_val_at = {m: true_roi * spend * ratio_at[m] for m in CURVE_MULTIPLIERS}
    true_1x = true_val_at[1.0]
    for m in CURVE_MULTIPLIERS:
      rows.append({'seed': seed, 'variant': 'truth', 'spend_multiplier': m,
                   'mean': true_val_at[m] / true_1x * 100,
                   'ci_lo': np.nan, 'ci_hi': np.nan})

    for variant in ('default', 'ec_alpha_only'):
      spec = build_model_spec(
          variant, sim, cfg, media_prior_type='roi', knots=8,
          max_lag=max_lag_by_variant.get(variant))
      idata = az.from_netcdf(os.path.join(
          MODELS_DIR, f'seed_{seed}_{variant}_inference_data.nc'))
      mmm = model.Meridian(input_data=data, model_spec=spec, inference_data=idata)
      az_obj = analyzer.Analyzer(mmm)
      for m in CURVE_MULTIPLIERS:
        inc = az_obj.incremental_outcome(
            use_posterior=True, scaling_factor0=0.0, scaling_factor1=m,
            inverse_transform_outcome=True, include_non_paid_channels=False)
        draws = np.asarray(inc)[..., 0].flatten() / true_1x * 100
        rows.append({
            'seed': seed, 'variant': variant, 'spend_multiplier': m,
            'mean': float(np.mean(draws)),
            'ci_lo': float(np.quantile(draws, lo_q)),
            'ci_hi': float(np.quantile(draws, hi_q)),
        })
      del mmm, az_obj

  df = pd.DataFrame(rows)
  df.to_csv(os.path.join(
      HERE, 'fitted_models', 'scratch_check_mroi_recovery_r90',
      'per_seed_response_curve_index.csv'), index=False)

  style = {
      'truth': ('Truth (known DGP)', TRUTH, 2.6),
      'default': ('Meridian default prior', DEFAULT, 2.2),
      'ec_alpha_only': ('Reach-informed prior', INFORMED, 2.2),
  }
  fig, axes = plt.subplots(1, 3, figsize=(13.33, 4.6), sharey=False)
  for col, seed in enumerate(SEEDS):
    ax = axes[col]
    sub = df[df.seed == seed]
    for variant, (label, color, lw) in style.items():
      s = sub[sub.variant == variant].sort_values('spend_multiplier')
      is_truth = variant == 'truth'
      ax.plot(s.spend_multiplier, s['mean'], marker='o', color=color, lw=lw,
              label=label if col == 0 else None, markersize=4,
              zorder=5 if is_truth else 3,
              linestyle=TRUTH_DASH if is_truth else '-')
      if not is_truth:
        ax.fill_between(s.spend_multiplier, s.ci_lo, s.ci_hi,
                         color=color, alpha=0.18, lw=0)

    wide = sub.pivot(index='spend_multiplier', columns='variant', values='mean')
    gap = wide['default'] - wide['truth']
    sign_change = np.where(np.diff(np.sign(gap.values)) != 0)[0]
    if sign_change.size:
      k = sign_change[0]
      x0, x1 = gap.index[k], gap.index[k + 1]
      g0, g1 = gap.values[k], gap.values[k + 1]
      cross = x0 + (x1 - x0) * abs(g0) / (abs(g0) + abs(g1))
      ax.axvline(cross, color='#666666', lw=1.0, ls='-.', zorder=1)
      ax.annotate(f'cross {cross:.1f}×', xy=(cross, ax.get_ylim()[1] * 0.06),
                  fontsize=9, color='#444444')

    ax.set_title(f'seed {seed}', fontsize=12, color='#555555')
    ax.set_xlabel('Spend, as a multiple of today', fontsize=10)
    if col == 0:
      ax.set_ylabel('TV incremental outcome index\n(100 = truth at 1x)')

  fig.legend(*axes[0].get_legend_handles_labels(),
             loc='upper center', ncol=3, fontsize=10, frameon=False,
             bbox_to_anchor=(0.5, 1.08))
  fig.suptitle(
      'Response curve per seed (unpooled) — oracle_r2=0.9, alpha_m=0.3',
      fontsize=13, fontweight='bold', y=1.16)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
  print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
  main()
