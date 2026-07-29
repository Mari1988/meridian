"""Helpers shared across the synthetic-data demo notebooks.

Covers building a Meridian `InputData` from simulated data, slicing an
ArviZ posterior summary table by parameter name, constructing `ec_m` prior
variants for the prior-recovery case studies, and Hill-curve diagnostics.
"""

import arviz as az
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from meridian import constants
from meridian.analysis import analyzer
from meridian.data import data_frame_input_data_builder
from meridian.data import input_data as input_data_lib
from meridian.model import adstock_hill
from meridian.model import prior_distribution
from meridian.model import spec

from data_simulator import GeoMediaDataSimulator


def build_input_data(
    df: pd.DataFrame,
    channel_names: list[str],
    control_col_names: list[str],
) -> input_data_lib.InputData:
  """Builds a Meridian `InputData` from a simulated geo x time DataFrame."""
  builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
      kpi_type='non_revenue'
  )
  builder = (
      builder.with_kpi(df, kpi_col='conversions')
      .with_revenue_per_kpi(df, revenue_per_kpi_col='revenue_per_conversion')
      .with_population(df)
      .with_controls(df, control_cols=control_col_names)
      .with_media(
          df,
          media_cols=[f'{channel}_impression' for channel in channel_names],
          media_spend_cols=[f'{channel}_spend' for channel in channel_names],
          media_channels=channel_names,
      )
  )
  return builder.build()


def filter_param_summary(
    summary_df: pd.DataFrame, prefix: str, suffix: str | None = None
) -> pd.DataFrame:
  """Returns ArviZ summary rows whose `index` matches the prefix/suffix."""
  cond = summary_df['index'].str.startswith(prefix)
  if suffix is not None:
    cond &= summary_df['index'].str.endswith(suffix)
  return summary_df.loc[cond, :]


def roi_posterior_summary(
    mmm, channel_names: list[str], hdi_prob: float = 0.9
) -> pd.DataFrame:
  """ArviZ-style posterior summary for `roi_m`, computed post-hoc.

  Under `media_prior_type='roi'` (or `'mroi'`/`'contribution'`), `roi_m` is
  sampled directly and lands in `mmm.inference_data.posterior` already, so
  `az.summary(mmm.inference_data, ...)` picks it up on its own. Under
  `'coefficient'`, `beta_m` is the free sampled parameter instead and `roi_m`
  is never added to `inference_data` -- it has to be recomputed from the
  fitted `beta_gm`/`ec_m`/`alpha_m`/spend via `analyzer.Analyzer.roi()`. This
  wraps that computation into the same `az.summary`-shaped table (`index`,
  `mean`, `sd`, `hdi_{lo}%`, `hdi_{hi}%`, `mcse_mean`, `mcse_sd`, `ess_bulk`,
  `ess_tail`, `r_hat`) so it's a drop-in replacement for
  `filter_param_summary(summary, 'roi_m')` regardless of `media_prior_type`.

  Args:
    mmm: A fitted `Meridian` model (posterior sampled).
    channel_names: Media channel names, in the same order as the model's
      media channel dimension.
    hdi_prob: HDI probability mass, matching the rest of the notebook's
      `az.summary` calls.

  Returns:
    A summary DataFrame with one row per channel.
  """
  roi_draws = analyzer.Analyzer(mmm).roi().numpy()  # (chains, draws, channels)
  idata = az.from_dict(
      posterior={'roi_m': roi_draws},
      dims={'roi_m': ['channel']},
      coords={'channel': channel_names},
  )
  return az.summary(idata, extend=True, hdi_prob=hdi_prob).reset_index()


def build_comparison_table(
    fitted_mmms: dict,
    config,
    sim,
    ground_truth: dict,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
  """One row per (channel, param), one column per fitted variant.

  Each cell is `"mean (hdi_lo-hdi_hi) mark"`, where `mark` is a check mark
  if the true value falls inside that variant's HDI at `hdi_prob`, else a
  cross mark. `roi_m` uses `roi_posterior_summary` (post-hoc, works
  regardless of `media_prior_type`); `alpha_m`/`ec_m` use `az.summary`
  directly since they're always native posterior variables.

  Args:
    fitted_mmms: `{variant_name: fitted Meridian model}`.
    config: The `SimulationConfig` used to build every model in `fitted_mmms`
      (must share the same `channel_names`).
    sim: The simulated `GeoMediaDataSimulator` used to build `config`'s data
      (source of true `alpha_m`/`ec_m`).
    ground_truth: `sim.compute_ground_truth()`'s output (source of true
      `roi_m`).
    hdi_prob: HDI probability mass.

  Returns:
    A wide comparison `DataFrame`.
  """
  true_vals = {
      'alpha_m': dict(zip(config.channel_names, sim.alpha_m.numpy())),
      'ec_m': dict(zip(config.channel_names, sim.ec_m.numpy())),
      'roi_m': dict(zip(config.channel_names, ground_truth['roi_m'])),
  }
  variant_summaries = {}
  for variant, mmm in fitted_mmms.items():
    az_summary = az.summary(
        mmm.inference_data, extend=True, hdi_prob=hdi_prob
    ).reset_index()
    variant_summaries[variant] = {
        'alpha_m': az_summary,
        'ec_m': az_summary,
        'roi_m': roi_posterior_summary(
            mmm, config.channel_names, hdi_prob=hdi_prob
        ),
    }

  rows = []
  for channel in config.channel_names:
    for param in ['alpha_m', 'ec_m', 'roi_m']:
      true_val = true_vals[param][channel]
      row = {
          'channel': channel,
          'param': param,
          'true': round(float(true_val), 3),
      }
      for variant in fitted_mmms:
        summary = variant_summaries[variant][param]
        hdi_cols = sorted(c for c in summary.columns if c.startswith('hdi_'))
        lo_col, hi_col = hdi_cols[0], hdi_cols[1]
        record = summary.loc[summary['index'] == f'{param}[{channel}]'].iloc[0]
        passed = record[lo_col] <= true_val <= record[hi_col]
        mark = '✓' if passed else '✗'
        row[variant] = (
            f"{record['mean']:.3f} ({record[lo_col]:.2f}-{record[hi_col]:.2f})"
            f' {mark}'
        )
      rows.append(row)
  return pd.DataFrame(rows)


def build_reach_based_ec_prior(
    sim,
    config,
    audience_noise_scale: float = 0.0,
    base_scale: float = 0.1,
    rng=None,
) -> tfp.distributions.Distribution:
  """LogNormal `ec_m` prior centered on the audience/frequency-derived value.

  Uses `sim.ec_m` (already derived in `simulate_adstock_hill_params()` from
  audience size, frequency, and population -- not from any fitted model) as
  the prior mean, with a spread that reflects uncertainty in the underlying
  planning assumptions: a real advertiser's stated confidence in their
  audience/reach/frequency inputs should translate into how tight a prior
  they're entitled to claim, so the prior's `scale` grows with
  `audience_noise_scale` rather than staying fixed regardless of it.

  Args:
    sim: A simulated `GeoMediaDataSimulator` (post `simulate_adstock_hill_
      params()`).
    config: The `SimulationConfig` used to build `sim`.
    audience_noise_scale: If > 0, the assumed `ec_m` used to center the prior
      is multiplicatively perturbed by `LogNormal(0, audience_noise_scale)`
      noise before use -- emulating an advertiser's independent audience/
      reach planning assumption being only approximately, not exactly,
      right. `0.0` (the default) reproduces the exact ground truth. The
      prior's `scale` (see `base_scale`) widens along with this value, so a
      noisier assumption also yields an appropriately less confident prior.
    base_scale: Residual log-scale uncertainty assumed even when
      `audience_noise_scale` is `0` (e.g. measurement error in population/
      frequency inputs that persists regardless). Combined with
      `audience_noise_scale` via sqrt-sum-of-squares, treating the two as
      independent log-normal uncertainty sources.
    rng: `np.random.Generator` for the perturbation draw. Defaults to a
      fresh, unseeded generator if `audience_noise_scale > 0` and no `rng`
      is given.

  Returns:
    A batched `LogNormal` distribution over `ec_m`.
  """
  assumed_ec_m = sim.ec_m.numpy()
  if audience_noise_scale:
    rng = rng or np.random.default_rng()
    assumed_ec_m = assumed_ec_m * rng.lognormal(
        mean=0.0, sigma=audience_noise_scale, size=assumed_ec_m.shape
    )
  ec50_log_loc = np.log(assumed_ec_m)
  prior_scale = float(np.sqrt(base_scale**2 + audience_noise_scale**2))
  return tfp.distributions.LogNormal(
      loc=[float(x) for x in ec50_log_loc],
      scale=[prior_scale] * config.n_imp_channels,
      name=constants.EC_M,
  )


def build_alpha_prior(
    sim, config, base_scale: float | dict[str, float] = 0.1
) -> tfp.distributions.Distribution:
  """TruncatedNormal `alpha_m` prior centered on the ground-truth `sim.alpha_m`.

  Analogous to `build_reach_based_ec_prior` but bounded to `[0, 1]` since
  `alpha_m` is a retention rate rather than a `LogNormal`-suited positive
  scale. Represents an advertiser anchoring on category/channel adstock-decay
  benchmarks (e.g. TV creative carrying over several weeks vs. digital's
  near-immediate decay) rather than knowing the exact simulated value with
  certainty -- `base_scale` is the residual uncertainty around that
  benchmark.

  Args:
    sim: A simulated `GeoMediaDataSimulator` (post `simulate_adstock_hill_
      params()`).
    config: The `SimulationConfig` used to build `sim`.
    base_scale: Standard deviation of the truncated normal around each
      channel's true `alpha_m`. Either one value applied to every channel, or
      a `{channel_name: scale}` dict to tighten/loosen individual channels
      (e.g. for pressure-testing whether a channel's poor `alpha_m` recovery
      is a weak-prior issue or a likelihood/identifiability issue).

  Returns:
    A batched `TruncatedNormal` distribution over `alpha_m`, bounded to
    `[0, 1]`.
  """
  assumed_alpha_m = sim.alpha_m.numpy()
  if isinstance(base_scale, dict):
    scales = [base_scale[ch] for ch in config.channel_names]
  else:
    scales = [base_scale] * config.n_imp_channels
  return tfp.distributions.TruncatedNormal(
      loc=[float(x) for x in assumed_alpha_m],
      scale=scales,
      low=0.0,
      high=1.0,
      name=constants.ALPHA_M,
  )


def build_slope_prior(
    sim, config, base_scale: float = 0.1
) -> tfp.distributions.Distribution:
  """LogNormal `slope_m` prior centered on the ground-truth `sim.slope_m`.

  `slope_m` is Meridian's Hill curve-shape exponent -- the same role as
  Robyn's `alpha` hyperparameter (see `SimulationConfig.slope_range` in
  `data_simulator.py`; not to be confused with this simulator's own
  `alpha_m`/adstock retention, which is Robyn's `theta`). Meridian's own
  default `slope_m` prior is a hard `Deterministic(1.0)` for plain media
  channels -- not just uninformative, but literally unfittable -- so using
  this prior means a notebook is explicitly opting into estimating it,
  anchored on an advertiser's planning assumption about each channel's
  response-curve shape (e.g. TV needing a real frequency threshold before
  eliciting response, vs. digital's more immediate, concave response).
  Analogous to `build_reach_based_ec_prior` for `ec_m`.

  Args:
    sim: A simulated `GeoMediaDataSimulator` (post `simulate_adstock_hill_
      params()`).
    config: The `SimulationConfig` used to build `sim`.
    base_scale: Log-scale standard deviation around each channel's true
      `slope_m`.

  Returns:
    A batched `LogNormal` distribution over `slope_m`.
  """
  assumed_slope_m = sim.slope_m.numpy()
  return tfp.distributions.LogNormal(
      loc=[float(x) for x in np.log(assumed_slope_m)],
      scale=[base_scale] * config.n_imp_channels,
      name=constants.SLOPE_M,
  )


def build_eta_prior(
    sim, config, base_scale: float = 0.03
) -> tfp.distributions.Distribution:
  """TruncatedNormal `eta_m` prior centered on the ground-truth `sim.eta_m`.

  `eta_m` governs the hierarchical spread of geo-level media effects
  (`beta_gm = exp(beta_m + eta_m * dev_g)` under `media_effects_dist=
  'log_normal'`). Unlike `ec_m`/`alpha_m`, none of the `PRIOR_VARIANTS` above
  inform it -- under `media_prior_type='coefficient'` it's left at
  Meridian's generic default `HalfNormal(1.0)`, uninformed by anything about
  the true DGP. Because `exp` is convex, an overestimated `eta_m` doesn't
  just widen geo-level uncertainty symmetrically -- it multiplicatively
  inflates the population-weighted *average* effect (and hence the derived
  `roi_m`), since a few geos with large positive `dev_g` draws end up with
  disproportionately large `beta_gm` that dominate the population-weighted
  sum. This lets us test whether that specific failure mode (observed for
  `'coefficient'`'s Display fit: fitted `eta_m` ~4.0 vs a true value of
  ~0.165, well-converged) is fixable the same way `ec_m`/`alpha_m` are, by
  anchoring the prior on the true simulated value.

  Args:
    sim: A simulated `GeoMediaDataSimulator` (post `simulate_coefficients()`,
      which sets `sim.eta_m`).
    config: The `SimulationConfig` used to build `sim`.
    base_scale: Standard deviation of the truncated normal around each
      channel's true `eta_m`.

  Returns:
    A batched `TruncatedNormal` distribution over `eta_m`, bounded to
    `[0, 10]` (`eta_m` has no natural upper bound the way `alpha_m` does).
  """
  assumed_eta_m = sim.eta_m.numpy()
  scales = [base_scale] * config.n_imp_channels
  return tfp.distributions.TruncatedNormal(
      loc=[float(x) for x in assumed_eta_m],
      scale=scales,
      low=0.0,
      high=10.0,
      name=constants.ETA_M,
  )


# Category/channel adstock-decay benchmark ranges for `ec_alpha_range` --
# deliberately *not* read from `sim.alpha_m`, unlike `build_alpha_prior`.
# These represent what a practitioner might plausibly believe from general
# channel-type experience (TV creative persisting longer than digital) with
# no knowledge of this specific simulation's exact `adstock_retention_range`
# -- each is a defensible guess that overlaps, but doesn't exactly match, the
# true range (TV (0.6, 0.8), Display (0.0, 0.3), Social (0.1, 0.4) in
# `SimulationConfig.adstock_retention_range`), to test whether an
# imperfect-but-plausible range still helps.
ALPHA_RANGE_PRIOR = {
    'TV': (0.4, 0.7),
    'Display': (0.0, 0.2),
    'Social': (0.2, 0.5),
}


def build_alpha_range_prior(
    config, alpha_ranges: dict[str, tuple[float, float]] = ALPHA_RANGE_PRIOR
) -> tfp.distributions.Distribution:
  """Uniform `alpha_m` prior over a category-benchmark range per channel.

  Unlike `build_alpha_prior`, this doesn't read the simulator's ground truth
  at all -- it represents an advertiser who knows a plausible *range* for
  each channel's adstock decay (e.g. "TV creative decays somewhere between
  0.4 and 0.7") but, realistically, doesn't know the precise value well
  enough to anchor a `TruncatedNormal` mean on it.

  Args:
    config: The `SimulationConfig` (only `channel_names` is used).
    alpha_ranges: `{channel_name: (low, high)}` bounds for each channel's
      `Uniform` prior. Defaults to `ALPHA_RANGE_PRIOR`.

  Returns:
    A batched `Uniform` distribution over `alpha_m`.
  """
  low = [alpha_ranges[ch][0] for ch in config.channel_names]
  high = [alpha_ranges[ch][1] for ch in config.channel_names]
  return tfp.distributions.Uniform(low=low, high=high, name=constants.ALPHA_M)


# Each variant is a function of (sim, config, rng) -> PriorDistribution |
# None. `None` means "no override", i.e. Meridian's own defaults for that
# param. `rng` is only used by `ec_noisy`; the others ignore it.
PRIOR_VARIANTS = {
    'default': lambda sim, config, rng=None: None,
    'ec_only': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(sim, config)
        )
    ),
    'ec_noisy': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(
                sim, config, audience_noise_scale=0.25, rng=rng
            )
        )
    ),
    'ec_alpha_only': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(sim, config),
            alpha_m=build_alpha_prior(sim, config),
        )
    ),
    # Pressure test for `ec_alpha_only`'s poor `alpha_m[Social]` recovery:
    # tightens just Social's `alpha_m` prior scale (0.1 -> 0.03, roughly a
    # 90% CI of [0.32, 0.44] around the true 0.379) while leaving TV/Display
    # at the same scale as `ec_alpha_only`. If the posterior still drifts
    # away from that much tighter prior, it confirms Social's `alpha_m`
    # miss is a likelihood/identifiability issue (a near-flat direct
    # likelihood for adstock decay, dominated by confounding with
    # `roi_m`/`beta_m`) rather than the prior simply not being informative
    # enough.
    'ec_alpha_social_tight': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(sim, config),
            alpha_m=build_alpha_prior(
                sim,
                config,
                base_scale={'TV': 0.03, 'Display': 0.03, 'Social': 0.03},
            ),
        )
    ),
    # A more realistic advertiser: knows a plausible category-benchmark
    # *range* for adstock decay per channel (not the exact simulated value,
    # and not even a best-guess mean to anchor a Normal on -- just bounds),
    # via `ALPHA_RANGE_PRIOR`. `ec_m` stays exact (as in `ec_only`) so this
    # isolates the effect of the `alpha_m` prior's shape/informativeness.
    'ec_alpha_range': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(sim, config),
            alpha_m=build_alpha_range_prior(config),
        )
    ),
    # Pressure test for the `eta_m`-driven `roi_m` blow-up observed under
    # `media_prior_type='coefficient'` (Display's fitted `eta_m` ~4.0 vs
    # true ~0.165): adds a tight, truth-centered `eta_m` prior on top of
    # `ec_alpha_social_tight`'s tight `ec_m`/`alpha_m` priors, to see whether
    # informing all three shape/dispersion parameters is enough to pull
    # `roi_m` inside its 90% HDI of the true value.
    'ec_alpha_eta_tight': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(sim, config),
            alpha_m=build_alpha_prior(
                sim,
                config,
                base_scale={'TV': 0.03, 'Display': 0.03, 'Social': 0.03},
            ),
            eta_m=build_eta_prior(sim, config, base_scale=0.03),
        )
    ),
    # `ec_alpha_social_tight` only tightens `alpha_m` (scale 0.1 -> 0.03,
    # all channels despite the name) -- `ec_m` there is still at
    # `build_reach_based_ec_prior`'s looser default `base_scale=0.1`. This
    # variant tightens `ec_m` to the same 0.03 scale too, for all channels,
    # to see whether matching `ec_m`'s informativeness to `alpha_m`'s
    # improves on `ec_alpha_social_tight`'s AKS/unmatched-DGP recovery
    # (7/9 cells inside their 95% HDI, notably still missing TV's `alpha_m`).
    'ec_alpha_both_tight': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(sim, config, base_scale=0.03),
            alpha_m=build_alpha_prior(
                sim,
                config,
                base_scale={'TV': 0.03, 'Display': 0.03, 'Social': 0.03},
            ),
        )
    ),
    # Extends `ec_alpha_only` with an informed `slope_m` prior, for DGPs
    # built with a non-default `SimulationConfig.slope_range` (Meridian's
    # own default `slope_m` prior is a hard `Deterministic(1.0)`, so this
    # is the only variant that lets the fitted model represent anything but
    # a concave Hill curve). Tests whether recovering the extra shape
    # parameter is what it takes to fix `ec_m`/`roi_m` recovery once the
    # true curve is a genuine S-curve, not just informing `ec_m`/`alpha_m`
    # under a fixed-at-1 `slope_m` assumption.
    'ec_alpha_slope_only': (
        lambda sim, config, rng=None: prior_distribution.PriorDistribution(
            ec_m=build_reach_based_ec_prior(sim, config),
            alpha_m=build_alpha_prior(sim, config),
            slope_m=build_slope_prior(sim, config),
        )
    ),
}


def build_model_spec(
    variant: str,
    sim,
    config,
    rng=None,
    media_prior_type: str = 'roi',
    knots: int | None = 8,
    enable_aks: bool = False,
) -> spec.ModelSpec:
  """Builds a `ModelSpec` for one of `PRIOR_VARIANTS`'s keys.

  Args:
    variant: A key into `PRIOR_VARIANTS`.
    sim: A simulated `GeoMediaDataSimulator` (post `simulate_adstock_hill_
      params()`).
    config: The `SimulationConfig` used to build `sim`.
    rng: Passed through to the `PRIOR_VARIANTS` builder (only used by
      `ec_noisy`).
    media_prior_type: `ModelSpec.media_prior_type` -- `'roi'` (the default)
      samples `roi_m` directly and derives `beta_m` from it, `ec_m`,
      `alpha_m`, and `slope_m`; `'coefficient'` samples `beta_m` directly
      (default prior `HalfNormal(5.0)`) with no such dependency on the
      Hill/adstock shape parameters. `ec_m`/`alpha_m` prior variants above are
      unaffected by this choice either way.
    knots: `ModelSpec.knots` for the time-effects spline. Default `8` is a
      heavily smoothed fit, *not* Meridian's own default. The DGP's own
      time-varying baseline (`mu_t` in `data_simulator.py`) uses
      `n_knots_simul = config.n_times` -- i.e. one independent, unsmoothed
      `Normal(0, 2.0)` shock per week -- so `8` is a large flexibility
      mismatch against the true generating process. `None` reproduces
      Meridian's actual default (one knot per time period, per
      `spec.ModelSpec`'s own docstring) -- maximum flexibility, matching the
      DGP's mismatch structurally without needing to alter the DGP itself.
      Ignored (must be left at `None`) when `enable_aks=True`.
    enable_aks: `ModelSpec.enable_aks` -- use Meridian's Automatic Knot
      Selection instead of a fixed `knots` count. Mutually exclusive with
      `knots` (Meridian requires `knots=None` when this is `True`).
  """
  prior = PRIOR_VARIANTS[variant](sim, config, rng)
  kwargs = dict(
      media_prior_type=media_prior_type,
      media_effects_dist='log_normal',
      max_lag=config.max_lag,
  )
  if enable_aks:
    kwargs['enable_aks'] = True
  else:
    kwargs['knots'] = knots
  if prior is not None:
    kwargs['prior'] = prior
  return spec.ModelSpec(**kwargs)


def hill_value(x, ec, slope=None) -> np.ndarray:
  """Evaluates Meridian's Hill saturation formula, reusing `HillTransformer`.

  Args:
    x: A scalar, or a 1-D array of media values. If 1-D, the same grid is
      applied to every channel in `ec`.
    ec: Half-saturation value(s): a scalar or a 1-D array of length
      `n_channels`.
    slope: Hill slope(s), matching `ec`'s shape. Defaults to `1.0` for every
      channel (as used by `data_simulator.py`'s ground-truth `slope_m`).

  Returns:
    `Hill(x)`. Shape `(n_channels,)` if `x` is a scalar, else
    `(len(x), n_channels)`.
  """
  ec_arr = np.atleast_1d(np.asarray(ec, dtype=np.float32))
  slope_arr = (
      np.ones_like(ec_arr)
      if slope is None
      else np.broadcast_to(np.asarray(slope, dtype=np.float32), ec_arr.shape)
  )
  x_arr = np.atleast_1d(np.asarray(x, dtype=np.float32))
  media = tf.constant(
      np.broadcast_to(x_arr[:, np.newaxis], (x_arr.shape[0], ec_arr.shape[0]))[
          np.newaxis, :, :
      ]
  )  # shape (1 geo, n_points, n_channels)
  transformer = adstock_hill.HillTransformer(
      ec=tf.constant(ec_arr), slope=tf.constant(slope_arr)
  )
  result = transformer.forward(media).numpy()[0]  # (n_points, n_channels)
  return result[0] if np.ndim(x) == 0 else result


def ceiling_fraction_at_median(ec, slope=None) -> np.ndarray:
  """Fraction of ceiling (max) effect achieved at a channel's own historical
  median execution level (`x=1` in Meridian's population-scaled units)."""
  return hill_value(1.0, ec, slope)


def posterior_param_mean(mmm, param_name: str) -> np.ndarray:
  """Posterior-mean array for `param_name` (e.g. `'ec_m'`, `'slope_m'`),
  averaged over chains and draws."""
  return mmm.inference_data.posterior[param_name].values.mean(axis=(0, 1))


def build_simulated_input(config):
  """Runs the full simulator pipeline and returns (sim, data, ground_truth)."""
  sim = GeoMediaDataSimulator(config)
  sim.simulate_population()
  sim.simulate_controls()
  sim.simulate_media(verbose=False)
  sim.simulate_cost_and_unit_value()
  sim.simulate_intercepts()
  sim.simulate_coefficients()
  sim.simulate_adstock_hill_params()
  sim.transform_media()
  sim.calibrate_channel_effects()
  sim.generate_kpi_and_revenue()
  df = sim.to_dataframe()
  ground_truth = sim.compute_ground_truth(verbose=False)
  data = build_input_data(df, config.channel_names, sim.control_col_names)
  return sim, data, ground_truth


def build_real_augmented_input(
    config,
    real_df: pd.DataFrame,
    rf_source_map: dict[str, str],
    plain_source_map: dict[str, str],
):
  """Runs the simulator pipeline with real-data-sourced, rescaled media.

  Identical to `build_simulated_input` except Section 1 (population) and
  Section 3 (media) are replaced by `simulate_population_from_real`/
  `align_time_index_to_real`/`simulate_media_from_real`: real per-geo
  population/dates and each channel's actual real-data reach/frequency (or
  impression) geo x time texture, rescaled to `config`'s assumed
  `current_reach_frac`/`frequency_range` target per channel. Every other
  step (controls, coefficients, adstock/Hill `ec_m` derivation, ROI
  calibration, KPI generation) is unchanged, so the same known-ground-truth
  recovery check used by `build_simulated_input` still applies.

  Args:
    config: The `SimulationConfig` to build the scenario from.
    real_df: A geo x time DataFrame with real population/reach/frequency/
      impression columns (e.g. the demo's `geo_media_rf.csv`), following
      that dataset's `<Channel>_reach`/`<Channel>_frequency`/`<Channel>_
      impression` naming convention. Its week count must match
      `config.n_times`.
    rf_source_map: `{config_channel_name: real_channel_prefix}` for channels
      backed by a real `(reach, frequency)` pair.
    plain_source_map: `{config_channel_name: real_channel_prefix}` for
      channels backed by only a real impression column.

  Returns:
    `(sim, data, ground_truth)`, matching `build_simulated_input`.
  """
  sim = GeoMediaDataSimulator(config)
  sim.simulate_population_from_real(real_df)
  sim.align_time_index_to_real(real_df)
  sim.simulate_controls()
  sim.simulate_media_from_real(
      real_df, rf_source_map, plain_source_map, verbose=False
  )
  sim.simulate_cost_and_unit_value()
  sim.simulate_intercepts()
  sim.simulate_coefficients()
  sim.simulate_adstock_hill_params()
  sim.transform_media()
  sim.calibrate_channel_effects()
  sim.generate_kpi_and_revenue()
  df = sim.to_dataframe()
  ground_truth = sim.compute_ground_truth(verbose=False)
  data = build_input_data(df, config.channel_names, sim.control_col_names)
  return sim, data, ground_truth
