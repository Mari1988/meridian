"""Helpers shared across the synthetic-data demo notebooks.

Covers building a Meridian `InputData` from simulated data, slicing an
ArviZ posterior summary table by parameter name, constructing `ec_m` prior
variants for the prior-recovery case studies, and Hill-curve diagnostics.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from meridian import constants
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
}


def build_model_spec(variant: str, sim, config, rng=None) -> spec.ModelSpec:
  """Builds a `ModelSpec` for one of `PRIOR_VARIANTS`'s keys."""
  prior = PRIOR_VARIANTS[variant](sim, config, rng)
  kwargs = dict(
      media_prior_type='roi',
      media_effects_dist='log_normal',
      knots=8,
      max_lag=config.max_lag,
  )
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
