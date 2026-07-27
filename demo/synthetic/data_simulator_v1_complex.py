"""Synthetic geo x time media data simulator (archival pre-simplification copy).

This is a snapshot of `data_simulator.py` kept for reference before it was
simplified (fewer stacked media-execution layers) to support a cleaner,
easier-to-defend "default priors vs. informed priors" case study. Import this
module directly (not via `data_simulator`) if you need the older, richer
media-execution mechanics (seasonal jitter, ramp-smoothed flighting, trend,
promo spikes).

Reproduces the causal, ground-truth-parameterized data-generating process
used by `simulate_media_data_reach_frequency_state_geo.ipynb`: geo
populations, control variables, reach x frequency media, adstock/Hill
transforms, and a KPI/revenue outcome with known ground-truth coefficients,
so posterior estimates from a fitted Meridian model can be checked against
the values used to generate the data.

Each `simulate_*` / `transform_media` / `generate_kpi_and_revenue` /
`to_dataframe` / `compute_ground_truth` method on `GeoMediaDataSimulator`
reproduces one numbered section of the original notebook and stores its
outputs as attributes on `self`, so a notebook can call them step by step and
inspect or plot intermediate tensors in between.
"""

import dataclasses
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr

from meridian.model import adstock_hill
from meridian.model import knots
from meridian.model import transformers

# U.S. Census Bureau, Vintage 2025 population estimates (50 states + DC).
# https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population
STATE_POPULATION = {
    'California': 39_355_309,
    'Texas': 31_709_821,
    'Florida': 23_462_518,
    'New York': 20_002_427,
    'Pennsylvania': 13_059_432,
    'Illinois': 12_719_141,
    'Ohio': 11_900_510,
    'Georgia': 11_302_748,
    'North Carolina': 11_197_968,
    'Michigan': 10_127_884,
    'New Jersey': 9_548_215,
    'Virginia': 8_880_107,
    'Washington': 8_001_020,
    'Arizona': 7_623_818,
    'Tennessee': 7_315_076,
    'Massachusetts': 7_154_084,
    'Indiana': 6_973_333,
    'Missouri': 6_270_541,
    'Maryland': 6_265_347,
    'Colorado': 6_012_561,
    'Wisconsin': 5_972_787,
    'Minnesota': 5_830_405,
    'South Carolina': 5_570_274,
    'Alabama': 5_193_088,
    'Louisiana': 4_618_189,
    'Kentucky': 4_606_864,
    'Oregon': 4_273_586,
    'Oklahoma': 4_123_288,
    'Connecticut': 3_688_496,
    'Utah': 3_538_904,
    'Nevada': 3_282_188,
    'Iowa': 3_238_387,
    'Arkansas': 3_114_791,
    'Kansas': 2_977_220,
    'Mississippi': 2_954_160,
    'New Mexico': 2_125_498,
    'Idaho': 2_029_733,
    'Nebraska': 2_018_006,
    'West Virginia': 1_766_147,
    'Hawaii': 1_432_820,
    'New Hampshire': 1_415_342,
    'Maine': 1_414_874,
    'Montana': 1_144_694,
    'Rhode Island': 1_114_521,
    'Delaware': 1_059_952,
    'South Dakota': 935_094,
    'North Dakota': 799_358,
    'Alaska': 737_270,
    'District of Columbia': 693_645,
    'Vermont': 644_663,
    'Wyoming': 588_753,
}

CHANNEL_DIM_NAME = 'channel'
CONTROL_DIM_NAME = 'control'
GEO_DIM_NAME = 'geo'
TIME_DIM_NAME = 'time'

# A control named this is treated specially by `simulate_controls()`: instead
# of an iid random draw, it's populated with the shared national
# demand-seasonality index also used to modulate media execution in
# `simulate_media()` -- making seasonality a genuine confounder (it drives
# both media timing and the KPI baseline) rather than a media-only effect.
SEASONALITY_CONTROL_NAME = 'seasonality_index'

KPI_COL_NAME = 'conversions'
POPULATION_COL_NAME = 'population'
UNIT_VALUE_COL_NAME = 'revenue_per_conversion'

SPEND_COL_SUFFIX = 'spend'
IMPRESSIONS_COL_SUFFIX = 'impression'
REACH_COL_SUFFIX = 'reach'
FREQUENCY_COL_SUFFIX = 'frequency'
CONTROL_COL_SUFFIX = 'control'


def _add_suffix(name: str, suffix: str) -> str:
  return '_'.join([name, suffix])


@dataclasses.dataclass
class SimulationConfig:
  """Tunable parameters for `GeoMediaDataSimulator`."""

  n_imp_channels: int = 3
  n_controls: int = 3
  n_times: int = 156  # 3 years of weekly data.
  seed_num: int = 1320
  channel_names: list[str] = dataclasses.field(
      default_factory=lambda: ['TV', 'Display', 'Social']
  )
  # The last control is special-cased by `simulate_controls()` -- see
  # `SEASONALITY_CONTROL_NAME` -- and shares its underlying calendar index
  # with each channel's seasonal execution multiplier, so it acts as a
  # genuine confounder rather than an independent nuisance covariate.
  control_names: list[str] = dataclasses.field(
      default_factory=lambda: [
          'sentiment_score',
          'competitor_activity_score',
          SEASONALITY_CONTROL_NAME,
      ]
  )
  # Each channel's target audience, as a fraction of a geo's own population
  # (rather than a fixed national headcount).
  target_audience_pop_frac: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.40, 'Display': 0.10, 'Social': 0.15}
  )
  # Current reach, as a fraction of each channel's target audience.
  current_reach_frac: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.20, 'Display': 0.5, 'Social': 0.9}
  )
  # Weekly frequency target (impressions per person reached), all channels.
  frequency_range: tuple[float, float] = (1.0, 5.0)
  enable_ramp: bool = False
  ramp_weeks: int = 10
  time_reach_noise_sd: float = 0.03
  frequency_noise_sd: float = 0.1
  geo_audience_heterogeneity_sd: float = 0.12

  # --- Time-varying execution multipliers layered on top of the flat
  # `current_reach_frac` target below (Section 3), each an independently
  # justified real media-planning behavior. All multipliers have mean ~1
  # over the full horizon except `flight_floor` below, which deliberately
  # pulls the realized average below the nominal `current_reach_frac`
  # target during dark flighting periods.

  # Seasonality: a single shared national demand-seasonality index -- an
  # annual (52-week) cycle anchored to actual calendar weeks, peaking at
  # `demand_seasonal_peak_week` (week 47 = the Nov/Dec holiday season by
  # default) -- also populates the `seasonality_index` control (see
  # `SEASONALITY_CONTROL_NAME`), so it drives both the KPI baseline and
  # media timing. Each channel reacts to that *same* underlying signal with
  # its own sensitivity (`seasonal_amplitude`): TV reacts most strongly
  # (holiday-driven linear-TV buying), Display moderately (promo-calendar
  # buying), Social least (always-on, comparatively flat year-round).
  demand_seasonal_peak_week: int = 47
  seasonal_amplitude: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.6, 'Display': 0.3, 'Social': 0.1}
  )

  # Flighting: channels are bought in discrete on/off bursts rather than
  # continuously (a two-state Markov chain with average "on"/"off" run
  # lengths `flight_burst_weeks`/`flight_dark_weeks`). `flight_floor` is the
  # residual multiplier during a dark period -- never exactly 0, since
  # there's usually some always-on baseline activity. TV is bought in
  # short, sharp flights with long dark gaps (typical linear-TV campaign
  # buying); Display flights around promo weeks with a moderate floor;
  # Social is modeled as effectively always-on, with a high floor, matching
  # always-on programmatic/social buying. Its burst length isn't set here --
  # `default_factory` lambdas can't see sibling fields -- but tied to
  # `n_times` in `__post_init__` below, so it scales with the horizon
  # instead of being hardcoded to a specific `n_times` value.
  flight_burst_weeks: dict[str, int] = dataclasses.field(
      default_factory=lambda: {'TV': 8, 'Display': 2, 'Social': 2}
  )
  flight_dark_weeks: dict[str, int] = dataclasses.field(
      default_factory=lambda: {'TV': 8, 'Display': 2, 'Social': 2}
  )
  flight_floor: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.2, 'Display': 0.5, 'Social': 0.5}
  )
  # How strongly each channel's flighting Markov chain (above) is pulled
  # toward the shared seasonal calendar (`seasonal_index_t`, see
  # `seasonal_amplitude` above): at 0, burst/dark transitions are calendar-
  # blind and `flight_burst_weeks`/`flight_dark_weeks` are the exact average
  # on/off durations (the old behavior); above 0, transitioning "on" gets
  # more likely and "off" less likely as the calendar approaches
  # `demand_seasonal_peak_week` (and vice versa near the trough), so those
  # durations become baselines that stretch near-peak and compress off-peak
  # -- i.e. bursts cluster around the same calendar window every year,
  # matching how real campaigns are deliberately planned around seasonal
  # demand rather than firing at random. TV is most calendar-driven
  # (holiday-driven linear-TV buying), Display moderately so (promo-
  # calendar buying); Social is small since it's already modeled as
  # always-on, with just a slight seasonal lean on top.
  flight_seasonal_bias: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.6, 'Display': 0.1, 'Social': 0.1}
  )
  # Number of weeks over which a flight's on/off transition ramps rather than
  # jumping instantly between `flight_floor` and 1.0 -- real campaigns are
  # trafficked/wound down gradually, not switched on a single week. Applied
  # as a symmetric triangular smoothing kernel over the raw Markov state.
  flight_ramp_weeks: dict[str, int] = dataclasses.field(
      default_factory=lambda: {'TV': 2, 'Display': 1, 'Social': 1}
  )

  # Year-to-year jitter on the shared seasonality signal (`seasonal_index_t`):
  # without it, every calendar year produces an identical seasonal curve,
  # which reads as obviously synthetic. `seasonal_peak_jitter_weeks_sd` shifts
  # each year's peak week by a per-year Normal draw; `seasonal_amplitude_jitter_sd`
  # scales that year's whole seasonal amplitude by a per-year draw around 1.0.
  seasonal_peak_jitter_weeks_sd: float = 3.0
  seasonal_amplitude_jitter_sd: float = 0.15

  # Trend: slow secular drift in execution intensity over the full horizon
  # (e.g. budget shifting away from linear TV toward social over time),
  # expressed as total fractional change from the first to the last week.
  trend_pct_total: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': -0.15, 'Display': 0.05, 'Social': 0.25}
  )

  # Autocorrelation of the additive week-to-week reach noise (an AR(1)
  # process instead of iid draws, since real execution noise persists
  # across adjacent weeks rather than resetting every week).
  reach_noise_ar1_phi: float = 0.4
  # Degrees of freedom for the AR(1) reach noise's innovation distribution.
  # Student-t rather than Gaussian, so noise is occasionally "jagged" (a few
  # larger week-to-week swings) instead of uniformly smooth -- the scale is
  # rederived in `_simulate_ar1_reach_noise()` so the stationary sd still
  # matches `time_reach_noise_sd` regardless of this value.
  reach_noise_df: float = 4.0

  # Sparse, short-lived one-off execution spikes (a flash promo, a launch
  # burst, a competitive response) layered on top of the flat multipliers
  # above -- without them, weekly impressions never show the "why did that
  # happen" outliers real delivery data has. `promo_spike_prob` is the
  # per-week probability of a new spike starting on a given channel;
  # `promo_spike_duration_weeks` is how many weeks it lasts once triggered;
  # `promo_spike_mult_range` is the multiplier applied to `reach_frac` while
  # active (independent of `flight_mult`/`seasonal_mult`, so a spike can
  # occur during an otherwise-dark flighting period too).
  promo_spike_prob: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.01, 'Display': 0.03, 'Social': 0.02}
  )
  promo_spike_duration_weeks: dict[str, int] = dataclasses.field(
      default_factory=lambda: {'TV': 1, 'Display': 1, 'Social': 1}
  )
  promo_spike_mult_range: tuple[float, float] = (1.4, 2.2)

  # Realistic CPM ($ per 1,000 impressions) per channel, +/- a variability
  # band to account for auction/market price fluctuation. TV/Display/Social
  # defaults reflect typical relative pricing (linear TV >> social >>
  # programmatic display).
  cpm_dollars: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 10.0, 'Display': 2.0, 'Social': 5.0}
  )
  cpm_variability_frac: float = 0.05

  # Target incremental ROI per channel, used by `calibrate_channel_effects()`
  # to rescale `beta_m`/`beta_gm` after media/cost are simulated. Without
  # this, all channels draw from the same beta_m hyperprior (Section 6) and
  # end up with an essentially arbitrary relative ROI split. Defaults
  # reflect TV's high CPM/broad-reach inefficiency vs. digital channels'
  # lower CPM and tighter targeting.
  target_roi: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 2.5, 'Display': 1.5, 'Social': 2.0}
  )

  # Range each channel's adstock decay rate (`alpha_m`) is drawn uniformly
  # from in `simulate_adstock_hill_params()`, reflecting ad memory persisting
  # longer for channels with richer/longer creative exposure (TV) than for
  # brief, disposable exposure (Display), with Social in between. TV's range
  # is set apart from the other two, but Display's and Social's ranges
  # overlap, so the TV > Social > Display ordering holds on average but isn't
  # guaranteed every run.
  adstock_retention_range: dict[str, tuple[float, float]] = dataclasses.field(
      default_factory=lambda: {
          'TV': (0.6, 0.8),
          'Display': (0.0, 0.3),
          'Social': (0.1, 0.4),
      }
  )

  # Multiplicative scale applied to the whole non-media baseline term (tau_g
  # + gamma_gc*controls + eps_gt + mu_t) in `generate_kpi_and_revenue()`,
  # before the `max(..., 0)` floor. Without it, media (calibrated only to
  # hit `target_roi` in dollar terms) ends up under 2% of total KPI, since
  # `tau_g`'s default N(15.0, 1.2) draw dwarfs media's contribution. A
  # multiplicative scale shrinks the baseline's mean and noise together
  # (unlike shrinking `tau_g`'s mean alone, which pushes a growing share of
  # geo-weeks below zero and clips them at a hard floor). 0.15 was
  # calibrated by search against this config/seed to land media at ~10% of
  # total KPI, roughly matching typical real-world MMM base/media splits.
  baseline_scale: float = 0.15

  # 156 weeks (n_times) from this date lands the last simulated week at
  # 2025-12-29, i.e. data runs through the end of 2025.
  start_date: datetime.date = datetime.date(2023, 1, 9)

  def __post_init__(self):
    if len(self.channel_names) != self.n_imp_channels:
      raise ValueError(f'channel_names must have length {self.n_imp_channels}')
    if len(self.control_names) != self.n_controls:
      raise ValueError(f'control_names must have length {self.n_controls}')
    self.flight_burst_weeks.setdefault('Social', self.n_times)

  @classmethod
  def from_dict(cls, overrides: dict) -> 'SimulationConfig':
    """Builds a config from a dict of field overrides, defaults for the rest."""
    return cls(**overrides)

  def __repr__(self) -> str:
    fields = dataclasses.fields(self)
    name_width = max(len(f.name) for f in fields)
    lines = (
        f'  {f.name:<{name_width}} = {getattr(self, f.name)!r}' for f in fields
    )
    return 'SimulationConfig(\n' + '\n'.join(lines) + '\n)'


class GeoMediaDataSimulator:
  """Simulates geo x time media/KPI data with known ground-truth params."""

  def __init__(self, config: SimulationConfig):
    self.config = config
    tf.random.set_seed(config.seed_num)
    dates = [
        config.start_date + datetime.timedelta(weeks=w)
        for w in range(config.n_times)
    ]
    self.time_index = pd.DatetimeIndex(dates)
    self.time_names = [d.strftime('%Y-%m-%d') for d in dates]
    self.control_col_names = [
        _add_suffix(c, CONTROL_COL_SUFFIX) for c in config.control_names
    ]

  # 1. Population.
  def simulate_population(self) -> tf.Tensor:
    """Simulates geo population (actual U.S. state populations)."""
    self.geo_names = list(STATE_POPULATION.keys())
    self.n_geos = len(self.geo_names)
    self.p_g = tf.constant(list(STATE_POPULATION.values()), dtype=tf.float32)
    self.national_population = tf.reduce_sum(self.p_g)
    return self.p_g

  # 2. Controls.
  def simulate_controls(self) -> tf.Tensor:
    """Simulates control variables and their population-scaled transform.

    One control (`SEASONALITY_CONTROL_NAME`) is not an iid random draw: it's
    the shared national demand-seasonality index, also read by
    `simulate_media()` (via `self.seasonal_index_t`) to modulate each
    channel's execution -- so it acts as a genuine confounder between media
    timing and the KPI baseline, rather than an independent covariate.
    """
    config = self.config
    week_of_year = self.time_index.isocalendar().week.to_numpy(dtype=np.float64)

    # Per-year jitter on the peak week and amplitude of the shared
    # seasonality signal, so successive calendar years don't produce an
    # identical, obviously-repeating curve (see `seasonal_peak_jitter_weeks_sd`
    # / `seasonal_amplitude_jitter_sd`).
    years = self.time_index.year.to_numpy()
    unique_years, year_idx_t = np.unique(years, return_inverse=True)
    n_years = len(unique_years)
    peak_shift_y = (
        tfp.distributions.Normal(0, config.seasonal_peak_jitter_weeks_sd)
        .sample(n_years)
        .numpy()
    )
    amplitude_jitter_y = (
        tfp.distributions.TruncatedNormal(
            1.0, config.seasonal_amplitude_jitter_sd, 0.5, 1.5
        )
        .sample(n_years)
        .numpy()
    )
    peak_shift_t = peak_shift_y[year_idx_t]
    amplitude_jitter_t = amplitude_jitter_y[year_idx_t]

    self.seasonal_index_t = tf.constant(
        amplitude_jitter_t
        * np.cos(
            2
            * np.pi
            * (week_of_year - (config.demand_seasonal_peak_week + peak_shift_t))
            / 52.0
        ),
        dtype=self.p_g.dtype,
    )
    seasonal_control_gtc = tf.tile(
        self.seasonal_index_t[tf.newaxis, :, tf.newaxis], [self.n_geos, 1, 1]
    )

    control_columns = []
    for name in config.control_names:
      if name == SEASONALITY_CONTROL_NAME:
        control_columns.append(seasonal_control_gtc)
      else:
        control_columns.append(
            tfp.distributions.Normal(0, 3).sample(
                [self.n_geos, config.n_times, 1]
            )
        )
    self.control_gtc = tf.concat(control_columns, axis=-1)

    control_transformer = transformers.CenteringAndScalingTransformer(
        tensor=self.control_gtc,
        population=self.p_g,
        population_scaling_id=None,
    )
    self.transformed_control_gtc = control_transformer.forward(self.control_gtc)
    return self.transformed_control_gtc

  def _simulate_seasonal_multiplier(self) -> tf.Tensor:
    """Per-channel sensitivity to the shared national seasonality index.

    Requires `simulate_controls()` to have run first (it populates
    `self.seasonal_index_t`), so that media execution and the KPI's
    `seasonality_index` control are driven by the exact same signal.
    """
    config = self.config
    amplitude_m = np.array(
        [config.seasonal_amplitude[ch] for ch in config.channel_names]
    )
    seasonal_mult_tm = (
        1.0
        + amplitude_m[np.newaxis, :]
        * self.seasonal_index_t.numpy()[:, np.newaxis]
    )
    return tf.constant(seasonal_mult_tm, dtype=self.p_g.dtype)

  def _simulate_flighting_multiplier(self) -> tf.Tensor:
    """On/off campaign flighting via a per-channel two-state Markov chain.

    Requires `simulate_controls()` to have run first (it populates
    `self.seasonal_index_t`): `flight_seasonal_bias` biases the chain's
    transition probabilities toward that same calendar signal, so bursts
    tend to cluster around `demand_seasonal_peak_week` every year rather
    than firing at calendar-blind random points.
    """
    config = self.config
    n_channels = config.n_imp_channels
    burst_m = np.array(
        [config.flight_burst_weeks[ch] for ch in config.channel_names],
        dtype=np.float64,
    )
    dark_m = np.array(
        [config.flight_dark_weeks[ch] for ch in config.channel_names],
        dtype=np.float64,
    )
    floor_m = np.array(
        [config.flight_floor[ch] for ch in config.channel_names],
        dtype=np.float64,
    )
    bias_m = np.array(
        [config.flight_seasonal_bias[ch] for ch in config.channel_names],
        dtype=np.float64,
    )
    base_p_on_to_off_m = 1.0 / burst_m
    base_p_off_to_on_m = 1.0 / dark_m

    # Scale the base transition probabilities by the seasonal index: easier
    # to turn "on" and harder to turn "off" near the seasonal peak, and vice
    # versa near the trough. At bias=0 this reduces to the flat, calendar-
    # blind probabilities used previously.
    seasonal_index_t = self.seasonal_index_t.numpy()
    p_off_to_on_tm = np.clip(
        base_p_off_to_on_m[np.newaxis, :]
        * (1.0 + bias_m[np.newaxis, :] * seasonal_index_t[:, np.newaxis]),
        0.0,
        1.0,
    )
    p_on_to_off_tm = np.clip(
        base_p_on_to_off_m[np.newaxis, :]
        * (1.0 - bias_m[np.newaxis, :] * seasonal_index_t[:, np.newaxis]),
        0.0,
        1.0,
    )

    transition_draws_tm = (
        tfp.distributions.Uniform(0, 1)
        .sample((config.n_times, n_channels))
        .numpy()
    )
    state_tm = np.ones((config.n_times, n_channels), dtype=np.float64)
    for m in range(n_channels):
      on = True
      for t in range(config.n_times):
        state_tm[t, m] = 1.0 if on else 0.0
        if on and transition_draws_tm[t, m] < p_on_to_off_tm[t, m]:
          on = False
        elif not on and transition_draws_tm[t, m] < p_off_to_on_tm[t, m]:
          on = True

    # Smooth the hard 0/1 Markov state into a gradual ramp via a symmetric
    # triangular kernel, so on/off transitions no longer read as an
    # instantaneous step (see `flight_ramp_weeks`) -- real campaigns are
    # trafficked/wound down over a few weeks, not switched on a single one.
    smoothed_state_tm = np.empty_like(state_tm)
    for m, ch in enumerate(config.channel_names):
      ramp = config.flight_ramp_weeks.get(ch, 0)
      if ramp <= 0:
        smoothed_state_tm[:, m] = state_tm[:, m]
        continue
      kernel = np.concatenate(
          [np.arange(1, ramp + 2), np.arange(ramp, 0, -1)]
      ).astype(np.float64)
      kernel /= kernel.sum()
      smoothed_state_tm[:, m] = np.convolve(state_tm[:, m], kernel, mode='same')

    flight_mult_tm = (
        floor_m[np.newaxis, :]
        + (1.0 - floor_m[np.newaxis, :]) * smoothed_state_tm
    )
    return tf.constant(flight_mult_tm, dtype=self.p_g.dtype)

  def _simulate_trend_multiplier(self) -> tf.Tensor:
    """Slow secular drift in execution intensity, per channel."""
    config = self.config
    trend_pct_m = np.array(
        [config.trend_pct_total[ch] for ch in config.channel_names]
    )
    t_frac = np.arange(config.n_times, dtype=np.float64) / max(
        config.n_times - 1, 1
    )
    trend_mult_tm = 1.0 + trend_pct_m[np.newaxis, :] * t_frac[:, np.newaxis]
    return tf.constant(trend_mult_tm, dtype=self.p_g.dtype)

  def _simulate_ar1_reach_noise(self) -> tf.Tensor:
    """AR(1) week-to-week reach noise (replaces iid noise)."""
    config = self.config
    phi = config.reach_noise_ar1_phi
    df = config.reach_noise_df
    # Student-t has Var = scale^2 * df/(df-2) (df > 2); rescale the
    # innovation's `scale` so the AR(1) process's stationary sd still equals
    # `time_reach_noise_sd`, regardless of `df`.
    variance_inflation = df / (df - 2)
    innovation_scale = float(
        config.time_reach_noise_sd
        * np.sqrt(max(1.0 - phi**2, 1e-6) / variance_inflation)
    )
    innovations_tm = tfp.distributions.StudentT(
        df=tf.constant(df, dtype=self.p_g.dtype),
        loc=tf.constant(0, dtype=self.p_g.dtype),
        scale=tf.constant(innovation_scale, dtype=self.p_g.dtype),
    ).sample((config.n_times, config.n_imp_channels))

    def step(prev_noise_m, innovation_m):
      return phi * prev_noise_m + innovation_m

    return tf.scan(
        step,
        innovations_tm,
        initializer=tf.zeros(
            [config.n_imp_channels], dtype=innovations_tm.dtype
        ),
    )

  def _simulate_promo_spike_multiplier(self) -> tf.Tensor:
    """Sparse, short-lived one-off execution spikes (promo/launch bursts)."""
    config = self.config
    n_channels = config.n_imp_channels
    prob_m = np.array(
        [config.promo_spike_prob.get(ch, 0.0) for ch in config.channel_names]
    )
    duration_m = np.array(
        [
            config.promo_spike_duration_weeks.get(ch, 1)
            for ch in config.channel_names
        ]
    )
    trigger_draws_tm = (
        tfp.distributions.Uniform(0, 1)
        .sample((config.n_times, n_channels))
        .numpy()
    )
    mult_draws_tm = (
        tfp.distributions.Uniform(*config.promo_spike_mult_range)
        .sample((config.n_times, n_channels))
        .numpy()
    )

    spike_mult_tm = np.ones((config.n_times, n_channels), dtype=np.float64)
    remaining_m = np.zeros(n_channels, dtype=np.int64)
    active_mult_m = np.ones(n_channels, dtype=np.float64)
    for t in range(config.n_times):
      for m in range(n_channels):
        if remaining_m[m] <= 0 and trigger_draws_tm[t, m] < prob_m[m]:
          remaining_m[m] = int(duration_m[m])
          active_mult_m[m] = mult_draws_tm[t, m]
        if remaining_m[m] > 0:
          spike_mult_tm[t, m] = active_mult_m[m]
          remaining_m[m] -= 1

    return tf.constant(spike_mult_tm, dtype=self.p_g.dtype)

  # 3. Media channels (reach x frequency).
  def simulate_media(self, verbose: bool = True) -> tf.Tensor:
    """Simulates weekly impressions per channel via reach x frequency."""
    config = self.config
    n_channels = config.n_imp_channels
    target_audience_pop_frac_m = tf.constant(
        [config.target_audience_pop_frac[ch] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    current_reach_frac_m = tf.constant(
        [config.current_reach_frac[ch] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )

    # Target-audience size per geo (channel's population share x geo's own
    # population, with per-geo/channel heterogeneity in that share).
    geo_audience_mult_gm = tfp.distributions.TruncatedNormal(
        1.0, config.geo_audience_heterogeneity_sd, 0.5, 1.5
    ).sample((self.n_geos, n_channels))
    self.audience_g_m = (
        self.p_g[:, tf.newaxis]
        * target_audience_pop_frac_m[tf.newaxis, :]
        * geo_audience_mult_gm
    )
    audience_gtm = self.audience_g_m[:, tf.newaxis, :]
    self.audience_national_m = tf.reduce_sum(self.audience_g_m, axis=0)

    # Reach % (of target audience) per time/channel (no geo dimension --
    # geo-level heterogeneity lives entirely in the audience size above).
    if config.enable_ramp:
      ramp_t = tf.minimum(
          tf.range(config.n_times, dtype=self.p_g.dtype) / config.ramp_weeks,
          1.0,
      )
    else:
      ramp_t = tf.ones([config.n_times], dtype=self.p_g.dtype)

    # Time-varying execution multipliers layered on top of the flat
    # `current_reach_frac` target: annual seasonality, on/off flighting,
    # and a secular budget trend, each independently parameterized per
    # channel (see `SimulationConfig`). Noise is AR(1) rather than iid so
    # week-to-week execution persists instead of resetting every week.
    self.seasonal_mult_tm = self._simulate_seasonal_multiplier()
    self.flight_mult_tm = self._simulate_flighting_multiplier()
    self.trend_mult_tm = self._simulate_trend_multiplier()
    self.promo_spike_mult_tm = self._simulate_promo_spike_multiplier()
    ar1_reach_noise_tm = self._simulate_ar1_reach_noise()

    reach_frac_tm = (
        current_reach_frac_m[tf.newaxis, :]
        * ramp_t[:, tf.newaxis]
        * self.seasonal_mult_tm
        * self.flight_mult_tm
        * self.trend_mult_tm
        * self.promo_spike_mult_tm
        + ar1_reach_noise_tm
    )
    self.reach_frac_gtm = tf.clip_by_value(
        reach_frac_tm[tf.newaxis, :, :], 0.01, 0.95
    )

    # Frequency (impressions per person reached, per week).
    self.frequency_gtm = tf.maximum(
        tfp.distributions.Uniform(*config.frequency_range).sample(
            (self.n_geos, config.n_times, n_channels)
        )
        + tfp.distributions.Normal(0, config.frequency_noise_sd).sample(
            (self.n_geos, config.n_times, n_channels)
        ),
        0.5,
    )

    # Reach count and impressions.
    self.reach_count_gtm = self.reach_frac_gtm * audience_gtm
    self.impression_gtm = tf.round(self.reach_count_gtm * self.frequency_gtm)

    if verbose:
      ipc_sparsity = np.sum(self.impression_gtm.numpy() == 0.0, axis=(0, 1)) / (
          self.n_geos * config.n_times
      )
      print(
          'percentage of sparsity of impression_gtm for each channel:'
          f' {[f"{s * 100:.2f}%" for s in ipc_sparsity]}'
      )
      realized_reach_frac_m = tf.reduce_mean(
          self.reach_count_gtm, axis=(0, 1)
      ) / tf.reduce_mean(self.audience_g_m, axis=0)
      realized_frequency_m = tf.reduce_mean(self.frequency_gtm, axis=(0, 1))
      for i, ch in enumerate(config.channel_names):
        print(
            f'{ch}: mean reach % = {realized_reach_frac_m[i]:.3f}'
            f' (nominal target {config.current_reach_frac[ch]} -- seasonality'
            ' /flighting/trend shift the realized average), mean frequency ='
            f' {realized_frequency_m[i]:.2f} (target range'
            f' {config.frequency_range})'
        )

    # Scale impressions (by population and by median of population-scaled
    # impressions) -- also needed by simulate_adstock_hill_params() below.
    self.impression_transformer = transformers.MediaTransformer(
        media=self.impression_gtm, population=self.p_g
    )
    self.transformed_ipc_gtm = self.impression_transformer.forward(
        self.impression_gtm
    )
    return self.impression_gtm

  # 4. Time-varying intercepts (mu_t) and geo effects (tau_g).
  def simulate_intercepts(self) -> tf.Tensor:
    """Simulates the time-varying intercept and geo-effect terms."""
    config = self.config
    self.tau_g = tfp.distributions.Normal(15.0, 1.2).sample(self.n_geos)
    n_knots_simul = config.n_times
    knots_k = tfp.distributions.Normal(0, 2.0).sample(n_knots_simul)
    knots_object = knots.get_knot_info(config.n_times, n_knots_simul, False)
    self.mu_t = tfp.distributions.Deterministic(
        tf.einsum(
            '...k,kt->...t',
            knots_k,
            tf.convert_to_tensor(knots_object.weights),
        )
    ).sample()
    return self.mu_t

  # 5. Cost and unit-value.
  def simulate_cost_and_unit_value(self) -> tf.Tensor:
    """Simulates per-impression cost (from realistic per-channel CPMs)."""
    config = self.config
    cpm_dollars_m = tf.constant(
        [config.cpm_dollars[ch] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    self.cpm_m = tfp.distributions.Uniform(
        cpm_dollars_m * (1 - config.cpm_variability_frac),
        cpm_dollars_m * (1 + config.cpm_variability_frac),
    ).sample()
    self.cost_gtm = self.impression_gtm * self.cpm_m / 1000.0
    self.unit_value = tfp.distributions.Uniform(0.0345, 0.0355).sample(
        (self.n_geos, config.n_times)
    )
    return self.cost_gtm

  # 6. Coefficients (beta_g,m and gamma_g,c) and error term epsilon_g,t.
  def simulate_coefficients(self) -> tf.Tensor:
    """Simulates media/control coefficients and the residual error term."""
    config = self.config
    self.beta_m = tfp.distributions.Normal(-1.4, 0.1).sample(
        config.n_imp_channels
    )
    self.eta_m = tfp.distributions.HalfNormal(0.18).sample(
        config.n_imp_channels
    )
    beta_gm_dev = tfp.distributions.Normal(0, 1).sample(
        [self.n_geos, config.n_imp_channels]
    )
    self.beta_gm = tf.exp(self.beta_m + self.eta_m * beta_gm_dev)

    self.gamma_c = tfp.distributions.Normal(3.5, 0.5).sample(config.n_controls)
    self.xi_c = tfp.distributions.HalfNormal(0.3).sample(config.n_controls)
    gamma_gc_dev = tfp.distributions.Normal(0, 1).sample(
        [self.n_geos, config.n_controls]
    )
    self.gamma_gc = self.gamma_c + self.xi_c * gamma_gc_dev

    self.sigma = tf.fill([1], 0.5)
    self.eps_gt = tfp.distributions.Normal(0, self.sigma[0]).sample(
        [self.n_geos, config.n_times]
    )
    return self.beta_gm

  # 7. Adstock / Hill parameters.
  def simulate_adstock_hill_params(self) -> tf.Tensor:
    """Derives ec_m from a "half-saturation at 50% of audience" assumption."""
    config = self.config
    half_sat_reach_count_national_m = 0.5 * self.audience_national_m
    # Convert that reach count to an impression count using the mean of
    # `frequency_range`, not a fresh Uniform draw over the same range: weekly
    # frequency is itself ~Uniform(frequency_range), so its realized average
    # across geos/times (and hence the `median_m` "current execution" level
    # below) already converges to that mean. Drawing an independent Uniform
    # sample here would inject noise into `ec_m` uncorrelated with what
    # `median_m` is actually built on.
    mean_frequency = (config.frequency_range[0] + config.frequency_range[1]) / 2
    ec_impressions_national_m = mean_frequency * half_sat_reach_count_national_m
    ec_per_capita_m = ec_impressions_national_m / self.national_population
    median_m = self.impression_transformer.population_scaled_median_m

    alpha_low_m = tf.constant(
        [config.adstock_retention_range[ch][0] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    alpha_high_m = tf.constant(
        [config.adstock_retention_range[ch][1] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    self.alpha_m = tfp.distributions.Uniform(alpha_low_m, alpha_high_m).sample()
    self.ec_m = ec_per_capita_m / median_m
    self.slope_m = tf.ones([config.n_imp_channels])
    print(f'ec_m = {self.ec_m.numpy()}')
    return self.ec_m

  # 8. Transform the media.
  def transform_media(self) -> tf.Tensor:
    """Applies the adstock decay and Hill saturation transforms to media."""
    hill_transformer = adstock_hill.HillTransformer(
        ec=self.ec_m, slope=self.slope_m
    )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=self.alpha_m, max_lag=8, n_times_output=self.config.n_times
    )
    self.media_transformed = hill_transformer.forward(
        adstock_transformer.forward(self.transformed_ipc_gtm)
    )
    return self.media_transformed

  # 8b. Calibrate channel effect sizes to hit realistic target ROIs.
  def calibrate_channel_effects(self) -> tf.Tensor:
    """Rescales beta_m/beta_gm so realized ROI matches `config.target_roi`.

    Must run after `transform_media()` and `simulate_cost_and_unit_value()`,
    and before `generate_kpi_and_revenue()`. `beta_m` is drawn from an
    identical hyperprior across channels (Section 6), so without this step
    each channel's relative contribution/ROI is essentially arbitrary --
    this reverse-engineers the scale the same way `ec_m` is already
    reverse-engineered from a "50% of audience" assumption (Section 7).
    Incremental revenue is linear in `beta_gm` (media/cost/unit-value are
    already fixed at this point), so a single rescale hits the target
    exactly, in expectation over the geo-level `beta_gm` heterogeneity.
    """
    config = self.config
    target_roi_m = tf.constant(
        [config.target_roi[ch] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    current_incremental_revenue_m = tf.einsum(
        'g,gt,gtm,gm->m',
        self.p_g,
        self.unit_value,
        self.media_transformed,
        self.beta_gm,
    )
    current_cost_m = tf.einsum('gtm->m', self.cost_gtm)
    scale_m = (target_roi_m * current_cost_m) / current_incremental_revenue_m

    self.beta_m = self.beta_m + tf.math.log(scale_m)
    self.beta_gm = self.beta_gm * scale_m
    return self.beta_gm

  # 9. Generate KPI and revenue.
  def generate_kpi_and_revenue(self) -> tf.Tensor:
    """Generates the KPI and revenue outcome from all simulated components."""
    kpi_per_capita_gt = tf.maximum(
        self.config.baseline_scale
        * (
            self.tau_g[..., tf.newaxis]
            + tf.einsum(
                'gtm,gm->gt', self.transformed_control_gtc, self.gamma_gc
            )
            + self.eps_gt
            + self.mu_t[tf.newaxis, ...]
        ),
        0.0,
    ) + tf.einsum('gtm,gm->gt', self.media_transformed, self.beta_gm)
    self.kpi_gt = kpi_per_capita_gt * self.p_g[..., tf.newaxis]
    self.revenue_gt = self.kpi_gt * tf.ones_like(self.unit_value)
    return self.kpi_gt

  # 10. Combine tensors into a Pandas DataFrame.
  def to_dataframe(self) -> pd.DataFrame:
    """Assembles all simulated tensors into a single geo x time DataFrame."""
    coords_gtm = {
        GEO_DIM_NAME: self.geo_names,
        TIME_DIM_NAME: self.time_names,
        CHANNEL_DIM_NAME: self.config.channel_names,
    }

    def channel_df(tensor: tf.Tensor, suffix: str) -> pd.DataFrame:
      da = xr.DataArray(
          tensor,
          dims=[GEO_DIM_NAME, TIME_DIM_NAME, CHANNEL_DIM_NAME],
          coords=coords_gtm,
          name=suffix,
      )
      return (
          da.to_dataframe()
          .reset_index()
          .pivot(
              index=[GEO_DIM_NAME, TIME_DIM_NAME],
              columns=CHANNEL_DIM_NAME,
              values=suffix,
          )
          .rename(columns=lambda x: _add_suffix(x, suffix))
          .reset_index()
      )

    media_df = channel_df(self.impression_gtm, IMPRESSIONS_COL_SUFFIX)
    spend_df = channel_df(self.cost_gtm, SPEND_COL_SUFFIX)
    reach_df = channel_df(self.reach_count_gtm, REACH_COL_SUFFIX)
    frequency_df = channel_df(self.frequency_gtm, FREQUENCY_COL_SUFFIX)

    control_data_name = 'control_value'
    control_df = (
        xr.DataArray(
            self.transformed_control_gtc,
            dims=[GEO_DIM_NAME, TIME_DIM_NAME, CONTROL_DIM_NAME],
            coords={
                GEO_DIM_NAME: self.geo_names,
                TIME_DIM_NAME: self.time_names,
                CONTROL_DIM_NAME: self.control_col_names,
            },
            name=control_data_name,
        )
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[GEO_DIM_NAME, TIME_DIM_NAME],
            columns=CONTROL_DIM_NAME,
            values=control_data_name,
        )
        .reset_index()
    )

    kpi_df = (
        xr.DataArray(
            self.kpi_gt,
            dims=[GEO_DIM_NAME, TIME_DIM_NAME],
            coords={
                GEO_DIM_NAME: self.geo_names,
                TIME_DIM_NAME: self.time_names,
            },
            name=KPI_COL_NAME,
        )
        .to_dataframe()
        .reset_index()
    )
    unit_value_df = (
        xr.DataArray(
            self.unit_value,
            dims=[GEO_DIM_NAME, TIME_DIM_NAME],
            coords={
                GEO_DIM_NAME: self.geo_names,
                TIME_DIM_NAME: self.time_names,
            },
            name=UNIT_VALUE_COL_NAME,
        )
        .to_dataframe()
        .reset_index()
    )
    population_df = (
        xr.DataArray(
            self.p_g,
            dims=[GEO_DIM_NAME],
            coords={GEO_DIM_NAME: self.geo_names},
            name=POPULATION_COL_NAME,
        )
        .to_dataframe()
        .reset_index()
    )

    self.geo_data_df = (
        media_df.merge(control_df)
        .merge(spend_df)
        .merge(reach_df)
        .merge(frequency_df)
        .merge(kpi_df)
        .merge(unit_value_df)
        .merge(population_df)
    )
    return self.geo_data_df

  # 11. Ground truth (raw-scale params, KPI-scaled ground truth, ROI).
  def compute_ground_truth(self, verbose: bool = True) -> dict[str, np.ndarray]:
    """Scales raw simulated params to the posterior's units and computes ROI."""
    config = self.config
    self.raw_scale_params = {
        'alpha_m': self.alpha_m.numpy(),
        'beta_gm': self.beta_gm.numpy()[:, : config.n_imp_channels],
        'beta_m': self.beta_m.numpy()[: config.n_imp_channels],
        'ec_m': self.ec_m.numpy(),
        'eta_m': self.eta_m.numpy()[: config.n_imp_channels],
        'gamma_c': self.gamma_c.numpy(),
        'gamma_gc': self.gamma_gc.numpy(),
        'mu_t': self.mu_t.numpy(),
        'sigma': self.sigma.numpy(),
        'slope_m': self.slope_m.numpy(),
        'tau_g': self.tau_g.numpy(),
        'xi_c': self.xi_c.numpy(),
        'intercept_gt': (
            (self.tau_g[..., tf.newaxis] + self.mu_t[tf.newaxis, ...]).numpy()
        ),
    }

    kpi_transformer = transformers.KpiTransformer(
        kpi=self.kpi_gt, population=self.p_g
    )
    self.kpi_mean = kpi_transformer.population_scaled_mean.numpy().item()
    self.kpi_stdev = kpi_transformer.population_scaled_stdev.numpy().item()

    ground_truth = {}
    for param in ['beta_gm', 'gamma_c', 'gamma_gc', 'sigma', 'xi_c']:
      ground_truth[param] = self.raw_scale_params[param] / self.kpi_stdev
    # If log(X) ~ N(mu, sigma), then log(X/k) ~ N(mu - log(k), sigma).
    ground_truth['beta_m'] = self.raw_scale_params['beta_m'] - np.log(
        self.kpi_stdev
    )
    ground_truth['intercept_gt'] = (
        self.raw_scale_params['intercept_gt'] - self.kpi_mean
    ) / self.kpi_stdev

    # ROI formula: https://developers.google.com/meridian/docs/basics/
    # roi-and-mroi-parameterization#roi
    incremental_revenue_m = self.kpi_stdev * tf.einsum(
        'g,gt,gtm,gm->m',
        self.p_g,
        self.unit_value,
        self.media_transformed,
        ground_truth['beta_gm'],
    )
    ground_truth_roi = incremental_revenue_m / tf.einsum(
        'gtm->m', self.cost_gtm
    )
    ground_truth['roi_m'] = ground_truth_roi.numpy()[: config.n_imp_channels]
    self.incremental_revenue_m = incremental_revenue_m

    if verbose:
      total_incremental_roi = np.sum(incremental_revenue_m) / np.sum(
          self.cost_gtm
      )
      total_revenue = np.sum(self.revenue_gt)
      print(f'ground-truth ROI for every channel = {ground_truth_roi.numpy()}')
      print(f'ground-truth total incremental ROI = {total_incremental_roi:.2f}')
      print()
      print(f'Total revenue = {total_revenue / 1e6:.2f}M')
      print(f'Total ROI = {total_revenue / np.sum(self.cost_gtm)}')

    self.ground_truth = ground_truth
    return ground_truth

  def _aggregate_over_geos(
      self, tensor: tf.Tensor, aggregate: str
  ) -> np.ndarray:
    if aggregate == 'sum':
      return np.sum(tensor.numpy(), axis=0)
    if aggregate == 'mean':
      return np.mean(tensor.numpy(), axis=0)
    raise ValueError(f"aggregate must be 'sum' or 'mean', got {aggregate!r}")

  def plot_media_time_series(
      self, aggregate: str = 'sum', ax: plt.Axes | None = None
  ) -> plt.Axes:
    """Plots weekly raw impressions by channel, aggregated across geos."""
    if ax is None:
      _, ax = plt.subplots(figsize=(10, 5))
    series_tm = self._aggregate_over_geos(self.impression_gtm, aggregate)
    for i, channel in enumerate(self.config.channel_names):
      ax.plot(self.time_index, series_tm[:, i], label=channel)
    ax.set_xlabel('Time')
    ax.set_ylabel('Impressions')
    ax.set_title(f'Weekly impressions by channel ({aggregate} over geos)')
    ax.legend()
    return ax

  def plot_spend_time_series(
      self, aggregate: str = 'sum', ax: plt.Axes | None = None
  ) -> plt.Axes:
    """Plots weekly spend by channel, aggregated across geos."""
    if ax is None:
      _, ax = plt.subplots(figsize=(10, 5))
    series_tm = self._aggregate_over_geos(self.cost_gtm, aggregate)
    for i, channel in enumerate(self.config.channel_names):
      ax.plot(self.time_index, series_tm[:, i], label=channel)
    ax.set_xlabel('Time')
    ax.set_ylabel('Spend ($)')
    ax.set_title(f'Weekly spend by channel ({aggregate} over geos)')
    ax.legend()
    return ax

  def plot_kpi_time_series(
      self, aggregate: str = 'mean', ax: plt.Axes | None = None
  ) -> plt.Axes:
    """Plots weekly KPI, aggregated across geos."""
    if ax is None:
      _, ax = plt.subplots(figsize=(10, 5))
    series_t = self._aggregate_over_geos(self.kpi_gt, aggregate)
    ax.plot(self.time_index, series_t)
    ax.set_xlabel('Time')
    ax.set_ylabel('KPI (conversions)')
    ax.set_title(f'Weekly KPI ({aggregate} over geos)')
    return ax
