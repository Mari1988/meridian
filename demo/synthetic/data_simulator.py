"""Synthetic geo x time media data simulator.

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

Media execution (`simulate_media`) is deliberately kept to three named,
independently-justified layers -- a shared seasonal signal, per-channel
on/off flighting, and per-channel AR(1) noise -- rather than a larger stack
of mechanisms, so the DGP stays simple enough to state and audit in a single
paragraph. See `data_simulator_v1_complex.py` for an earlier, richer version
(seasonal jitter, ramp-smoothed flighting, secular trend, promo spikes) kept
for reference.
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
  # Weekly frequency target (impressions per person reached), per channel.
  frequency_range: dict[str, tuple[float, float]] = dataclasses.field(
      default_factory=lambda: {
          'TV': (1.0, 5.0),
          'Display': (1.0, 5.0),
          'Social': (1.0, 5.0),
      }
  )
  frequency_noise_sd: float = 0.1
  geo_audience_heterogeneity_sd: float = 0.12
  # The fixed weekly "effective frequency" at which half of a channel's
  # target audience is assumed to produce half-saturation, used to derive
  # the ground-truth `ec_m` in `simulate_adstock_hill_params()`. Deliberately
  # a constant, literature-motivated threshold (Krugman's three-hit theory /
  # Naples' effective-frequency guidance, commonly cited around 3-4
  # exposures/week) rather than each channel's own `mean_frequency_m` -- so
  # `ec_m` no longer automatically tracks a channel's current execution
  # frequency, and a channel's actual weekly frequency genuinely competes
  # against this threshold instead of canceling out of the `ec_m` ratio.
  saturation_frequency: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 4.0, 'Display': 4.0, 'Social': 4.0}
  )

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

  # Flighting: channels are bought in discrete on/off runs rather than
  # continuously. Each channel picks one of two textures via `flight_style`,
  # both built from the same renewal-process mechanism (alternating runs
  # whose *lengths* are randomly drawn, not their week-by-week on/off state):
  #  - 'flighting' (TV, Display): burst/dark run lengths are drawn from a
  #    Gamma distribution with mean `flight_burst_weeks`/`flight_dark_weeks`
  #    and spread `flight_burst_cv`/`flight_dark_cv`, rounded to whole weeks.
  #    A *memoryless* (geometric) run-length model was tried first and
  #    rejected: with mean 8, ~1/3 of its runs land at 1-3 weeks purely by
  #    chance, which no real media plan would produce -- a planner commits
  #    to a flight length in advance, so real/practitioner run lengths
  #    cluster tightly around their target instead of spanning 1-26 weeks in
  #    a single realization. The default CVs (0.6 burst, 0.9 dark) are fit
  #    against the empirical on/off run-length spread in the Robyn/Garve
  #    open MMM demo dataset (github.com/Garve/datasets, mmm.csv) -- a
  #    synthetic dataset, but one built by MMM practitioners specifically to
  #    mimic real client delivery, and the closest freely-available reference
  #    for this shape. `flight_floor` is the residual multiplier during a
  #    dark run -- never exactly 0, since there's usually some always-on
  #    baseline activity.
  #  - 'continuity' (Social): always-on, with short sharp promotional spikes
  #    (`continuity_spike_weeks`/`continuity_spike_cv`) recurring every
  #    `continuity_gap_weeks` (+/- `continuity_gap_cv`) at
  #    `continuity_spike_amplitude` above baseline, then renormalized to
  #    mean 1 over the horizon. Calibrated against a real (not synthetic)
  #    weekly TV-GRP series -- a Shenzhen TV-manufacturer dataset
  #    (github.com/jamesrawlins1000/Market-mix-modelling-data) -- which
  #    shows recurring peaks roughly every 5 weeks (CV ~0.57) at ~1.9x the
  #    baseline level, rather than a single smooth annual cycle.
  #
  # TV is bought in short, sharp flights with long dark gaps (typical
  # linear-TV campaign buying); Display is given long, slow-changing cycles
  # matching real programmatic-display delivery (see `reach_noise_ar1_phi`
  # below). Social's `flight_burst_weeks`/`flight_dark_weeks`/`flight_floor`
  # entries below are unused defaults, kept only so the dicts have a value
  # for every channel -- switching Social's `flight_style` back to
  # 'flighting' would make them active again.
  flight_style: dict[str, str] = dataclasses.field(
      default_factory=lambda: {
          'TV': 'flighting',
          'Display': 'flighting',
          'Social': 'continuity',
      }
  )
  flight_burst_weeks: dict[str, int] = dataclasses.field(
      default_factory=lambda: {'TV': 8, 'Display': 12, 'Social': 2}
  )
  flight_dark_weeks: dict[str, int] = dataclasses.field(
      default_factory=lambda: {'TV': 8, 'Display': 12, 'Social': 2}
  )
  flight_burst_cv: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.6, 'Display': 0.6, 'Social': 0.6}
  )
  flight_dark_cv: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.9, 'Display': 0.9, 'Social': 0.9}
  )
  flight_floor: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.2, 'Display': 0.5, 'Social': 0.5}
  )
  continuity_spike_weeks: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 1, 'Display': 1, 'Social': 1}
  )
  continuity_spike_cv: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.3, 'Display': 0.3, 'Social': 0.3}
  )
  continuity_gap_weeks: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 5, 'Display': 5, 'Social': 5}
  )
  continuity_gap_cv: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.57, 'Display': 0.57, 'Social': 0.57}
  )
  continuity_spike_amplitude: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 1.9, 'Display': 1.9, 'Social': 1.9}
  )

  # Autocorrelation of the additive week-to-week reach noise (an AR(1)
  # process instead of iid draws, since real execution noise persists
  # across adjacent weeks rather than resetting every week), per channel.
  # These were tuned against a real client's weekly geo media data: TV is
  # given the *lowest* phi/highest sd of the three, since real linear-TV
  # delivery is the choppiest, lowest-autocorrelation channel (week-to-week
  # spikes, not smooth multi-week blocks); Display is given the *highest*
  # phi/lowest sd, matching real programmatic-display delivery, which is the
  # smoothest/most persistent of the observed channels.
  reach_noise_ar1_phi: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.08, 'Display': 0.92, 'Social': 0.3}
  )
  time_reach_noise_sd: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 0.13, 'Display': 0.01, 'Social': 0.05}
  )
  # Degrees of freedom for the AR(1) reach noise's innovation distribution.
  # Student-t rather than Gaussian, so noise is occasionally "jagged" (a few
  # larger week-to-week swings) instead of uniformly smooth -- the scale is
  # rederived in `_simulate_ar1_reach_noise()` so the stationary sd still
  # matches `time_reach_noise_sd` regardless of this value.
  reach_noise_df: float = 4.0

  # Realistic CPM ($ per 1,000 impressions) per channel, +/- a variability
  # band to account for auction/market price fluctuation. TV/Display/Social
  # defaults reflect typical relative pricing (linear TV >> social >>
  # programmatic display).
  cpm_dollars: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 10.0, 'Display': 2.0, 'Social': 5.0}
  )
  cpm_variability_frac: float = 0.05

  # Baseline target incremental ROI per channel, used by
  # `calibrate_channel_effects()` to rescale `beta_m`/`beta_gm` after
  # media/cost are simulated. Without this, all channels draw from the same
  # beta_m hyperprior (Section 6) and end up with an essentially arbitrary
  # relative ROI split. Defaults reflect TV's high CPM/broad-reach
  # inefficiency vs. digital channels' lower CPM and tighter targeting.
  # `roi_ec_elasticity`/`roi_alpha_elasticity` below further adjust this
  # baseline by each channel's own curve shape.
  target_roi: dict[str, float] = dataclasses.field(
      default_factory=lambda: {'TV': 2.5, 'Display': 1.5, 'Social': 2.0}
  )

  # A channel's true incremental ROI plausibly isn't independent of its
  # adstock/saturation shape: a channel that saturates faster (lower ec_m,
  # often a more narrowly-targeted audience) tends to convert better per
  # exposure, and a channel with richer/more memorable creative (higher
  # alpha_m, more retained) tends to persuade more per exposure too --
  # rather than every channel's ROI being an assumption independent of its
  # curve shape. `simulate_adstock_hill_params()` adjusts each channel's
  # `target_roi` baseline above by
  # `(ec_m / cross-channel geometric mean) ** roi_ec_elasticity *
  #  (alpha_m / cross-channel geometric mean) ** roi_alpha_elasticity`
  # before `calibrate_channel_effects()` calibrates to it. `0.0` (the
  # default for both) is an exact no-op -- multiplier is 1.0 for every
  # channel -- reproducing the fully independent behavior used elsewhere in
  # this notebook series. Expected signs: `roi_ec_elasticity` negative
  # (larger ec_m -> lower ROI), `roi_alpha_elasticity` positive (larger
  # alpha_m -> higher ROI).
  roi_ec_elasticity: float = 0.0
  roi_alpha_elasticity: float = 0.0

  # Range each channel's adstock decay rate (`alpha_m`) is drawn uniformly
  # from in `simulate_adstock_hill_params()`, reflecting ad memory persisting
  # longer for channels with richer/longer creative exposure (TV) than for
  # brief, disposable exposure (Display), with Social in between. TV's range
  # is set apart from the other two, but Display's and Social's ranges
  # overlap, so the TV > Social > Display ordering holds on average but isn't
  # guaranteed every run.
  adstock_retention_range: dict[str, tuple[float, float]] = dataclasses.field(
      default_factory=lambda: {
          'TV': (0.4, 0.6),
          'Display': (0.0, 0.3),
          'Social': (0.1, 0.4),
      }
  )

  # Range each channel's Hill `slope_m` (curve-shape exponent -- the same
  # role as Robyn's `alpha` hyperparameter: https://facebookexperimental.
  # github.io/Robyn/docs/analysts-guide-to-MMM, not to be confused with
  # this simulator's own `alpha_m`/adstock retention, which is Robyn's
  # `theta`) is drawn uniformly from in `simulate_adstock_hill_params()`.
  # `slope=1` (the default for every channel below) gives Meridian's usual
  # concave-only Hill curve (`Hill(x)=x/(x+ec)`, maximal marginal value at
  # `x=0`, monotonically diminishing); `slope>1` gives a genuine S-curve
  # (near-zero response *and* near-zero marginal value close to `x=0`, a
  # real "needs a threshold of exposure before eliciting any response"
  # regime, before marginal value rises, peaks, then decays). `(1.0, 1.0)`
  # for every channel (the default) is an exact no-op, matching every other
  # notebook in this series, since Meridian's own default `slope_m` prior
  # is a hard `Deterministic(1.0)` for plain (non-RF) media channels
  # (`prior_distribution.py`) -- not just uninformative, but literally
  # unfittable unless a notebook explicitly overrides it. So a channel
  # given a true `slope_m != 1` here is one Meridian's default fitted
  # model is structurally incapable of representing, regardless of prior
  # informativeness on `ec_m`/`alpha_m`.
  slope_range: dict[str, tuple[float, float]] = dataclasses.field(
      default_factory=lambda: {
          'TV': (1.0, 1.0),
          'Display': (1.0, 1.0),
          'Social': (1.0, 1.0),
      }
  )

  # Adstock truncation window (weeks), shared by the ground-truth transform
  # below and `model_utils.build_model_spec`'s fitted `ModelSpec` -- both
  # must agree for a fair "recovered vs. true" comparison. Kept at Meridian's
  # own default (8) unless overridden; a channel whose true `adstock_
  # retention_range` sits high enough that an 8-week window truncates a
  # non-trivial tail of its geometric decay may need a larger value.
  max_lag: int = 8

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

  # Number of knots used to generate the true time-varying baseline `mu_t`
  # in `simulate_intercepts()`. `None` (the default) reproduces the
  # historical behavior: `n_knots_simul = n_times`, i.e. one independent,
  # unsmoothed `Normal(0, 2.0)` shock per week. Set this to match (or
  # deliberately mismatch) `model_utils.build_model_spec`'s fitted `knots`
  # argument -- e.g. `n_knots_mu_t=8` alongside a fitted model also using
  # `knots=8` -- to test whether a DGP/model flexibility mismatch in the
  # baseline spline (as opposed to `ec_m`/`alpha_m`/`eta_m` mis-specification)
  # is responsible for a channel's `roi_m` not recovering.
  n_knots_mu_t: int | None = None

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

  # 1b. Population from real demo data (alternative to Section 1, for use
  # with `simulate_media_from_real` below).
  def simulate_population_from_real(self, real_df: pd.DataFrame) -> tf.Tensor:
    """Sets geo_names/p_g from a real geo x time DataFrame's own population.

    Alternative entry point to `simulate_population()` for notebooks that
    want the rest of the DGP (controls, coefficients, adstock/Hill `ec_m`
    derivation, KPI generation) to run over the same real geos/populations
    that `simulate_media_from_real()` sources its media texture from, rather
    than the synthetic US-state population table.

    Args:
      real_df: A geo x time DataFrame with `geo` and `population` columns
        (one population value per geo, e.g. the demo's `geo_media_rf.csv`).
    """
    geo_pop = real_df.drop_duplicates('geo').set_index('geo')['population']
    self.geo_names = sorted(geo_pop.index)
    self.n_geos = len(self.geo_names)
    self.p_g = tf.constant(
        [geo_pop[g] for g in self.geo_names], dtype=tf.float32
    )
    self.national_population = tf.reduce_sum(self.p_g)
    return self.p_g

  def align_time_index_to_real(self, real_df: pd.DataFrame) -> None:
    """Overrides the config-derived time index with the real data's own dates.

    Must be called (after `simulate_population_from_real`) before
    `simulate_controls`/`simulate_media_from_real`, so every downstream
    tensor is indexed by the real demo dataset's actual calendar weeks
    instead of `config.start_date`. `config.n_times` must already match the
    real data's week count -- this only aligns the calendar dates, not the
    tensor lengths.

    Args:
      real_df: A geo x time DataFrame with a `time` column of week-start
        dates (any format `pd.to_datetime` accepts).
    """
    real_times = sorted(pd.to_datetime(real_df['time'].unique()))
    if len(real_times) != self.config.n_times:
      raise ValueError(
          f'Real data has {len(real_times)} weeks but config.n_times ='
          f' {self.config.n_times} -- set n_times to match the real data.'
      )
    self.time_index = pd.DatetimeIndex(real_times)
    self.time_names = [d.strftime('%Y-%m-%d') for d in real_times]

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

    self.seasonal_index_t = tf.constant(
        np.cos(
            2 * np.pi * (week_of_year - config.demand_seasonal_peak_week) / 52.0
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

  def _sample_renewal_run_lengths(
      self, mean: float, cv: float, n_draws: int
  ) -> np.ndarray:
    """Whole-week run lengths from a Gamma(`mean`, `cv`), floored at 1 week.

    A low `cv` concentrates run lengths tightly around `mean` (a flight
    calendar planned in advance); `cv` near 1 approaches the spread of a
    memoryless (geometric) process.
    """
    cv = max(cv, 1e-3)
    shape = 1.0 / (cv**2)
    rate = shape / mean
    draws = (
        tfp.distributions.Gamma(concentration=shape, rate=rate)
        .sample(n_draws)
        .numpy()
    )
    return np.maximum(1, np.round(draws)).astype(int)

  def _build_renewal_state(
      self,
      n_times: int,
      on_mean: float,
      on_cv: float,
      off_mean: float,
      off_cv: float,
  ) -> np.ndarray:
    """An alternating on(1)/off(0) run-length renewal process, `n_times` long."""
    # Worst case every draw lands at the length-1 floor, so `n_times` draws
    # per state always covers the full horizon.
    on_lengths = self._sample_renewal_run_lengths(on_mean, on_cv, n_times)
    off_lengths = self._sample_renewal_run_lengths(off_mean, off_cv, n_times)
    segments = []
    total = 0
    on = True
    i_on = i_off = 0
    while total < n_times:
      if on:
        length = on_lengths[i_on]
        i_on += 1
        segments.append(np.ones(length))
      else:
        length = off_lengths[i_off]
        i_off += 1
        segments.append(np.zeros(length))
      total += length
      on = not on
    return np.concatenate(segments)[:n_times]

  def _simulate_flighting_multiplier(self) -> tf.Tensor:
    """Per-channel execution texture: 'flighting' or 'continuity' (see config).

    Both styles share the same renewal-process mechanism -- alternating runs
    whose *lengths* are drawn from a Gamma distribution rather than a
    memoryless geometric one (see `_build_renewal_state`). 'flighting'
    channels alternate full-strength/`flight_floor` runs sized by
    `flight_burst_weeks`/`flight_dark_weeks`. 'continuity' channels alternate
    short `continuity_spike_amplitude` spikes and baseline runs, then
    renormalize to mean 1 so the texture doesn't shift the channel's overall
    execution level (matching the other multipliers' mean-~1 convention).
    """
    config = self.config
    mult_tm = np.empty(
        (config.n_times, config.n_imp_channels), dtype=np.float64
    )
    for m, ch in enumerate(config.channel_names):
      style = config.flight_style[ch]
      if style == 'flighting':
        state_t = self._build_renewal_state(
            config.n_times,
            config.flight_burst_weeks[ch],
            config.flight_burst_cv[ch],
            config.flight_dark_weeks[ch],
            config.flight_dark_cv[ch],
        )
        floor = config.flight_floor[ch]
        mult_tm[:, m] = floor + (1.0 - floor) * state_t
      elif style == 'continuity':
        state_t = self._build_renewal_state(
            config.n_times,
            config.continuity_spike_weeks[ch],
            config.continuity_spike_cv[ch],
            config.continuity_gap_weeks[ch],
            config.continuity_gap_cv[ch],
        )
        amplitude = config.continuity_spike_amplitude[ch]
        raw_t = 1.0 + (amplitude - 1.0) * state_t
        mult_tm[:, m] = raw_t / raw_t.mean()
      else:
        raise ValueError(f'Unknown flight_style {style!r} for channel {ch!r}')
    return tf.constant(mult_tm, dtype=self.p_g.dtype)

  def _simulate_ar1_reach_noise(self) -> tf.Tensor:
    """Per-channel AR(1) week-to-week reach noise (replaces iid noise)."""
    config = self.config
    df = config.reach_noise_df
    # Student-t has Var = scale^2 * df/(df-2) (df > 2); rescale each
    # channel's innovation `scale` so its AR(1) process's stationary sd still
    # equals `time_reach_noise_sd[channel]`, regardless of `df`.
    variance_inflation = df / (df - 2)
    phi_m = np.array(
        [config.reach_noise_ar1_phi[ch] for ch in config.channel_names]
    )
    sd_m = np.array(
        [config.time_reach_noise_sd[ch] for ch in config.channel_names]
    )
    innovation_scale_m = sd_m * np.sqrt(
        np.maximum(1.0 - phi_m**2, 1e-6) / variance_inflation
    )
    innovations_tm = tfp.distributions.StudentT(
        df=tf.constant(df, dtype=self.p_g.dtype),
        loc=tf.constant(0, dtype=self.p_g.dtype),
        scale=tf.constant(innovation_scale_m, dtype=self.p_g.dtype),
    ).sample(config.n_times)
    phi_m_t = tf.constant(phi_m, dtype=self.p_g.dtype)

    def step(prev_noise_m, innovation_m):
      return phi_m_t * prev_noise_m + innovation_m

    return tf.scan(
        step,
        innovations_tm,
        initializer=tf.zeros(
            [config.n_imp_channels], dtype=innovations_tm.dtype
        ),
    )

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

    # Time-varying execution multipliers layered on top of the flat
    # `current_reach_frac` target: annual seasonality and on/off flighting,
    # each independently parameterized per channel (see `SimulationConfig`).
    # Noise is AR(1) and per-channel rather than iid/global, so week-to-week
    # execution persists (or doesn't) with a strength tuned per channel.
    self.seasonal_mult_tm = self._simulate_seasonal_multiplier()
    self.flight_mult_tm = self._simulate_flighting_multiplier()
    ar1_reach_noise_tm = self._simulate_ar1_reach_noise()

    reach_frac_tm = (
        current_reach_frac_m[tf.newaxis, :]
        * self.seasonal_mult_tm
        * self.flight_mult_tm
        + ar1_reach_noise_tm
    )
    self.reach_frac_gtm = tf.clip_by_value(
        reach_frac_tm[tf.newaxis, :, :], 0.01, 0.95
    )

    # Frequency (impressions per person reached, per week).
    freq_low_m = tf.constant(
        [config.frequency_range[ch][0] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    freq_high_m = tf.constant(
        [config.frequency_range[ch][1] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    self.mean_frequency_m = (freq_low_m + freq_high_m) / 2
    self.frequency_gtm = tf.maximum(
        tfp.distributions.Uniform(freq_low_m, freq_high_m).sample(
            (self.n_geos, config.n_times)
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
            '/flighting shift the realized average), mean frequency ='
            f' {realized_frequency_m[i]:.2f} (target range'
            f' {config.frequency_range[ch]})'
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

  def _pivot_real_column(self, real_df: pd.DataFrame, col: str) -> np.ndarray:
    """Pivots a real-data column to a `(n_geos, n_times)` array, ordered to
    match `self.geo_names`/`self.time_names` (set by
    `simulate_population_from_real`/`align_time_index_to_real`)."""
    real_df = real_df.copy()
    real_df['time'] = pd.to_datetime(real_df['time']).dt.strftime('%Y-%m-%d')
    pivot = real_df.pivot(index='geo', columns='time', values=col)
    pivot = pivot.reindex(index=self.geo_names, columns=self.time_names)
    return pivot.to_numpy(dtype=np.float64)

  # 3b. Media from real demo data (alternative to Section 3).
  def simulate_media_from_real(
      self,
      real_df: pd.DataFrame,
      rf_source_map: dict[str, str],
      plain_source_map: dict[str, str],
      verbose: bool = True,
  ) -> tf.Tensor:
    """Builds media the same way `simulate_media()` does, but sources each
    channel's real-world execution *texture* (its actual geo x time
    reach/frequency, or impression, noise/variation) from a real dataset
    instead of synthetic seasonality/flighting/AR(1) generation --
    normalized per geo to a mean of 1 and rescaled to this channel's
    `config`-assumed `current_reach_frac`/`frequency_range` target, exactly
    as the synthetic version would target on average. Everything else
    (target audience per geo, adstock/Hill `ec_m` derivation, KPI
    generation) is unaffected, so the DGP's known ground truth still holds.

    Must be called after `simulate_population_from_real` and
    `align_time_index_to_real` (so `self.geo_names`/`self.time_names` match
    `real_df`).

    Args:
      real_df: A geo x time DataFrame with real reach/frequency/impression
        columns (e.g. the demo's `geo_media_rf.csv`), following that
        dataset's `<Channel>_reach`/`<Channel>_frequency`/`<Channel>_
        impression` naming convention.
      rf_source_map: `{config_channel_name: real_channel_prefix}` for
        channels backed by a real `(reach, frequency)` pair -- e.g.
        `{'TV': 'Channel3', 'Social': 'Channel3'}` reuses the same real
        reach/frequency series (independently rescaled per target channel).
      plain_source_map: `{config_channel_name: real_channel_prefix}` for
        channels backed by only a real impression column (no real
        reach/frequency split available) -- e.g. `{'Display': 'Channel0'}`.
        Reach/frequency for these channels are back-derived placeholders
        (`frequency = mean_frequency_m`, `reach_count = impressions /
        frequency`) purely so `to_dataframe()`'s output columns stay
        populated; they carry no independent real-data information.
      verbose: Whether to print sparsity/reach/frequency diagnostics,
        matching `simulate_media()`'s verbose block.

    Returns:
      `self.impression_gtm`.
    """
    config = self.config
    covered = set(rf_source_map) | set(plain_source_map)
    if covered != set(config.channel_names):
      raise ValueError(
          'rf_source_map/plain_source_map together must cover exactly'
          f' {config.channel_names}, got {sorted(covered)}'
      )

    target_audience_pop_frac_m = tf.constant(
        [config.target_audience_pop_frac[ch] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    freq_low_m = tf.constant(
        [config.frequency_range[ch][0] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    freq_high_m = tf.constant(
        [config.frequency_range[ch][1] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    self.mean_frequency_m = (freq_low_m + freq_high_m) / 2

    # Target-audience size per geo -- identical mechanism to
    # `simulate_media()`, just driven by the real per-geo population set by
    # `simulate_population_from_real`.
    geo_audience_mult_gm = tfp.distributions.TruncatedNormal(
        1.0, config.geo_audience_heterogeneity_sd, 0.5, 1.5
    ).sample((self.n_geos, config.n_imp_channels))
    self.audience_g_m = (
        self.p_g[:, tf.newaxis]
        * target_audience_pop_frac_m[tf.newaxis, :]
        * geo_audience_mult_gm
    )
    audience_gm = self.audience_g_m.numpy()
    self.audience_national_m = tf.reduce_sum(self.audience_g_m, axis=0)

    reach_frac_gtm = np.zeros(
        (self.n_geos, config.n_times, config.n_imp_channels)
    )
    frequency_gtm = np.zeros(
        (self.n_geos, config.n_times, config.n_imp_channels)
    )
    impression_gtm = np.zeros(
        (self.n_geos, config.n_times, config.n_imp_channels)
    )

    for ch_idx, ch in enumerate(config.channel_names):
      target_reach_frac = config.current_reach_frac[ch]
      target_freq = self.mean_frequency_m.numpy()[ch_idx]
      if ch in rf_source_map:
        prefix = rf_source_map[ch]
        reach_raw_gt = self._pivot_real_column(
            real_df, _add_suffix(prefix, REACH_COL_SUFFIX)
        )
        freq_raw_gt = self._pivot_real_column(
            real_df, _add_suffix(prefix, FREQUENCY_COL_SUFFIX)
        )
        # Per-geo normalization to mean 1 -- preserves each geo's own real
        # relative geo x time variation (noise, autocorrelation, ramp-up
        # zeros) while the target level below sets the overall scale.
        reach_shape_gt = reach_raw_gt / np.maximum(
            reach_raw_gt.mean(axis=1, keepdims=True), 1e-9
        )
        freq_shape_gt = freq_raw_gt / np.maximum(
            freq_raw_gt.mean(axis=1, keepdims=True), 1e-9
        )
        reach_frac_ch_gt = np.clip(
            target_reach_frac * reach_shape_gt, 0.0, 0.98
        )
        frequency_ch_gt = np.maximum(target_freq * freq_shape_gt, 0.5)
        reach_count_ch_gt = (
            reach_frac_ch_gt * audience_gm[:, ch_idx : ch_idx + 1]
        )
        impression_ch_gt = np.round(reach_count_ch_gt * frequency_ch_gt)
      else:
        prefix = plain_source_map[ch]
        impr_raw_gt = self._pivot_real_column(
            real_df, _add_suffix(prefix, IMPRESSIONS_COL_SUFFIX)
        )
        impr_shape_gt = impr_raw_gt / np.maximum(
            impr_raw_gt.mean(axis=1, keepdims=True), 1e-9
        )
        target_level_g = (
            audience_gm[:, ch_idx] * target_reach_frac * target_freq
        )
        impression_ch_gt = np.round(
            target_level_g[:, np.newaxis] * impr_shape_gt
        )
        # No real reach/frequency split exists for a plain-impression source
        # channel -- back-derive placeholders at a constant frequency purely
        # so `to_dataframe()` stays populated (see docstring).
        frequency_ch_gt = np.full_like(impression_ch_gt, target_freq)
        reach_count_ch_gt = impression_ch_gt / target_freq
        reach_frac_ch_gt = (
            reach_count_ch_gt / audience_gm[:, ch_idx : ch_idx + 1]
        )

      reach_frac_gtm[:, :, ch_idx] = reach_frac_ch_gt
      frequency_gtm[:, :, ch_idx] = frequency_ch_gt
      impression_gtm[:, :, ch_idx] = impression_ch_gt

    self.reach_frac_gtm = tf.constant(reach_frac_gtm, dtype=self.p_g.dtype)
    self.frequency_gtm = tf.constant(frequency_gtm, dtype=self.p_g.dtype)
    self.reach_count_gtm = (
        self.reach_frac_gtm * self.audience_g_m[:, tf.newaxis, :]
    )
    self.impression_gtm = tf.constant(impression_gtm, dtype=self.p_g.dtype)

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
        source = rf_source_map.get(ch) or plain_source_map.get(ch)
        print(
            f'{ch} (from real {source}): mean reach % ='
            f' {realized_reach_frac_m[i]:.3f} (target'
            f' {config.current_reach_frac[ch]}), mean frequency ='
            f' {realized_frequency_m[i]:.2f} (target range'
            f' {config.frequency_range[ch]})'
        )

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
    n_knots_simul = config.n_knots_mu_t or config.n_times
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
    """Derives ec_m from "half of audience at a fixed effective frequency",
    and adjusts `target_roi` by each channel's own resulting `ec_m`/`alpha_m`
    (see `roi_ec_elasticity`/`roi_alpha_elasticity` on `SimulationConfig`).
    """
    config = self.config
    half_sat_reach_count_national_m = 0.5 * self.audience_national_m
    # Convert that reach count to an impression count using a fixed
    # literature-motivated "effective frequency" (`config.saturation_
    # frequency`), not each channel's own current `mean_frequency_m`. Using
    # the channel's own current frequency here would make it cancel out of
    # `ec_m` entirely (both `ec_impressions` and `median_m` below scale
    # linearly with it), so a channel's current weekly frequency would never
    # actually affect how saturated it looks. Anchoring to a fixed threshold
    # instead decouples `ec_m` from current execution frequency and lets
    # `frequency_range` genuinely compete against that threshold.
    saturation_frequency_m = tf.constant(
        [config.saturation_frequency[ch] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    ec_impressions_national_m = (
        saturation_frequency_m * half_sat_reach_count_national_m
    )
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

    slope_low_m = tf.constant(
        [config.slope_range[ch][0] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    slope_high_m = tf.constant(
        [config.slope_range[ch][1] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    self.slope_m = tfp.distributions.Uniform(slope_low_m, slope_high_m).sample()
    print(f'ec_m = {self.ec_m.numpy()}')

    target_roi_base_m = tf.constant(
        [config.target_roi[ch] for ch in config.channel_names],
        dtype=self.p_g.dtype,
    )
    ec_m_gmean = tf.exp(tf.reduce_mean(tf.math.log(self.ec_m)))
    alpha_m_gmean = tf.exp(tf.reduce_mean(tf.math.log(self.alpha_m)))
    roi_shape_multiplier_m = (
        self.ec_m / ec_m_gmean
    ) ** config.roi_ec_elasticity * (
        self.alpha_m / alpha_m_gmean
    ) ** config.roi_alpha_elasticity
    self.target_roi_m = target_roi_base_m * roi_shape_multiplier_m
    return self.ec_m

  # 8. Transform the media.
  def transform_media(self) -> tf.Tensor:
    """Applies the adstock decay and Hill saturation transforms to media."""
    hill_transformer = adstock_hill.HillTransformer(
        ec=self.ec_m, slope=self.slope_m
    )
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=self.alpha_m,
        max_lag=self.config.max_lag,
        n_times_output=self.config.n_times,
    )
    self.media_transformed = hill_transformer.forward(
        adstock_transformer.forward(self.transformed_ipc_gtm)
    )
    return self.media_transformed

  # 8b. Calibrate channel effect sizes to hit realistic target ROIs.
  def calibrate_channel_effects(self) -> tf.Tensor:
    """Rescales beta_m/beta_gm so realized ROI matches `self.target_roi_m`
    (the shape-adjusted target computed in `simulate_adstock_hill_params()`).

    Must run after `simulate_adstock_hill_params()`, `transform_media()`,
    and `simulate_cost_and_unit_value()`, and before
    `generate_kpi_and_revenue()`. `beta_m` is drawn from an identical
    hyperprior across channels (Section 6), so without this step each
    channel's relative contribution/ROI is essentially arbitrary -- this
    reverse-engineers the scale the same way `ec_m` is already
    reverse-engineered from a "50% of audience" assumption (Section 7).
    Incremental revenue is linear in `beta_gm` (media/cost/unit-value are
    already fixed at this point), so a single rescale hits the target
    exactly, in expectation over the geo-level `beta_gm` heterogeneity.
    """
    current_incremental_revenue_m = tf.einsum(
        'g,gt,gtm,gm->m',
        self.p_g,
        self.unit_value,
        self.media_transformed,
        self.beta_gm,
    )
    current_cost_m = tf.einsum('gtm->m', self.cost_gtm)
    scale_m = (
        self.target_roi_m * current_cost_m
    ) / current_incremental_revenue_m

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
    # Revenue is KPI valued at `unit_value` (~0.035), matching the units
    # `compute_ground_truth`'s `incremental_revenue_m` is computed in. An
    # earlier `tf.ones_like(self.unit_value)` here left `revenue_gt` equal to
    # `kpi_gt`, so any media-contribution share taken as
    # `incremental_revenue_m / revenue_gt` came out ~1/unit_value too small.
    self.revenue_gt = self.kpi_gt * self.unit_value
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
