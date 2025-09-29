import pandas as pd
import xarray as xr
import natsort
import tensorflow as tf
import numpy as np
import dataclasses
import functools

from meridian import constants as c
from meridian.model import knots

# -----------------------------------------------------------------------------
# InputDataBuilderMini
# -----------------------------------------------------------------------------

class InputDataBuilderMini:
  """Builds `InputData` from DataFrames."""
  def __init__(self, model_config: dict) -> None:
    self.time_col: str = model_config.get('time_col', c.TIME)
    self.geo_col: str = model_config.get('geo_col', c.GEO)
    self.population_col: str = model_config.get('population_col', c.POPULATION)
    self.kpi_col: str = model_config.get('kpi_col', c.KPI)
    self.kpi_type: str = model_config.get('kpi_type')
    self.revenue_per_kpi_col: str = model_config.get('revenue_per_kpi_col', c.REVENUE_PER_KPI)

    # media inputs
    self.media_cols: list[str] = model_config.get('media_cols', [])
    self.media_spend_cols: list[str] = model_config.get('media_spend_cols', [])
    self.media_channels: list[str] = model_config.get('media_channels', [])

    # reach based media inputs
    self.reach_cols: list[str] = model_config.get('reach_cols', [])
    self.frequency_cols: list[str] = model_config.get('frequency_cols', [])
    self.rf_spend_cols: list[str] = model_config.get('rf_spend_cols', [])
    self.rf_channels: list[str] = model_config.get('rf_channels', [])

    # control inputs
    self.control_cols: list[str] = model_config.get('control_cols', [])

  @property
  def n_times(self):
    return len(self.time)

  @property
  def n_geos(self):
    return len(self.geo)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  def knot_info(self, n_knots: int) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=n_knots or self.n_times,
        is_national=self.is_national,
    )

  def with_time(self, df: pd.DataFrame):
    time_df = df.copy()
    time_idx = pd.to_datetime(time_df[self.time_col].unique())
    time_str = time_idx.sort_values().strftime('%Y-%m-%d').values.tolist()
    time_sorted = natsort.natsorted(time_str)

    self.time = xr.DataArray(time_sorted, dims=[c.TIME], coords={c.TIME: time_sorted})
    return self

  def with_geo(self, df:pd.DataFrame):
    geo_df = df.copy()
    geo_list = geo_df[self.geo_col].unique().tolist()
    geo_sorted = natsort.natsorted(geo_list)
    self.geo = xr.DataArray(geo_sorted, dims=[c.GEO], coords={c.GEO: geo_sorted})
    return self

  def with_kpi(self, df: pd.DataFrame):
    kpi_df = df.copy()
    data = kpi_df.set_index([self.geo_col, self.time_col])[self.kpi_col]
    self.kpi = data. \
      rename(c.KPI). \
      rename_axis([c.GEO, c.TIME]). \
      to_xarray().reindex(geo=self.geo, time=self.time)
    return self

  def with_population(self, df: pd.DataFrame):
    pop_df = df.copy()
    data = pop_df. \
      set_index([self.geo_col])[self.population_col]. \
      groupby(self.geo_col).mean()
    self.population = data. \
      rename(c.POPULATION). \
      rename_axis([c.GEO]). \
      to_xarray().reindex(geo=self.geo)
    return self

  def with_revenue_per_kpi(self, df: pd.DataFrame):
    revenue_per_kpi_df = df.copy()
    data = revenue_per_kpi_df.set_index([self.geo_col, self.time_col])[self.revenue_per_kpi_col]
    self.revenue_per_kpi = data. \
      rename(c.POPULATION). \
      rename_axis([c.GEO, c.TIME]). \
      to_xarray().reindex(geo=self.geo, time=self.time)
    return self

  def with_controls(self, df: pd.DataFrame):
    controls_df = df.copy()
    data = controls_df.set_index([self.geo_col, self.time_col])[self.control_cols].stack()
    self.controls = data. \
      rename(c.CONTROLS). \
      rename_axis([c.GEO, c.TIME, c.CONTROL_VARIABLE]). \
      to_xarray().reindex(geo=self.geo, time=self.time)
    return self

  def with_media(self, df: pd.DataFrame):
    media_df = df.copy()

    # impressions
    data = media_df.set_index([self.geo_col, self.time_col])[self.media_cols].stack()
    self.media = data. \
      rename(c.MEDIA). \
      rename_axis([c.GEO, c.TIME, c.MEDIA_CHANNEL]). \
      to_xarray().reindex(geo=self.geo, time=self.time)

    # spend
    data = media_df.set_index([self.geo_col, self.time_col])[self.media_spend_cols].stack()
    self.media_spend = data. \
      rename(c.MEDIA_SPEND). \
      rename_axis([c.GEO, c.TIME, c.MEDIA_CHANNEL]). \
      to_xarray().reindex(geo=self.geo, time=self.time)
    return self

  def with_reach(self, df: pd.DataFrame):
    reach_df = df.copy()

    # reach
    data = reach_df.set_index([self.geo_col, self.time_col])[self.reach_cols].stack()
    self.reach = data. \
      rename(c.REACH). \
      rename_axis([c.GEO, c.TIME, c.RF_CHANNEL]). \
      to_xarray().reindex(geo=self.geo, time=self.time)

    # frequency
    data = reach_df.set_index([self.geo_col, self.time_col])[self.frequency_cols].stack()
    self.frequency = data. \
      rename(c.FREQUENCY). \
      rename_axis([c.GEO, c.TIME, c.RF_CHANNEL]). \
      to_xarray().reindex(geo=self.geo, time=self.time)

    # spend
    data = reach_df.set_index([self.geo_col, self.time_col])[self.rf_spend_cols].stack()
    self.rf_spend = data. \
      rename(c.RF_SPEND). \
      rename_axis([c.GEO, c.TIME, c.RF_CHANNEL]). \
      to_xarray().reindex(geo=self.geo, time=self.time)
    return self

  def build(self, df: pd.DataFrame):
    input_data = self. \
      with_time(df). \
      with_geo(df). \
      with_kpi(df). \
      with_population(df). \
      with_revenue_per_kpi(df). \
      with_controls(df). \
      with_media(df). \
      with_reach(df)
    return input_data

# -----------------------------------------------------------------------------
# MediaTensors Container
# -----------------------------------------------------------------------------

class MediaScalerMini:
  """ scaler for media tensors """
  def __init__(self, media: tf.Tensor, population: tf.Tensor):
    # media scaled by population
    per_cap_media = tf.math.divide_no_nan(media, population[:, tf.newaxis, tf.newaxis])
    per_cap_media = tf.where(per_cap_media == 0, np.nan, per_cap_media)

    # median of media scaled by population
    self.population_scaled_median_m = tf.numpy_function(
      func=lambda x: np.nanmedian(x, axis=(0, 1)),
      inp=[per_cap_media],
      Tout=tf.float32
    )

    # scaling factors (population * median)
    self.scaling_factors_gm = tf.einsum('...g, ...m -> ...gm', population, self.population_scaled_median_m)

  def forward(self, media: tf.Tensor):
    media_scaled = tf.math.divide_no_nan(media, self.scaling_factors_gm[:, tf.newaxis, :])
    return media_scaled
  def inverse(self, media_scaled: tf.Tensor):
    media = tf.math.multiply_no_nan(media_scaled, self.scaling_factors_gm[:, tf.newaxis, :])
    return media


@dataclasses.dataclass(frozen=True)
class MediaTensorsMini:
  """ container class for media tensors """
  media: tf.Tensor | None = None
  media_spend: tf.Tensor | None = None
  media_transformer: MediaScalerMini | None = None
  media_scaled: tf.Tensor | None = None

  @staticmethod
  def build(input_data: InputDataBuilderMini):
    media = tf.convert_to_tensor(input_data.media, dtype=tf.float32)
    media_spend = tf.convert_to_tensor(input_data.media_spend, dtype=tf.float32)
    _population = tf.convert_to_tensor(input_data.population, dtype=tf.float32)
    media_transformer = MediaScalerMini(media, _population)
    media_scaled = media_transformer.forward(media)
    return MediaTensorsMini(media=media, media_spend=media_spend, media_transformer=media_transformer, media_scaled=media_scaled)


@dataclasses.dataclass(frozen=True)
class RFTensorsMini:
  """ container class for reach and frequency tensors """
  reach: tf.Tensor | None = None
  frequency: tf.Tensor | None = None
  rf_impressions: tf.Tensor | None = None
  rf_spend: tf.Tensor | None = None
  reach_transformer: MediaScalerMini | None = None
  reach_scaled: tf.Tensor | None = None

  @staticmethod
  def build(input_data: InputDataBuilderMini):
    reach = tf.convert_to_tensor(input_data.reach, dtype=tf.float32)
    frequency = tf.convert_to_tensor(input_data.frequency, dtype=tf.float32)
    rf_impressions = reach * frequency
    rf_spend = tf.convert_to_tensor(input_data.rf_spend, dtype=tf.float32)
    _population = tf.convert_to_tensor(input_data.population, dtype=tf.float32)
    reach_transformer = MediaScalerMini(reach, _population)
    reach_scaled = reach_transformer.forward(reach)
    return RFTensorsMini(reach=reach, frequency=frequency, rf_impressions=rf_impressions, rf_spend=rf_spend, reach_transformer=reach_transformer, reach_scaled=reach_scaled)


# -----------------------------------------------------------------------------
# KPITensors Container
# -----------------------------------------------------------------------------

class KPITransformerMini:
  """ scaler for kpi tensors """
  def __init__(self, kpi: tf.Tensor, population: tf.Tensor):
    per_cap_kpi = tf.math.divide_no_nan(kpi, population[:, tf.newaxis])
    self.population = population
    self.population_scaled_kpi_mean = tf.reduce_mean(per_cap_kpi)
    self.population_scaled_kpi_std = tf.math.reduce_std(per_cap_kpi)

  def forward(self, kpi: tf.Tensor):
    kpi_scaled = tf.math.divide_no_nan(
      tf.math.divide_no_nan(kpi, self.population[:, tf.newaxis]) - self.population_scaled_kpi_mean,
      self.population_scaled_kpi_std
    )
    return kpi_scaled

  def inverse(self, kpi_scaled: tf.Tensor):
    kpi = tf.math.multiply_no_nan(
      kpi_scaled * self.population_scaled_kpi_std + self.population_scaled_kpi_mean,
      self.population[:, tf.newaxis]
    )
    return kpi

@dataclasses.dataclass(frozen=True)
class KPITensorsMini:
  kpi: tf.Tensor | None = None
  kpi_scaled: tf.Tensor | None = None
  kpi_transformer: KPITransformerMini | None = None

  @staticmethod
  def build(input_data: InputDataBuilderMini):
    kpi = tf.convert_to_tensor(input_data.kpi, dtype = tf.float32)  # (g, t)
    _population = tf.convert_to_tensor(input_data.population, dtype=tf.float32)  # (g)
    kpi_transformer = KPITransformerMini(kpi, _population)
    kpi_scaled = kpi_transformer.forward(kpi)
    return KPITensorsMini(kpi=kpi, kpi_scaled=kpi_scaled, kpi_transformer=kpi_transformer)

# -----------------------------------------------------------------------------
# ControlsTensors Container
# -----------------------------------------------------------------------------

class ControlsTransformerMini:
  """ scaler for controls tensors """
  def __init__(self, controls: tf.Tensor):
    self._means = tf.reduce_mean(controls, axis=(0, 1))  # axis along geo and time
    self._stdevs = tf.math.reduce_std(controls, axis=(0, 1))

  def forward(self, controls: tf.Tensor):
    controls_scaled = tf.math.divide_no_nan(controls - self._means, self._stdevs)
    return controls_scaled

  def inverse(self, controls_scaled: tf.Tensor):
    controls = tf.math.multiply_no_nan(controls_scaled, self._stdevs) + self._means
    return controls

@dataclasses.dataclass(frozen=True)
class ControlsTensorsMini:
  controls: tf.Tensor | None = None
  controls_scaled: tf.Tensor | None = None
  controls_transformer: ControlsTransformerMini | None = None

  @staticmethod
  def build(input_data: InputDataBuilderMini):
    controls = tf.convert_to_tensor(input_data.controls, dtype=tf.float32)
    controls_transformer = ControlsTransformerMini(controls)
    controls_scaled = controls_transformer.forward(controls)
    return ControlsTensorsMini(controls=controls, controls_scaled=controls_scaled, controls_transformer=controls_transformer)
