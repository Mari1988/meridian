import tensorflow as tf # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from meridian.model import model, adstock_hill
from meridian.analysis import visualizer, analyzer

class ExportMediaTransformations:
  """Class to export the media transformations."""

  def __init__(self, mmm: model.Meridian):
    self.mmm = mmm
    self.media_channels = self.mmm.input_data.media.media_channel.values.tolist()

    # get the point estimates based on the median of the posterior
    posterior = self.mmm.inference_data.posterior
    self.alpha_m = posterior.alpha_m.median(dim=['chain', 'draw'])
    self.ec_m = posterior.ec_m.median(dim=['chain', 'draw'])
    self.slope_m = posterior.slope_m.median(dim=['chain', 'draw'])

    # get the media parameters
    self.media_parameters_df = self.get_media_parameters_df()
    self.media_scaled, self.adstocked_media, self.saturated_media, self.media_trans_journey_df = self.get_media_transformations_df()
    self.media_sum_to_debug_df = self.get_media_sum_to_debug_df()

  def plot_media_journey_all(self):
    for chnl in self.media_channels:
      fig = self.plot_media_journey(self.media_trans_journey_df, chnl)
      display(fig)

  def get_media_transformations_df(self):

    # 1. get the scaled media data
    media_scaled = self.mmm.media_tensors.media_scaled
    media_scaled_tm = self.get_geo_weighted_media_data(media_scaled, weights=self.mmm.input_data.media)

    # 2. get the adstocked media data
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=tf.constant(self.alpha_m, dtype=tf.float32),
        max_lag=self.mmm.model_spec.max_lag,
        n_times_output=self.mmm.n_times,
        decay_function=self.mmm.model_spec.adstock_decay_function,
    )
    adstocked_media = adstock_transformer.forward(media_scaled)
    adstocked_media_tm = self.get_geo_weighted_media_data(adstocked_media, weights=self.mmm.input_data.media)

    # 3. get the saturated media data
    hill_transformer = adstock_hill.HillTransformer(
        ec=tf.constant(self.ec_m, dtype=tf.float32),
        slope=tf.constant(self.slope_m, dtype=tf.float32)
    )
    saturated_media = hill_transformer.forward(adstocked_media)
    saturated_media_tm = self.get_geo_weighted_media_data(saturated_media, weights=self.mmm.input_data.media)

    # 4. finally wrap everything to a dataframe
    media_channels = self.mmm.input_data.media.media_channel.values.tolist()
    time_coords = self.mmm.input_data.time.values

    media_scaled_tm_df = pd.DataFrame(media_scaled_tm, columns=[f'{m}_scaled' for m in media_channels])
    media_scaled_tm_df.insert(0, 'time', time_coords)

    adstocked_media_tm_df = pd.DataFrame(adstocked_media_tm, columns=[f'{m}_adstocked' for m in media_channels])
    adstocked_media_tm_df.insert(0, 'time', time_coords)

    saturated_media_tm_df = pd.DataFrame(saturated_media_tm, columns=[f'{m}_saturated' for m in media_channels])
    saturated_media_tm_df.insert(0, 'time', time_coords)

    media_trans_journey_df = media_scaled_tm_df. \
      merge(adstocked_media_tm_df, on='time', how='left'). \
        merge(saturated_media_tm_df, on='time', how='left')
    media_trans_journey_df['time'] = pd.to_datetime(media_trans_journey_df['time'])

    return media_scaled, adstocked_media, saturated_media, media_trans_journey_df

  def get_media_parameters_df(self):
    media_parameters_df = pd.DataFrame({
      'media_channel': self.media_channels,
      'alpha': self.alpha_m,
      'ec': self.ec_m,
      'slope': self.slope_m
    })
    return media_parameters_df

  def get_media_sum_to_debug_df(self):
    media_sum_to_debug_df = pd.DataFrame({
      'media_channel': self.media_channels,
      'media_scaled_sum': tf.reduce_sum(self.media_scaled, axis=(0, 1)).numpy(),
      'adstocked_media_sum': tf.reduce_sum(self.adstocked_media, axis=(0, 1)).numpy(),
      'saturated_media_sum': tf.reduce_sum(self.saturated_media, axis=(0, 1)).numpy()
    })
    return media_sum_to_debug_df

  @staticmethod
  def get_geo_weighted_media_data(media_transformed, weights):
    raw_media_gtm = tf.convert_to_tensor(weights, dtype=media_transformed.dtype)
    media_times_transformed_tm = tf.reduce_sum(tf.multiply(raw_media_gtm, media_transformed), axis=0)  # (time, channel)
    raw_media_sum_tm = tf.reduce_sum(raw_media_gtm, axis=0)  # (time, channel)
    return tf.math.divide_no_nan(media_times_transformed_tm, raw_media_sum_tm)

  @staticmethod
  def plot_media_journey(data, chnl):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=data, x='time', y=f'{chnl}_scaled', ax=ax, label='scaled')
    sns.lineplot(data=data, x='time', y=f'{chnl}_adstocked', ax=ax, label='adstocked', color='green')
    secondary_ax = ax.twinx()
    sns.lineplot(data=data, x='time', y=f'{chnl}_saturated', linestyle='--', ax=secondary_ax, label='saturated', color='red')
    secondary_ax.legend(loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    ax.set_title(f'{chnl} scaled vs adstocked')
    plt.close()
    return fig
