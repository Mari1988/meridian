# Copyright 2025 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fast response curves computation using median parameters.

This module provides a computationally efficient way to generate media response
curves by using median parameter values from the posterior distribution instead
of full Bayesian sampling. This approach provides significant speed improvements
while maintaining mathematical accuracy for point estimates.
"""

from collections.abc import Sequence
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr
import altair as alt

from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import formatter
from meridian.analysis import summary_text
from meridian.model import adstock_hill
from meridian.model import model


__all__ = [
    'FastResponseCurves',
]


class FastResponseCurves:
  """Generates fast weekly spend response curves using median parameter estimates.

  This class provides an efficient alternative to the full Bayesian response
  curve computation by using median parameter values and weekly median spend 
  baselines, creating actionable weekly spend scenarios for media execution teams.
  This approach offers significant computational speedup while maintaining 
  accuracy for point estimates.
  """

  def __init__(self, meridian: model.Meridian):
    """Initializes the FastResponseCurves with median parameters.

    Args:
      meridian: Fitted Meridian model with posterior samples.

    Raises:
      ValueError: If the model hasn't been fitted (no posterior data).
    """
    if not hasattr(meridian.inference_data, c.POSTERIOR):
      raise ValueError(
          'FastResponseCurves requires a fitted model with posterior data.'
      )

    self._meridian = meridian
    self._extract_median_parameters()
    self._validate_parameters()

  def _extract_median_parameters(self):
    """Extracts median parameter values from posterior samples."""
    posterior = self._meridian.inference_data.posterior

    # Initialize parameter storage
    self.media_params = {}
    self.rf_params = {}
    self.organic_media_params = {}
    self.organic_rf_params = {}
    self.non_media_params = {}

    # Extract media parameters if they exist
    if self._meridian.n_media_channels > 0:
      self.media_params = {
          'ec': np.median(posterior[c.EC_M].values, axis=(0, 1)),
          'slope': np.median(posterior[c.SLOPE_M].values, axis=(0, 1)),
          'alpha': np.median(posterior[c.ALPHA_M].values, axis=(0, 1)),
          'beta': np.median(posterior[c.BETA_GM].values, axis=(0, 1)),
      }

    # Extract R&F parameters if they exist
    if self._meridian.n_rf_channels > 0:
      self.rf_params = {
          'ec': np.median(posterior[c.EC_RF].values, axis=(0, 1)),
          'slope': np.median(posterior[c.SLOPE_RF].values, axis=(0, 1)),
          'alpha': np.median(posterior[c.ALPHA_RF].values, axis=(0, 1)),
          'beta': np.median(posterior[c.BETA_GRF].values, axis=(0, 1)),
      }

    # Extract organic media parameters if they exist
    if self._meridian.n_organic_media_channels > 0:
      self.organic_media_params = {
          'ec': np.median(posterior[c.EC_OM].values, axis=(0, 1)),
          'slope': np.median(posterior[c.SLOPE_OM].values, axis=(0, 1)),
          'alpha': np.median(posterior[c.ALPHA_OM].values, axis=(0, 1)),
          'beta': np.median(posterior[c.BETA_GOM].values, axis=(0, 1)),
      }

    # Extract organic R&F parameters if they exist
    if self._meridian.n_organic_rf_channels > 0:
      self.organic_rf_params = {
          'ec': np.median(posterior[c.EC_ORF].values, axis=(0, 1)),
          'slope': np.median(posterior[c.SLOPE_ORF].values, axis=(0, 1)),
          'alpha': np.median(posterior[c.ALPHA_ORF].values, axis=(0, 1)),
          'beta': np.median(posterior[c.BETA_GORF].values, axis=(0, 1)),
      }

    # Extract non-media treatment parameters if they exist
    if self._meridian.non_media_treatments is not None:
      self.non_media_params = {
          'gamma': np.median(posterior[c.GAMMA_GN].values, axis=(0, 1)),
      }

  def _validate_parameters(self):
    """Validates that extracted parameters have correct shapes."""
    # Validate media parameters
    if self.media_params:
      expected_channels = self._meridian.n_media_channels
      for param_name, param_value in self.media_params.items():
        if param_name == 'beta':
          expected_shape = (self._meridian.n_geos, expected_channels)
        else:
          expected_shape = (expected_channels,)

        if param_value.shape != expected_shape:
          raise ValueError(
              f'Media parameter {param_name} has shape {param_value.shape}, '
              f'expected {expected_shape}'
          )

    # Validate R&F parameters
    if self.rf_params:
      expected_channels = self._meridian.n_rf_channels
      for param_name, param_value in self.rf_params.items():
        if param_name == 'beta':
          expected_shape = (self._meridian.n_geos, expected_channels)
        else:
          expected_shape = (expected_channels,)

        if param_value.shape != expected_shape:
          raise ValueError(
              f'R&F parameter {param_name} has shape {param_value.shape}, '
              f'expected {expected_shape}'
          )

  def _scale_media_data(self, multiplier: float, by_reach: bool = True) -> tuple[
      Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]
  ]:
    """Scales media execution data by the given multiplier.

    Args:
      multiplier: Factor to scale media execution by.
      by_reach: For R&F channels, whether to scale reach (True) or frequency.

    Returns:
      Tuple of (scaled_media, scaled_reach, scaled_frequency) tensors.
    """
    scaled_media = None
    scaled_reach = None
    scaled_frequency = None

    # Scale standard media if it exists
    if self._meridian.media_tensors.media is not None:
      scaled_media = self._meridian.media_tensors.media * multiplier

    # Scale R&F data if it exists
    if (self._meridian.rf_tensors.reach is not None and
        self._meridian.rf_tensors.frequency is not None):
      if by_reach:
        scaled_reach = self._meridian.rf_tensors.reach * multiplier
        scaled_frequency = self._meridian.rf_tensors.frequency
      else:
        scaled_reach = self._meridian.rf_tensors.reach
        scaled_frequency = self._meridian.rf_tensors.frequency * multiplier

    return scaled_media, scaled_reach, scaled_frequency

  def _compute_spend_amounts(self,
                           spend_multipliers: Sequence[float],
                           selected_times: Optional[Sequence[str]] = None,
                           selected_geos: Optional[Sequence[str]] = None) -> tf.Tensor:
    """Computes actual weekly spend amounts for each multiplier and channel.

    Args:
      spend_multipliers: List of multipliers to apply to weekly median spend.
      selected_times: Optional subset of time periods to include.
      selected_geos: Optional subset of geos to include.

    Returns:
      Tensor of shape (n_multipliers, n_channels) with weekly spend amounts.
    """
    # Get historical spend data
    spend_tensors = []
    if self._meridian.media_tensors.media_spend is not None:
      spend_tensors.append(self._meridian.media_tensors.media_spend)
    if self._meridian.rf_tensors.rf_spend is not None:
      spend_tensors.append(self._meridian.rf_tensors.rf_spend)

    if not spend_tensors:
      raise ValueError('No spend data available in the model.')

    total_spend = tf.concat(spend_tensors, axis=-1)

    # Filter by selected dimensions if needed
    if tf.rank(total_spend) == 3:  # Has geo and time dimensions
      # Apply geo/time filtering similar to existing analyzer
      if selected_geos is not None:
        geo_indices = [
            i for i, geo in enumerate(self._meridian.input_data.geo.values)
            if geo in selected_geos
        ]
        total_spend = tf.gather(total_spend, geo_indices, axis=0)

      if selected_times is not None:
        time_indices = [
            i for i, time in enumerate(self._meridian.input_data.time.values)
            if time in selected_times
        ]
        total_spend = tf.gather(total_spend, time_indices, axis=1)

      # Calculate weekly median spend: sum across geos, then median across time
      total_spend_by_geo_time = tf.reduce_sum(total_spend, axis=0)  # Sum across geos
      weekly_median_spend = tfp.stats.percentile(total_spend_by_geo_time, 50.0, axis=0)  # Median across time
      total_spend = weekly_median_spend

    # Apply multipliers to get spend matrix
    spend_multipliers_tensor = tf.constant(spend_multipliers, dtype=tf.float32)
    spend_amounts = tf.einsum('k,m->km', spend_multipliers_tensor, total_spend)

    return spend_amounts

  def _compute_geo_spend_scenarios(self,
                                 max_multiplier: float = 2.0,
                                 num_steps: int = 100,
                                 selected_times: Optional[Sequence[str]] = None,
                                 selected_geos: Optional[Sequence[str]] = None) -> tf.Tensor:
    """Computes geo-level spend scenarios for simulation.

    Creates synthetic spend ranges from 0 to max_multiplier * geo_max_weekly_spend
    for each geo independently, following the MPA simulation approach.

    Args:
      max_multiplier: Maximum multiplier for geo spend ranges.
      num_steps: Number of simulation steps per geo.
      selected_times: Optional subset of time periods.
      selected_geos: Optional subset of geos.

    Returns:
      Tensor of shape (n_geos, n_steps, n_channels) with spend scenarios.
    """
    # Get historical spend data
    spend_tensors = []
    if self._meridian.media_tensors.media_spend is not None:
      spend_tensors.append(self._meridian.media_tensors.media_spend)
    if self._meridian.rf_tensors.rf_spend is not None:
      spend_tensors.append(self._meridian.rf_tensors.rf_spend)

    if not spend_tensors:
      raise ValueError('No spend data available in the model.')

    total_spend = tf.concat(spend_tensors, axis=-1)

    # Filter by selected dimensions if needed
    if tf.rank(total_spend) == 3:  # Has geo and time dimensions
      if selected_geos is not None:
        geo_indices = [
            i for i, geo in enumerate(self._meridian.input_data.geo.values)
            if geo in selected_geos
        ]
        total_spend = tf.gather(total_spend, geo_indices, axis=0)

      if selected_times is not None:
        time_indices = [
            i for i, time in enumerate(self._meridian.input_data.time.values)
            if time in selected_times
        ]
        total_spend = tf.gather(total_spend, time_indices, axis=1)

      # Calculate geo-level maximum weekly spend per channel
      # Shape: (n_geos, n_channels)
      geo_max_weekly_spend = tf.reduce_max(total_spend, axis=1)  # Max across time dimension
      
    else:
      # If data doesn't have geo/time dimensions, use as is
      geo_max_weekly_spend = total_spend

    # Create simulation scenarios: 0 to max_multiplier * geo_max_spend
    # Shape: (n_geos, n_steps, n_channels)
    n_geos, n_channels = geo_max_weekly_spend.shape
    
    # Create linear ranges from 0 to geo_max * max_multiplier for each geo/channel
    spend_scenarios_list = []
    
    for geo_idx in range(n_geos):
      geo_scenarios_list = []
      for channel_idx in range(n_channels):
        max_spend = geo_max_weekly_spend[geo_idx, channel_idx] * max_multiplier
        channel_scenarios = tf.linspace(0.0, max_spend, num_steps)
        geo_scenarios_list.append(channel_scenarios)
      
      # Stack channels for this geo: (n_steps, n_channels)
      geo_scenarios = tf.stack(geo_scenarios_list, axis=1)
      spend_scenarios_list.append(geo_scenarios)
    
    # Stack geos: (n_geos, n_steps, n_channels)
    spend_scenarios = tf.stack(spend_scenarios_list, axis=0)

    return spend_scenarios

  def _simulate_geo_media_transformations(self,
                                        spend_scenarios: tf.Tensor,
                                        selected_times: Optional[Sequence[str]] = None,
                                        by_reach: bool = True) -> tf.Tensor:
    """Applies media transformations to geo-level spend scenarios.

    This method converts spend scenarios into media execution data and applies
    the full adstock and hill transformation pipeline at the geo level,
    following the MPA transformation approach.

    Args:
      spend_scenarios: Tensor of shape (n_geos, n_steps, n_channels).
      selected_times: Optional subset of time periods for transformation.
      by_reach: For R&F channels, whether to scale reach (True) or frequency.

    Returns:
      Tensor of shape (n_geos, n_steps, n_channels) with transformed effects.
    """
    n_geos, n_steps, n_channels = spend_scenarios.shape

    # Convert spend scenarios to media execution data
    # For this simulation, we assume spend scenarios represent weekly spend amounts
    # We need to create synthetic media execution data that would produce these spends
    
    # Initialize results tensor
    all_geo_effects = []

    # Process each geo independently
    for geo_idx in range(n_geos):
      geo_spend_scenarios = spend_scenarios[geo_idx]  # Shape: (n_steps, n_channels)
      
      # Convert spend to media execution for this geo
      # For standard media channels, assume spend = impressions * cpm
      # For simplicity, we'll treat spend scenarios as impression-level data
      geo_media_data = geo_spend_scenarios  # Shape: (n_steps, n_channels)
      
      # Reshape to match expected input format: (n_times, n_channels)
      # We'll simulate each step as a separate time period
      geo_media_data_reshaped = tf.expand_dims(geo_media_data, axis=0)  # (1, n_steps, n_channels)
      geo_media_data_reshaped = tf.transpose(geo_media_data_reshaped, [1, 0, 2])  # (n_steps, 1, n_channels)
      
      # Apply media transformations for this geo
      if self.has_media_data and geo_idx < len(self.media_params['ec']):
        # Extract parameters for media channels
        n_media_channels = len(self.media_params['ec'])
        media_channels_end = min(n_media_channels, n_channels)
        
        if media_channels_end > 0:
          geo_media_subset = geo_media_data_reshaped[:, :, :media_channels_end]
          
          # Apply adstock and hill transformations
          geo_media_effects = self._transform_media_channels_simulation(
              media_data=geo_media_subset,
              params=self.media_params,
              selected_times=selected_times
          )
        else:
          geo_media_effects = tf.zeros((n_steps, 1, 0))
      else:
        geo_media_effects = tf.zeros((n_steps, 1, 0))

      # Handle R&F channels if they exist
      if self.has_rf_data and n_channels > len(self.media_params.get('ec', [])):
        rf_start_idx = len(self.media_params.get('ec', []))
        rf_channels = n_channels - rf_start_idx
        
        if rf_channels > 0:
          # For R&F simulation, we'll treat spend scenarios as reach*frequency
          geo_rf_subset = geo_media_data_reshaped[:, :, rf_start_idx:]
          
          # Apply R&F transformations (simplified approach)
          geo_rf_effects = self._transform_media_channels_simulation(
              media_data=geo_rf_subset,
              params=self.rf_params,
              selected_times=selected_times
          )
          
          # Concatenate media and RF effects
          if geo_media_effects.shape[2] > 0:
            geo_effects = tf.concat([geo_media_effects, geo_rf_effects], axis=2)
          else:
            geo_effects = geo_rf_effects
        else:
          geo_effects = geo_media_effects
      else:
        geo_effects = geo_media_effects

      # Reshape geo effects for storage - remove the singleton geo dimension
      if geo_effects.shape[1] == 1:
        geo_effects_squeezed = tf.squeeze(geo_effects, axis=1)  # Shape: (n_steps, n_channels)
      else:
        geo_effects_squeezed = geo_effects[:, 0, :]  # Take first geo dimension
      all_geo_effects.append(geo_effects_squeezed)

    # Stack all geo effects: (n_geos, n_steps, n_channels)
    geo_effects_tensor = tf.stack(all_geo_effects, axis=0)

    return geo_effects_tensor

  def _transform_media_channels_simulation(self,
                                         media_data: tf.Tensor,
                                         params: dict[str, np.ndarray],
                                         selected_times: Optional[Sequence[str]] = None) -> tf.Tensor:
    """Applies adstock and hill transformations for simulation data.

    Args:
      media_data: Media data with shape (n_times, n_geos, n_channels).
      params: Dictionary containing transformation parameters.
      selected_times: Optional subset of time periods.

    Returns:
      Transformed media effects tensor.
    """
    if not params or 'ec' not in params:
      # Return zeros if no parameters available
      return tf.zeros_like(media_data)

    # Extract median parameters for fast computation
    ec_params = tf.constant(params['ec'], dtype=tf.float32)
    slope_params = tf.constant(params['slope'], dtype=tf.float32)
    alpha_params = tf.constant(params['alpha'], dtype=tf.float32)  # adstock
    beta_params = tf.constant(params['beta'], dtype=tf.float32)   # coefficients

    n_times, n_geos, n_channels = media_data.shape
    transformed_effects = []

    for channel_idx in range(n_channels):
      if channel_idx >= len(ec_params):
        continue

      # Extract parameters for this channel
      ec = ec_params[channel_idx]
      slope = slope_params[channel_idx]
      alpha = alpha_params[channel_idx]
      beta = beta_params[channel_idx]

      # Extract media data for this channel: (n_times, n_geos)
      channel_media = media_data[:, :, channel_idx]

      # Apply adstock transformation along time dimension
      adstocked_media = self._apply_adstock_simulation(channel_media, alpha)

      # Apply hill transformation
      saturated_media = self._apply_hill_simulation(adstocked_media, ec, slope)

      # Apply coefficient
      channel_effects = saturated_media * beta

      transformed_effects.append(channel_effects)

    if transformed_effects:
      # Stack channels: (n_times, n_geos, n_channels)
      effects_tensor = tf.stack(transformed_effects, axis=2)
    else:
      effects_tensor = tf.zeros((n_times, n_geos, 0))

    return effects_tensor

  def _apply_adstock_simulation(self, data: tf.Tensor, alpha: float) -> tf.Tensor:
    """Applies adstock transformation for simulation data."""
    # Simple geometric adstock: y_t = alpha * y_{t-1} + x_t
    # For simulation purposes, use a simplified approach
    n_times = data.shape[0]
    
    # Initialize output tensor
    adstocked = tf.TensorArray(dtype=tf.float32, size=n_times)
    
    # Initialize with first data point
    carry = data[0]
    adstocked = adstocked.write(0, carry)
    
    # Apply adstock recursively
    for t in range(1, n_times):
      carry = carry * alpha + data[t]
      adstocked = adstocked.write(t, carry)
    
    return adstocked.stack()

  def _apply_hill_simulation(self, data: tf.Tensor, ec: float, slope: float) -> tf.Tensor:
    """Applies hill saturation transformation for simulation data."""
    # Hill transformation: data^slope / (data^slope + ec^slope)
    data_safe = tf.maximum(data, 1e-8)  # Avoid division by zero
    numerator = tf.pow(data_safe, slope)
    denominator = numerator + tf.pow(ec, slope)
    return tf.where(tf.equal(data, 0), 0.0, numerator / denominator)

  @property
  def has_media_data(self) -> bool:
    """Returns True if the model has standard media channels."""
    return bool(self.media_params)

  @property
  def has_rf_data(self) -> bool:
    """Returns True if the model has reach & frequency channels."""
    return bool(self.rf_params)

  @property
  def n_total_channels(self) -> int:
    """Returns total number of paid media channels."""
    return self._meridian.n_media_channels + self._meridian.n_rf_channels

  @property
  def channel_names(self) -> list[str]:
    """Returns list of all paid media channel names."""
    names = []
    if self._meridian.input_data.media_channel is not None:
      names.extend(self._meridian.input_data.media_channel.values.tolist())
    if self._meridian.input_data.rf_channel is not None:
      names.extend(self._meridian.input_data.rf_channel.values.tolist())
    return names

  def calculate_inflection_points(self, 
                                selected_geos: Optional[Sequence[str]] = None,
                                selected_times: Optional[Sequence[str]] = None) -> dict[str, float]:
    """Calculate weekly inflection spend points (half-saturation) for each channel.
    
    The inflection point represents the weekly spend level where diminishing returns 
    begin to accelerate in the Hill saturation curve. It's calculated as:
    inflection_spend = ec_parameter * weekly_median_spend
    
    Args:
      selected_geos: Optional subset of geos to include in median calculation.
      selected_times: Optional subset of time periods to include.
      
    Returns:
      Dictionary mapping channel names to weekly inflection spend values.
    """
    inflection_points = {}
    
    # Get historical spend data
    spend_tensors = []
    channel_names = []
    ec_params = []
    
    # Process standard media channels
    if self._meridian.media_tensors.media_spend is not None and self.media_params:
      spend_tensors.append(self._meridian.media_tensors.media_spend)
      if self._meridian.input_data.media_channel is not None:
        channel_names.extend(self._meridian.input_data.media_channel.values.tolist())
      ec_params.extend(self.media_params['ec'])
      
    # Process R&F channels 
    if self._meridian.rf_tensors.rf_spend is not None and self.rf_params:
      spend_tensors.append(self._meridian.rf_tensors.rf_spend)
      if self._meridian.input_data.rf_channel is not None:
        channel_names.extend(self._meridian.input_data.rf_channel.values.tolist())
      ec_params.extend(self.rf_params['ec'])
    
    if not spend_tensors:
      return inflection_points
      
    # Concatenate spend data
    total_spend = tf.concat(spend_tensors, axis=-1)
    
    # Apply geo/time filtering if specified
    if tf.rank(total_spend) == 3:  # Has geo and time dimensions
      if selected_geos is not None:
        geo_indices = [
            i for i, geo in enumerate(self._meridian.input_data.geo.values)
            if geo in selected_geos
        ]
        total_spend = tf.gather(total_spend, geo_indices, axis=0)
        
      if selected_times is not None:
        time_indices = [
            i for i, time in enumerate(self._meridian.input_data.time.values)  
            if time in selected_times
        ]
        total_spend = tf.gather(total_spend, time_indices, axis=1)
        
      # Calculate weekly median spend to match response curve calculation
      # Sum across geos, then median across time to get weekly median spend
      total_spend_by_geo_time = tf.reduce_sum(total_spend, axis=0)  # Sum across geos
      weekly_median_spend = tfp.stats.percentile(total_spend_by_geo_time, 50.0, axis=0)  # Median across time
      historical_spend = weekly_median_spend.numpy()
    else:
      # Assume already aggregated data
      historical_spend = total_spend.numpy()
    
    # Calculate weekly inflection points: ec * weekly_median_spend
    for channel, ec, weekly_spend in zip(channel_names, ec_params, historical_spend):
      inflection_points[channel] = float(ec * weekly_spend)
      
    return inflection_points

  def _transform_media(self,
                      scaled_media: Optional[tf.Tensor] = None,
                      scaled_reach: Optional[tf.Tensor] = None,
                      scaled_frequency: Optional[tf.Tensor] = None,
                      selected_times: Optional[Sequence[str]] = None) -> tf.Tensor:
    """Applies adstock and hill transformations to scaled media data.

    Args:
      scaled_media: Scaled standard media data tensor.
      scaled_reach: Scaled reach data tensor.
      scaled_frequency: Scaled frequency data tensor.
      selected_times: Optional subset of time periods for output.

    Returns:
      Tensor of transformed media effects with shape (n_geos, n_times, n_channels).
    """
    transformed_effects = []

    # Transform standard media channels
    if scaled_media is not None and self.media_params:
      media_effects = self._transform_media_channels(
          media_data=scaled_media,
          params=self.media_params,
          selected_times=selected_times
      )
      transformed_effects.append(media_effects)

    # Transform R&F channels
    if (scaled_reach is not None and scaled_frequency is not None and
        self.rf_params):
      # For R&F, we need to combine reach and frequency into impressions
      rf_impressions = scaled_reach * scaled_frequency
      rf_effects = self._transform_media_channels(
          media_data=rf_impressions,
          params=self.rf_params,
          selected_times=selected_times
      )
      transformed_effects.append(rf_effects)

    if not transformed_effects:
      raise ValueError('No media data to transform.')

    # Concatenate all channel effects along the last dimension
    return tf.concat(transformed_effects, axis=-1)

  def _transform_media_channels(self,
                               media_data: tf.Tensor,
                               params: dict[str, np.ndarray],
                               selected_times: Optional[Sequence[str]] = None) -> tf.Tensor:
    """Transforms a single type of media data (media or R&F).

    Args:
      media_data: Media execution data with shape (n_geos, n_times, n_channels).
      params: Dictionary containing 'ec', 'slope', 'alpha', 'beta' parameters.
      selected_times: Optional subset of time periods for output.

    Returns:
      Transformed media effects tensor.
    """
    # Determine output time periods
    n_times_output = len(selected_times) if selected_times else len(self._meridian.input_data.kpi.time)

    # Step 0: Apply per-capita and median scaling (same as model does)
    # This is critical - the original model uses media_scaled, not raw media
    if self._meridian.media_tensors.media_transformer is not None:
      media_scaled = self._meridian.media_tensors.media_transformer.forward(media_data)
    else:
      media_scaled = media_data

    # Step 1: Apply adstock transformation
    adstock_transformer = adstock_hill.AdstockTransformer(
        alpha=tf.constant(params['alpha'], dtype=tf.float32),
        max_lag=self._meridian.model_spec.max_lag,
        n_times_output=n_times_output
    )
    adstocked_media = adstock_transformer.forward(media_scaled)

    # Step 2: Apply hill saturation transformation
    hill_transformer = adstock_hill.HillTransformer(
        ec=tf.constant(params['ec'], dtype=tf.float32),
        slope=tf.constant(params['slope'], dtype=tf.float32)
    )
    saturated_media = hill_transformer.forward(adstocked_media)

    # Step 3: Apply effectiveness coefficients (beta parameters)
    # saturated_media shape: (n_geos, n_times, n_channels)
    # params['beta'] shape: (n_geos, n_channels)
    # Result shape: (n_geos, n_times, n_channels)
    beta_tensor = tf.constant(params['beta'], dtype=tf.float32)
    media_effects = saturated_media * beta_tensor[:, tf.newaxis, :]

    return media_effects

  def _aggregate_media_effects(self,
                              media_effects: tf.Tensor,
                              selected_times: Optional[Sequence[str]] = None,
                              selected_geos: Optional[Sequence[str]] = None,
                              aggregate_geos: bool = True,
                              aggregate_times: bool = True) -> tf.Tensor:
    """Aggregates media effects across geos and/or times.

    Args:
      media_effects: Transformed media effects tensor.
      selected_times: Optional subset of time periods to include.
      selected_geos: Optional subset of geos to include.
      aggregate_geos: Whether to sum across geos.
      aggregate_times: Whether to sum across times.

    Returns:
      Aggregated media effects tensor.
    """
    # Apply time filtering if specified
    if selected_times is not None:
      time_indices = [
          i for i, time in enumerate(self._meridian.input_data.time.values)
          if time in selected_times
      ]
      media_effects = tf.gather(media_effects, time_indices, axis=1)

    # Apply geo filtering if specified
    if selected_geos is not None:
      geo_indices = [
          i for i, geo in enumerate(self._meridian.input_data.geo.values)
          if geo in selected_geos
      ]
      media_effects = tf.gather(media_effects, geo_indices, axis=0)

    # Aggregate dimensions as requested
    aggregation_axes = []
    if aggregate_geos:
      aggregation_axes.append(0)  # geo dimension
    if aggregate_times:
      aggregation_axes.append(1)  # time dimension

    if aggregation_axes:
      media_effects = tf.reduce_sum(media_effects, axis=aggregation_axes)

    return media_effects


  def _inverse_outcome(self,
                      modeled_incremental_outcome: tf.Tensor,
                      use_kpi: bool) -> tf.Tensor:
    """Applies inverse transformation to convert media effects back to original scale.
    
    This replicates the _inverse_outcome method from analyzer.py to ensure our
    FastResponseCurves implementation matches the original response_curves behavior.
    
    Args:
      modeled_incremental_outcome: Tensor with shape (n_geos, n_times, n_channels).
      use_kpi: Whether to return KPI (True) or revenue (False).
      
    Returns:
      Tensor of incremental outcome in original KPI or revenue units.
    """
    # Media effects have shape (n_geos, n_times, n_channels)
    # We need to apply inverse transformation per channel
    n_geos, n_times, n_channels = modeled_incremental_outcome.shape
    
    # Initialize output tensor
    kpi_effects = tf.zeros_like(modeled_incremental_outcome)
    
    # Apply inverse transformation for each channel independently
    for ch in range(n_channels):
      # Extract channel effects: (n_geos, n_times)
      channel_effects = modeled_incremental_outcome[:, :, ch]
      
      # Apply inverse KPI transformation (matching analyzer.py approach)
      # The transformation is: inverse(effects) - inverse(zeros)
      t1 = self._meridian.kpi_transformer.inverse(channel_effects)
      t2 = self._meridian.kpi_transformer.inverse(tf.zeros_like(channel_effects))
      kpi_channel = t1 - t2
      
      # Store the transformed channel effects
      kpi_effects = tf.tensor_scatter_nd_update(
          kpi_effects,
          tf.stack([
              tf.repeat(tf.range(n_geos), n_times),
              tf.tile(tf.range(n_times), [n_geos]),
              tf.fill([n_geos * n_times], ch)
          ], axis=1),
          tf.reshape(kpi_channel, [-1])
      )

    if use_kpi:
      return kpi_effects
    
    # Convert to revenue if requested and revenue_per_kpi is available
    if self._meridian.revenue_per_kpi is not None:
      return tf.einsum("gt,gtm->gtm", self._meridian.revenue_per_kpi, kpi_effects)
    else:
      # If no revenue_per_kpi, return KPI regardless of use_kpi setting
      return kpi_effects

  def compute_response_curves(self,
                             spend_multipliers: Optional[Sequence[float]] = None,
                             selected_times: Optional[Sequence[str]] = None,
                             selected_geos: Optional[Sequence[str]] = None,
                             by_reach: bool = True,
                             use_kpi: bool = True) -> xr.Dataset:
    """Computes weekly spend response curves using median parameter estimates.

    This method efficiently generates response curves by using median parameter
    values and weekly median spend baselines instead of full Bayesian sampling,
    providing significant computational speedup while maintaining accuracy for
    point estimates and creating actionable weekly spend scenarios.

    Args:
      spend_multipliers: List of multipliers to apply to weekly median spend.
        Defaults to np.arange(0, 2.2, 0.01).
      selected_times: Optional subset of time periods to include. By default,
        all time periods are included.
      selected_geos: Optional subset of geos to include. By default, all geos
        are included.
      by_reach: For R&F channels, whether to scale reach (True) or frequency.
      use_kpi: Whether to use KPI instead of revenue for the outcome.

    Returns:
      xarray.Dataset with coordinates: channel, spend_multiplier
      and data variables: spend (weekly amounts), incremental_outcome, roi
    """
    # Set default spend multipliers if not provided
    if spend_multipliers is None:
      spend_multipliers = list(np.arange(0, 2.2, c.RESPONSE_CURVE_STEP_SIZE))

    # Determine if we should use KPI
    if use_kpi is None:
      use_kpi = self._meridian.input_data.revenue_per_kpi is None

    # Compute spend amounts for each multiplier
    spend_amounts = self._compute_spend_amounts(
        spend_multipliers=spend_multipliers,
        selected_times=selected_times,
        selected_geos=selected_geos
    )

    # Initialize results storage
    n_multipliers = len(spend_multipliers)
    n_channels = self.n_total_channels

    incremental_outcomes = np.zeros((n_multipliers, n_channels))
    spend_matrix = spend_amounts.numpy()

    # Compute incremental outcome for each multiplier
    for i, multiplier in enumerate(spend_multipliers):
      # Scale media data
      scaled_media, scaled_reach, scaled_frequency = self._scale_media_data(
          multiplier=multiplier, by_reach=by_reach
      )  # scales raw media execution by the multiplier

      # Transform media data to get effects
      media_effects = self._transform_media(
          scaled_media=scaled_media,
          scaled_reach=scaled_reach,
          scaled_frequency=scaled_frequency,
          selected_times=selected_times
      )

      # Apply inverse transformation BEFORE aggregation (to match original implementation)
      # This converts from model KPI scale back to original KPI/revenue scale
      transformed_effects = self._inverse_outcome(media_effects, use_kpi)

      # Aggregate effects across geos and times
      aggregated_effects = self._aggregate_media_effects(
          media_effects=transformed_effects,
          selected_times=selected_times,
          selected_geos=selected_geos,
          aggregate_geos=True,
          aggregate_times=True
      )

      # Store per-channel incremental outcomes
      incremental_outcomes[i, :] = aggregated_effects.numpy()

    # Calculate ROI for each channel and multiplier
    roi_matrix = np.zeros_like(incremental_outcomes)
    # Avoid division by zero by setting ROI to 0 where spend is 0
    nonzero_spend = spend_matrix > 0
    roi_matrix[nonzero_spend] = (
        incremental_outcomes[nonzero_spend] / spend_matrix[nonzero_spend]
    )

    # Create coordinate arrays
    channel_names = self.channel_names
    spend_multiplier_coords = np.array(spend_multipliers)

    # Create xarray Dataset
    coords = {
        c.CHANNEL: channel_names,
        c.SPEND_MULTIPLIER: spend_multiplier_coords,
        c.METRIC: [c.MEAN]  # Since we only have point estimates
    }

    # Reshape data to include metric dimension for consistency with existing API
    spend_data = spend_matrix[:, np.newaxis, :]  # (multipliers, metrics, channels)
    incremental_data = incremental_outcomes[:, np.newaxis, :]
    roi_data = roi_matrix[:, np.newaxis, :]

    # Transpose to match expected dimensions: (channels, metrics, multipliers)
    spend_data = np.transpose(spend_data, (2, 1, 0))
    incremental_data = np.transpose(incremental_data, (2, 1, 0))
    roi_data = np.transpose(roi_data, (2, 1, 0))

    data_vars = {
        c.SPEND: ([c.CHANNEL, c.METRIC, c.SPEND_MULTIPLIER], spend_data),
        c.INCREMENTAL_OUTCOME: ([c.CHANNEL, c.METRIC, c.SPEND_MULTIPLIER], incremental_data),
        c.ROI: ([c.CHANNEL, c.METRIC, c.SPEND_MULTIPLIER], roi_data),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords)

  def response_curves_data(self,
                          spend_multipliers: Optional[Sequence[float]] = None,
                          selected_times: Optional[Sequence[str]] = None,
                          by_reach: bool = True) -> xr.Dataset:
    """Alias for compute_response_curves to match existing API.

    This method provides the same interface as MediaEffects.response_curves_data
    for compatibility with existing code.
    """
    # Convert selected_times to frozenset if needed for consistency
    selected_times_frozen = None
    if selected_times is not None:
      selected_times_frozen = frozenset(selected_times)

    # Determine use_kpi based on model configuration
    use_kpi = self._meridian.input_data.revenue_per_kpi is None

    return self.compute_response_curves(
        spend_multipliers=spend_multipliers,
        selected_times=list(selected_times_frozen) if selected_times_frozen else None,
        by_reach=by_reach,
        use_kpi=use_kpi
    )

  def compute_geo_level_response_curves(self,
                                      max_multiplier: float = 2.0,
                                      num_steps: int = 100,
                                      selected_times: Optional[Sequence[str]] = None,
                                      selected_geos: Optional[Sequence[str]] = None,
                                      aggregation_level: str = "national",
                                      by_reach: bool = True,
                                      use_kpi: bool = True) -> xr.Dataset:
    """Computes geo-level weekly spend response curves with national aggregation.

    This method creates synthetic spend scenarios for each geo independently,
    applies media transformations at the geo level, and aggregates results
    to produce national-level response curves. This approach provides more
    granular insights into geo-specific saturation patterns while producing
    actionable national-level media planning insights.

    Args:
      max_multiplier: Maximum simulation range as multiplier of geo max spend.
        E.g., 2.0 creates scenarios from 0 to 2x geo maximum weekly spend.
      num_steps: Number of simulation points for smooth curves.
      selected_times: Optional subset of time periods to include.
      selected_geos: Optional subset of geos to include.
      aggregation_level: Level of aggregation - "national", "geo", or "both".
      by_reach: For R&F channels, whether to scale reach (True) or frequency.
      use_kpi: Whether to use KPI instead of revenue for the outcome.

    Returns:
      xarray.Dataset with coordinates: channel, spend_multiplier_equiv
      and data variables: spend (national equivalent), incremental_outcome, roi
    """
    # Determine if we should use KPI
    if use_kpi is None:
      use_kpi = self._meridian.input_data.revenue_per_kpi is None

    # Generate geo-level spend scenarios  
    spend_scenarios = self._compute_geo_spend_scenarios(
        max_multiplier=max_multiplier,
        num_steps=num_steps,
        selected_times=selected_times,
        selected_geos=selected_geos
    )

    # Apply geo-level media transformations
    geo_effects = self._simulate_geo_media_transformations(
        spend_scenarios=spend_scenarios,
        selected_times=selected_times,
        by_reach=by_reach
    )

    # Apply inverse transformation and aggregate
    transformed_effects = self._inverse_outcome(geo_effects, use_kpi)
    
    # Aggregate to desired level
    if aggregation_level == "national":
      aggregated_effects = tf.reduce_sum(transformed_effects, axis=0)  # Sum across geos
    elif aggregation_level == "geo":
      aggregated_effects = transformed_effects  # Keep geo dimension
    else:
      raise ValueError(f"Unsupported aggregation_level: {aggregation_level}")

    # Calculate equivalent national spend amounts
    national_spend_amounts = tf.reduce_sum(spend_scenarios, axis=0)  # Sum across geos
    
    # Convert to numpy for dataset creation
    incremental_outcomes = aggregated_effects.numpy()
    spend_matrix = national_spend_amounts.numpy()

    # Calculate ROI for each channel and scenario
    roi_matrix = np.divide(
        incremental_outcomes, 
        spend_matrix,
        out=np.zeros_like(incremental_outcomes),
        where=spend_matrix != 0
    )

    # Create equivalent multipliers for compatibility
    baseline_national_spend = tf.reduce_sum(
        tf.reduce_max(spend_scenarios, axis=1),  # Max across steps per geo
        axis=0  # Sum across geos
    ).numpy()
    
    equivalent_multipliers = []
    for step in range(num_steps):
      if np.any(baseline_national_spend > 0):
        eq_mult = spend_matrix[step] / np.maximum(baseline_national_spend, 1e-8)
        equivalent_multipliers.append(np.mean(eq_mult))  # Average across channels
      else:
        equivalent_multipliers.append(step / num_steps * max_multiplier)

    # Create xarray dataset
    channel_names = self.channel_names
    
    coords = {
        'channel': channel_names,
        'spend_multiplier_equiv': equivalent_multipliers
    }

    data_vars = {
        'spend': (['spend_multiplier_equiv', 'channel'], spend_matrix),
        'incremental_outcome': (['spend_multiplier_equiv', 'channel'], incremental_outcomes),
        'roi': (['spend_multiplier_equiv', 'channel'], roi_matrix)
    }

    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    dataset.attrs['method'] = 'geo_level_simulation'
    dataset.attrs['max_multiplier'] = max_multiplier
    dataset.attrs['num_steps'] = num_steps
    dataset.attrs['aggregation_level'] = aggregation_level

    return dataset

  def plot_response_curves(self,
                          spend_multipliers: Optional[Sequence[float]] = None,
                          selected_times: Optional[Sequence[str]] = None,
                          by_reach: bool = True,
                          plot_separately: bool = True,
                          num_channels_displayed: Optional[int] = None,
                          show_inflection_points: bool = False) -> alt.Chart:
    """Plots the response curves for each channel using median parameters.

    This method creates visualizations similar to MediaEffects.plot_response_curves
    but using deterministic median parameter values instead of confidence intervals.

    Args:
      spend_multipliers: List of multipliers to apply to historical spend.
      selected_times: Optional subset of time periods to include.
      by_reach: For R&F channels, whether to scale reach (True) or frequency.
      plot_separately: If True, plots are faceted. If False, plots are layered.
      num_channels_displayed: Number of channels to show on layered plot.
      show_inflection_points: If True, adds vertical lines marking inflection points.

    Returns:
      Altair chart showing the response curves per channel, optionally with 
      inflection point markers.
    """
    # Get response curve data
    response_data = self.compute_response_curves(
        spend_multipliers=spend_multipliers,
        selected_times=selected_times,
        by_reach=by_reach
    )

    # Calculate inflection points if requested
    inflection_points_dict = None
    if show_inflection_points:
      try:
        inflection_points_dict = self.calculate_inflection_points(selected_times=selected_times)
      except Exception as e:
        print(f"Warning: Could not calculate inflection points: {e}")

    # Transform data for plotting
    response_curves_df = self._transform_response_curve_data_for_plotting(
        response_data=response_data,
        num_channels_displayed=num_channels_displayed,
        plot_separately=plot_separately,
        inflection_points=inflection_points_dict
    )

    # Determine axis labels
    if self._meridian.input_data.revenue_per_kpi is not None:
      y_axis_label = summary_text.INC_OUTCOME_LABEL
    else:
      y_axis_label = summary_text.INC_KPI_LABEL

    # Determine chart title
    total_num_channels = self.n_total_channels
    if plot_separately:
      title = f"Weekly spend response curves by marketing channel"
    else:
      if num_channels_displayed is None:
        num_channels_displayed = min(total_num_channels, 7)
      title = f"Weekly spend response curves by marketing channel (top {num_channels_displayed})"

    # Create base chart
    base = (
        alt.Chart(response_curves_df, width=c.VEGALITE_FACET_DEFAULT_WIDTH)
        .transform_calculate(
            spend_level=(
                'datum.spend_multiplier >= 1.0 ? "Above current spend" : "Below'
                ' current spend"'
            )
        )
        .encode(
            x=alt.X(
                f'{c.SPEND}:Q',
                title="Weekly Spend",
                axis=alt.Axis(
                    labelExpr=formatter.compact_number_expr(),
                    **formatter.AXIS_CONFIG,
                ),
            ),
            y=alt.Y(
                f'{c.MEAN}:Q',
                title=y_axis_label,
                axis=alt.Axis(
                    labelExpr=formatter.compact_number_expr(),
                    **formatter.Y_AXIS_TITLE_CONFIG,
                ),
            ),
            color=f'{c.CHANNEL}:N',
        )
    )

    # Create line plot
    line = base.mark_line().encode(
        strokeDash=alt.StrokeDash(
            f'{c.SPEND_LEVEL}:N',
            sort='descending',
            legend=alt.Legend(title=None),
        )
    )

    # Create historic spend point
    historic_spend_point = (
        base.mark_point(filled=True, size=c.POINT_SIZE, opacity=1)
        .encode(
            tooltip=[c.SPEND, c.MEAN],
            shape=alt.Shape(
                f'{c.CURRENT_SPEND}:N', legend=alt.Legend(title=None)
            ),
        )
        .transform_filter(alt.datum.spend_multiplier == 1.0)
    )

    # Create inflection point markers if they exist in the data
    inflection_points_chart = None
    inflection_labels_chart = None
    if show_inflection_points and 'is_inflection_point' in response_curves_df.columns:
      inflection_data = response_curves_df[response_curves_df['is_inflection_point'] == True]
      
      if len(inflection_data) > 0:
        # Green dot markers
        inflection_points_chart = (
          base.mark_circle(
            size=150,
            color='green',
            stroke='darkgreen', 
            strokeWidth=3,
            opacity=1.0
          )
          .encode(
            tooltip=[f'{c.CHANNEL}:N', f'{c.SPEND}:Q', f'{c.MEAN}:Q', 'point_label:N', 'inflection_point_type:N']
          )
          .transform_filter(alt.datum.is_inflection_point == True)
        )
        
        # Text labels for inflection points
        inflection_labels_chart = (
          base.mark_text(
            align='left',
            baseline='bottom',
            dx=10,  # Offset text more to the right of the point
            dy=-15,  # Offset text more above the point
            fontSize=12,
            color='darkgreen',
            fontWeight='bold',
            stroke='white',
            strokeWidth=1  # Add white stroke for better visibility
          )
          .encode(
            text='point_label:N'
          )
          .transform_filter(alt.datum.is_inflection_point == True)
        )

    # Combine line, historic point, and inflection points
    chart_layers = [line, historic_spend_point]
    if inflection_points_chart is not None:
      chart_layers.append(inflection_points_chart)
    if inflection_labels_chart is not None:
      chart_layers.append(inflection_labels_chart)
    
    plot = alt.layer(*chart_layers)

    # Apply faceting if requested
    if plot_separately:
      plot = plot.facet(c.CHANNEL, columns=3).resolve_scale(
          x=c.INDEPENDENT, y=c.INDEPENDENT
      )

    # Update title to indicate inflection points if shown
    if show_inflection_points:
      title = title + " (with Inflection Points)"

    return plot.properties(
        title=formatter.custom_title_params(title)
    ).configure_axis(**formatter.TEXT_CONFIG)

  def _transform_response_curve_data_for_plotting(self,
                                                 response_data: xr.Dataset,
                                                 num_channels_displayed: Optional[int] = None,
                                                 plot_separately: bool = True,
                                                 inflection_points: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """Transforms response curve data for plotting.

    Args:
      response_data: xarray Dataset from compute_response_curves.
      num_channels_displayed: Number of top channels to include.
      plot_separately: Whether plotting separately (affects channel selection).
      inflection_points: Optional dict mapping channel names to inflection spend values.

    Returns:
      DataFrame formatted for Altair plotting, with inflection point markers if provided.
    """
    # Determine number of channels to display
    total_channels = len(response_data.channel)
    if plot_separately:
      num_channels_displayed = total_channels
    else:
      if num_channels_displayed is None:
        num_channels_displayed = min(total_channels, 7)
      num_channels_displayed = min(num_channels_displayed, total_channels)

    # Sort channels by spend at multiplier=1.0 (historical spend)
    historical_spend = response_data.sel(spend_multiplier=1.0)[c.SPEND].sel(metric=c.MEAN)
    sorted_channels = historical_spend.sortby(historical_spend, ascending=False).channel.values

    # Select top channels
    selected_channels = sorted_channels[:num_channels_displayed]

    # Filter data to selected channels
    filtered_data = response_data.sel(channel=selected_channels)

    # Convert to DataFrame
    df = (
        filtered_data[[c.SPEND, c.INCREMENTAL_OUTCOME]]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[c.CHANNEL, c.SPEND_MULTIPLIER],
            columns=c.METRIC,
            values=[c.SPEND, c.INCREMENTAL_OUTCOME],
        )
        .reset_index()
    )

    # Flatten column names and rename
    df.columns = [
        f'{col[0]}_{col[1]}' if col[1] else col[0]
        for col in df.columns
    ]
    df = df.rename(columns={
        f'{c.SPEND}_{c.MEAN}': c.SPEND,
        f'{c.INCREMENTAL_OUTCOME}_{c.MEAN}': c.MEAN,
    })

    # Add current spend indicator
    df[c.CURRENT_SPEND] = np.where(
        df[c.SPEND_MULTIPLIER] == 1.0,
        summary_text.CURRENT_SPEND_LABEL,
        pd.NA,
    )

    # Add inflection point indicators and labels if provided
    if inflection_points:
      df['is_inflection_point'] = False
      df['inflection_point_type'] = pd.NA
      df['point_label'] = pd.NA
      
      # Get current median spend (spend at multiplier = 1.0)
      current_median_spends = {}
      for channel in df[c.CHANNEL].unique():
        current_spend_data = df[(df[c.CHANNEL] == channel) & (df[c.SPEND_MULTIPLIER] == 1.0)]
        if len(current_spend_data) > 0:
          current_median_spends[channel] = current_spend_data[c.SPEND].iloc[0]
      
      for channel, inflection_spend in inflection_points.items():
        if channel in df[c.CHANNEL].values:
          # Find the closest spend point to the inflection spend for this channel
          channel_data = df[df[c.CHANNEL] == channel].copy()
          if len(channel_data) > 0:
            closest_idx = (channel_data[c.SPEND] - inflection_spend).abs().idxmin()
            df.loc[closest_idx, 'is_inflection_point'] = True
            df.loc[closest_idx, 'inflection_point_type'] = 'Inflection Point'
            
            # Add detailed label with current spend and half-saturation spend
            current_spend = current_median_spends.get(channel, 0)
            label = f"Weekly Inflection: ${inflection_spend/1000:.0f}K (Current: ${current_spend/1000:.0f}K)"
            df.loc[closest_idx, 'point_label'] = label

    return df

  def plot_geo_level_response_curves(self,
                                    max_multiplier: float = 2.0,
                                    num_steps: int = 50,
                                    selected_times: Optional[Sequence[str]] = None,
                                    selected_geos: Optional[Sequence[str]] = None,
                                    aggregation_level: str = "national",
                                    plot_separately: bool = True,
                                    num_channels_displayed: Optional[int] = None,
                                    show_inflection_points: bool = False) -> alt.Chart:
    """Plots response curves using geo-level simulation with national aggregation.

    This method generates response curves by simulating spend scenarios at the geo level
    (0 to max_multiplier * geo_max_spend per geo), then aggregating to national level.
    This approach provides more realistic saturation patterns while maintaining
    weekly actionable spend recommendations.

    Args:
      max_multiplier: Maximum spend multiplier to simulate per geo.
      num_steps: Number of spend steps to simulate.
      selected_times: Optional subset of time periods to include.
      selected_geos: Optional subset of geos to include.
      aggregation_level: Level to aggregate results ("national" only supported).
      plot_separately: If True, plots are faceted. If False, plots are layered.
      num_channels_displayed: Number of channels to show on layered plot.
      show_inflection_points: If True, adds green dots marking inflection points.

    Returns:
      Altair chart showing geo-enhanced response curves per channel.
    """
    # Get geo-level response curve data
    response_data = self.compute_geo_level_response_curves(
        max_multiplier=max_multiplier,
        num_steps=num_steps,
        selected_times=selected_times,
        selected_geos=selected_geos,
        aggregation_level=aggregation_level
    )

    # Calculate inflection points if requested
    inflection_points = None
    if show_inflection_points:
      try:
        inflection_points = self.calculate_inflection_points(
            selected_geos=selected_geos,
            selected_times=selected_times
        )
      except Exception as e:
        print(f"Warning: Could not calculate inflection points: {e}")

    # Use geo-specific data transformation for plotting
    response_curves_df = self._transform_geo_response_data_for_plotting(
        response_data=response_data,
        num_channels_displayed=num_channels_displayed,
        plot_separately=plot_separately,
        inflection_points=inflection_points
    )

    # Create base chart
    base_chart = alt.Chart(response_curves_df).add_params(
        alt.param("highlight", select=alt.SelectionPoint(on="mouseover"))
    )

    # Create main response curve lines
    line_chart = base_chart.mark_line().encode(
        x=alt.X('spend:Q', title='Weekly Spend ($)', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('mean:Q', title='Incremental Outcome', axis=alt.Axis(format=',.0f')),
        color=alt.Color('channel:N', title='Channel', legend=alt.Legend(titleLimit=0))
    )

    charts = [line_chart]

    # Add inflection point markers if available
    if inflection_points:
      inflection_df = response_curves_df[response_curves_df['is_inflection_point'] == True]
      if not inflection_df.empty:
        inflection_chart = base_chart.transform_filter(
            alt.datum.is_inflection_point == True
        ).mark_circle(
            size=100,
            color='green',
            stroke='darkgreen',
            strokeWidth=2
        ).encode(
            x='spend:Q',
            y='mean:Q',
            color=alt.value('green'),
            tooltip=['channel:N', 'spend:Q', 'mean:Q', 'point_label:N']
        )
        
        # Add text labels for inflection points
        inflection_labels = base_chart.transform_filter(
            alt.datum.is_inflection_point == True
        ).mark_text(
            dx=10,
            dy=-10,
            fontSize=9,
            fontWeight='bold',
            color='darkgreen'
        ).encode(
            x='spend:Q',
            y='mean:Q',
            text='point_label:N'
        )
        
        charts.extend([inflection_chart, inflection_labels])

    # Combine charts
    combined_chart = alt.layer(*charts)

    # Apply faceting or layering
    if plot_separately:
      chart = combined_chart.facet(
          column=alt.Column('channel:N', title=None, header=alt.Header(labelAngle=0))
      ).resolve_scale(
          x='independent',
          y='independent'
      )
    else:
      chart = combined_chart

    # Apply styling and title
    title = f"Geo-Enhanced Weekly Response Curves ({aggregation_level.title()} Level)"
    if show_inflection_points:
      title += " with Inflection Points"

    return chart.properties(
        title=formatter.custom_title_params(title),
        width=300 if plot_separately else 500,
        height=200 if plot_separately else 300
    ).configure_axis(**formatter.TEXT_CONFIG)

  def _transform_geo_response_data_for_plotting(self,
                                              response_data: xr.Dataset,
                                              num_channels_displayed: Optional[int] = None,
                                              plot_separately: bool = True,
                                              inflection_points: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """Transforms geo-level response curve data for plotting.

    Similar to _transform_response_curve_data_for_plotting but handles geo-specific
    coordinate names and data structure.

    Args:
      response_data: xarray Dataset from compute_geo_level_response_curves.
      num_channels_displayed: Number of top channels to include.
      plot_separately: Whether plotting separately (affects channel selection).
      inflection_points: Optional dict mapping channel names to inflection spend values.

    Returns:
      DataFrame formatted for Altair plotting with geo-enhanced data.
    """
    # Determine number of channels to display
    total_channels = len(response_data.channel)
    if plot_separately:
      num_channels_displayed = total_channels
    else:
      if num_channels_displayed is None:
        num_channels_displayed = min(total_channels, 7)
      num_channels_displayed = min(num_channels_displayed, total_channels)

    # Sort channels by maximum spend (geo data may not have spend_multiplier=1.0)
    max_spend_by_channel = response_data[c.SPEND].max(dim='spend_multiplier_equiv')
    sorted_channels = max_spend_by_channel.sortby(max_spend_by_channel, ascending=False).channel.values

    # Select top channels
    selected_channels = sorted_channels[:num_channels_displayed]

    # Filter data to selected channels
    filtered_data = response_data.sel(channel=selected_channels)

    # Convert to DataFrame (geo data uses spend_multiplier_equiv instead of spend_multiplier)
    df = (
        filtered_data[[c.SPEND, c.INCREMENTAL_OUTCOME]]
        .to_dataframe()
        .reset_index()
    )

    # Rename columns for consistency with plotting code
    df = df.rename(columns={
        c.SPEND: 'spend',
        c.INCREMENTAL_OUTCOME: 'mean',
        'spend_multiplier_equiv': 'spend_multiplier'
    })

    # Add inflection point markers if provided
    if inflection_points:
      df['is_inflection_point'] = False
      df['point_label'] = ''

      for channel, inflection_spend in inflection_points.items():
        if channel in selected_channels:
          channel_df = df[df['channel'] == channel].copy()
          if not channel_df.empty:
            # Find closest spend point to inflection
            spend_diff = np.abs(channel_df['spend'] - inflection_spend)
            closest_idx = channel_df.index[spend_diff.argmin()]

            # Mark the point
            df.loc[closest_idx, 'is_inflection_point'] = True
            
            # Create label with current and inflection values
            current_median = float(channel_df[channel_df['spend_multiplier'].abs().idxmin()]['spend'])
            label = f'Current: ${current_median:,.0f}\nInflection: ${inflection_spend:,.0f}'
            df.loc[closest_idx, 'point_label'] = label
    else:
      df['is_inflection_point'] = False
      df['point_label'] = ''

    return df
