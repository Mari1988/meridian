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

"""Media transformation functions for independent optimizer."""

from typing import Optional
import numpy as np
import tensorflow as tf


__all__ = [
    "MediaScaler",
    "apply_adstock",
    "apply_hill_saturation", 
    "apply_adstock_hill",
    "apply_rf_transformation",
    "compute_incremental_outcome",
    "compute_rf_incremental_outcome",
]


class MediaScaler:
  """Media scaler equivalent to Meridian's MediaTransformer.
  
  Applies population scaling and median normalization to media data.
  Based on Meridian's transformers.py MediaTransformer implementation.
  """
  
  def __init__(self, media: tf.Tensor, population: tf.Tensor):
    """Initialize MediaScaler with media data and population.
    
    Args:
      media: Media tensor with shape (n_geos, n_times, n_channels).
      population: Population tensor with shape (n_geos,).
    """
    # Scale media by population: media / population for each geo
    population_scaled_media = tf.math.divide_no_nan(
        media, population[:, tf.newaxis, tf.newaxis]
    )
    
    # Replace zeros with NaNs for median calculation
    population_scaled_media_nan = tf.where(
        population_scaled_media == 0, np.nan, population_scaled_media
    )
    
    # Compute per-channel medians across geos and times
    self._population_scaled_median_m = tf.numpy_function(
        func=lambda x: np.nanmedian(x, axis=[0, 1]),
        inp=[population_scaled_media_nan],
        Tout=tf.float32,
    )
    
    # Create scale factors: population * median for each geo-channel combination
    self._scale_factors_gm = tf.einsum(
        "g,m->gm", population, self._population_scaled_median_m
    )
  
  @property
  def population_scaled_median_m(self):
    """Population-scaled median values per channel."""
    return self._population_scaled_median_m
  
  def forward(self, tensor: tf.Tensor) -> tf.Tensor:
    """Scale tensor using computed scale factors."""
    return tensor / self._scale_factors_gm[:, tf.newaxis, :]
  
  def inverse(self, tensor: tf.Tensor) -> tf.Tensor:
    """Reverse scaling using inverse scale factors."""
    return tensor * self._scale_factors_gm[:, tf.newaxis, :]


def apply_adstock(
    media: tf.Tensor,
    alpha: tf.Tensor,
    max_lag: int,
    n_times_output: int,
) -> tf.Tensor:
  """Apply adstock transformation to media data.
  
  Args:
    media: Media tensor with shape (n_geos, n_times, n_channels).
    alpha: Adstock decay parameters with shape (n_channels,).
    max_lag: Maximum lag periods to consider.
    n_times_output: Number of output time periods.
    
  Returns:
    Adstock-transformed media tensor.
  """
  # Validate arguments
  if n_times_output > media.shape[-2]:
    raise ValueError("n_times_output cannot exceed media time periods")
  if media.shape[-1] != alpha.shape[-1]:
    raise ValueError("Media channels must match alpha parameters")
  if n_times_output <= 0:
    raise ValueError("n_times_output must be positive")
  if max_lag < 0:
    raise ValueError("max_lag must be non-negative")
  
  n_media_times = media.shape[-2]
  window_size = min(max_lag + 1, n_media_times)
  
  # Drop excess historical periods
  required_n_media_times = n_times_output + window_size - 1
  if n_media_times > required_n_media_times:
    media = media[..., -required_n_media_times:, :]
  
  # Pad with zeros if necessary
  if n_media_times < required_n_media_times:
    pad_shape = (
        media.shape[:-2]
        + (required_n_media_times - n_media_times,)
        + (media.shape[-1],)
    )
    media = tf.concat([tf.zeros(pad_shape), media], axis=-2)
  
  # Create windowed view for adstock calculation
  window_list = []
  for i in range(window_size):
    window_list.append(media[..., i:i+n_times_output, :])
  windowed = tf.stack(window_list)
  
  # Calculate adstock weights
  l_range = tf.range(window_size - 1, -1, -1, dtype=tf.float32)
  weights = tf.expand_dims(alpha, -1) ** l_range
  normalization_factors = tf.expand_dims(
      (1 - alpha ** window_size) / (1 - alpha), -1
  )
  weights = tf.divide(weights, normalization_factors)
  
  return tf.einsum('...mw,w...gtm->...gtm', weights, windowed)


def apply_hill_saturation(
    media: tf.Tensor,
    ec50: tf.Tensor,
    slope: tf.Tensor,
) -> tf.Tensor:
  """Apply Hill saturation transformation to media data.
  
  Args:
    media: Media tensor with shape (n_geos, n_times, n_channels).
    ec50: EC50 parameters with shape (n_channels,).
    slope: Slope parameters with shape (n_channels,).
    
  Returns:
    Hill-transformed media tensor.
  """
  # Validate arguments
  if slope.shape != ec50.shape:
    raise ValueError("slope and ec50 dimensions must match")
  if media.shape[-1] != slope.shape[-1]:
    raise ValueError("Media channels must match slope/ec50 parameters")
  
  # Hill saturation: media^slope / (media^slope + ec50^slope)
  t1 = media ** slope[..., tf.newaxis, tf.newaxis, :]
  t2 = (ec50**slope)[..., tf.newaxis, tf.newaxis, :]
  return t1 / (t1 + t2)


def apply_adstock_hill(
    media: tf.Tensor,
    alpha: tf.Tensor,
    ec50: tf.Tensor,
    slope: tf.Tensor,
    max_lag: int,
    n_times_output: int,
) -> tf.Tensor:
  """Apply combined adstock and hill transformation.
  
  Args:
    media: Media tensor with shape (n_geos, n_times, n_channels).
    alpha: Adstock decay parameters with shape (n_channels,).
    ec50: EC50 parameters with shape (n_channels,).
    slope: Slope parameters with shape (n_channels,).
    max_lag: Maximum lag periods to consider.
    n_times_output: Number of output time periods.
    
  Returns:
    Adstock + Hill transformed media tensor.
  """
  # Apply adstock first, then hill saturation
  adstock_media = apply_adstock(media, alpha, max_lag, n_times_output)
  return apply_hill_saturation(adstock_media, ec50, slope)


def apply_rf_transformation(
    reach: tf.Tensor,
    frequency: tf.Tensor,
    alpha: tf.Tensor,
    ec50: tf.Tensor,
    slope: tf.Tensor,
    max_lag: int,
    n_times_output: int,
) -> tf.Tensor:
  """Apply R&F-specific transformation: Hill(frequency) first, then Adstock(reach × adj_frequency).
  
  This implements the correct R&F transformation order as used in Meridian:
  1. Apply Hill saturation to frequency (optimal frequency modeling)
  2. Multiply adjusted frequency by reach 
  3. Apply Adstock decay to the combined reach × adj_frequency
  
  Args:
    reach: Reach tensor with shape (n_geos, n_times, n_rf_channels).
    frequency: Frequency tensor with shape (n_geos, n_times, n_rf_channels).
    alpha: Adstock decay parameters with shape (n_rf_channels,).
    ec50: EC50 parameters for frequency with shape (n_rf_channels,).
    slope: Slope parameters for frequency with shape (n_rf_channels,).
    max_lag: Maximum lag periods for adstock.
    n_times_output: Number of output time periods.
    
  Returns:
    R&F transformed media tensor with shape (n_geos, n_times_output, n_rf_channels).
  """
  # Validate inputs
  if reach.shape != frequency.shape:
    raise ValueError("Reach and frequency must have the same shape")
  if reach.shape[-1] != alpha.shape[-1]:
    raise ValueError("R&F channels must match parameter dimensions")
  
  # Step 1: Apply Hill saturation to frequency (optimal frequency modeling)
  adj_frequency = apply_hill_saturation(frequency, ec50, slope)
  
  # Step 2: Multiply reach by adjusted frequency
  reach_times_adj_frequency = reach * adj_frequency
  
  # Step 3: Apply Adstock decay to the combined metric
  rf_out = apply_adstock(reach_times_adj_frequency, alpha, max_lag, n_times_output)
  
  return rf_out


def compute_rf_incremental_outcome(
    reach: tf.Tensor,
    frequency: tf.Tensor,
    rf_coefficients: tf.Tensor,
    alpha: tf.Tensor,
    ec50: tf.Tensor,
    slope: tf.Tensor,
    max_lag: int = 13,
    selected_times: Optional[slice] = None,
    selected_geos: Optional[slice] = None,
) -> tf.Tensor:
  """Compute incremental outcome from R&F data and model parameters.
  
  Args:
    reach: Reach tensor with shape (n_geos, n_times, n_rf_channels).
    frequency: Frequency tensor with shape (n_geos, n_times, n_rf_channels).
    rf_coefficients: R&F effect coefficients with shape (n_geos, n_rf_channels).
    alpha: Adstock decay parameters with shape (n_rf_channels,).
    ec50: EC50 parameters for frequency with shape (n_rf_channels,).
    slope: Slope parameters for frequency with shape (n_rf_channels,).
    max_lag: Maximum lag periods for adstock.
    selected_times: Time slice for aggregation.
    selected_geos: Geo slice for aggregation.
    
  Returns:
    Incremental outcome per R&F channel with shape (n_rf_channels,).
  """
  n_times_output = reach.shape[-2]
  if selected_times is not None:
    n_times_output = len(range(*selected_times.indices(reach.shape[-2])))
  
  # Apply R&F-specific transformations
  transformed_rf = apply_rf_transformation(
      reach=reach,
      frequency=frequency,
      alpha=alpha,
      ec50=ec50,
      slope=slope,
      max_lag=max_lag,
      n_times_output=n_times_output,
  )
  
  # Apply R&F coefficients
  rf_contribution = tf.einsum(
      'gtm,gm->gtm', 
      transformed_rf, 
      rf_coefficients
  )
  
  # Sum over time and geo dimensions
  if selected_times is not None:
    rf_contribution = rf_contribution[..., selected_times, :]
  if selected_geos is not None:
    rf_contribution = rf_contribution[selected_geos, ...]
  
  return tf.reduce_sum(rf_contribution, axis=[0, 1])  # Sum over geos and times


def compute_incremental_outcome(
    media: tf.Tensor,
    media_coefficients: tf.Tensor,
    alpha: tf.Tensor,
    ec50: tf.Tensor,
    slope: tf.Tensor,
    max_lag: int = 13,
    selected_times: Optional[slice] = None,
    selected_geos: Optional[slice] = None,
) -> tf.Tensor:
  """Compute incremental outcome from media spend and model parameters.
  
  Args:
    media: Media tensor with shape (n_geos, n_times, n_channels).
    media_coefficients: Media effect coefficients with shape (n_geos, n_channels).
    alpha: Adstock decay parameters with shape (n_channels,).
    ec50: EC50 parameters with shape (n_channels,).
    slope: Slope parameters with shape (n_channels,).
    max_lag: Maximum lag periods for adstock.
    selected_times: Time slice for aggregation.
    selected_geos: Geo slice for aggregation.
    
  Returns:
    Incremental outcome per channel with shape (n_channels,).
  """
  n_times_output = media.shape[-2]
  if selected_times is not None:
    n_times_output = len(range(*selected_times.indices(media.shape[-2])))
  
  # Apply adstock and hill transformations
  transformed_media = apply_adstock_hill(
      media=media,
      alpha=alpha,
      ec50=ec50,
      slope=slope,
      max_lag=max_lag,
      n_times_output=n_times_output,
  )
  
  # Apply media coefficients
  media_contribution = tf.einsum(
      'gtm,gm->gtm', 
      transformed_media, 
      media_coefficients
  )
  
  # Sum over time and geo dimensions
  if selected_times is not None:
    media_contribution = media_contribution[..., selected_times, :]
  if selected_geos is not None:
    media_contribution = media_contribution[selected_geos, ...]
  
  return tf.reduce_sum(media_contribution, axis=[0, 1])  # Sum over geos and times