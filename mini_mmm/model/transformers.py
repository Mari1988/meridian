"""Media transformation functions for Mini MMM.

This module implements the core media transformations used in Marketing Mix Modeling:
- Adstock transformation (carryover effects)
- Hill saturation transformation (diminishing returns)

These transformations are based on the Meridian methodology but simplified
for clarity and computational efficiency.
"""

import numpy as np
from typing import Union, Optional
from scipy.optimize import minimize_scalar


class AdstockTransformer:
  """Applies adstock (carryover) transformation to media data.
  
  Adstock represents the carryover effect of media activities, where the impact
  of advertising extends beyond the immediate period. This implementation uses
  a geometric adstock function:
  
  adstocked[t] = media[t] + retention_rate * adstocked[t-1]
  
  This is equivalent to convolving the media series with a geometric decay kernel.
  """
  
  @staticmethod
  def transform(media: np.ndarray, 
                retention_rate: Union[float, np.ndarray],
                max_lag: int = 8,
                normalizing: bool = True) -> np.ndarray:
    """Applies adstock transformation to media data.
    
    Args:
      media: Media data array of shape (n_weeks,) or (n_weeks, n_channels)
      retention_rate: Retention rate(s) between 0 and 1. Can be scalar or
                     array matching number of channels
      max_lag: Maximum number of lags to consider (for computational efficiency)
      normalizing: Whether to normalize to preserve total media volume
      
    Returns:
      Adstocked media data with same shape as input
    """
    media = np.atleast_2d(media.T).T  # Ensure 2D: (n_weeks, n_channels)
    n_weeks, n_channels = media.shape
    
    # Handle retention rate broadcasting
    retention_rate = np.atleast_1d(retention_rate)
    if len(retention_rate) == 1 and n_channels > 1:
      retention_rate = np.repeat(retention_rate, n_channels)
    elif len(retention_rate) != n_channels:
      raise ValueError(f"retention_rate length {len(retention_rate)} doesn't "
                      f"match number of channels {n_channels}")
    
    # Initialize output
    adstocked = np.zeros_like(media)
    
    # Apply adstock transformation for each channel
    for c in range(n_channels):
      retention = retention_rate[c]
      
      # Simple geometric adstock (can be optimized with convolution)
      for t in range(n_weeks):
        adstocked[t, c] = media[t, c]
        # Add carryover from previous periods (up to max_lag)
        for lag in range(1, min(t + 1, max_lag + 1)):
          adstocked[t, c] += media[t - lag, c] * (retention ** lag)
    
    # Normalize to preserve total volume if requested
    if normalizing:
      for c in range(n_channels):
        original_sum = np.sum(media[:, c])
        adstocked_sum = np.sum(adstocked[:, c])
        if adstocked_sum > 0:
          adstocked[:, c] *= original_sum / adstocked_sum
    
    # Return in original shape
    return adstocked.squeeze() if media.ndim == 1 else adstocked
  
  @staticmethod
  def get_adstock_multiplier(retention_rate: float, max_lag: int = 8) -> float:
    """Calculates the total adstock multiplier for a given retention rate.
    
    This is the factor by which the total media volume is increased due to
    carryover effects.
    
    Args:
      retention_rate: Retention rate between 0 and 1
      max_lag: Maximum number of lags considered
      
    Returns:
      Adstock multiplier (always >= 1.0)
    """
    if retention_rate == 0:
      return 1.0
    
    # Geometric series sum: 1 + r + r^2 + ... + r^max_lag
    if retention_rate == 1:
      return max_lag + 1
    else:
      return (1 - retention_rate**(max_lag + 1)) / (1 - retention_rate)


class HillTransformer:
  """Applies Hill saturation transformation to media data.
  
  The Hill transformation models the diminishing returns effect in marketing,
  where additional spend has decreasing marginal impact:
  
  hill_transformed = slope * media^ec / (half_saturation^ec + media^ec)
  
  Where:
  - slope: Maximum possible effect (saturation level)
  - ec: Shape parameter controlling curvature 
  - half_saturation: Media level at which effect is half of slope
  """
  
  @staticmethod
  def transform(media: np.ndarray,
                ec: Union[float, np.ndarray],
                slope: Union[float, np.ndarray],
                half_saturation: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """Applies Hill saturation transformation to media data.
    
    Args:
      media: Media data array of shape (n_weeks,) or (n_weeks, n_channels)
      ec: Shape parameter(s), typically between 0.3 and 3.0
      slope: Maximum effect parameter(s)
      half_saturation: Half-saturation point(s). If None, estimated as median
                      of non-zero media values
      
    Returns:
      Hill-transformed media data with same shape as input
    """
    media = np.atleast_2d(media.T).T  # Ensure 2D: (n_weeks, n_channels)
    n_weeks, n_channels = media.shape
    
    # Handle parameter broadcasting
    ec = np.atleast_1d(ec)
    slope = np.atleast_1d(slope)
    
    if len(ec) == 1 and n_channels > 1:
      ec = np.repeat(ec, n_channels)
    if len(slope) == 1 and n_channels > 1:
      slope = np.repeat(slope, n_channels)
    
    # Estimate half_saturation if not provided
    if half_saturation is None:
      half_saturation = np.zeros(n_channels)
      for c in range(n_channels):
        non_zero_media = media[:, c][media[:, c] > 0]
        if len(non_zero_media) > 0:
          half_saturation[c] = np.median(non_zero_media)
        else:
          half_saturation[c] = 1.0  # Default value
    else:
      half_saturation = np.atleast_1d(half_saturation)
      if len(half_saturation) == 1 and n_channels > 1:
        half_saturation = np.repeat(half_saturation, n_channels)
    
    # Validate parameter lengths
    if len(ec) != n_channels:
      raise ValueError(f"ec length {len(ec)} doesn't match channels {n_channels}")
    if len(slope) != n_channels:
      raise ValueError(f"slope length {len(slope)} doesn't match channels {n_channels}")
    if len(half_saturation) != n_channels:
      raise ValueError(f"half_saturation length {len(half_saturation)} doesn't "
                      f"match channels {n_channels}")
    
    # Apply Hill transformation
    hill_transformed = np.zeros_like(media)
    
    for c in range(n_channels):
      media_c = media[:, c]
      ec_c = ec[c]
      slope_c = slope[c]
      half_sat_c = half_saturation[c]
      
      # Handle zero half_saturation
      if half_sat_c <= 0:
        half_sat_c = 1.0
      
      # Hill transformation: slope * x^ec / (half_sat^ec + x^ec)
      media_powered = np.power(media_c, ec_c)
      half_sat_powered = np.power(half_sat_c, ec_c)
      
      hill_transformed[:, c] = (slope_c * media_powered / 
                               (half_sat_powered + media_powered))
    
    # Return in original shape
    return hill_transformed.squeeze() if media.ndim == 1 else hill_transformed
  
  @staticmethod
  def get_marginal_effect(media: Union[float, np.ndarray],
                         ec: float,
                         slope: float, 
                         half_saturation: float) -> Union[float, np.ndarray]:
    """Calculates marginal effect (derivative) of Hill transformation.
    
    Args:
      media: Media spend level(s)
      ec: Shape parameter
      slope: Maximum effect parameter
      half_saturation: Half-saturation point
      
    Returns:
      Marginal effect at given media level(s)
    """
    if half_saturation <= 0:
      half_saturation = 1.0
    
    media_powered = np.power(media, ec)
    half_sat_powered = np.power(half_saturation, ec)
    
    # Derivative of Hill function
    numerator = slope * ec * np.power(media, ec - 1) * half_sat_powered
    denominator = np.power(half_sat_powered + media_powered, 2)
    
    return numerator / denominator
  
  @staticmethod
  def find_optimal_spend(target_efficiency: float,
                        ec: float,
                        slope: float,
                        half_saturation: float,
                        max_spend: float = 1000) -> float:
    """Finds optimal spend level for a target marginal efficiency.
    
    Args:
      target_efficiency: Target marginal ROI (effect per unit spend)
      ec: Shape parameter
      slope: Maximum effect parameter  
      half_saturation: Half-saturation point
      max_spend: Maximum spend to consider in search
      
    Returns:
      Optimal spend level
    """
    def efficiency_diff(spend):
      marginal = HillTransformer.get_marginal_effect(
          spend, ec, slope, half_saturation)
      return abs(marginal - target_efficiency)
    
    result = minimize_scalar(efficiency_diff, bounds=(0.01, max_spend), 
                           method='bounded')
    return result.x


def apply_media_transformations(media: np.ndarray,
                               retention_rates: np.ndarray,
                               ec_values: np.ndarray,
                               slope_values: np.ndarray,
                               half_saturation: Optional[np.ndarray] = None,
                               adstock_first: bool = True,
                               max_lag: int = 8) -> np.ndarray:
  """Applies both adstock and Hill transformations to media data.
  
  Args:
    media: Media data of shape (n_weeks, n_channels)
    retention_rates: Adstock retention rates for each channel
    ec_values: Hill shape parameters for each channel
    slope_values: Hill slope parameters for each channel
    half_saturation: Optional Hill half-saturation parameters
    adstock_first: Whether to apply adstock before Hill (recommended)
    max_lag: Maximum lag for adstock transformation
    
  Returns:
    Transformed media data with same shape as input
  """
  if adstock_first:
    # Adstock first, then Hill
    adstocked = AdstockTransformer.transform(
        media, retention_rates, max_lag=max_lag)
    transformed = HillTransformer.transform(
        adstocked, ec_values, slope_values, half_saturation)
  else:
    # Hill first, then Adstock
    hill_transformed = HillTransformer.transform(
        media, ec_values, slope_values, half_saturation)
    transformed = AdstockTransformer.transform(
        hill_transformed, retention_rates, max_lag=max_lag)
  
  return transformed