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

"""ROIToCoefficientsConverter for converting ROI data to equivalent coefficients."""

import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf

from meridian.data import input_data
from meridian.model import model
from meridian.model import spec
from meridian import constants


__all__ = [
    'ROIToCoefficientsConverter',
]


class ROIToCoefficientsConverter:
  """Converts ROI data to equivalent coefficient values for Meridian modeling.

  This class takes ROI values from Excel input and converts them to equivalent
  coefficient values that can be used with the existing MediaParameterLoader
  workflow. The conversion uses Meridian's media transformation functions
  (adstock and hill) to calculate the relationship between media inputs and
  incremental outcomes.

  The conversion process:
  1. Creates a dummy Meridian model for proper scaling and transformations
  2. Applies media transformations using provided parameters (adstock, hill)
  3. Calculates denominators from transformed media data
  4. Calculates numerators from ROI × Spend data
  5. Computes coefficients as the ratio numerator / denominator
  """

  def __init__(self, 
               roi_df: pd.DataFrame, 
               parameters_df: pd.DataFrame, 
               input_data_obj: input_data.InputData,
               model_config: Dict[str, Any]):
    """Initialize the ROIToCoefficientsConverter.

    Args:
      roi_df: DataFrame containing ROI values from Excel ROI sheet.
        Expected format: same as coefficients sheet but with ROI values.
        Must have 'geo' column and channel columns matching model_config.
      parameters_df: DataFrame containing parameters from Excel Parameters sheet.
        Expected columns: 'MediaVariable', 'Adstock', 'Inflexion', 'Slope'
      input_data_obj: Meridian InputData object containing the model data.
        Used for creating dummy model and extracting scaling information.
      model_config: Dictionary containing model configuration with keys:
        - media_channels: List of media channel names
        - rf_channels: List of R&F channel names (optional)
        - media_cols: List of media column names in data
        - media_spend_cols: List of media spend column names
        - rf_cols: List of R&F column names (optional)
        - rf_spend_cols: List of R&F spend column names (optional)
    """
    self.roi_df = roi_df
    self.parameters_df = parameters_df
    self.input_data_obj = input_data_obj
    self.model_config = model_config
    
    # Extract channel configurations
    self.media_channels = model_config.get('media_channels', [])
    self.rf_channels = model_config.get('rf_channels', [])
    
    # Validate inputs
    self._validate_inputs()
    
    # Initialize internal state
    self.dummy_model = None
    self.converted_coefficients_df = None
    
    logging.info(f"Initialized ROIToCoefficientsConverter with {len(self.media_channels)} media channels and {len(self.rf_channels)} RF channels")

  def _validate_inputs(self) -> None:
    """Validate input data for ROI conversion."""
    if self.roi_df is None or self.roi_df.empty:
      raise ValueError("ROI DataFrame is None or empty")
    
    if self.parameters_df is None or self.parameters_df.empty:
      raise ValueError("Parameters DataFrame is None or empty")
    
    if self.input_data_obj is None:
      raise ValueError("InputData object is None")
    
    if not self.media_channels and not self.rf_channels:
      raise ValueError("At least one of media_channels or rf_channels must be specified")

    # Validate ROI DataFrame structure
    required_roi_cols = ['geo']
    required_roi_cols.extend(self.media_channels)
    required_roi_cols.extend(self.rf_channels)
    
    missing_cols = [col for col in required_roi_cols if col not in self.roi_df.columns]
    if missing_cols:
      raise ValueError(f"Missing required columns in ROI data: {missing_cols}")
    
    # Validate ROI values are positive and finite
    self._validate_roi_values()
    
    # Validate parameters DataFrame structure and values
    self._validate_parameters()
    
    # Validate model configuration for required fields
    self._validate_model_config()

  def _validate_roi_values(self) -> None:
    """Validate that ROI values are positive and finite."""
    all_channels = self.media_channels + self.rf_channels
    
    for channel in all_channels:
      if channel in self.roi_df.columns:
        values = self.roi_df[channel].values
        
        # Check for NaN values
        if np.isnan(values).any():
          raise ValueError(f"ROI data for channel '{channel}' contains NaN values")
        
        # Check for infinite values
        if np.isinf(values).any():
          raise ValueError(f"ROI data for channel '{channel}' contains infinite values")
        
        # Check for negative values (ROI should be positive)
        if (values < 0).any():
          raise ValueError(f"ROI data for channel '{channel}' contains negative values. ROI should be positive.")
        
        # Check for zero values (may indicate data issues)
        if (values == 0).all():
          logging.warning(f"All ROI values for channel '{channel}' are zero. This may indicate data issues.")

  def _validate_parameters(self) -> None:
    """Validate parameters DataFrame structure and value ranges."""
    required_param_cols = ['MediaVariable', 'Adstock', 'Inflexion', 'Slope']
    missing_param_cols = [col for col in required_param_cols if col not in self.parameters_df.columns]
    if missing_param_cols:
      raise ValueError(f"Missing required columns in Parameters data: {missing_param_cols}")
    
    # Validate that all required channels are present in parameters
    all_channels = self.media_channels + self.rf_channels
    param_channels = set(self.parameters_df['MediaVariable'].values)
    missing_channels = [ch for ch in all_channels if ch not in param_channels]
    if missing_channels:
      raise ValueError(f"Missing channels in Parameters data: {missing_channels}")
    
    # Validate parameter value ranges
    for _, row in self.parameters_df.iterrows():
      channel = row['MediaVariable']
      adstock = row['Adstock']
      inflexion = row['Inflexion']
      slope = row['Slope']
      
      # Adstock should be between 0 and 1
      if not (0 <= adstock <= 1):
        raise ValueError(f"Adstock parameter for '{channel}' must be between 0 and 1, got {adstock}")
      
      # Inflexion should be positive
      if inflexion <= 0:
        raise ValueError(f"Inflexion parameter for '{channel}' must be positive, got {inflexion}")
      
      # Slope should be positive
      if slope <= 0:
        raise ValueError(f"Slope parameter for '{channel}' must be positive, got {slope}")

  def _validate_model_config(self) -> None:
    """Validate model configuration contains required fields for ROI conversion."""
    required_config_keys = ['media_channels']
    
    # Add media spend validation if media channels exist
    if self.media_channels:
      required_config_keys.extend(['media_cols', 'media_spend_cols'])
      
      # Validate lengths match
      if len(self.model_config.get('media_cols', [])) != len(self.media_channels):
        raise ValueError(f"Length of media_cols must match media_channels")
      
      if len(self.model_config.get('media_spend_cols', [])) != len(self.media_channels):
        raise ValueError(f"Length of media_spend_cols must match media_channels")
    
    # Add RF validation if RF channels exist
    if self.rf_channels:
      required_config_keys.extend(['rf_channels'])
      
      if 'reach_cols' in self.model_config and 'frequency_cols' in self.model_config:
        if len(self.model_config.get('reach_cols', [])) != len(self.rf_channels):
          raise ValueError(f"Length of reach_cols must match rf_channels")
        
        if len(self.model_config.get('frequency_cols', [])) != len(self.rf_channels):
          raise ValueError(f"Length of frequency_cols must match rf_channels")
      
      if 'rf_spend_cols' in self.model_config:
        if len(self.model_config.get('rf_spend_cols', [])) != len(self.rf_channels):
          raise ValueError(f"Length of rf_spend_cols must match rf_channels")
    
    # Check for missing keys
    missing_keys = [key for key in required_config_keys if key not in self.model_config]
    if missing_keys:
      raise ValueError(f"Missing required keys in model_config: {missing_keys}")

  def convert_roi_to_coefficients(self) -> pd.DataFrame:
    """Convert ROI values to equivalent coefficient values.

    This is the main method that orchestrates the conversion process:
    1. Creates dummy Meridian model for transformations
    2. Processes media and RF transformations separately  
    3. Calculates coefficient values using the conversion formula
    4. Returns coefficients DataFrame in the same format as input ROI

    Returns:
      DataFrame with coefficient values in the same format as ROI input.
      Can be used directly as coefficients_df in MediaParameterLoader.
      
    Raises:
      ValueError: If conversion fails due to data issues or calculation errors.
    """
    if self.converted_coefficients_df is not None:
      return self.converted_coefficients_df
    
    try:
      # Step 1: Create dummy model for transformations and scaling
      self.dummy_model = self._create_dummy_model()
      
      # Step 2: Process media channels if present
      media_coefficients = None
      if self.media_channels:
        media_coefficients = self._convert_media_roi_to_coefficients()
      
      # Step 3: Process RF channels if present  
      rf_coefficients = None
      if self.rf_channels:
        rf_coefficients = self._convert_rf_roi_to_coefficients()
      
      # Step 4: Combine results into final DataFrame
      self.converted_coefficients_df = self._combine_coefficient_results(
          media_coefficients, rf_coefficients)
      
      logging.info(f"Successfully converted ROI to coefficients for {len(self.converted_coefficients_df)} geos")
      return self.converted_coefficients_df
      
    except Exception as e:
      raise ValueError(f"Error converting ROI to coefficients: {str(e)}")

  def _create_dummy_model(self) -> model.Meridian:
    """Create a dummy Meridian model for transformation and scaling operations.
    
    The dummy model is used to:
    - Access proper scaling factors through the model's transformers
    - Use the model's media transformation functions (adstock_hill_media_fn, etc.)
    - Ensure consistent scaling with what Meridian would use in practice
    
    Returns:
      Meridian model instance with default ModelSpec but using our InputData.
    """
    try:
      model_spec = spec.ModelSpec()
      dummy_model = model.Meridian(
          input_data=self.input_data_obj,
          model_spec=model_spec
      )
      logging.info("Successfully created dummy Meridian model for ROI conversion")
      return dummy_model
      
    except Exception as e:
      raise ValueError(f"Error creating dummy Meridian model: {str(e)}")

  def _convert_media_roi_to_coefficients(self) -> Optional[Dict[str, np.ndarray]]:
    """Convert ROI values to coefficients for media channels.
    
    Uses the conversion formula for impression-based media channels:
    1. Apply adstock and hill transformations to media data
    2. Calculate denominator using transformed media data  
    3. Calculate numerator from ROI × Spend
    4. Return coefficients as numerator / denominator
    
    Returns:
      Dictionary mapping channel names to coefficient arrays (per geo).
      None if no media channels are configured.
    """
    if not self.media_channels:
      return None
    
    try:
      # Step 1: Extract media parameters from parameters DataFrame
      media_params = self._extract_media_parameters()
      alpha_m, ec_m, slope_m = media_params['alpha_m'], media_params['ec_m'], media_params['slope_m']
      
      # Step 2: Get scaled media data from dummy model
      media_scaled = self.dummy_model.media_tensors.media_scaled
      if media_scaled is None:
        raise ValueError("Media scaled data not available from dummy model")
      
      # Step 3: Apply media transformations using dummy model
      media_transformed = self.dummy_model.adstock_hill_media(
          media=media_scaled,
          alpha=tf.constant(alpha_m, dtype=tf.float32),
          ec=tf.constant(ec_m, dtype=tf.float32),
          slope=tf.constant(slope_m, dtype=tf.float32)
      )
      logging.info(f"Applied media transformations. Transformed shape: {media_transformed.shape}")
      
      # Step 4: Calculate denominator using einsum operation
      denominator_gx = self._calculate_media_denominator(media_transformed)
      
      # Step 5: Extract ROI and spend data for media channels
      roi_gx, spend_gx = self._extract_media_roi_and_spend()
      
      # Step 6: Calculate numerator (ROI × Spend)
      numerator_gx = tf.multiply(roi_gx, spend_gx)
      logging.info(f"Calculated numerator shape: {numerator_gx.shape}")
      
      # Step 7: Calculate coefficients (numerator / denominator) with protection against division by zero
      # Add small epsilon to avoid division by zero
      epsilon = tf.constant(1e-8, dtype=tf.float32)
      denominator_safe = tf.maximum(denominator_gx, epsilon)
      coefficients_gx = tf.divide(numerator_gx, denominator_safe)
      
      # Check for potential division by zero issues
      near_zero_mask = tf.less(tf.abs(denominator_gx), epsilon)
      if tf.reduce_any(near_zero_mask):
        logging.warning("Some denominator values are very small, coefficient calculations may be unstable")
      
      logging.info(f"Calculated coefficients shape: {coefficients_gx.shape}")
      
      # Validate coefficient values
      self._validate_coefficient_results(coefficients_gx, "media")
      
      # Step 8: Convert to dictionary format with channel names
      return self._format_media_coefficients(coefficients_gx)
      
    except Exception as e:
      raise ValueError(f"Error converting media ROI to coefficients: {str(e)}")

  def _convert_rf_roi_to_coefficients(self) -> Optional[Dict[str, np.ndarray]]:
    """Convert ROI values to coefficients for reach & frequency channels.
    
    Uses the conversion formula for reach/frequency-based channels:
    1. Apply adstock and hill transformations to RF data
    2. Calculate denominator using transformed RF data
    3. Calculate numerator from ROI × Spend  
    4. Return coefficients as numerator / denominator
    
    Returns:
      Dictionary mapping RF channel names to coefficient arrays (per geo).
      None if no RF channels are configured.
    """
    if not self.rf_channels:
      return None
      
    try:
      # Step 1: Extract RF parameters from parameters DataFrame
      rf_params = self._extract_rf_parameters()
      alpha_rf, ec_rf, slope_rf = rf_params['alpha_rf'], rf_params['ec_rf'], rf_params['slope_rf']
      
      # Step 2: Get scaled RF data from dummy model
      reach_scaled = self.dummy_model.rf_tensors.reach_scaled
      frequency_data = self.dummy_model.rf_tensors.frequency
      
      if reach_scaled is None or frequency_data is None:
        raise ValueError("RF scaled data not available from dummy model")
      
      # Step 3: Apply RF transformations using dummy model
      rf_transformed = self.dummy_model.adstock_hill_rf(
          reach=reach_scaled,
          frequency=frequency_data,
          alpha=tf.constant(alpha_rf, dtype=tf.float32),
          ec=tf.constant(ec_rf, dtype=tf.float32),
          slope=tf.constant(slope_rf, dtype=tf.float32)
      )
      logging.info(f"Applied RF transformations. Transformed shape: {rf_transformed.shape}")
      
      # Step 4: Calculate denominator using einsum operation
      denominator_gx = self._calculate_rf_denominator(rf_transformed)
      
      # Step 5: Extract ROI and spend data for RF channels
      roi_gx, spend_gx = self._extract_rf_roi_and_spend()
      
      # Step 6: Calculate numerator (ROI × Spend)
      numerator_gx = tf.multiply(roi_gx, spend_gx)
      logging.info(f"Calculated RF numerator shape: {numerator_gx.shape}")
      
      # Step 7: Calculate coefficients (numerator / denominator) with protection against division by zero
      # Add small epsilon to avoid division by zero
      epsilon = tf.constant(1e-8, dtype=tf.float32)
      denominator_safe = tf.maximum(denominator_gx, epsilon)
      coefficients_gx = tf.divide(numerator_gx, denominator_safe)
      
      # Check for potential division by zero issues
      near_zero_mask = tf.less(tf.abs(denominator_gx), epsilon)
      if tf.reduce_any(near_zero_mask):
        logging.warning("Some RF denominator values are very small, coefficient calculations may be unstable")
      
      logging.info(f"Calculated RF coefficients shape: {coefficients_gx.shape}")
      
      # Validate coefficient values
      self._validate_coefficient_results(coefficients_gx, "RF")
      
      # Step 8: Convert to dictionary format with channel names
      return self._format_rf_coefficients(coefficients_gx)
      
    except Exception as e:
      raise ValueError(f"Error converting RF ROI to coefficients: {str(e)}")

  def _combine_coefficient_results(self, 
                                   media_coefficients: Optional[Dict[str, np.ndarray]],
                                   rf_coefficients: Optional[Dict[str, np.ndarray]]) -> pd.DataFrame:
    """Combine media and RF coefficient results into a single DataFrame.
    
    Args:
      media_coefficients: Dictionary of media channel coefficients by geo.
      rf_coefficients: Dictionary of RF channel coefficients by geo.
      
    Returns:
      DataFrame with same structure as original ROI DataFrame but with 
      coefficient values instead of ROI values.
    """
    # Start with geo column from original ROI data
    result_data = {'geo': self.roi_df['geo'].tolist()}
    
    # Add media coefficients
    if media_coefficients:
      for channel in self.media_channels:
        if channel in media_coefficients:
          result_data[channel] = media_coefficients[channel]
        else:
          logging.warning(f"Media channel {channel} not found in converted coefficients")
          result_data[channel] = np.zeros(len(self.roi_df))
    
    # Add RF coefficients  
    if rf_coefficients:
      for channel in self.rf_channels:
        if channel in rf_coefficients:
          result_data[channel] = rf_coefficients[channel]
        else:
          logging.warning(f"RF channel {channel} not found in converted coefficients")
          result_data[channel] = np.zeros(len(self.roi_df))
    
    return pd.DataFrame(result_data)

  def _extract_media_parameters(self) -> Dict[str, List[float]]:
    """Extract media transformation parameters from parameters DataFrame.
    
    Returns:
      Dictionary with lists of parameter values for media channels:
      - alpha_m: List of adstock retention rates
      - ec_m: List of hill inflection points  
      - slope_m: List of hill slope values
    """
    # Filter parameters for media channels only
    media_params_df = self.parameters_df[
        self.parameters_df['MediaVariable'].isin(self.media_channels)
    ].copy()
    
    # Ensure channels are in the correct order
    media_params_df['channel_order'] = media_params_df['MediaVariable'].apply(
        lambda x: self.media_channels.index(x)
    )
    media_params_df = media_params_df.sort_values('channel_order')
    
    return {
        'alpha_m': media_params_df['Adstock'].tolist(),
        'ec_m': media_params_df['Inflexion'].tolist(),
        'slope_m': media_params_df['Slope'].tolist()
    }

  def _extract_rf_parameters(self) -> Dict[str, List[float]]:
    """Extract RF transformation parameters from parameters DataFrame.
    
    Returns:
      Dictionary with lists of parameter values for RF channels:
      - alpha_rf: List of adstock retention rates
      - ec_rf: List of hill inflection points
      - slope_rf: List of hill slope values
    """
    if not self.rf_channels:
      return {'alpha_rf': [], 'ec_rf': [], 'slope_rf': []}
    
    # Filter parameters for RF channels only
    rf_params_df = self.parameters_df[
        self.parameters_df['MediaVariable'].isin(self.rf_channels)
    ].copy()
    
    # Ensure channels are in the correct order
    rf_params_df['channel_order'] = rf_params_df['MediaVariable'].apply(
        lambda x: self.rf_channels.index(x)
    )
    rf_params_df = rf_params_df.sort_values('channel_order')
    
    return {
        'alpha_rf': rf_params_df['Adstock'].tolist(),
        'ec_rf': rf_params_df['Inflexion'].tolist(),
        'slope_rf': rf_params_df['Slope'].tolist()
    }

  def _calculate_media_denominator(self, media_transformed: tf.Tensor) -> tf.Tensor:
    """Calculate denominator for media coefficient calculation using einsum.
    
    Args:
      media_transformed: Transformed media tensor with shape (n_geos, n_times, n_media_channels).
      
    Returns:
      Denominator tensor with shape (n_geos, n_media_channels).
    """
    # Get required data from dummy model and input data
    revenue_per_kpi = self.input_data_obj.revenue_per_kpi  # Shape: (geo, time)
    population = self.input_data_obj.population  # Shape: (geo,)
    population_scaled_stdev = self.dummy_model.kpi_transformer.population_scaled_stdev  # Scalar
    
    # Convert to TensorFlow tensors
    if revenue_per_kpi is not None:
      revenue_per_kpi_tensor = tf.constant(revenue_per_kpi.values, dtype=tf.float32)
    else:
      # If no revenue_per_kpi, use ones (assuming non-revenue KPI)
      n_geos, n_times = media_transformed.shape[0], media_transformed.shape[1]
      revenue_per_kpi_tensor = tf.ones((n_geos, n_times), dtype=tf.float32)
      
    population_tensor = tf.constant(population.values, dtype=tf.float32)
    
    # Apply einsum operation: "gtx,gt,g,->gx"
    # media_transformed: (geo, time, channel)
    # revenue_per_kpi: (geo, time)  
    # population: (geo,)
    # population_scaled_stdev: scalar
    denominator_gx = tf.einsum(
        'gtx,gt,g->gx',
        media_transformed,
        revenue_per_kpi_tensor,
        population_tensor
    ) * population_scaled_stdev
    
    return denominator_gx

  def _extract_media_roi_and_spend(self) -> Tuple[tf.Tensor, tf.Tensor]:
    """Extract ROI and spend data for media channels.
    
    Returns:
      Tuple of (roi_tensor, spend_tensor) both with shape (n_geos, n_media_channels).
    """
    # Extract ROI data for media channels (from ROI DataFrame)
    roi_data = []
    for channel in self.media_channels:
      if channel not in self.roi_df.columns:
        raise ValueError(f"Media channel {channel} not found in ROI data")
      roi_data.append(self.roi_df[channel].values)
    
    roi_array = np.array(roi_data).T  # Transpose to get (geo, channel) shape
    roi_tensor = tf.constant(roi_array, dtype=tf.float32)
    
    # Extract spend data for media channels (from InputData)
    # The spend data is in the media_spend columns of the original data
    spend_data = []
    media_spend_cols = self.model_config.get('media_spend_cols', [])
    
    if len(media_spend_cols) != len(self.media_channels):
      raise ValueError(f"Number of media_spend_cols ({len(media_spend_cols)}) must match media_channels ({len(self.media_channels)})")
    
    # Get spend data by summing across time for each geo-channel combination  
    for channel in self.media_channels:
      # Sum spend across time periods for each geo
      spend_by_geo = []
      for geo in self.input_data_obj.geo.values:
        geo_data = self.input_data_obj.media_spend.sel(geo=geo, media_channel=channel).values
        total_spend = float(np.sum(geo_data))
        spend_by_geo.append(total_spend)
      
      spend_data.append(spend_by_geo)
    
    spend_array = np.array(spend_data).T  # Transpose to get (geo, channel) shape
    spend_tensor = tf.constant(spend_array, dtype=tf.float32)
    
    return roi_tensor, spend_tensor

  def _format_media_coefficients(self, coefficients_tensor: tf.Tensor) -> Dict[str, np.ndarray]:
    """Format media coefficients tensor into dictionary with channel names.
    
    Args:
      coefficients_tensor: Tensor with shape (n_geos, n_media_channels).
      
    Returns:
      Dictionary mapping channel names to coefficient arrays per geo.
    """
    coefficients_array = coefficients_tensor.numpy()
    result = {}
    
    for i, channel in enumerate(self.media_channels):
      result[channel] = coefficients_array[:, i]  # Extract column for this channel
    
    return result

  def _calculate_rf_denominator(self, rf_transformed: tf.Tensor) -> tf.Tensor:
    """Calculate denominator for RF coefficient calculation using einsum.
    
    Args:
      rf_transformed: Transformed RF tensor with shape (n_geos, n_times, n_rf_channels).
      
    Returns:
      Denominator tensor with shape (n_geos, n_rf_channels).
    """
    # Use the same logic as media denominator but for RF channels
    revenue_per_kpi = self.input_data_obj.revenue_per_kpi  
    population = self.input_data_obj.population  
    population_scaled_stdev = self.dummy_model.kpi_transformer.population_scaled_stdev  
    
    # Convert to TensorFlow tensors
    if revenue_per_kpi is not None:
      revenue_per_kpi_tensor = tf.constant(revenue_per_kpi.values, dtype=tf.float32)
    else:
      # If no revenue_per_kpi, use ones (assuming non-revenue KPI)
      n_geos, n_times = rf_transformed.shape[0], rf_transformed.shape[1]
      revenue_per_kpi_tensor = tf.ones((n_geos, n_times), dtype=tf.float32)
      
    population_tensor = tf.constant(population.values, dtype=tf.float32)
    
    # Apply einsum operation for RF: "gtx,gt,g->gx"
    denominator_gx = tf.einsum(
        'gtx,gt,g->gx',
        rf_transformed,
        revenue_per_kpi_tensor,
        population_tensor
    ) * population_scaled_stdev
    
    return denominator_gx

  def _extract_rf_roi_and_spend(self) -> Tuple[tf.Tensor, tf.Tensor]:
    """Extract ROI and spend data for RF channels.
    
    Returns:
      Tuple of (roi_tensor, spend_tensor) both with shape (n_geos, n_rf_channels).
    """
    # Extract ROI data for RF channels (from ROI DataFrame)
    roi_data = []
    for channel in self.rf_channels:
      if channel not in self.roi_df.columns:
        raise ValueError(f"RF channel {channel} not found in ROI data")
      roi_data.append(self.roi_df[channel].values)
    
    roi_array = np.array(roi_data).T  # Transpose to get (geo, channel) shape
    roi_tensor = tf.constant(roi_array, dtype=tf.float32)
    
    # Extract spend data for RF channels (from InputData)
    spend_data = []
    rf_spend_cols = self.model_config.get('rf_spend_cols', [])
    
    if len(rf_spend_cols) != len(self.rf_channels):
      raise ValueError(f"Number of rf_spend_cols ({len(rf_spend_cols)}) must match rf_channels ({len(self.rf_channels)})")
    
    # Get spend data by summing across time for each geo-channel combination  
    for channel in self.rf_channels:
      # Sum spend across time periods for each geo
      spend_by_geo = []
      for geo in self.input_data_obj.geo.values:
        geo_data = self.input_data_obj.rf_spend.sel(geo=geo, rf_channel=channel).values
        total_spend = float(np.sum(geo_data))
        spend_by_geo.append(total_spend)
      
      spend_data.append(spend_by_geo)
    
    spend_array = np.array(spend_data).T  # Transpose to get (geo, channel) shape
    spend_tensor = tf.constant(spend_array, dtype=tf.float32)
    
    return roi_tensor, spend_tensor

  def _format_rf_coefficients(self, coefficients_tensor: tf.Tensor) -> Dict[str, np.ndarray]:
    """Format RF coefficients tensor into dictionary with channel names.
    
    Args:
      coefficients_tensor: Tensor with shape (n_geos, n_rf_channels).
      
    Returns:
      Dictionary mapping RF channel names to coefficient arrays per geo.
    """
    coefficients_array = coefficients_tensor.numpy()
    result = {}
    
    for i, channel in enumerate(self.rf_channels):
      result[channel] = coefficients_array[:, i]  # Extract column for this channel
    
    return result

  def _validate_coefficient_results(self, coefficients_tensor: tf.Tensor, channel_type: str) -> None:
    """Validate the calculated coefficient results for potential issues.
    
    Args:
      coefficients_tensor: Calculated coefficients tensor.
      channel_type: Type of channels ("media" or "RF") for logging.
    """
    coefficients_array = coefficients_tensor.numpy()
    
    # Check for NaN values
    if np.isnan(coefficients_array).any():
      raise ValueError(f"Calculated {channel_type} coefficients contain NaN values. This indicates a numerical issue in the conversion.")
    
    # Check for infinite values
    if np.isinf(coefficients_array).any():
      raise ValueError(f"Calculated {channel_type} coefficients contain infinite values. This indicates a numerical issue in the conversion.")
    
    # Check for very large values (potential numerical instability)
    max_coeff = np.max(np.abs(coefficients_array))
    if max_coeff > 1e6:
      logging.warning(f"Some {channel_type} coefficients are very large (max: {max_coeff:.2e}). This may indicate numerical instability.")
    
    # Check for very small values
    min_nonzero_coeff = np.min(np.abs(coefficients_array[coefficients_array != 0]))
    if min_nonzero_coeff < 1e-10:
      logging.warning(f"Some {channel_type} coefficients are very small (min non-zero: {min_nonzero_coeff:.2e}). This may indicate numerical precision issues.")
    
    # Log coefficient statistics
    logging.info(f"{channel_type} coefficient statistics - Mean: {np.mean(coefficients_array):.6f}, "
                f"Std: {np.std(coefficients_array):.6f}, "
                f"Min: {np.min(coefficients_array):.6f}, "
                f"Max: {np.max(coefficients_array):.6f}")

  def get_conversion_summary(self) -> Dict[str, Any]:
    """Get summary of the ROI to coefficients conversion process.
    
    Returns:
      Dictionary containing conversion statistics and information for debugging.
    """
    if self.converted_coefficients_df is None:
      return {"status": "not_converted", "message": "Conversion not yet performed"}
    
    summary = {
        "status": "converted",
        "num_geos": len(self.converted_coefficients_df),
        "media_channels": self.media_channels,
        "rf_channels": self.rf_channels,
        "coefficient_columns": list(self.converted_coefficients_df.columns),
    }
    
    # Add statistics for converted values
    for channel in self.media_channels + self.rf_channels:
      if channel in self.converted_coefficients_df.columns:
        values = self.converted_coefficients_df[channel].values
        summary[f"{channel}_stats"] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)), 
            "max": float(np.max(values)),
            "std": float(np.std(values))
        }
    
    return summary