"""MediaParameterLoader for processing Parameters sheet data."""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import xarray as xr

from meridian import constants


__all__ = [
    'MediaParameterLoader',
]


class MediaParameterLoader:
  """Loads and processes media parameters from Excel Parameters sheet.

  This class handles the Parameters sheet data and organizes parameter values
  according to Meridian constants for media and R&F channels. It validates
  the data structure and ensures proper alignment with model configuration.
  """

  def __init__(self, parameters_df: pd.DataFrame, model_config: Dict[str, Any], 
               coefficients_df: Optional[pd.DataFrame] = None, 
               data_df: Optional[pd.DataFrame] = None,
               auto_filter_geos: bool = True):
    """Initialize the MediaParameterLoader.

    Args:
      parameters_df: DataFrame containing parameters from Excel Parameters sheet.
        Expected columns: 'MediaVariable', 'Adstock', 'Inflexion', 'Slope'
      model_config: Dictionary containing model configuration with keys:
        - media_channels: List of media channel names
        - rf_channels: List of R&F channel names (optional)
      coefficients_df: Optional DataFrame containing coefficients from Excel Coefficients sheet.
        Expected columns: 'geo' + media_channels + rf_channels
      data_df: Optional DataFrame containing data from Excel Data sheet (for geo validation).
      auto_filter_geos: Whether to skip strict geo validation. If True (default), 
        assumes data has already been filtered. If False, enforces strict geo matching.
    """
    self.parameters_df = parameters_df.copy()
    self.coefficients_df = coefficients_df.copy() if coefficients_df is not None else None
    self.data_df = data_df
    self.model_config = model_config
    self.auto_filter_geos = auto_filter_geos

    # Validate inputs
    self._validate_inputs()

    # Get combined channel list
    self.media_channels = self.model_config.get('media_channels', [])
    self.rf_channels = self.model_config.get('rf_channels', [])
    self.all_channels = self.media_channels + self.rf_channels

    # Processed data
    self.processed_parameters: Optional[Dict[str, List[float]]] = None
    self.processed_coefficients: Optional[Dict[str, xr.DataArray]] = None

  def _validate_inputs(self) -> None:
    """Validate input DataFrame and model configuration."""
    if self.parameters_df is None or self.parameters_df.empty:
      raise ValueError("parameters_df cannot be None or empty")

    if not isinstance(self.model_config, dict):
      raise ValueError("model_config must be a dictionary")

    if 'media_channels' not in self.model_config:
      raise ValueError("model_config must contain 'media_channels' key")

  def validate_parameters(self) -> None:
    """Validate the parameters DataFrame structure and content."""
    # Check required columns
    required_columns = ['MediaVariable', 'Adstock', 'Inflexion', 'Slope']
    missing_columns = [col for col in required_columns if col not in self.parameters_df.columns]
    if missing_columns:
      raise ValueError(f"Missing required columns in Parameters sheet: {missing_columns}")

    # Check for duplicate MediaVariables
    if self.parameters_df['MediaVariable'].duplicated().any():
      duplicates = self.parameters_df[self.parameters_df['MediaVariable'].duplicated()]['MediaVariable'].tolist()
      raise ValueError(f"Duplicate MediaVariable entries found: {duplicates}")

    # Get MediaVariable values from DataFrame
    parameter_channels = set(self.parameters_df['MediaVariable'].tolist())
    expected_channels = set(self.all_channels)

    # Check that all expected channels are present
    missing_channels = expected_channels - parameter_channels
    if missing_channels:
      raise ValueError(f"Missing MediaVariable entries for channels: {list(missing_channels)}")

    # Check for unexpected channels
    extra_channels = parameter_channels - expected_channels
    if extra_channels:
      raise ValueError(f"Unexpected MediaVariable entries found: {list(extra_channels)}. "
                       f"Expected channels: {self.all_channels}")

    # Validate parameter value ranges
    self._validate_parameter_values()

    logging.info("Parameters validation passed - all requirements satisfied")

  def _validate_parameter_values(self) -> None:
    """Validate that parameter values are within expected ranges."""
    # Check for non-numeric values
    numeric_columns = ['Adstock', 'Inflexion', 'Slope']
    for col in numeric_columns:
      if not pd.api.types.is_numeric_dtype(self.parameters_df[col]):
        raise ValueError(f"Column '{col}' must contain numeric values")

      # Check for NaN values
      if self.parameters_df[col].isna().any():
        raise ValueError(f"Column '{col}' contains missing values")

    # Validate Adstock values (should be between 0 and 1)
    adstock_values = self.parameters_df['Adstock']
    if (adstock_values < 0).any() or (adstock_values > 1).any():
      invalid_rows = self.parameters_df[(adstock_values < 0) | (adstock_values > 1)]
      raise ValueError(f"Adstock values must be between 0 and 1. Invalid values found for: "
                       f"{invalid_rows['MediaVariable'].tolist()}")

    # Validate Inflexion values (should be positive)
    inflexion_values = self.parameters_df['Inflexion']
    if (inflexion_values <= 0).any():
      invalid_rows = self.parameters_df[inflexion_values <= 0]
      raise ValueError(f"Inflexion values must be positive. Invalid values found for: "
                       f"{invalid_rows['MediaVariable'].tolist()}")

    # Validate Slope values (should be positive)
    slope_values = self.parameters_df['Slope']
    if (slope_values <= 0).any():
      invalid_rows = self.parameters_df[slope_values <= 0]
      raise ValueError(f"Slope values must be positive. Invalid values found for: "
                       f"{invalid_rows['MediaVariable'].tolist()}")

  def reorder_parameters(self) -> pd.DataFrame:
    """Reorder parameters DataFrame to match channel order in model_config.

    Returns:
      DataFrame with parameters reordered to match media_channels + rf_channels order.
    """
    # Create a mapping for desired order
    channel_order = {channel: idx for idx, channel in enumerate(self.all_channels)}

    # Add order column temporarily
    self.parameters_df['_order'] = self.parameters_df['MediaVariable'].map(channel_order)

    # Sort by order
    reordered_df = self.parameters_df.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)

    # Verify the order matches expected
    expected_order = reordered_df['MediaVariable'].tolist()
    if expected_order != self.all_channels:
      raise ValueError(f"Failed to reorder parameters. Expected: {self.all_channels}, "
                       f"Got: {expected_order}")

    logging.info(f"Parameters reordered to match channel sequence: {self.all_channels}")
    return reordered_df

  def process_media_parameters(self, reordered_df: pd.DataFrame) -> Dict[str, List[float]]:
    """Extract parameters for media channels.

    Args:
      reordered_df: DataFrame with parameters in correct order.

    Returns:
      Dictionary with media channel parameters using Meridian constants.
    """
    media_params = {
      constants.ALPHA_M: [],
      constants.EC_M: [],
      constants.SLOPE_M: []
    }

    for channel in self.media_channels:
      channel_row = reordered_df[reordered_df['MediaVariable'] == channel]
      if len(channel_row) != 1:
        raise ValueError(f"Expected exactly one row for media channel {channel}, "
                         f"found {len(channel_row)}")

      row = channel_row.iloc[0]
      media_params[constants.ALPHA_M].append(float(row['Adstock']))
      media_params[constants.EC_M].append(float(row['Inflexion']))
      media_params[constants.SLOPE_M].append(float(row['Slope']))

    logging.info(f"Processed {len(self.media_channels)} media channel parameters")
    return media_params

  def process_rf_parameters(self, reordered_df: pd.DataFrame) -> Dict[str, List[float]]:
    """Extract parameters for R&F channels.

    Args:
      reordered_df: DataFrame with parameters in correct order.

    Returns:
      Dictionary with R&F channel parameters using Meridian constants.
    """
    rf_params = {
      constants.ALPHA_RF: [],
      constants.EC_RF: [],
      constants.SLOPE_RF: []
    }

    for channel in self.rf_channels:
      channel_row = reordered_df[reordered_df['MediaVariable'] == channel]
      if len(channel_row) != 1:
        raise ValueError(f"Expected exactly one row for R&F channel {channel}, "
                         f"found {len(channel_row)}")

      row = channel_row.iloc[0]
      rf_params[constants.ALPHA_RF].append(float(row['Adstock']))
      rf_params[constants.EC_RF].append(float(row['Inflexion']))
      rf_params[constants.SLOPE_RF].append(float(row['Slope']))

    logging.info(f"Processed {len(self.rf_channels)} R&F channel parameters")
    return rf_params

  def get_parameter_dict(self) -> Dict[str, List[float]]:
    """Process parameters and return organized parameter dictionary.

    Returns:
      Dictionary with parameter lists organized by Meridian constants:
      - ALPHA_M, EC_M, SLOPE_M: Lists for media channels
      - ALPHA_RF, EC_RF, SLOPE_RF: Lists for R&F channels (if any)
    """
    if self.processed_parameters is not None:
      return self.processed_parameters

    # Validate parameters
    self.validate_parameters()

    # Reorder parameters to match channel sequence
    reordered_df = self.reorder_parameters()

    # Process parameters by channel type
    result = {}

    # Process media channel parameters
    if self.media_channels:
      media_params = self.process_media_parameters(reordered_df)
      result.update(media_params)

    # Process R&F channel parameters
    if self.rf_channels:
      rf_params = self.process_rf_parameters(reordered_df)
      result.update(rf_params)

    # Cache the result
    self.processed_parameters = result

    logging.info(f"Successfully processed parameters for {len(self.all_channels)} channels")
    return result

  def get_channel_parameter_summary(self) -> pd.DataFrame:
    """Get a summary of parameters by channel and type.

    Returns:
      DataFrame with channel names, types, and parameter values for easy inspection.
    """
    if self.processed_parameters is None:
      self.get_parameter_dict()

    summary_data = []

    # Add media channels
    for i, channel in enumerate(self.media_channels):
      summary_data.append({
        'Channel': channel,
        'Type': 'Media',
        'Adstock': self.processed_parameters[constants.ALPHA_M][i],
        'Inflexion': self.processed_parameters[constants.EC_M][i],
        'Slope': self.processed_parameters[constants.SLOPE_M][i]
      })

    # Add R&F channels
    for i, channel in enumerate(self.rf_channels):
      summary_data.append({
        'Channel': channel,
        'Type': 'R&F',
        'Adstock': self.processed_parameters[constants.ALPHA_RF][i],
        'Inflexion': self.processed_parameters[constants.EC_RF][i],
        'Slope': self.processed_parameters[constants.SLOPE_RF][i]
      })

    return pd.DataFrame(summary_data)
  
  def get_parameter_data_arrays(self) -> Dict[str, xr.DataArray]:
    """Process parameters and return organized xarray.DataArray objects.
    
    Returns:
      Dictionary with parameter DataArrays organized by Meridian constants:
      - ALPHA_M, EC_M, SLOPE_M: DataArrays with media_channel coordinates
      - ALPHA_RF, EC_RF, SLOPE_RF: DataArrays with rf_channel coordinates (if any)
    """
    # Get processed parameter lists first
    param_dict = self.get_parameter_dict()
    
    result = {}
    
    # Create DataArrays for media channel parameters
    if self.media_channels:
      result[constants.ALPHA_M] = xr.DataArray(
        data=param_dict[constants.ALPHA_M],
        dims=['media_channel'],
        coords={'media_channel': self.media_channels},
        name=constants.ALPHA_M
      )
      
      result[constants.EC_M] = xr.DataArray(
        data=param_dict[constants.EC_M],
        dims=['media_channel'],
        coords={'media_channel': self.media_channels},
        name=constants.EC_M
      )
      
      result[constants.SLOPE_M] = xr.DataArray(
        data=param_dict[constants.SLOPE_M],
        dims=['media_channel'],
        coords={'media_channel': self.media_channels},
        name=constants.SLOPE_M
      )
      
    # Create DataArrays for R&F channel parameters  
    if self.rf_channels:
      result[constants.ALPHA_RF] = xr.DataArray(
        data=param_dict[constants.ALPHA_RF],
        dims=['rf_channel'],
        coords={'rf_channel': self.rf_channels},
        name=constants.ALPHA_RF
      )
      
      result[constants.EC_RF] = xr.DataArray(
        data=param_dict[constants.EC_RF],
        dims=['rf_channel'],
        coords={'rf_channel': self.rf_channels},
        name=constants.EC_RF
      )
      
      result[constants.SLOPE_RF] = xr.DataArray(
        data=param_dict[constants.SLOPE_RF],
        dims=['rf_channel'],
        coords={'rf_channel': self.rf_channels},
        name=constants.SLOPE_RF
      )
    
    logging.info(f"Successfully created DataArrays for {len(result)} parameter types")
    return result

  def validate_coefficients(self) -> None:
    """Validate the coefficients DataFrame structure and content."""
    if self.coefficients_df is None:
      raise ValueError("coefficients_df cannot be None for coefficients validation")

    # Check required columns
    required_columns = ['geo'] + self.all_channels
    missing_columns = [col for col in required_columns if col not in self.coefficients_df.columns]
    if missing_columns:
      raise ValueError(f"Missing required columns in Coefficients sheet: {missing_columns}")

    # Check for unexpected columns
    expected_columns = set(required_columns)
    actual_columns = set(self.coefficients_df.columns)
    extra_columns = actual_columns - expected_columns
    if extra_columns:
      raise ValueError(f"Unexpected columns in Coefficients sheet: {list(extra_columns)}. "
                       f"Expected columns: {required_columns}")

    # Validate geo values if data_df is available and strict validation is enabled
    if self.data_df is not None and not self.auto_filter_geos:
      geo_col = self.model_config.get('geo_col', 'geo')
      if geo_col in self.data_df.columns:
        data_geos = set(self.data_df[geo_col].unique())
        coeff_geos = set(self.coefficients_df['geo'].unique())
        
        missing_geos = data_geos - coeff_geos
        if missing_geos:
          raise ValueError(f"Missing geo values in Coefficients sheet: {list(missing_geos)}")
          
        extra_geos = coeff_geos - data_geos
        if extra_geos:
          raise ValueError(f"Unexpected geo values in Coefficients sheet: {list(extra_geos)}. "
                           f"Should match geo values from Data sheet")
    elif self.auto_filter_geos:
      logging.info("Geo validation skipped - auto_filter_geos is enabled, assuming data already filtered")

    # Check for duplicate geo entries
    if self.coefficients_df['geo'].duplicated().any():
      duplicates = self.coefficients_df[self.coefficients_df['geo'].duplicated()]['geo'].tolist()
      raise ValueError(f"Duplicate geo entries found in Coefficients sheet: {duplicates}")

    # Validate coefficient values (numeric and non-negative)
    for channel in self.all_channels:
      channel_values = self.coefficients_df[channel]
      
      # Check for non-numeric values
      if not pd.api.types.is_numeric_dtype(channel_values):
        raise ValueError(f"Column '{channel}' in Coefficients sheet must contain numeric values")
      
      # Check for NaN values
      if channel_values.isna().any():
        invalid_geos = self.coefficients_df[channel_values.isna()]['geo'].tolist()
        raise ValueError(f"Column '{channel}' contains missing values for geos: {invalid_geos}")
      
      # Check for negative values
      if (channel_values < 0).any():
        invalid_geos = self.coefficients_df[channel_values < 0]['geo'].tolist()
        raise ValueError(f"Column '{channel}' contains negative values for geos: {invalid_geos}. "
                         f"MMM coefficients must be non-negative")

    logging.info("Coefficients validation passed - all requirements satisfied")

  def get_coefficients_data_arrays(self) -> Dict[str, xr.DataArray]:
    """Process coefficients and return organized xarray.DataArray objects.
    
    Returns:
      Dictionary with coefficients DataArrays organized by Meridian constants:
      - BETA_GM: DataArray with (geo, media_channel) coordinates for media channels
      - BETA_GRF: DataArray with (geo, rf_channel) coordinates for R&F channels (if any)
    """
    if self.coefficients_df is None:
      raise ValueError("coefficients_df is None. Cannot process coefficients.")
      
    if self.processed_coefficients is not None:
      return self.processed_coefficients

    # Validate coefficients
    self.validate_coefficients()

    # Sort coefficients by geo to ensure consistent ordering
    sorted_coeffs_df = self.coefficients_df.sort_values('geo').reset_index(drop=True)
    geo_list = sorted_coeffs_df['geo'].tolist()
    
    result = {}

    # Create DataArray for media channel coefficients
    if self.media_channels:
      media_coeff_data = []
      for channel in self.media_channels:
        channel_values = sorted_coeffs_df[channel].values
        media_coeff_data.append(channel_values)
      
      # Transpose to get (geo, media_channel) shape
      media_coeff_array = np.array(media_coeff_data).T
      
      result[constants.BETA_GM] = xr.DataArray(
        data=media_coeff_array,
        dims=['geo', 'media_channel'],
        coords={
          'geo': geo_list,
          'media_channel': self.media_channels
        },
        name=constants.BETA_GM
      )

    # Create DataArray for R&F channel coefficients
    if self.rf_channels:
      rf_coeff_data = []
      for channel in self.rf_channels:
        channel_values = sorted_coeffs_df[channel].values
        rf_coeff_data.append(channel_values)
      
      # Transpose to get (geo, rf_channel) shape
      rf_coeff_array = np.array(rf_coeff_data).T
      
      result[constants.BETA_GRF] = xr.DataArray(
        data=rf_coeff_array,
        dims=['geo', 'rf_channel'],
        coords={
          'geo': geo_list,
          'rf_channel': self.rf_channels
        },
        name=constants.BETA_GRF
      )

    # Cache the result
    self.processed_coefficients = result

    logging.info(f"Successfully created coefficients DataArrays for {len(result)} coefficient types")
    return result
