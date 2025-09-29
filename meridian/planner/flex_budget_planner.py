"""FlexibleBudgetPlanner for creating Meridian InputData from Excel files and budget optimization."""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import arviz as az
import numpy as np

from meridian.data import data_frame_input_data_builder
from meridian.data import input_data
from meridian.model import model
from meridian.model import spec
from meridian.analysis import optimizer
from meridian.planner import media_parameter_loader
from meridian.planner import point_inference_data
from meridian.planner import roi_to_coefficients_converter
from meridian.analysis.optimizer import OptimizationResults

__all__ = [
    'FlexibleBudgetPlanner',
]


class FlexibleBudgetPlanner:
  """Loads Excel file data, creates Meridian InputData objects, and runs budget optimization.

  This class handles Excel files with the expected structure:
  - 'Data' sheet: Contains MMM input data (time series data)
  - 'Coefficients' or 'ROI' sheet: Contains geo-level coefficients or ROI values
  - 'Parameters' sheet: Contains media parameters (adstock, hill parameters)

  ROI sheets are automatically detected and converted to equivalent coefficients
  using Meridian's media transformation functions. Exactly one of 'Coefficients'
  or 'ROI' sheet must be present.

  The class uses DataFrameInputDataBuilder to create properly structured
  Meridian InputData objects and provides budget optimization functionality.

  Geo Filtering:
  By default (auto_filter_geos=True), the system automatically filters the Data
  sheet to only include geos that have corresponding entries in the Coefficients
  or ROI sheet. This allows working with partial geo coverage. Set
  auto_filter_geos=False to enforce strict geo matching validation instead.
  """

  def __init__(self, file_name: str, model_config: Dict[str, Any], auto_filter_geos: bool = True):
    """Initialize the FlexibleBudgetPlanner.

    Args:
      file_name: Path to the Excel file containing MMM data.
      model_config: Dictionary containing model configuration with keys:
        - time_col: Name of time column
        - geo_col: Name of geo column
        - population_col: Name of population column
        - kpi_type: Type of KPI ('revenue' or 'non_revenue')
        - kpi_col: Name of KPI column
        - revenue_per_kpi_col: Name of revenue per KPI column
        - media_cols: List of media impression columns
        - media_spend_cols: List of media spend columns
        - media_channels: List of media channel names
        - reach_cols: List of reach columns (optional)
        - frequency_cols: List of frequency columns (optional)
        - rf_spend_cols: List of R&F spend columns (optional)
        - rf_channels: List of R&F channel names (optional)
        - control_cols: List of control variable columns (optional)
        - is_roi_input: Boolean indicating ROI input (auto-detected, do not set manually)
      auto_filter_geos: Whether to automatically filter Data sheet to geos present
        in Coefficients/ROI sheet. If True (default), Data will be filtered to
        intersection of geos. If False, strict geo matching validation is enforced.
    """
    self.file_name = file_name
    self.model_config = model_config
    self.auto_filter_geos = auto_filter_geos

    # Data containers
    self.data_df: Optional[pd.DataFrame] = None
    self.coefficients_df: Optional[pd.DataFrame] = None
    self.parameters_df: Optional[pd.DataFrame] = None
    self.roi_df: Optional[pd.DataFrame] = None
    self.input_type: Optional[str] = None

    # Optimization intermediate outputs
    self.input_data: Optional[input_data.InputData] = None
    self.inference_data: Optional[Any] = None  # az.InferenceData
    self.model_obj: Optional[Any] = None  # model.Meridian

    # Validate required config keys
    self._validate_config()
    self._validate_optimization_config()

  def _validate_config(self) -> None:
    """Validate that model_config contains required keys."""
    # Core required keys (always needed)
    required_keys = [
      'time_col', 'geo_col', 'population_col', 'kpi_type', 'kpi_col'
    ]

    missing_keys = [key for key in required_keys if key not in self.model_config]
    if missing_keys:
      raise ValueError(f"Missing required config keys: {missing_keys}")

    # Set default empty lists for media keys if not provided (enables RF-only configurations)
    if 'media_cols' not in self.model_config:
      self.model_config['media_cols'] = []
    if 'media_spend_cols' not in self.model_config:
      self.model_config['media_spend_cols'] = []
    if 'media_channels' not in self.model_config:
      self.model_config['media_channels'] = []

    # Validate media list lengths match (only if media channels exist)
    if self.model_config.get('media_channels'):
      if len(self.model_config['media_cols']) != len(self.model_config['media_channels']):
        raise ValueError("media_cols and media_channels must have same length")

      if len(self.model_config['media_spend_cols']) != len(self.model_config['media_channels']):
        raise ValueError("media_spend_cols and media_channels must have same length")

    # Validate R&F channels if provided
    if any(key in self.model_config for key in ['reach_cols', 'frequency_cols', 'rf_spend_cols', 'rf_channels']):
      rf_keys = ['reach_cols', 'frequency_cols', 'rf_spend_cols', 'rf_channels']
      missing_rf_keys = [key for key in rf_keys if key not in self.model_config]
      if missing_rf_keys:
        raise ValueError(f"If using R&F channels, all R&F keys must be provided. Missing: {missing_rf_keys}")

      if not all(len(self.model_config[key]) == len(self.model_config['rf_channels']) for key in rf_keys[:-1]):
        raise ValueError("All R&F column lists must have same length as rf_channels")

      # Validate that media_channels and rf_channels don't overlap (only if both exist)
      if self.model_config.get('media_channels') and self.model_config.get('rf_channels', []):
        media_channels = set(self.model_config['media_channels'])
        rf_channels = set(self.model_config['rf_channels'])
        overlapping_channels = media_channels.intersection(rf_channels)
        if overlapping_channels:
          raise ValueError(f"Channels cannot be both media and R&F channels. Overlapping channels: {list(overlapping_channels)}")

    # Ensure at least one channel type is provided
    has_media_channels = bool(self.model_config.get('media_channels'))
    has_rf_channels = bool(self.model_config.get('rf_channels', []))

    if not has_media_channels and not has_rf_channels:
      raise ValueError("At least one of media_channels or rf_channels must be provided")

  def _validate_optimization_config(self) -> None:
    """Validate model_config for optimization-specific requirements."""
    # Validate kpi_type
    if 'kpi_type' in self.model_config:
      kpi_type = self.model_config['kpi_type']
      if kpi_type not in ['revenue', 'non_revenue']:
        raise ValueError(f"kpi_type must be either 'revenue' or 'non_revenue', got: {kpi_type}")

    logging.info("Optimization config validation passed")

  def _detect_input_type(self) -> str:
    """Detect whether Excel file contains Coefficients or ROI sheet.

    Returns:
      String indicating input type: 'coefficients' or 'roi'.

    Raises:
      ValueError: If neither sheet exists, both exist, or Excel file cannot be read.
    """
    try:
      excel_file = pd.ExcelFile(self.file_name)
      available_sheets = excel_file.sheet_names

      has_coefficients = 'Coefficients' in available_sheets
      has_roi = 'ROI' in available_sheets

      if has_coefficients and has_roi:
        raise ValueError("Excel file cannot contain both 'Coefficients' and 'ROI' sheets. Please provide exactly one.")

      if has_coefficients:
        return 'coefficients'
      elif has_roi:
        return 'roi'
      else:
        raise ValueError("Excel file must contain either 'Coefficients' or 'ROI' sheet.")

    except Exception as e:
      if "cannot contain both" in str(e) or "must contain either" in str(e):
        raise  # Re-raise our validation errors
      raise ValueError(f"Error reading Excel file structure: {str(e)}")

  def _validate_sheet_requirements(self, input_type: str) -> None:
    """Validate that required sheets exist for the detected input type.

    Args:
      input_type: Type of input detected ('coefficients' or 'roi').

    Raises:
      ValueError: If required sheets are missing.
    """
    try:
      excel_file = pd.ExcelFile(self.file_name)
      available_sheets = excel_file.sheet_names

      # Data sheet is always required
      if 'Data' not in available_sheets:
        raise ValueError("Excel file must contain 'Data' sheet")

      # Parameters sheet is required for ROI conversion
      if input_type == 'roi' and 'Parameters' not in available_sheets:
        raise ValueError("ROI input requires 'Parameters' sheet for conversion")

      # Validate the appropriate coefficient/ROI sheet exists
      if input_type == 'coefficients' and 'Coefficients' not in available_sheets:
        raise ValueError("Coefficients input type detected but 'Coefficients' sheet not found")
      elif input_type == 'roi' and 'ROI' not in available_sheets:
        raise ValueError("ROI input type detected but 'ROI' sheet not found")

    except Exception as e:
      if "Excel file must contain" in str(e) or "input type detected" in str(e) or "ROI input requires" in str(e):
        raise  # Re-raise our validation errors
      raise ValueError(f"Error validating sheet requirements: {str(e)}")

  def _convert_roi_to_coefficients(self) -> pd.DataFrame:
    """Convert ROI data to equivalent coefficients using ROIToCoefficientsConverter.

    This method creates an InputData object first, then uses the converter to
    transform ROI values into coefficient values that can be used with the
    existing MediaParameterLoader workflow.

    Returns:
      DataFrame with coefficient values in same format as original ROI data.

    Raises:
      ValueError: If ROI conversion fails.
    """
    try:
      # First need to create InputData to pass to the converter
      # This uses the same logic as build_input_data() but simplified
      builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
          kpi_type=self.model_config['kpi_type'],
          default_geo_column=self.model_config['geo_col'],
          default_time_column=self.model_config['time_col'],
          default_population_column=self.model_config['population_col'],
          default_kpi_column=self.model_config['kpi_col']
      )

      # Add basic data required for ROI conversion
      builder = builder.with_kpi(
          self.data_df,
          kpi_col=self.model_config['kpi_col'],
          time_col=self.model_config['time_col'],
          geo_col=self.model_config['geo_col']
      )

      # Add population data
      builder = builder.with_population(
          self.data_df,
          population_col=self.model_config['population_col'],
          geo_col=self.model_config['geo_col']
      )

      # Add revenue per KPI if available
      if 'revenue_per_kpi_col' in self.model_config:
        builder = builder.with_revenue_per_kpi(
            self.data_df,
            revenue_per_kpi_col=self.model_config['revenue_per_kpi_col'],
            time_col=self.model_config['time_col'],
            geo_col=self.model_config['geo_col']
        )

      # Add media data
      builder = builder.with_media(
          self.data_df,
          media_cols=self.model_config['media_cols'],
          media_spend_cols=self.model_config['media_spend_cols'],
          media_channels=self.model_config['media_channels'],
          time_col=self.model_config['time_col'],
          geo_col=self.model_config['geo_col']
      )

      # Add RF data if present
      if 'rf_channels' in self.model_config and self.model_config['rf_channels']:
        builder = builder.with_reach(
            self.data_df,
            reach_cols=self.model_config['reach_cols'],
            frequency_cols=self.model_config['frequency_cols'],
            rf_spend_cols=self.model_config['rf_spend_cols'],
            rf_channels=self.model_config['rf_channels'],
            time_col=self.model_config['time_col'],
            geo_col=self.model_config['geo_col']
        )

      input_data_obj = builder.build()

      # Create converter and perform conversion
      converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=self.roi_df,
          parameters_df=self.parameters_df,
          input_data_obj=input_data_obj,
          model_config=self.model_config
      )

      converted_coefficients = converter.convert_roi_to_coefficients()
      logging.info(f"Successfully converted ROI to coefficients for {len(converted_coefficients)} geos")

      return converted_coefficients

    except Exception as e:
      raise ValueError(f"Error converting ROI to coefficients: {str(e)}")

  def _filter_data_by_available_geos(self) -> None:
    """Filter Data sheet to only include geos present in Coefficients/ROI sheet.

    This method filters self.data_df to only include geos that have corresponding
    entries in the coefficients_df or roi_df. This allows the system to work
    with partial geo coverage in the coefficients/ROI sheets.

    The filtering is performed in-place on self.data_df.
    """
    if not self.auto_filter_geos:
      return  # Skip filtering if disabled

    if self.data_df is None:
      raise ValueError("Data DataFrame not loaded. Cannot perform geo filtering.")

    # Determine which coefficient/ROI DataFrame to use for filtering
    filter_df = None
    filter_sheet_name = ""

    if self.input_type == 'coefficients' and self.coefficients_df is not None:
      filter_df = self.coefficients_df
      filter_sheet_name = "Coefficients"
    elif self.input_type == 'roi' and self.roi_df is not None:
      filter_df = self.roi_df
      filter_sheet_name = "ROI"
    else:
      logging.warning("No Coefficients or ROI data available for geo filtering. Skipping filtering.")
      return

    # Get geo column name
    geo_col = self.model_config.get('geo_col', 'geo')

    if geo_col not in self.data_df.columns:
      raise ValueError(f"Geo column '{geo_col}' not found in Data sheet")

    if 'geo' not in filter_df.columns:
      raise ValueError(f"'geo' column not found in {filter_sheet_name} sheet")

    # Get available geos from both sheets
    data_geos = set(self.data_df[geo_col].unique())
    available_geos = set(filter_df['geo'].unique())

    # Calculate intersection and differences
    common_geos = data_geos.intersection(available_geos)
    missing_from_coeffs = data_geos - available_geos
    extra_in_coeffs = available_geos - data_geos

    # Log filtering information
    original_geo_count = len(data_geos)
    filtered_geo_count = len(common_geos)

    logging.info(f"Geo filtering summary:")
    logging.info(f"  - Original geos in Data sheet: {original_geo_count}")
    logging.info(f"  - Available geos in {filter_sheet_name} sheet: {len(available_geos)}")
    logging.info(f"  - Common geos (intersection): {filtered_geo_count}")

    if missing_from_coeffs:
      logging.info(f"  - Geos in Data but not in {filter_sheet_name}: {sorted(list(missing_from_coeffs))}")

    if extra_in_coeffs:
      logging.info(f"  - Geos in {filter_sheet_name} but not in Data: {sorted(list(extra_in_coeffs))}")

    if not common_geos:
      raise ValueError(f"No common geos found between Data sheet and {filter_sheet_name} sheet. "
                       f"Cannot proceed with empty geo intersection.")

    # Filter data_df to only include common geos
    original_rows = len(self.data_df)
    self.data_df = self.data_df[self.data_df[geo_col].isin(common_geos)].copy()
    filtered_rows = len(self.data_df)

    logging.info(f"  - Data rows before filtering: {original_rows}")
    logging.info(f"  - Data rows after filtering: {filtered_rows}")
    logging.info(f"Geo filtering completed. Data sheet filtered to {filtered_geo_count} geos.")

  def load_excel_data(self) -> None:
    """Load data from all sheets in the Excel file with ROI/Coefficients detection."""
    try:
      # Step 1: Detect input type (ROI vs Coefficients)
      self.input_type = self._detect_input_type()
      logging.info(f"Detected input type: {self.input_type}")

      # Step 2: Validate sheet requirements for detected type
      self._validate_sheet_requirements(self.input_type)

      # Step 3: Set is_roi_input flag in model config
      self.model_config['is_roi_input'] = (self.input_type == 'roi')

      # Step 4: Load Data sheet (always required)
      self.data_df = pd.read_excel(self.file_name, sheet_name='Data')
      logging.info(f"Loaded Data sheet with shape: {self.data_df.shape}")

      # Step 5: Load Parameters sheet (required for ROI, optional for Coefficients)
      try:
        self.parameters_df = pd.read_excel(self.file_name, sheet_name='Parameters')
        logging.info(f"Loaded Parameters sheet with shape: {self.parameters_df.shape}")
      except ValueError:
        if self.input_type == 'roi':
          raise ValueError("Parameters sheet is required for ROI input but not found")
        logging.warning("Parameters sheet not found - skipping")

      # Step 6: Load coefficient/ROI sheet based on detected type
      if self.input_type == 'coefficients':
        self.coefficients_df = pd.read_excel(self.file_name, sheet_name='Coefficients')
        logging.info(f"Loaded Coefficients sheet with shape: {self.coefficients_df.shape}")
        self.roi_df = None

      elif self.input_type == 'roi':
        self.roi_df = pd.read_excel(self.file_name, sheet_name='ROI')
        logging.info(f"Loaded ROI sheet with shape: {self.roi_df.shape}")

        # Convert ROI to coefficients using the converter
        logging.info("Converting ROI data to equivalent coefficients...")
        self.coefficients_df = self._convert_roi_to_coefficients()
        logging.info(f"ROI conversion completed. Generated coefficients with shape: {self.coefficients_df.shape}")

      # Step 7: Apply geo filtering if enabled
      self._filter_data_by_available_geos()

    except Exception as e:
      raise ValueError(f"Error loading Excel file {self.file_name}: {str(e)}")

  def validate_data_columns(self) -> None:
    """Validate that the data sheet contains all required columns."""
    if self.data_df is None:
      raise ValueError("Data not loaded. Call load_excel_data() first.")

    required_cols = [
      self.model_config['time_col'],
      self.model_config['geo_col'],
      self.model_config['population_col'],
      self.model_config['kpi_col'],
    ]

    # Add media columns (only if media channels are configured)
    if self.model_config.get('media_channels'):
      required_cols.extend(self.model_config['media_cols'])
      required_cols.extend(self.model_config['media_spend_cols'])

    # Add revenue per KPI if specified
    if 'revenue_per_kpi_col' in self.model_config and self.model_config['revenue_per_kpi_col'] is not None:
      required_cols.append(self.model_config['revenue_per_kpi_col'])

    # Add R&F columns if specified
    if 'reach_cols' in self.model_config:
      required_cols.extend(self.model_config['reach_cols'])
      required_cols.extend(self.model_config['frequency_cols'])
      required_cols.extend(self.model_config['rf_spend_cols'])

    # Add control columns if specified
    if 'control_cols' in self.model_config:
      required_cols.extend(self.model_config['control_cols'])

    missing_cols = [col for col in required_cols if col not in self.data_df.columns]
    if missing_cols:
      raise ValueError(f"Missing required columns in Data sheet: {missing_cols}")

    logging.info("Data validation passed - all required columns present")

  def build_input_data(self) -> input_data.InputData:
    """Build InputData object using DataFrameInputDataBuilder.

    Returns:
      InputData object ready for Meridian model fitting.
    """
    if self.data_df is None:
      self.load_excel_data()

    self.validate_data_columns()

    # Initialize builder with KPI type
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
      kpi_type=self.model_config['kpi_type'],
      default_geo_column=self.model_config['geo_col'],
      default_time_column=self.model_config['time_col'],
      default_population_column=self.model_config['population_col'],
      default_kpi_column=self.model_config['kpi_col']
    )

    # Add KPI data
    builder = builder.with_kpi(
      self.data_df,
      kpi_col=self.model_config['kpi_col'],
      time_col=self.model_config['time_col'],
      geo_col=self.model_config['geo_col']
    )

    # Add revenue per KPI if specified
    if 'revenue_per_kpi_col' in self.model_config and self.model_config['revenue_per_kpi_col'] is not None:
      builder = builder.with_revenue_per_kpi(
        self.data_df,
        revenue_per_kpi_col=self.model_config['revenue_per_kpi_col'],
        time_col=self.model_config['time_col'],
        geo_col=self.model_config['geo_col']
      )

    # Add population
    builder = builder.with_population(
      self.data_df,
      population_col=self.model_config['population_col'],
      geo_col=self.model_config['geo_col']
    )

    # Add control variables if specified
    if 'control_cols' in self.model_config and self.model_config['control_cols']:
      builder = builder.with_controls(
        self.data_df,
        control_cols=self.model_config['control_cols'],
        time_col=self.model_config['time_col'],
        geo_col=self.model_config['geo_col']
      )

    # Add media channels (only if media channels are configured)
    if self.model_config.get('media_channels'):
      builder = builder.with_media(
        self.data_df,
        media_cols=self.model_config['media_cols'],
        media_spend_cols=self.model_config['media_spend_cols'],
        media_channels=self.model_config['media_channels'],
        time_col=self.model_config['time_col'],
        geo_col=self.model_config['geo_col']
      )

    # Add R&F channels if specified
    if 'reach_cols' in self.model_config:
      builder = builder.with_reach(
        self.data_df,
        reach_cols=self.model_config['reach_cols'],
        frequency_cols=self.model_config['frequency_cols'],
        rf_spend_cols=self.model_config['rf_spend_cols'],
        rf_channels=self.model_config['rf_channels'],
        time_col=self.model_config['time_col'],
        geo_col=self.model_config['geo_col']
      )

    # Build and return the InputData object
    input_data_obj = builder.build()
    logging.info("Successfully created InputData object")

    return input_data_obj

  def get_coefficients_data(self) -> Optional[pd.DataFrame]:
    """Get coefficients data if available.

    Returns:
      DataFrame containing coefficients data or None if not available.
    """
    return self.coefficients_df

  def get_parameters_data(self) -> Optional[pd.DataFrame]:
    """Get parameters data if available.

    Returns:
      DataFrame containing parameters data or None if not available.
    """
    return self.parameters_df

  def get_processed_parameters(self) -> Optional[Dict[str, Any]]:
    """Get processed parameters organized by Meridian constants.

    Returns:
      Dictionary with parameter lists organized by constants:
      - ALPHA_M, EC_M, SLOPE_M: Lists for media channels
      - ALPHA_RF, EC_RF, SLOPE_RF: Lists for R&F channels (if any)
      Returns None if no parameters sheet is available.
    """
    if self.parameters_df is None:
      logging.warning("No parameters data available")
      return None

    try:
      param_loader = media_parameter_loader.MediaParameterLoader(
        self.parameters_df,
        self.model_config,
        coefficients_df=self.coefficients_df,
        data_df=self.data_df,
        auto_filter_geos=self.auto_filter_geos
      )
      processed_params = param_loader.get_parameter_dict()
      logging.info("Successfully processed media parameters")
      return processed_params

    except Exception as e:
      raise ValueError(f"Error processing parameters: {str(e)}")

  def get_parameter_summary(self) -> Optional[pd.DataFrame]:
    """Get a summary of parameters by channel and type.

    Returns:
      DataFrame with channel names, types, and parameter values for easy inspection.
      Returns None if no parameters sheet is available.
    """
    if self.parameters_df is None:
      return None

    try:
      param_loader = media_parameter_loader.MediaParameterLoader(
        self.parameters_df,
        self.model_config,
        coefficients_df=self.coefficients_df,
        data_df=self.data_df,
        auto_filter_geos=self.auto_filter_geos
      )
      return param_loader.get_channel_parameter_summary()

    except Exception as e:
      raise ValueError(f"Error creating parameter summary: {str(e)}")

  def get_processed_parameter_arrays(self) -> Optional[Dict[str, Any]]:
    """Get processed parameters as xarray.DataArray objects organized by Meridian constants.

    Returns:
      Dictionary with parameter DataArrays organized by constants:
      - ALPHA_M, EC_M, SLOPE_M: DataArrays with media_channel coordinates
      - ALPHA_RF, EC_RF, SLOPE_RF: DataArrays with rf_channel coordinates (if any)
      Returns None if no parameters sheet is available.
    """
    if self.parameters_df is None:
      logging.warning("No parameters data available")
      return None

    try:
      param_loader = media_parameter_loader.MediaParameterLoader(
        self.parameters_df,
        self.model_config,
        coefficients_df=self.coefficients_df,
        data_df=self.data_df,
        auto_filter_geos=self.auto_filter_geos
      )
      processed_arrays = param_loader.get_parameter_data_arrays()
      logging.info("Successfully processed media parameters as DataArrays")
      return processed_arrays

    except Exception as e:
      raise ValueError(f"Error processing parameter arrays: {str(e)}")

  def get_processed_coefficients_arrays(self) -> Optional[Dict[str, Any]]:
    """Get processed coefficients as xarray.DataArray objects organized by Meridian constants.

    Returns:
      Dictionary with coefficients DataArrays organized by constants:
      - BETA_GM: DataArray with (geo, media_channel) coordinates for media channels
      - BETA_GRF: DataArray with (geo, rf_channel) coordinates for R&F channels (if any)
      Returns None if no coefficients sheet is available.
    """
    if self.coefficients_df is None:
      logging.warning("No coefficients data available")
      return None

    try:
      param_loader = media_parameter_loader.MediaParameterLoader(
        self.parameters_df,
        self.model_config,
        coefficients_df=self.coefficients_df,
        data_df=self.data_df,
        auto_filter_geos=self.auto_filter_geos
      )
      processed_coeffs = param_loader.get_coefficients_data_arrays()
      logging.info("Successfully processed media coefficients as DataArrays")
      return processed_coeffs

    except Exception as e:
      raise ValueError(f"Error processing coefficients arrays: {str(e)}")

  def get_inference_data(self) -> Optional[az.InferenceData]:
    """Create ArviZ InferenceData from parameters and coefficients.

    Combines parameter and coefficient arrays into a unified ArviZ InferenceData
    object that is compatible with Meridian analysis functions expecting
    Bayesian inference results.

    Returns:
      ArviZ InferenceData object with posterior and sample_stats groups,
      or None if parameters or coefficients are not available.
    """
    # Check if both parameters and coefficients sheets are available
    if self.parameters_df is None or self.coefficients_df is None:
      logging.warning("Missing parameters or coefficients sheets for InferenceData creation")
      return None

    parameter_arrays = self.get_processed_parameter_arrays()
    coefficient_arrays = self.get_processed_coefficients_arrays()

    if parameter_arrays is None or coefficient_arrays is None:
      logging.warning("Missing processed parameter or coefficient arrays for InferenceData creation")
      return None

    try:
      # Build input_data if not already built
      if self.data_df is None:
        self.load_excel_data()

      if self.input_data is None:
        self.input_data = self.build_input_data()

      # Create PointInferenceData with input_data for complete structure
      point_data = point_inference_data.PointInferenceData(
        parameter_arrays, coefficient_arrays,
        input_data_obj=self.input_data
      )
      inference_data = point_data.get_inference_data()
      logging.info("Successfully created ArviZ InferenceData from Excel data")
      return inference_data

    except Exception as e:
      raise ValueError(f"Error creating InferenceData: {str(e)}")

  def _format_optimizer_kwargs(self, optimizer_kwargs: dict) -> dict:
    """Format optimizer kwargs."""
    keys_to_build = ['spend_constraint_lower', 'spend_constraint_upper', 'pct_of_spend']
    paid_channels_argument_builder = self.input_data.get_paid_channels_argument_builder()

    for key in keys_to_build:
      if key in optimizer_kwargs and isinstance(optimizer_kwargs[key], dict):
        optimizer_kwargs[key] = paid_channels_argument_builder(**optimizer_kwargs[key])

    return optimizer_kwargs


  def optimize(self, optimizer_kwargs: dict | None = None) -> Any:
    """Run budget optimization using Excel data with Meridian model.

    Creates a Meridian model using the Excel data and runs budget optimization.
    Handles use_kpi parameter based on model configuration validation.

    Args:
      optimizer_kwargs: Optional dictionary of keyword arguments to pass to
        BudgetOptimizer.optimize(). User-provided parameters take precedence
        over auto-detected ones. Key supported parameters include:
        - fixed_budget (bool): Whether to use fixed budget optimization
        - budget (float): Budget amount for optimization
        - start_date/end_date: Time range for optimization
        - pct_of_spend (Sequence[float]): Percentage allocation per channel
        - spend_constraint_lower/upper: Spend constraint bounds
        - target_roi/target_mroi (float): Target ROI/marginal ROI values
        - use_kpi (bool): Whether to optimize for KPI vs revenue
        - confidence_level (float): Confidence level for optimization
        - And other BudgetOptimizer.optimize() parameters

    Returns:
      BudgetOptimizer results containing optimized budget allocation.

    Raises:
      ValueError: If optimization fails or data is incomplete.
    """
    try:
      # Build input data and inference data
      logging.info("Building input data and inference data for optimization...")
      self.input_data = self.build_input_data()
      point_inference_data = self.get_inference_data()

      if point_inference_data is None:
        raise ValueError("Cannot create inference data - missing parameters or coefficients sheets")

      # Store intermediate outputs
      self.inference_data = point_inference_data

      # Create Meridian model
      logging.info("Creating Meridian model...")
      model_spec = spec.ModelSpec(knots=1)
      model_obj = model.Meridian(
          input_data=self.input_data,
          model_spec=model_spec,
          inference_data=point_inference_data
      )

      # Store model object
      self.model_obj = model_obj

      # Sample prior (required for optimization)
      logging.info("Sampling prior distributions...")
      model_obj.sample_prior(n_draws=100, seed=42)

      # Create optimizer
      logging.info("Creating budget optimizer...")
      budget_optimizer = optimizer.BudgetOptimizer(model_obj)

      # Check if we need to use use_kpi parameter (auto-detected)
      auto_kwargs = {}
      kpi_type = self.model_config.get('kpi_type')
      revenue_per_kpi_col = self.model_config.get('revenue_per_kpi_col')

      if (kpi_type == 'non_revenue' and
          ('revenue_per_kpi_col' not in self.model_config or revenue_per_kpi_col is None)):
        auto_kwargs['use_kpi'] = True
        logging.info("Using use_kpi=True due to non_revenue KPI type without revenue_per_kpi_col")

      # Merge user-provided kwargs with auto-detected ones (user kwargs take precedence)
      final_kwargs = auto_kwargs.copy()
      if optimizer_kwargs:
        final_kwargs.update(optimizer_kwargs)
        logging.info(f"Using user-provided optimizer kwargs: {optimizer_kwargs}")

      final_kwargs = self._format_optimizer_kwargs(final_kwargs)

      # Run optimization
      logging.info("Running budget optimization...")
      optimizer_results = budget_optimizer.optimize(**final_kwargs)

      logging.info("Budget optimization completed successfully")
      return optimizer_results

    except Exception as e:
      raise ValueError(f"Error during optimization: {str(e)}")



class CompareOptimizedVsNonOptimized:
  def __init__(self, opt_results: OptimizationResults) -> None:
    self.optimized_data = opt_results.optimized_data.sel(metric='mean')
    self.nonoptimized_data = opt_results.nonoptimized_data.sel(metric='mean')

  @property
  def opt_df(self):
    return self.optimized_data.to_dataframe().reset_index()

  @property
  def nonopt_df(self):
    return self.nonoptimized_data.to_dataframe().reset_index()

  @property
  def opt_attrs(self):
    return self.optimized_data.attrs

  @property
  def nonopt_attrs(self):
    return self.nonoptimized_data.attrs

  def get_total_level_comparison(self) -> pd.DataFrame:
    """ Overall summary of optimized vs non-optimized results """
    total_opt_df = pd.DataFrame([self.opt_attrs]). \
      rename(columns={
        'budget': 'optimized_budget',
        'total_incremental_outcome': 'optimized_total_incremental_outcome',
          'total_cpik': 'optimized_total_cpa'}
          )[['start_date', 'end_date', 'optimized_budget', 'optimized_total_incremental_outcome', 'optimized_total_cpa']]

    total_nonopt_df = pd.DataFrame([self.nonopt_attrs]). \
      rename(columns={
        'budget': 'nonoptimized_budget',
        'total_incremental_outcome': 'nonoptimized_total_incremental_outcome',
          'total_cpik': 'nonoptimized_total_cpa'}
          )[['nonoptimized_budget', 'nonoptimized_total_incremental_outcome', 'nonoptimized_total_cpa']]

    total_opt_vs_nonopt_df = total_opt_df.join(total_nonopt_df)
    total_opt_vs_nonopt_df['budget_change'] = total_opt_vs_nonopt_df['optimized_budget'] / total_opt_vs_nonopt_df['nonoptimized_budget'] - 1.0
    total_opt_vs_nonopt_df['outcome_change'] = total_opt_vs_nonopt_df['optimized_total_incremental_outcome'] / total_opt_vs_nonopt_df['nonoptimized_total_incremental_outcome'] - 1.0
    total_opt_vs_nonopt_df['cpa_change'] = total_opt_vs_nonopt_df['optimized_total_cpa'] / total_opt_vs_nonopt_df['nonoptimized_total_cpa'] - 1.0

    return total_opt_vs_nonopt_df

  def get_channel_level_comparison(self) -> pd.DataFrame:
    """ Channel level summary - Optimized vs Non-Optimized """
    opt_df_formatted = self.opt_df.rename(
      columns={
      'spend': 'optimized_spend',
      'incremental_outcome': 'optimized_incremental_outcome',
      'effectiveness': 'optimized_effectiveness',
      'cpik': 'optimized_cpa',
      })[['channel', 'optimized_spend', 'optimized_incremental_outcome', 'optimized_effectiveness', 'optimized_cpa']].copy()


    nonopt_df_formatted = self.nonopt_df.rename(
      columns={
      'spend': 'nonoptimized_spend',
      'incremental_outcome': 'nonoptimized_incremental_outcome',
      'effectiveness': 'nonoptimized_effectiveness',
      'cpik': 'nonoptimized_cpa',
      })[['channel', 'nonoptimized_spend', 'nonoptimized_incremental_outcome', 'nonoptimized_effectiveness', 'nonoptimized_cpa']].copy()

    channel_opt_vs_nonopt_df = opt_df_formatted.merge(nonopt_df_formatted, on='channel', how='left')
    channel_opt_vs_nonopt_df['budget_change'] = channel_opt_vs_nonopt_df['optimized_spend'] / channel_opt_vs_nonopt_df['nonoptimized_spend'] - 1.0
    channel_opt_vs_nonopt_df['outcome_change'] = channel_opt_vs_nonopt_df['optimized_incremental_outcome'] / channel_opt_vs_nonopt_df['nonoptimized_incremental_outcome'] - 1.0
    channel_opt_vs_nonopt_df['effectiveness_change'] = channel_opt_vs_nonopt_df['optimized_effectiveness'] / channel_opt_vs_nonopt_df['nonoptimized_effectiveness'] - 1.0
    channel_opt_vs_nonopt_df['cpa_change'] = channel_opt_vs_nonopt_df['optimized_cpa'] / channel_opt_vs_nonopt_df['nonoptimized_cpa'] - 1.0

    return channel_opt_vs_nonopt_df
