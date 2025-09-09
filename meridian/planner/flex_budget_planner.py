"""FlexibleBudgetPlanner for creating Meridian InputData from Excel files and budget optimization."""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import arviz as az

from meridian.data import data_frame_input_data_builder
from meridian.data import input_data
from meridian.model import model
from meridian.model import spec
from meridian.analysis import optimizer
from meridian.planner import media_parameter_loader
from meridian.planner import point_inference_data
from meridian.planner import roi_to_coefficients_converter


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
  """

  def __init__(self, file_name: str, model_config: Dict[str, Any]):
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
    """
    self.file_name = file_name
    self.model_config = model_config

    # Data containers
    self.data_df: Optional[pd.DataFrame] = None
    self.coefficients_df: Optional[pd.DataFrame] = None
    self.parameters_df: Optional[pd.DataFrame] = None
    self.roi_df: Optional[pd.DataFrame] = None
    self.input_type: Optional[str] = None

    # Validate required config keys
    self._validate_config()
    self._validate_optimization_config()

  def _validate_config(self) -> None:
    """Validate that model_config contains required keys."""
    required_keys = [
      'time_col', 'geo_col', 'population_col', 'kpi_type',
      'kpi_col', 'media_cols', 'media_spend_cols', 'media_channels'
    ]

    missing_keys = [key for key in required_keys if key not in self.model_config]
    if missing_keys:
      raise ValueError(f"Missing required config keys: {missing_keys}")

    # Validate list lengths match
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

      # Validate that media_channels and rf_channels don't overlap
      media_channels = set(self.model_config['media_channels'])
      rf_channels = set(self.model_config['rf_channels'])
      overlapping_channels = media_channels.intersection(rf_channels)
      if overlapping_channels:
        raise ValueError(f"Channels cannot be both media and R&F channels. Overlapping channels: {list(overlapping_channels)}")

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

    # Add media columns
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

    # Add media channels
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
        data_df=self.data_df
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
        data_df=self.data_df
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
        data_df=self.data_df
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
        data_df=self.data_df
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

      input_data_obj = self.build_input_data()

      # Create PointInferenceData with input_data for complete structure
      point_data = point_inference_data.PointInferenceData(
        parameter_arrays, coefficient_arrays,
        input_data_obj=input_data_obj
      )
      inference_data = point_data.get_inference_data()
      logging.info("Successfully created ArviZ InferenceData from Excel data")
      return inference_data

    except Exception as e:
      raise ValueError(f"Error creating InferenceData: {str(e)}")

  def optimize(self) -> Any:
    """Run budget optimization using Excel data with Meridian model.

    Creates a Meridian model using the Excel data and runs budget optimization.
    Handles use_kpi parameter based on model configuration validation.

    Returns:
      BudgetOptimizer results containing optimized budget allocation.

    Raises:
      ValueError: If optimization fails or data is incomplete.
    """
    try:
      # Build input data and inference data
      logging.info("Building input data and inference data for optimization...")
      data = self.build_input_data()
      point_inference_data = self.get_inference_data()

      if point_inference_data is None:
        raise ValueError("Cannot create inference data - missing parameters or coefficients sheets")

      # Create Meridian model
      logging.info("Creating Meridian model...")
      model_spec = spec.ModelSpec()
      model_obj = model.Meridian(
          input_data=data,
          model_spec=model_spec,
          inference_data=point_inference_data
      )

      # Sample prior (required for optimization)
      logging.info("Sampling prior distributions...")
      model_obj.sample_prior(n_draws=100, seed=42)

      # Create optimizer
      logging.info("Creating budget optimizer...")
      budget_optimizer = optimizer.BudgetOptimizer(model_obj)

      # Check if we need to use use_kpi parameter
      optimize_kwargs = {}
      kpi_type = self.model_config.get('kpi_type')
      revenue_per_kpi_col = self.model_config.get('revenue_per_kpi_col')

      if (kpi_type == 'non_revenue' and
          ('revenue_per_kpi_col' not in self.model_config or revenue_per_kpi_col is None)):
        optimize_kwargs['use_kpi'] = True
        logging.info("Using use_kpi=True due to non_revenue KPI type without revenue_per_kpi_col")

      # Run optimization
      logging.info("Running budget optimization...")
      optimizer_results = budget_optimizer.optimize(**optimize_kwargs)

      logging.info("Budget optimization completed successfully")
      return optimizer_results

    except Exception as e:
      raise ValueError(f"Error during optimization: {str(e)}")
