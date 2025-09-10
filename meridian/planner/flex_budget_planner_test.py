"""Tests for FlexibleBudgetPlanner."""

import os
import tempfile
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import numpy as np
import xarray as xr
import arviz as az

from meridian.planner import flex_budget_planner
from meridian.planner.flex_budget_planner import FlexibleBudgetPlanner
from meridian import constants


class FlexibleBudgetPlannerTest(parameterized.TestCase):
  
  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    
    # Sample model configuration
    self.model_config = {
      'time_col': 'week',
      'geo_col': 'geo', 
      'population_col': 'population',
      'kpi_type': 'non_revenue',
      'kpi_col': 'conversions',
      'revenue_per_kpi_col': 'revenue_per_conversion',
      'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
      'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
      'media_channels': ['Channel0', 'Channel1', 'Channel2'],
      'reach_cols': ['Channel3_reach'],
      'frequency_cols': ['Channel3_frequency'], 
      'rf_spend_cols': ['Channel3_spend'],
      'rf_channels': ['Channel3'],
      'control_cols': ['sentiment_score_control', 'competitor_activity_score_control']
    }
    
    # Sample data for testing
    self.sample_data = pd.DataFrame({
      'week': ['2024-01-01', '2024-01-08', '2024-01-15'] * 2,
      'geo': ['US', 'US', 'US', 'CA', 'CA', 'CA'],
      'population': [100000, 100000, 100000, 50000, 50000, 50000],
      'conversions': [100, 120, 110, 80, 90, 85],
      'revenue_per_conversion': [10.0, 10.0, 10.0, 12.0, 12.0, 12.0],
      'Channel0_impression': [1000, 1200, 1100, 800, 900, 850],
      'Channel1_impression': [2000, 2200, 2100, 1600, 1800, 1700],
      'Channel2_impression': [1500, 1700, 1600, 1200, 1400, 1300],
      'Channel0_spend': [500, 600, 550, 400, 450, 425],
      'Channel1_spend': [800, 900, 850, 640, 720, 680],
      'Channel2_spend': [600, 700, 650, 480, 540, 520],
      'Channel3_reach': [5000, 5500, 5200, 4000, 4400, 4200],
      'Channel3_frequency': [2.5, 2.7, 2.6, 2.3, 2.5, 2.4],
      'Channel3_spend': [1000, 1100, 1050, 800, 900, 850],
      'sentiment_score_control': [0.6, 0.65, 0.7, 0.55, 0.6, 0.65],
      'competitor_activity_score_control': [0.4, 0.35, 0.3, 0.45, 0.4, 0.35]
    })
    
    self.sample_coefficients = pd.DataFrame({
      'geo': ['US', 'CA'],
      'Channel0': [0.1, 0.12], 
      'Channel1': [0.15, 0.18],
      'Channel2': [0.08, 0.10],
      'Channel3': [0.20, 0.22]
    })
    
    self.sample_parameters = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1', 'Channel2', 'Channel3'],
      'Adstock': [0.7, 0.6, 0.8, 0.5],
      'Inflexion': [0.5, 0.4, 0.6, 0.3],
      'Slope': [2.0, 1.8, 2.2, 1.5]
    })

  def _create_test_excel_file(self, include_coefficients=True, include_parameters=True):
    """Create a temporary Excel file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    temp_file.close()
    
    with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
      self.sample_data.to_excel(writer, sheet_name='Data', index=False)
      if include_coefficients:
        self.sample_coefficients.to_excel(writer, sheet_name='Coefficients', index=False)
      if include_parameters:
        self.sample_parameters.to_excel(writer, sheet_name='Parameters', index=False)
        
    return temp_file.name

  def test_init_valid_config(self):
    """Test initialization with valid config."""
    file_name = 'test_file.xlsx'
    loader = FlexibleBudgetPlanner(file_name, self.model_config)
    
    self.assertEqual(loader.file_name, file_name)
    self.assertEqual(loader.model_config, self.model_config)
    self.assertIsNone(loader.data_df)
    self.assertIsNone(loader.coefficients_df)
    self.assertIsNone(loader.parameters_df)

  def test_init_missing_required_config(self):
    """Test initialization with missing required config keys."""
    invalid_config = self.model_config.copy()
    del invalid_config['kpi_col']
    
    with self.assertRaises(ValueError) as cm:
      FlexibleBudgetPlanner('test.xlsx', invalid_config)
      
    self.assertIn('Missing required config keys', str(cm.exception))

  def test_init_mismatched_media_lengths(self):
    """Test initialization with mismatched media column lengths."""
    invalid_config = self.model_config.copy()
    invalid_config['media_channels'] = ['Channel0', 'Channel1']  # Different length
    
    with self.assertRaises(ValueError) as cm:
      FlexibleBudgetPlanner('test.xlsx', invalid_config)
      
    self.assertIn('media_cols and media_channels must have same length', str(cm.exception))

  def test_init_incomplete_rf_config(self):
    """Test initialization with incomplete R&F configuration."""
    invalid_config = self.model_config.copy()
    del invalid_config['frequency_cols']  # Missing R&F key
    
    with self.assertRaises(ValueError) as cm:
      FlexibleBudgetPlanner('test.xlsx', invalid_config)
      
    self.assertIn('all R&F keys must be provided', str(cm.exception))

  def test_init_overlapping_media_rf_channels(self):
    """Test initialization with overlapping media and R&F channels."""
    invalid_config = self.model_config.copy()
    # Make Channel0 appear in both media and R&F channels
    invalid_config['rf_channels'] = ['Channel0']
    
    with self.assertRaises(ValueError) as cm:
      FlexibleBudgetPlanner('test.xlsx', invalid_config)
      
    self.assertIn('Channels cannot be both media and R&F channels', str(cm.exception))
    self.assertIn('Channel0', str(cm.exception))

  def test_load_excel_data_success(self):
    """Test successful Excel data loading."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      # Check data was loaded
      self.assertIsNotNone(loader.data_df)
      self.assertIsNotNone(loader.coefficients_df)
      self.assertIsNotNone(loader.parameters_df)
      
      # Check data content (allow dtype differences due to Excel loading)
      pd.testing.assert_frame_equal(loader.data_df, self.sample_data, check_dtype=False)
      pd.testing.assert_frame_equal(loader.coefficients_df, self.sample_coefficients, check_dtype=False)
      pd.testing.assert_frame_equal(loader.parameters_df, self.sample_parameters, check_dtype=False)
      
    finally:
      os.unlink(excel_file)

  def test_load_excel_data_missing_required_sheets(self):
    """Test loading Excel data with missing required Coefficients/ROI sheets."""
    excel_file = self._create_test_excel_file(
      include_coefficients=False, 
      include_parameters=False
    )
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      
      # Should raise ValueError as Coefficients or ROI sheet is required
      with self.assertRaises(ValueError) as cm:
        loader.load_excel_data()
      
      self.assertIn("must contain either 'Coefficients' or 'ROI' sheet", str(cm.exception))
      
    finally:
      os.unlink(excel_file)

  def test_load_excel_data_missing_file(self):
    """Test loading from non-existent Excel file."""
    loader = FlexibleBudgetPlanner('nonexistent.xlsx', self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.load_excel_data()
      
    self.assertIn('Error loading Excel file', str(cm.exception))

  def test_validate_data_columns_success(self):
    """Test successful data column validation."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      loader.validate_data_columns()  # Should not raise
      
    finally:
      os.unlink(excel_file)

  def test_validate_data_columns_missing_columns(self):
    """Test validation with missing required columns."""
    # Create data missing a required column
    incomplete_data = self.sample_data.drop(columns=['conversions'])
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    temp_file.close()
    
    try:
      with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
        incomplete_data.to_excel(writer, sheet_name='Data', index=False)
        # Add required Coefficients sheet for successful loading
        self.sample_coefficients.to_excel(writer, sheet_name='Coefficients', index=False)
        self.sample_parameters.to_excel(writer, sheet_name='Parameters', index=False)
        
      loader = FlexibleBudgetPlanner(temp_file.name, self.model_config)
      loader.load_excel_data()
      
      with self.assertRaises(ValueError) as cm:
        loader.validate_data_columns()
        
      self.assertIn('Missing required columns', str(cm.exception))
      
    finally:
      os.unlink(temp_file.name)

  def test_validate_data_columns_no_data_loaded(self):
    """Test validation when no data is loaded."""
    loader = FlexibleBudgetPlanner('test.xlsx', self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_data_columns()
      
    self.assertIn('Data not loaded', str(cm.exception))

  def test_build_input_data_success(self):
    """Test successful InputData building."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      input_data = loader.build_input_data()
      
      # Check that we got an InputData object
      self.assertIsNotNone(input_data)
      
      # Check basic data structure
      self.assertIsNotNone(input_data.kpi)
      self.assertIsNotNone(input_data.population)
      self.assertIsNotNone(input_data.media)
      self.assertIsNotNone(input_data.media_spend)
      self.assertIsNotNone(input_data.reach)
      self.assertIsNotNone(input_data.frequency)
      self.assertIsNotNone(input_data.rf_spend)
      self.assertIsNotNone(input_data.controls)
      
      # Check dimensions
      self.assertEqual(list(input_data.kpi.dims), [constants.GEO, constants.TIME])
      self.assertEqual(list(input_data.media.dims), [constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL])
      self.assertEqual(list(input_data.reach.dims), [constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL])
      
    finally:
      os.unlink(excel_file)

  def test_build_input_data_without_rf_channels(self):
    """Test building InputData without R&F channels."""
    config_no_rf = self.model_config.copy()
    for key in ['reach_cols', 'frequency_cols', 'rf_spend_cols', 'rf_channels']:
      del config_no_rf[key]
      
    # Create simplified data without R&F columns  
    simple_data = self.sample_data.drop(columns=[
      'Channel3_reach', 'Channel3_frequency', 'Channel3_spend'
    ])
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    temp_file.close()
    
    try:
      with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
        simple_data.to_excel(writer, sheet_name='Data', index=False)
        # Add required Coefficients sheet (only media channels, no RF)
        simple_coefficients = self.sample_coefficients.drop(columns=['Channel3'])
        simple_coefficients.to_excel(writer, sheet_name='Coefficients', index=False)
        # Add Parameters sheet for media channels only
        simple_parameters = self.sample_parameters[self.sample_parameters['MediaVariable'] != 'Channel3']
        simple_parameters.to_excel(writer, sheet_name='Parameters', index=False)
        
      loader = FlexibleBudgetPlanner(temp_file.name, config_no_rf)
      input_data = loader.build_input_data()
      
      # Check basic data structure
      self.assertIsNotNone(input_data.kpi)
      self.assertIsNotNone(input_data.media)
      self.assertIsNone(input_data.reach)  # Should be None without R&F channels
      self.assertIsNone(input_data.frequency)
      self.assertIsNone(input_data.rf_spend)
      
    finally:
      os.unlink(temp_file.name)

  def test_build_input_data_without_controls(self):
    """Test building InputData without control variables."""
    config_no_controls = self.model_config.copy()
    del config_no_controls['control_cols']
    
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, config_no_controls)
      input_data = loader.build_input_data()
      
      # Check basic data structure 
      self.assertIsNotNone(input_data.kpi)
      self.assertIsNotNone(input_data.media)
      self.assertIsNone(input_data.controls)  # Should be None without controls
      
    finally:
      os.unlink(excel_file)

  def test_get_coefficients_data(self):
    """Test getting coefficients data."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      coefficients = loader.get_coefficients_data()
      self.assertIsNotNone(coefficients)
      pd.testing.assert_frame_equal(coefficients, self.sample_coefficients)
      
    finally:
      os.unlink(excel_file)

  def test_get_parameters_data(self):
    """Test getting parameters data."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      parameters = loader.get_parameters_data()
      self.assertIsNotNone(parameters)
      pd.testing.assert_frame_equal(parameters, self.sample_parameters)
      
    finally:
      os.unlink(excel_file)

  def test_get_data_not_loaded(self):
    """Test getting data when not loaded."""
    loader = FlexibleBudgetPlanner('test.xlsx', self.model_config)
    
    self.assertIsNone(loader.get_coefficients_data())
    self.assertIsNone(loader.get_parameters_data())

  def test_get_processed_parameters_success(self):
    """Test getting processed parameters successfully."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      processed_params = loader.get_processed_parameters()
      self.assertIsNotNone(processed_params)
      
      # Check that all expected keys are present
      from meridian import constants
      expected_keys = [constants.ALPHA_M, constants.EC_M, constants.SLOPE_M, 
                       constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF]
      for key in expected_keys:
        self.assertIn(key, processed_params)
        
      # Check media channel parameters (3 channels)
      self.assertEqual(len(processed_params[constants.ALPHA_M]), 3)
      self.assertEqual(len(processed_params[constants.EC_M]), 3)
      self.assertEqual(len(processed_params[constants.SLOPE_M]), 3)
      
      # Check R&F channel parameters (1 channel)
      self.assertEqual(len(processed_params[constants.ALPHA_RF]), 1)
      self.assertEqual(len(processed_params[constants.EC_RF]), 1)
      self.assertEqual(len(processed_params[constants.SLOPE_RF]), 1)
      
    finally:
      os.unlink(excel_file)

  def test_get_processed_parameters_no_parameters(self):
    """Test getting processed parameters when no parameters sheet exists."""
    excel_file = self._create_test_excel_file(include_parameters=False)
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      processed_params = loader.get_processed_parameters()
      self.assertIsNone(processed_params)
      
    finally:
      os.unlink(excel_file)

  def test_get_parameter_summary_success(self):
    """Test getting parameter summary successfully."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      summary = loader.get_parameter_summary()
      self.assertIsNotNone(summary)
      
      # Check summary structure
      self.assertEqual(len(summary), 4)  # 4 channels total
      self.assertEqual(list(summary.columns), ['Channel', 'Type', 'Adstock', 'Inflexion', 'Slope'])
      
      # Check media channels
      media_rows = summary[summary['Type'] == 'Media']
      self.assertEqual(len(media_rows), 3)
      
      # Check R&F channels
      rf_rows = summary[summary['Type'] == 'R&F']
      self.assertEqual(len(rf_rows), 1)
      
    finally:
      os.unlink(excel_file)

  def test_get_parameter_summary_no_parameters(self):
    """Test getting parameter summary when no parameters sheet exists."""
    excel_file = self._create_test_excel_file(include_parameters=False)
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      summary = loader.get_parameter_summary()
      self.assertIsNone(summary)
      
    finally:
      os.unlink(excel_file)

  def test_get_processed_parameter_arrays_success(self):
    """Test getting processed parameter DataArrays successfully."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      parameter_arrays = loader.get_processed_parameter_arrays()
      self.assertIsNotNone(parameter_arrays)
      
      # Check that all expected DataArrays are present
      expected_keys = [constants.ALPHA_M, constants.EC_M, constants.SLOPE_M, 
                       constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF]
      for key in expected_keys:
        self.assertIn(key, parameter_arrays)
        self.assertIsInstance(parameter_arrays[key], xr.DataArray)
        
      # Check media channel DataArrays (3 channels)
      alpha_m = parameter_arrays[constants.ALPHA_M]
      self.assertEqual(alpha_m.shape, (3,))
      self.assertEqual(alpha_m.dims, ('media_channel',))
      self.assertEqual(list(alpha_m.coords['media_channel'].values), ['Channel0', 'Channel1', 'Channel2'])
      
      # Check R&F channel DataArrays (1 channel) 
      alpha_rf = parameter_arrays[constants.ALPHA_RF]
      self.assertEqual(alpha_rf.shape, (1,))
      self.assertEqual(alpha_rf.dims, ('rf_channel',))
      self.assertEqual(list(alpha_rf.coords['rf_channel'].values), ['Channel3'])
      
    finally:
      os.unlink(excel_file)

  def test_get_processed_parameter_arrays_no_parameters(self):
    """Test getting processed parameter arrays when no parameters sheet exists."""
    excel_file = self._create_test_excel_file(include_parameters=False)
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      parameter_arrays = loader.get_processed_parameter_arrays()
      self.assertIsNone(parameter_arrays)
      
    finally:
      os.unlink(excel_file)

  def test_get_processed_parameter_arrays_consistency(self):
    """Test that DataArrays values match get_processed_parameters results."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      param_dict = loader.get_processed_parameters()
      param_arrays = loader.get_processed_parameter_arrays()
      
      self.assertIsNotNone(param_dict)
      self.assertIsNotNone(param_arrays)
      
      # Values should match exactly
      np.testing.assert_array_equal(param_arrays[constants.ALPHA_M].values, param_dict[constants.ALPHA_M])
      np.testing.assert_array_equal(param_arrays[constants.EC_M].values, param_dict[constants.EC_M])
      np.testing.assert_array_equal(param_arrays[constants.SLOPE_M].values, param_dict[constants.SLOPE_M])
      np.testing.assert_array_equal(param_arrays[constants.ALPHA_RF].values, param_dict[constants.ALPHA_RF])
      np.testing.assert_array_equal(param_arrays[constants.EC_RF].values, param_dict[constants.EC_RF])
      np.testing.assert_array_equal(param_arrays[constants.SLOPE_RF].values, param_dict[constants.SLOPE_RF])
      
    finally:
      os.unlink(excel_file)

  @parameterized.named_parameters(
    ('revenue_kpi', 'revenue'),
    ('non_revenue_kpi', 'non_revenue'),
  )
  def test_different_kpi_types(self, kpi_type):
    """Test with different KPI types."""
    config = self.model_config.copy()
    config['kpi_type'] = kpi_type
    
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, config)
      input_data = loader.build_input_data()
      
      self.assertIsNotNone(input_data)
      self.assertIsNotNone(input_data.kpi)
      
    finally:
      os.unlink(excel_file)

  def test_get_inference_data_success(self):
    """Test successful InferenceData creation."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      inference_data = loader.get_inference_data()
      
      self.assertIsNotNone(inference_data)
      self.assertIsInstance(inference_data, az.InferenceData)
      
      # Check groups
      expected_groups = ['posterior', 'sample_stats']
      self.assertEqual(sorted(inference_data.groups()), expected_groups)
      
      # Check posterior structure
      posterior = inference_data.posterior
      self.assertIsInstance(posterior, xr.Dataset)
      
      # Check dimensions
      expected_dims = ['chain', 'draw', 'media_channel', 'rf_channel', 'geo']
      for dim in expected_dims:
        self.assertIn(dim, posterior.dims)
        
      # Check dimension sizes
      self.assertEqual(posterior.sizes['chain'], 1)
      self.assertEqual(posterior.sizes['draw'], 1)
      self.assertEqual(posterior.sizes['media_channel'], 3)
      self.assertEqual(posterior.sizes['rf_channel'], 1)
      self.assertEqual(posterior.sizes['geo'], 2)  # Test file has 2 geos
      
      # Check that essential data variables from Excel are present
      essential_vars = {
        constants.ALPHA_M, constants.EC_M, constants.SLOPE_M,
        constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF,
        constants.BETA_GM, constants.BETA_GRF
      }
      actual_vars = set(posterior.data_vars.keys())
      self.assertTrue(essential_vars.issubset(actual_vars), 
                     f"Missing essential variables: {essential_vars - actual_vars}")
      
    finally:
      os.unlink(excel_file)

  def test_get_inference_data_no_parameters(self):
    """Test InferenceData creation with missing parameters."""
    excel_file = self._create_test_excel_file(include_parameters=False)
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      inference_data = loader.get_inference_data()
      
      self.assertIsNone(inference_data)
      
    finally:
      os.unlink(excel_file)

  def test_get_inference_data_no_coefficients(self):
    """Test InferenceData creation with missing required Coefficients sheet."""
    excel_file = self._create_test_excel_file(include_coefficients=False)
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      
      # Should fail during loading since Coefficients sheet is required
      with self.assertRaises(ValueError) as cm:
        loader.load_excel_data()
      
      self.assertIn("must contain either 'Coefficients' or 'ROI' sheet", str(cm.exception))
      
    finally:
      os.unlink(excel_file)

  def test_inference_data_compatibility(self):
    """Test that InferenceData is compatible with ArviZ functions."""
    excel_file = self._create_test_excel_file()
    
    try:
      loader = FlexibleBudgetPlanner(excel_file, self.model_config)
      loader.load_excel_data()
      
      inference_data = loader.get_inference_data()
      
      # Test ArviZ functions work
      self.assertIsNotNone(inference_data)
      
      # Test groups() method
      groups = inference_data.groups()
      self.assertIn('posterior', groups)
      self.assertIn('sample_stats', groups)
      
      # Test accessing posterior data
      posterior = inference_data.posterior
      alpha_m = posterior[constants.ALPHA_M]
      self.assertEqual(alpha_m.shape, (1, 1, 3))  # chain, draw, media_channel
      
    finally:
      os.unlink(excel_file)

  def test_rf_only_configuration(self):
    """Test RF-only configuration (no media channels, only reach/frequency)."""
    # Create RF-only configuration (no media keys specified)
    rf_only_config = {
      'time_col': 'week',
      'geo_col': 'geo',
      'population_col': 'population',
      'kpi_type': 'non_revenue',
      'kpi_col': 'conversions',
      'revenue_per_kpi_col': 'revenue_per_conversion',
      
      # Only RF channels, no media channels
      'reach_cols': ['Channel3_reach'],
      'frequency_cols': ['Channel3_frequency'],
      'rf_spend_cols': ['Channel3_spend'],
      'rf_channels': ['Channel3'],
    }
    
    # Create RF-only data and Excel file
    rf_only_data = self.sample_data[['week', 'geo', 'population', 'conversions', 'revenue_per_conversion',
                                     'Channel3_reach', 'Channel3_frequency', 'Channel3_spend']].copy()
    
    rf_only_coefficients = self.sample_coefficients[['geo', 'Channel3']].copy()
    rf_only_parameters = self.sample_parameters[self.sample_parameters['MediaVariable'] == 'Channel3'].copy()
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    temp_file.close()
    
    try:
      with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
        rf_only_data.to_excel(writer, sheet_name='Data', index=False)
        rf_only_coefficients.to_excel(writer, sheet_name='Coefficients', index=False)
        rf_only_parameters.to_excel(writer, sheet_name='Parameters', index=False)
      
      # Test initialization and validation
      loader = FlexibleBudgetPlanner(temp_file.name, rf_only_config)
      
      # Verify media channels are empty lists (auto-created)
      self.assertEqual(loader.model_config['media_channels'], [])
      self.assertEqual(loader.model_config['media_cols'], [])
      self.assertEqual(loader.model_config['media_spend_cols'], [])
      
      # Verify RF channels are configured
      self.assertEqual(loader.model_config['rf_channels'], ['Channel3'])
      
      # Test data loading
      loader.load_excel_data()
      self.assertIsNotNone(loader.data_df)
      self.assertIsNotNone(loader.coefficients_df)
      self.assertIsNotNone(loader.parameters_df)
      
      # Test InputData building
      input_data = loader.build_input_data()
      
      # Verify RF-only structure
      self.assertIsNotNone(input_data.kpi)
      self.assertIsNotNone(input_data.population)
      self.assertIsNone(input_data.media)  # No media channels
      self.assertIsNotNone(input_data.reach)  # RF channels present
      self.assertIsNotNone(input_data.frequency)
      self.assertIsNotNone(input_data.rf_spend)
      
      # Verify RF dimensions
      self.assertEqual(input_data.reach.shape[2], 1)  # 1 RF channel
      
      # Test inference data creation
      inference_data = loader.get_inference_data()
      self.assertIsNotNone(inference_data)
      
      # Verify posterior contains RF parameters but no media parameters
      posterior = inference_data.posterior
      self.assertIn(constants.ALPHA_RF, posterior.data_vars)
      self.assertIn(constants.EC_RF, posterior.data_vars)
      self.assertIn(constants.SLOPE_RF, posterior.data_vars)
      self.assertIn(constants.BETA_GRF, posterior.data_vars)
      
      # Verify RF dimensions in posterior
      self.assertEqual(posterior[constants.ALPHA_RF].shape, (1, 1, 1))  # chain, draw, rf_channel
      self.assertEqual(posterior[constants.BETA_GRF].shape, (1, 1, 2, 1))  # chain, draw, geo, rf_channel
      
    finally:
      os.unlink(temp_file.name)


if __name__ == '__main__':
  absltest.main()