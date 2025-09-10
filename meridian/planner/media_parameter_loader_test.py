"""Tests for MediaParameterLoader."""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import numpy as np
import xarray as xr

from meridian.planner import media_parameter_loader
from meridian import constants


class MediaParameterLoaderTest(parameterized.TestCase):
  
  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    
    # Sample model configuration
    self.model_config = {
      'media_channels': ['Channel0', 'Channel1', 'Channel2'],
      'rf_channels': ['Channel3'],
      'geo_col': 'geo'
    }
    
    # Sample parameters DataFrame (correct order)
    self.sample_parameters = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1', 'Channel2', 'Channel3'],
      'Adstock': [0.5, 0.3, 0.7, 0.6],
      'Inflexion': [1.5, 1.2, 1.8, 1.4],
      'Slope': [1.0, 2.0, 1.5, 3.0]
    })
    
    # Sample parameters with different order
    self.unordered_parameters = pd.DataFrame({
      'MediaVariable': ['Channel3', 'Channel0', 'Channel2', 'Channel1'],
      'Adstock': [0.6, 0.5, 0.7, 0.3],
      'Inflexion': [1.4, 1.5, 1.8, 1.2],
      'Slope': [3.0, 1.0, 1.5, 2.0]
    })
    
    # Sample coefficients DataFrame
    self.sample_coefficients = pd.DataFrame({
      'geo': ['Geo0', 'Geo1', 'Geo2'],
      'Channel0': [0.55, 0.45, 0.62],
      'Channel1': [0.32, 0.28, 0.35],
      'Channel2': [0.68, 0.72, 0.65],
      'Channel3': [0.42, 0.38, 0.45]
    })
    
    # Sample data DataFrame (for geo validation)
    self.sample_data = pd.DataFrame({
      'geo': ['Geo0', 'Geo1', 'Geo2'],
      'week': ['2021-01-01', '2021-01-08', '2021-01-15'],
      'kpi': [100, 110, 105]
    })
    
    # Sample with only media channels (no RF)
    self.media_only_config = {
      'media_channels': ['Channel0', 'Channel1', 'Channel2'],
      'rf_channels': [],
      'geo_col': 'geo'
    }
    
    self.media_only_parameters = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1', 'Channel2'],
      'Adstock': [0.5, 0.3, 0.7],
      'Inflexion': [1.5, 1.2, 1.8],
      'Slope': [1.0, 2.0, 1.5]
    })
    
    self.media_only_coefficients = pd.DataFrame({
      'geo': ['Geo0', 'Geo1', 'Geo2'],
      'Channel0': [0.55, 0.45, 0.62],
      'Channel1': [0.32, 0.28, 0.35],
      'Channel2': [0.68, 0.72, 0.65]
    })

  def test_init_valid_inputs(self):
    """Test initialization with valid inputs."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    
    self.assertEqual(loader.media_channels, ['Channel0', 'Channel1', 'Channel2'])
    self.assertEqual(loader.rf_channels, ['Channel3'])
    self.assertEqual(loader.all_channels, ['Channel0', 'Channel1', 'Channel2', 'Channel3'])
    self.assertIsNone(loader.processed_parameters)

  def test_init_empty_dataframe(self):
    """Test initialization with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with self.assertRaises(ValueError) as cm:
      media_parameter_loader.MediaParameterLoader(empty_df, self.model_config)
      
    self.assertIn('cannot be None or empty', str(cm.exception))

  def test_init_invalid_model_config(self):
    """Test initialization with invalid model config."""
    invalid_config = {}  # Missing media_channels key
    
    with self.assertRaises(ValueError) as cm:
      media_parameter_loader.MediaParameterLoader(self.sample_parameters, invalid_config)
      
    self.assertIn('must contain \'media_channels\' key', str(cm.exception))

  def test_init_media_only_config(self):
    """Test initialization with media channels only (no R&F)."""
    media_only_config = {'media_channels': ['Channel0', 'Channel1'], 'rf_channels': []}
    media_only_params = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1'],
      'Adstock': [0.5, 0.3],
      'Inflexion': [1.5, 1.2],
      'Slope': [1.0, 2.0]
    })
    
    loader = media_parameter_loader.MediaParameterLoader(media_only_params, media_only_config)
    
    self.assertEqual(loader.media_channels, ['Channel0', 'Channel1'])
    self.assertEqual(loader.rf_channels, [])
    self.assertEqual(loader.all_channels, ['Channel0', 'Channel1'])

  def test_validate_parameters_success(self):
    """Test successful parameter validation."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    loader.validate_parameters()  # Should not raise

  def test_validate_parameters_missing_columns(self):
    """Test validation with missing required columns."""
    invalid_params = self.sample_parameters.drop(columns=['Adstock'])
    loader = media_parameter_loader.MediaParameterLoader(invalid_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('Missing required columns', str(cm.exception))
    self.assertIn('Adstock', str(cm.exception))

  def test_validate_parameters_duplicate_media_variables(self):
    """Test validation with duplicate MediaVariable entries."""
    duplicate_params = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel0', 'Channel1', 'Channel2'],
      'Adstock': [0.5, 0.4, 0.3, 0.7],
      'Inflexion': [1.5, 1.4, 1.2, 1.8],
      'Slope': [1.0, 1.1, 2.0, 1.5]
    })
    loader = media_parameter_loader.MediaParameterLoader(duplicate_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('Duplicate MediaVariable entries', str(cm.exception))
    self.assertIn('Channel0', str(cm.exception))

  def test_validate_parameters_missing_channels(self):
    """Test validation with missing MediaVariable entries."""
    incomplete_params = self.sample_parameters[self.sample_parameters['MediaVariable'] != 'Channel3']
    loader = media_parameter_loader.MediaParameterLoader(incomplete_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('Missing MediaVariable entries', str(cm.exception))
    self.assertIn('Channel3', str(cm.exception))

  def test_validate_parameters_extra_channels(self):
    """Test validation with unexpected MediaVariable entries."""
    extra_params = pd.concat([
      self.sample_parameters,
      pd.DataFrame({
        'MediaVariable': ['Channel4'],
        'Adstock': [0.8],
        'Inflexion': [2.0],
        'Slope': [2.5]
      })
    ], ignore_index=True)
    loader = media_parameter_loader.MediaParameterLoader(extra_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('Unexpected MediaVariable entries', str(cm.exception))
    self.assertIn('Channel4', str(cm.exception))

  def test_validate_parameter_values_invalid_adstock(self):
    """Test validation with invalid Adstock values."""
    invalid_params = self.sample_parameters.copy()
    invalid_params.loc[0, 'Adstock'] = -0.1  # Negative value
    invalid_params.loc[1, 'Adstock'] = 1.5   # > 1
    
    loader = media_parameter_loader.MediaParameterLoader(invalid_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('Adstock values must be between 0 and 1', str(cm.exception))

  def test_validate_parameter_values_invalid_inflexion(self):
    """Test validation with invalid Inflexion values."""
    invalid_params = self.sample_parameters.copy()
    invalid_params.loc[0, 'Inflexion'] = -0.5  # Negative value
    
    loader = media_parameter_loader.MediaParameterLoader(invalid_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('Inflexion values must be positive', str(cm.exception))

  def test_validate_parameter_values_invalid_slope(self):
    """Test validation with invalid Slope values."""
    invalid_params = self.sample_parameters.copy()
    invalid_params.loc[0, 'Slope'] = 0  # Zero value
    
    loader = media_parameter_loader.MediaParameterLoader(invalid_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('Slope values must be positive', str(cm.exception))

  def test_validate_parameter_values_nan_values(self):
    """Test validation with NaN values."""
    invalid_params = self.sample_parameters.copy()
    invalid_params.loc[0, 'Adstock'] = np.nan
    
    loader = media_parameter_loader.MediaParameterLoader(invalid_params, self.model_config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('contains missing values', str(cm.exception))

  def test_reorder_parameters_correct_order(self):
    """Test parameter reordering when already in correct order."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    reordered = loader.reorder_parameters()
    
    expected_order = ['Channel0', 'Channel1', 'Channel2', 'Channel3']
    actual_order = reordered['MediaVariable'].tolist()
    self.assertEqual(actual_order, expected_order)

  def test_reorder_parameters_wrong_order(self):
    """Test parameter reordering when in wrong order."""
    loader = media_parameter_loader.MediaParameterLoader(self.unordered_parameters, self.model_config)
    reordered = loader.reorder_parameters()
    
    expected_order = ['Channel0', 'Channel1', 'Channel2', 'Channel3']
    actual_order = reordered['MediaVariable'].tolist()
    self.assertEqual(actual_order, expected_order)
    
    # Check that values were reordered correctly
    self.assertEqual(reordered[reordered['MediaVariable'] == 'Channel0']['Adstock'].iloc[0], 0.5)
    self.assertEqual(reordered[reordered['MediaVariable'] == 'Channel3']['Adstock'].iloc[0], 0.6)

  def test_process_media_parameters(self):
    """Test processing media channel parameters."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    reordered = loader.reorder_parameters()
    media_params = loader.process_media_parameters(reordered)
    
    expected_media_params = {
      constants.ALPHA_M: [0.5, 0.3, 0.7],
      constants.EC_M: [1.5, 1.2, 1.8],
      constants.SLOPE_M: [1.0, 2.0, 1.5]
    }
    
    self.assertEqual(media_params, expected_media_params)

  def test_process_rf_parameters(self):
    """Test processing R&F channel parameters."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    reordered = loader.reorder_parameters()
    rf_params = loader.process_rf_parameters(reordered)
    
    expected_rf_params = {
      constants.ALPHA_RF: [0.6],
      constants.EC_RF: [1.4],
      constants.SLOPE_RF: [3.0]
    }
    
    self.assertEqual(rf_params, expected_rf_params)

  def test_process_media_only_parameters(self):
    """Test processing parameters when only media channels exist."""
    media_only_config = {'media_channels': ['Channel0', 'Channel1'], 'rf_channels': []}
    media_only_params = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1'],
      'Adstock': [0.5, 0.3],
      'Inflexion': [1.5, 1.2],
      'Slope': [1.0, 2.0]
    })
    
    loader = media_parameter_loader.MediaParameterLoader(media_only_params, media_only_config)
    params = loader.get_parameter_dict()
    
    expected_params = {
      constants.ALPHA_M: [0.5, 0.3],
      constants.EC_M: [1.5, 1.2],
      constants.SLOPE_M: [1.0, 2.0]
    }
    
    self.assertEqual(params, expected_params)

  def test_get_parameter_dict_complete(self):
    """Test getting complete parameter dictionary."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    params = loader.get_parameter_dict()
    
    expected_params = {
      constants.ALPHA_M: [0.5, 0.3, 0.7],
      constants.EC_M: [1.5, 1.2, 1.8],
      constants.SLOPE_M: [1.0, 2.0, 1.5],
      constants.ALPHA_RF: [0.6],
      constants.EC_RF: [1.4],
      constants.SLOPE_RF: [3.0]
    }
    
    self.assertEqual(params, expected_params)

  def test_get_parameter_dict_caching(self):
    """Test that parameter dictionary is cached after first computation."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    
    # First call
    params1 = loader.get_parameter_dict()
    self.assertIsNotNone(loader.processed_parameters)
    
    # Second call should return cached result
    params2 = loader.get_parameter_dict()
    self.assertIs(params1, params2)

  def test_get_parameter_dict_with_reordering(self):
    """Test parameter dictionary with unordered input."""
    loader = media_parameter_loader.MediaParameterLoader(self.unordered_parameters, self.model_config)
    params = loader.get_parameter_dict()
    
    # Should produce same result as ordered input
    expected_params = {
      constants.ALPHA_M: [0.5, 0.3, 0.7],
      constants.EC_M: [1.5, 1.2, 1.8],
      constants.SLOPE_M: [1.0, 2.0, 1.5],
      constants.ALPHA_RF: [0.6],
      constants.EC_RF: [1.4],
      constants.SLOPE_RF: [3.0]
    }
    
    self.assertEqual(params, expected_params)

  def test_get_channel_parameter_summary(self):
    """Test getting channel parameter summary."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    summary = loader.get_channel_parameter_summary()
    
    self.assertEqual(len(summary), 4)  # 4 channels total
    
    # Check media channels
    media_rows = summary[summary['Type'] == 'Media']
    self.assertEqual(len(media_rows), 3)
    self.assertEqual(media_rows['Channel'].tolist(), ['Channel0', 'Channel1', 'Channel2'])
    
    # Check R&F channels
    rf_rows = summary[summary['Type'] == 'R&F']
    self.assertEqual(len(rf_rows), 1)
    self.assertEqual(rf_rows['Channel'].iloc[0], 'Channel3')
    
    # Check parameter values
    channel0_row = summary[summary['Channel'] == 'Channel0'].iloc[0]
    self.assertEqual(channel0_row['Adstock'], 0.5)
    self.assertEqual(channel0_row['Inflexion'], 1.5)
    self.assertEqual(channel0_row['Slope'], 1.0)

  @parameterized.named_parameters(
    ('string_values', pd.DataFrame({
        'MediaVariable': ['Channel0'], 
        'Adstock': ['invalid'], 
        'Inflexion': [1.5], 
        'Slope': [1.0]
    })),
    ('mixed_types', pd.DataFrame({
        'MediaVariable': ['Channel0'], 
        'Adstock': [0.5], 
        'Inflexion': ['invalid'], 
        'Slope': [1.0]
    })),
  )
  def test_validate_parameter_values_non_numeric(self, invalid_params):
    """Test validation with non-numeric parameter values."""
    config = {'media_channels': ['Channel0'], 'rf_channels': []}
    loader = media_parameter_loader.MediaParameterLoader(invalid_params, config)
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_parameters()
      
    self.assertIn('must contain numeric values', str(cm.exception))

  def test_edge_case_single_channel(self):
    """Test with single channel configuration."""
    single_config = {'media_channels': ['Channel0'], 'rf_channels': []}
    single_params = pd.DataFrame({
      'MediaVariable': ['Channel0'],
      'Adstock': [0.5],
      'Inflexion': [1.5],
      'Slope': [1.0]
    })
    
    loader = media_parameter_loader.MediaParameterLoader(single_params, single_config)
    params = loader.get_parameter_dict()
    
    expected_params = {
      constants.ALPHA_M: [0.5],
      constants.EC_M: [1.5],
      constants.SLOPE_M: [1.0]
    }
    
    self.assertEqual(params, expected_params)

  def test_edge_case_boundary_values(self):
    """Test with boundary parameter values."""
    boundary_params = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1'],
      'Adstock': [0.0, 1.0],  # Boundary values for adstock
      'Inflexion': [0.001, 100.0],  # Very small and large inflexion
      'Slope': [0.001, 1000.0],  # Very small and large slope
    })
    boundary_config = {'media_channels': ['Channel0', 'Channel1'], 'rf_channels': []}
    
    loader = media_parameter_loader.MediaParameterLoader(boundary_params, boundary_config)
    params = loader.get_parameter_dict()  # Should not raise
    
    self.assertEqual(len(params[constants.ALPHA_M]), 2)
    self.assertEqual(params[constants.ALPHA_M], [0.0, 1.0])

  def test_get_parameter_data_arrays_complete(self):
    """Test getting complete parameter DataArrays."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    arrays = loader.get_parameter_data_arrays()
    
    # Check that all expected DataArrays are present
    expected_keys = [constants.ALPHA_M, constants.EC_M, constants.SLOPE_M, 
                     constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF]
    for key in expected_keys:
      self.assertIn(key, arrays)
      self.assertIsInstance(arrays[key], xr.DataArray)
    
    # Check media DataArrays structure
    alpha_m = arrays[constants.ALPHA_M]
    self.assertEqual(alpha_m.dims, ('media_channel',))
    self.assertEqual(list(alpha_m.coords['media_channel'].values), ['Channel0', 'Channel1', 'Channel2'])
    self.assertEqual(alpha_m.name, constants.ALPHA_M)
    np.testing.assert_array_equal(alpha_m.values, [0.5, 0.3, 0.7])
    
    # Check R&F DataArrays structure
    alpha_rf = arrays[constants.ALPHA_RF]
    self.assertEqual(alpha_rf.dims, ('rf_channel',))
    self.assertEqual(list(alpha_rf.coords['rf_channel'].values), ['Channel3'])
    self.assertEqual(alpha_rf.name, constants.ALPHA_RF)
    np.testing.assert_array_equal(alpha_rf.values, [0.6])

  def test_get_parameter_data_arrays_media_only(self):
    """Test DataArrays with media channels only."""
    media_only_config = {'media_channels': ['Channel0', 'Channel1'], 'rf_channels': []}
    media_only_params = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1'],
      'Adstock': [0.5, 0.3],
      'Inflexion': [1.5, 1.2],
      'Slope': [1.0, 2.0]
    })
    
    loader = media_parameter_loader.MediaParameterLoader(media_only_params, media_only_config)
    arrays = loader.get_parameter_data_arrays()
    
    # Should only have media parameter DataArrays
    expected_keys = [constants.ALPHA_M, constants.EC_M, constants.SLOPE_M]
    self.assertEqual(set(arrays.keys()), set(expected_keys))
    
    # Check structure
    alpha_m = arrays[constants.ALPHA_M]
    self.assertEqual(alpha_m.dims, ('media_channel',))
    self.assertEqual(list(alpha_m.coords['media_channel'].values), ['Channel0', 'Channel1'])
    np.testing.assert_array_equal(alpha_m.values, [0.5, 0.3])

  def test_get_parameter_data_arrays_rf_only(self):
    """Test DataArrays with R&F channels only."""
    rf_only_config = {'media_channels': [], 'rf_channels': ['Channel3']}
    rf_only_params = pd.DataFrame({
      'MediaVariable': ['Channel3'],
      'Adstock': [0.6],
      'Inflexion': [1.4],
      'Slope': [3.0]
    })
    
    loader = media_parameter_loader.MediaParameterLoader(rf_only_params, rf_only_config)
    arrays = loader.get_parameter_data_arrays()
    
    # Should only have R&F parameter DataArrays
    expected_keys = [constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF]
    self.assertEqual(set(arrays.keys()), set(expected_keys))
    
    # Check structure
    alpha_rf = arrays[constants.ALPHA_RF]
    self.assertEqual(alpha_rf.dims, ('rf_channel',))
    self.assertEqual(list(alpha_rf.coords['rf_channel'].values), ['Channel3'])
    np.testing.assert_array_equal(alpha_rf.values, [0.6])

  def test_get_parameter_data_arrays_with_reordering(self):
    """Test DataArrays with parameter reordering."""
    loader = media_parameter_loader.MediaParameterLoader(self.unordered_parameters, self.model_config)
    arrays = loader.get_parameter_data_arrays()
    
    # Should produce same result as ordered input (values should be reordered correctly)
    alpha_m = arrays[constants.ALPHA_M]
    self.assertEqual(list(alpha_m.coords['media_channel'].values), ['Channel0', 'Channel1', 'Channel2'])
    np.testing.assert_array_equal(alpha_m.values, [0.5, 0.3, 0.7])  # Correctly reordered
    
    alpha_rf = arrays[constants.ALPHA_RF]
    self.assertEqual(list(alpha_rf.coords['rf_channel'].values), ['Channel3'])
    np.testing.assert_array_equal(alpha_rf.values, [0.6])

  def test_get_parameter_data_arrays_data_types(self):
    """Test that DataArrays have correct data types."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    arrays = loader.get_parameter_data_arrays()
    
    # All values should be float type
    for key, array in arrays.items():
      self.assertTrue(np.issubdtype(array.dtype, np.floating), f"{key} should have float dtype")
      
    # Test specific values
    alpha_m = arrays[constants.ALPHA_M]
    self.assertIsInstance(float(alpha_m.values[0]), float)

  def test_get_parameter_data_arrays_single_channel(self):
    """Test DataArrays with single channel configuration."""
    single_config = {'media_channels': ['Channel0'], 'rf_channels': []}
    single_params = pd.DataFrame({
      'MediaVariable': ['Channel0'],
      'Adstock': [0.5],
      'Inflexion': [1.5],
      'Slope': [1.0]
    })
    
    loader = media_parameter_loader.MediaParameterLoader(single_params, single_config)
    arrays = loader.get_parameter_data_arrays()
    
    # Check single element arrays
    alpha_m = arrays[constants.ALPHA_M]
    self.assertEqual(alpha_m.shape, (1,))
    self.assertEqual(list(alpha_m.coords['media_channel'].values), ['Channel0'])
    np.testing.assert_array_equal(alpha_m.values, [0.5])

  def test_get_parameter_data_arrays_consistency_with_dict(self):
    """Test that DataArrays values match get_parameter_dict results."""
    loader = media_parameter_loader.MediaParameterLoader(self.sample_parameters, self.model_config)
    
    param_dict = loader.get_parameter_dict()
    arrays = loader.get_parameter_data_arrays()
    
    # Values should match exactly
    np.testing.assert_array_equal(arrays[constants.ALPHA_M].values, param_dict[constants.ALPHA_M])
    np.testing.assert_array_equal(arrays[constants.EC_M].values, param_dict[constants.EC_M])
    np.testing.assert_array_equal(arrays[constants.SLOPE_M].values, param_dict[constants.SLOPE_M])
    np.testing.assert_array_equal(arrays[constants.ALPHA_RF].values, param_dict[constants.ALPHA_RF])
    np.testing.assert_array_equal(arrays[constants.EC_RF].values, param_dict[constants.EC_RF])
    np.testing.assert_array_equal(arrays[constants.SLOPE_RF].values, param_dict[constants.SLOPE_RF])

  # Coefficients Tests

  def test_validate_coefficients_valid(self):
    """Test coefficients validation with valid data."""
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=self.sample_coefficients,
      data_df=self.sample_data
    )
    
    # Should not raise any exception
    loader.validate_coefficients()

  def test_validate_coefficients_missing_columns(self):
    """Test coefficients validation with missing columns."""
    invalid_coefficients = self.sample_coefficients.drop(columns=['Channel0'])
    
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=invalid_coefficients
    )
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_coefficients()
      
    self.assertIn('Missing required columns', str(cm.exception))
    self.assertIn('Channel0', str(cm.exception))

  def test_validate_coefficients_extra_columns(self):
    """Test coefficients validation with unexpected columns."""
    invalid_coefficients = self.sample_coefficients.copy()
    invalid_coefficients['ExtraColumn'] = [1.0, 2.0, 3.0]
    
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=invalid_coefficients
    )
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_coefficients()
      
    self.assertIn('Unexpected columns', str(cm.exception))
    self.assertIn('ExtraColumn', str(cm.exception))

  def test_validate_coefficients_duplicate_geos(self):
    """Test coefficients validation with duplicate geo entries."""
    invalid_coefficients = self.sample_coefficients.copy()
    invalid_coefficients.loc[2, 'geo'] = 'Geo0'  # Create duplicate
    
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=invalid_coefficients
    )
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_coefficients()
      
    self.assertIn('Duplicate geo entries', str(cm.exception))

  def test_validate_coefficients_negative_values(self):
    """Test coefficients validation with negative values."""
    invalid_coefficients = self.sample_coefficients.copy()
    invalid_coefficients.loc[0, 'Channel0'] = -0.1
    
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=invalid_coefficients
    )
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_coefficients()
      
    self.assertIn('negative values', str(cm.exception))
    self.assertIn('must be non-negative', str(cm.exception))

  def test_validate_coefficients_missing_values(self):
    """Test coefficients validation with missing values."""
    invalid_coefficients = self.sample_coefficients.copy()
    invalid_coefficients.loc[0, 'Channel1'] = np.nan
    
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=invalid_coefficients
    )
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_coefficients()
      
    self.assertIn('missing values', str(cm.exception))

  def test_validate_coefficients_geo_mismatch(self):
    """Test coefficients validation with geo values not matching data."""
    invalid_coefficients = self.sample_coefficients.copy()
    # Add an invalid geo instead of replacing to test extra geos case
    invalid_coefficients = pd.concat([
      invalid_coefficients,
      pd.DataFrame({
        'geo': ['InvalidGeo'],
        'Channel0': [0.5],
        'Channel1': [0.3],
        'Channel2': [0.7],
        'Channel3': [0.4]
      })
    ], ignore_index=True)
    
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=invalid_coefficients,
      data_df=self.sample_data,
      auto_filter_geos=False  # Ensure strict validation for this test
    )
    
    with self.assertRaises(ValueError) as cm:
      loader.validate_coefficients()
      
    self.assertIn('Unexpected geo values', str(cm.exception))

  def test_get_coefficients_data_arrays_basic(self):
    """Test basic coefficients DataArray creation."""
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=self.sample_coefficients,
      data_df=self.sample_data
    )
    
    arrays = loader.get_coefficients_data_arrays()
    
    # Check structure
    self.assertIn(constants.BETA_GM, arrays)
    self.assertIn(constants.BETA_GRF, arrays)
    
    # Check media coefficients array
    beta_gm = arrays[constants.BETA_GM]
    self.assertEqual(beta_gm.dims, ('geo', 'media_channel'))
    self.assertEqual(beta_gm.shape, (3, 3))  # 3 geos, 3 media channels
    self.assertEqual(list(beta_gm.coords['geo'].values), ['Geo0', 'Geo1', 'Geo2'])
    self.assertEqual(list(beta_gm.coords['media_channel'].values), ['Channel0', 'Channel1', 'Channel2'])
    
    # Check R&F coefficients array
    beta_grf = arrays[constants.BETA_GRF]
    self.assertEqual(beta_grf.dims, ('geo', 'rf_channel'))
    self.assertEqual(beta_grf.shape, (3, 1))  # 3 geos, 1 rf channel
    self.assertEqual(list(beta_grf.coords['geo'].values), ['Geo0', 'Geo1', 'Geo2'])
    self.assertEqual(list(beta_grf.coords['rf_channel'].values), ['Channel3'])

  def test_get_coefficients_data_arrays_values(self):
    """Test coefficients DataArray values are correct."""
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=self.sample_coefficients,
      data_df=self.sample_data
    )
    
    arrays = loader.get_coefficients_data_arrays()
    
    # Check media coefficients values
    beta_gm = arrays[constants.BETA_GM]
    expected_media_values = np.array([
      [0.55, 0.32, 0.68],  # Geo0: Channel0, Channel1, Channel2
      [0.45, 0.28, 0.72],  # Geo1: Channel0, Channel1, Channel2
      [0.62, 0.35, 0.65]   # Geo2: Channel0, Channel1, Channel2
    ])
    np.testing.assert_array_equal(beta_gm.values, expected_media_values)
    
    # Check R&F coefficients values
    beta_grf = arrays[constants.BETA_GRF]
    expected_rf_values = np.array([
      [0.42],  # Geo0: Channel3
      [0.38],  # Geo1: Channel3
      [0.45]   # Geo2: Channel3
    ])
    np.testing.assert_array_equal(beta_grf.values, expected_rf_values)

  def test_get_coefficients_data_arrays_media_only(self):
    """Test coefficients DataArrays with media channels only."""
    loader = media_parameter_loader.MediaParameterLoader(
      self.media_only_parameters, 
      self.media_only_config,
      coefficients_df=self.media_only_coefficients,
      data_df=self.sample_data
    )
    
    arrays = loader.get_coefficients_data_arrays()
    
    # Should only have media coefficients
    self.assertIn(constants.BETA_GM, arrays)
    self.assertNotIn(constants.BETA_GRF, arrays)
    
    # Check media coefficients
    beta_gm = arrays[constants.BETA_GM]
    self.assertEqual(beta_gm.shape, (3, 3))  # 3 geos, 3 media channels
    self.assertEqual(list(beta_gm.coords['media_channel'].values), ['Channel0', 'Channel1', 'Channel2'])

  def test_get_coefficients_data_arrays_no_coefficients_df(self):
    """Test coefficients DataArrays with no coefficients DataFrame."""
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config
    )
    
    with self.assertRaises(ValueError) as cm:
      loader.get_coefficients_data_arrays()
      
    self.assertIn('coefficients_df is None', str(cm.exception))

  def test_get_coefficients_data_arrays_caching(self):
    """Test that coefficients DataArrays are cached properly."""
    loader = media_parameter_loader.MediaParameterLoader(
      self.sample_parameters, 
      self.model_config,
      coefficients_df=self.sample_coefficients,
      data_df=self.sample_data
    )
    
    # First call should process and cache
    arrays1 = loader.get_coefficients_data_arrays()
    
    # Second call should return cached result
    arrays2 = loader.get_coefficients_data_arrays()
    
    # Should be the same objects (cached)
    self.assertIs(arrays1[constants.BETA_GM], arrays2[constants.BETA_GM])
    self.assertIs(arrays1[constants.BETA_GRF], arrays2[constants.BETA_GRF])


if __name__ == '__main__':
  absltest.main()