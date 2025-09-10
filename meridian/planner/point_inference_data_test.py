"""Tests for PointInferenceData."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import xarray as xr
import arviz as az

from meridian.planner import point_inference_data
from meridian import constants


class PointInferenceDataTest(parameterized.TestCase):
  
  def setUp(self):
    """Set up test fixtures."""
    super().setUp()
    
    # Sample parameter arrays (matching MediaParameterLoader output)
    self.parameter_arrays = {
      constants.ALPHA_M: xr.DataArray(
        data=[0.5, 0.3, 0.7],
        dims=['media_channel'],
        coords={'media_channel': ['Channel0', 'Channel1', 'Channel2']},
        name=constants.ALPHA_M
      ),
      constants.EC_M: xr.DataArray(
        data=[1.5, 1.2, 1.8], 
        dims=['media_channel'],
        coords={'media_channel': ['Channel0', 'Channel1', 'Channel2']},
        name=constants.EC_M
      ),
      constants.SLOPE_M: xr.DataArray(
        data=[1.0, 2.0, 1.5],
        dims=['media_channel'], 
        coords={'media_channel': ['Channel0', 'Channel1', 'Channel2']},
        name=constants.SLOPE_M
      ),
      constants.ALPHA_RF: xr.DataArray(
        data=[0.6],
        dims=['rf_channel'],
        coords={'rf_channel': ['Channel3']},
        name=constants.ALPHA_RF
      ),
      constants.EC_RF: xr.DataArray(
        data=[1.4],
        dims=['rf_channel'],
        coords={'rf_channel': ['Channel3']}, 
        name=constants.EC_RF
      ),
      constants.SLOPE_RF: xr.DataArray(
        data=[3.0],
        dims=['rf_channel'],
        coords={'rf_channel': ['Channel3']},
        name=constants.SLOPE_RF
      ),
    }
    
    # Sample coefficient arrays (matching MediaParameterLoader output)
    self.coefficient_arrays = {
      constants.BETA_GM: xr.DataArray(
        data=np.array([
          [0.55, 0.32, 0.68],  # Geo0
          [0.45, 0.28, 0.72],  # Geo1
          [0.62, 0.35, 0.65]   # Geo2
        ]),
        dims=['geo', 'media_channel'],
        coords={
          'geo': ['Geo0', 'Geo1', 'Geo2'],
          'media_channel': ['Channel0', 'Channel1', 'Channel2']
        },
        name=constants.BETA_GM
      ),
      constants.BETA_GRF: xr.DataArray(
        data=np.array([
          [0.42],  # Geo0 
          [0.38],  # Geo1
          [0.45]   # Geo2
        ]),
        dims=['geo', 'rf_channel'],
        coords={
          'geo': ['Geo0', 'Geo1', 'Geo2'],
          'rf_channel': ['Channel3']
        },
        name=constants.BETA_GRF
      ),
    }
    
    # Media-only arrays for testing
    self.media_only_parameters = {
      constants.ALPHA_M: self.parameter_arrays[constants.ALPHA_M],
      constants.EC_M: self.parameter_arrays[constants.EC_M],
      constants.SLOPE_M: self.parameter_arrays[constants.SLOPE_M],
    }
    
    self.media_only_coefficients = {
      constants.BETA_GM: self.coefficient_arrays[constants.BETA_GM]
    }

  def test_init_valid_inputs(self):
    """Test initialization with valid parameter and coefficient arrays."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    # Check that inference data was created
    self.assertIsInstance(point_data.inference_data, az.InferenceData)
    self.assertIsNotNone(point_data.posterior)
    
  def test_init_empty_parameters(self):
    """Test initialization with empty parameter arrays."""
    with self.assertRaises(ValueError) as cm:
      point_inference_data.PointInferenceData({}, self.coefficient_arrays)
      
    self.assertIn('parameter_arrays cannot be empty', str(cm.exception))
    
  def test_init_empty_coefficients(self):
    """Test initialization with empty coefficient arrays."""
    with self.assertRaises(ValueError) as cm:
      point_inference_data.PointInferenceData(self.parameter_arrays, {})
      
    self.assertIn('coefficient_arrays cannot be empty', str(cm.exception))
    
  def test_init_missing_parameter_keys(self):
    """Test initialization with missing required parameter keys."""
    incomplete_params = {constants.ALPHA_M: self.parameter_arrays[constants.ALPHA_M]}
    
    with self.assertRaises(ValueError) as cm:
      point_inference_data.PointInferenceData(incomplete_params, self.coefficient_arrays)
      
    self.assertIn('Missing media parameter arrays', str(cm.exception))
    
  def test_init_missing_coefficient_keys(self):
    """Test initialization with missing required coefficient keys.""" 
    incomplete_coeffs = {}
    
    with self.assertRaises(ValueError) as cm:
      point_inference_data.PointInferenceData(self.parameter_arrays, incomplete_coeffs)
      
    self.assertIn('coefficient_arrays cannot be empty', str(cm.exception))

  def test_reshape_array_basic(self):
    """Test array reshaping adds chain and draw dimensions."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    # Test reshaping alpha_m array
    original_array = self.parameter_arrays[constants.ALPHA_M]
    reshaped_array = point_data._reshape_array(original_array)
    
    # Check dimensions
    expected_dims = ['chain', 'draw', 'media_channel']
    self.assertEqual(reshaped_array.dims, tuple(expected_dims))
    
    # Check shape
    expected_shape = (1, 1, 3)
    self.assertEqual(reshaped_array.shape, expected_shape)
    
    # Check coordinates
    self.assertEqual(reshaped_array.coords['chain'].values.tolist(), [0])
    self.assertEqual(reshaped_array.coords['draw'].values.tolist(), [0])
    self.assertEqual(
      reshaped_array.coords['media_channel'].values.tolist(), 
      ['Channel0', 'Channel1', 'Channel2']
    )
    
    # Check data values are preserved
    np.testing.assert_array_equal(
      reshaped_array.values[0, 0, :], 
      original_array.values
    )

  def test_reshape_array_2d(self):
    """Test array reshaping works with 2D arrays."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    # Test reshaping beta_gm array (2D)
    original_array = self.coefficient_arrays[constants.BETA_GM]
    reshaped_array = point_data._reshape_array(original_array)
    
    # Check dimensions
    expected_dims = ['chain', 'draw', 'geo', 'media_channel']
    self.assertEqual(reshaped_array.dims, tuple(expected_dims))
    
    # Check shape
    expected_shape = (1, 1, 3, 3)
    self.assertEqual(reshaped_array.shape, expected_shape)
    
    # Check data values are preserved
    np.testing.assert_array_equal(
      reshaped_array.values[0, 0, :, :],
      original_array.values
    )

  def test_create_posterior_dataset(self):
    """Test posterior dataset creation."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    posterior = point_data.posterior
    
    # Check that all expected data variables are present
    expected_vars = {
      constants.ALPHA_M, constants.EC_M, constants.SLOPE_M,
      constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF,
      constants.BETA_GM, constants.BETA_GRF
    }
    self.assertEqual(set(posterior.data_vars.keys()), expected_vars)
    
    # Check that all variables have chain and draw as first dimensions
    for var_name, data_var in posterior.data_vars.items():
      self.assertEqual(data_var.dims[0], 'chain')
      self.assertEqual(data_var.dims[1], 'draw')
      self.assertEqual(data_var.sizes['chain'], 1)
      self.assertEqual(data_var.sizes['draw'], 1)
      
    # Check data types are float32
    for data_var in posterior.data_vars.values():
      self.assertEqual(data_var.dtype, np.float32)

  def test_create_inference_data_structure(self):
    """Test InferenceData structure and groups."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    inference_data = point_data.get_inference_data()
    
    # Check it's an ArviZ InferenceData object
    self.assertIsInstance(inference_data, az.InferenceData)
    
    # Check groups
    expected_groups = ['posterior', 'sample_stats']
    self.assertEqual(sorted(inference_data.groups()), expected_groups)
    
    # Check posterior group
    self.assertIn('posterior', inference_data.groups())
    posterior = inference_data.posterior
    self.assertIsInstance(posterior, xr.Dataset)
    
    # Check sample_stats group
    self.assertIn('sample_stats', inference_data.groups())
    sample_stats = inference_data.sample_stats
    self.assertIsInstance(sample_stats, xr.Dataset)
    self.assertIn('diverging', sample_stats.data_vars)
    self.assertIn('energy', sample_stats.data_vars)

  def test_inference_data_dimensions(self):
    """Test that InferenceData has correct dimensions and coordinates.""" 
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    posterior = point_data.posterior
    
    # Check expected dimensions exist
    expected_dims = ['chain', 'draw', 'media_channel', 'rf_channel', 'geo']
    for dim in expected_dims:
      self.assertIn(dim, posterior.dims)
      
    # Check dimension sizes
    self.assertEqual(posterior.sizes['chain'], 1)
    self.assertEqual(posterior.sizes['draw'], 1)
    self.assertEqual(posterior.sizes['media_channel'], 3)
    self.assertEqual(posterior.sizes['rf_channel'], 1)
    self.assertEqual(posterior.sizes['geo'], 3)
    
    # Check coordinate values
    self.assertEqual(posterior.coords['chain'].values.tolist(), [0])
    self.assertEqual(posterior.coords['draw'].values.tolist(), [0])
    self.assertEqual(
      posterior.coords['media_channel'].values.tolist(),
      ['Channel0', 'Channel1', 'Channel2']
    )
    self.assertEqual(
      posterior.coords['rf_channel'].values.tolist(),
      ['Channel3']
    )
    self.assertEqual(
      posterior.coords['geo'].values.tolist(), 
      ['Geo0', 'Geo1', 'Geo2']
    )

  def test_data_values_preservation(self):
    """Test that original data values are preserved correctly."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    posterior = point_data.posterior
    
    # Check alpha_m values
    alpha_m_values = posterior[constants.ALPHA_M].values[0, 0, :]
    expected_alpha_m = self.parameter_arrays[constants.ALPHA_M].values.astype(np.float32)
    np.testing.assert_array_equal(alpha_m_values, expected_alpha_m)
    
    # Check beta_gm values
    beta_gm_values = posterior[constants.BETA_GM].values[0, 0, :, :]
    expected_beta_gm = self.coefficient_arrays[constants.BETA_GM].values.astype(np.float32)
    np.testing.assert_array_equal(beta_gm_values, expected_beta_gm)

  def test_media_only_configuration(self):
    """Test with media-only configuration (no R&F channels)."""
    point_data = point_inference_data.PointInferenceData(
      self.media_only_parameters, self.media_only_coefficients
    )
    
    posterior = point_data.posterior
    
    # Check media variables are present
    media_vars = {constants.ALPHA_M, constants.EC_M, constants.SLOPE_M, constants.BETA_GM}
    for var in media_vars:
      self.assertIn(var, posterior.data_vars)
      
    # Check R&F variables are not present
    rf_vars = {constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF, constants.BETA_GRF}
    for var in rf_vars:
      self.assertNotIn(var, posterior.data_vars)

  def test_repr_method(self):
    """Test string representation."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    repr_str = repr(point_data)
    
    # Check that it contains expected information
    self.assertIn('PointInferenceData', repr_str)
    self.assertIn('variables', repr_str)
    self.assertIn('posterior', repr_str)
    self.assertIn('sample_stats', repr_str)

  def test_get_inference_data_method(self):
    """Test get_inference_data method returns proper object."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    inference_data = point_data.get_inference_data()
    
    # Should be same object as internal inference_data
    self.assertIs(inference_data, point_data.inference_data)
    self.assertIsInstance(inference_data, az.InferenceData)

  def test_posterior_property(self):
    """Test posterior property access."""
    point_data = point_inference_data.PointInferenceData(
      self.parameter_arrays, self.coefficient_arrays
    )
    
    posterior = point_data.posterior
    
    # Should be same as inference_data.posterior
    self.assertIs(posterior, point_data.inference_data.posterior)
    self.assertIsInstance(posterior, xr.Dataset)


if __name__ == '__main__':
  absltest.main()