"""Tests for ROIToCoefficientsConverter."""

import tempfile
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf

from meridian.planner import roi_to_coefficients_converter
from meridian.data import input_data
from meridian.model import model
from meridian.model import spec
from meridian import constants


class ROIToCoefficientsConverterTest(parameterized.TestCase):
  
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
      'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
      'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
      'media_channels': ['Channel0', 'Channel1', 'Channel2'],
      'reach_cols': ['Channel3_reach'],
      'frequency_cols': ['Channel3_frequency'], 
      'rf_spend_cols': ['Channel3_spend'],
      'rf_channels': ['Channel3']
    }
    
    # Sample ROI data
    self.sample_roi_data = pd.DataFrame({
      'geo': ['Geo0', 'Geo1', 'Geo2'],
      'Channel0': [2.5, 3.2, 2.8],  # ROI values
      'Channel1': [1.8, 2.1, 1.9],
      'Channel2': [3.1, 2.7, 3.4],
      'Channel3': [4.2, 3.8, 4.5]   # RF channel ROI
    })
    
    # Sample parameters data
    self.sample_parameters = pd.DataFrame({
      'MediaVariable': ['Channel0', 'Channel1', 'Channel2', 'Channel3'],
      'Adstock': [0.51, 0.29, 0.17, 0.6],
      'Inflexion': [1.53, 1.23, 1.16, 1.4],
      'Slope': [1.0, 1.0, 1.0, 3.0]
    })
    
    # Create mock InputData object
    self.mock_input_data = self._create_mock_input_data()
    
  def _create_mock_input_data(self):
    """Create a mock InputData object for testing."""
    # Create mock data with proper dimensions
    n_geos, n_times, n_media_channels = 3, 52, 3  # 3 geos, 52 weeks, 3 media channels
    n_rf_channels = 1
    
    # Create mock arrays
    mock_input_data = mock.MagicMock(spec=input_data.InputData)
    
    # Set up geo coordinates
    mock_input_data.geo = xr.DataArray(['Geo0', 'Geo1', 'Geo2'], dims='geo')
    
    # Set up population data
    mock_input_data.population = xr.DataArray([10000, 15000, 12000], dims='geo')
    
    # Set up revenue per kpi (optional)
    mock_input_data.revenue_per_kpi = xr.DataArray(
        np.ones((n_geos, n_times)), 
        dims=['geo', 'time'],
        coords={'geo': ['Geo0', 'Geo1', 'Geo2'], 'time': range(n_times)}
    )
    
    # Set up media spend data
    media_spend_data = np.random.uniform(1000, 5000, (n_geos, n_times, n_media_channels))
    mock_input_data.media_spend = xr.DataArray(
        media_spend_data,
        dims=['geo', 'time', 'media_channel'],
        coords={
            'geo': ['Geo0', 'Geo1', 'Geo2'],
            'time': range(n_times),
            'media_channel': ['Channel0', 'Channel1', 'Channel2']
        }
    )
    
    # Set up RF spend data
    rf_spend_data = np.random.uniform(2000, 8000, (n_geos, n_times, n_rf_channels))
    mock_input_data.rf_spend = xr.DataArray(
        rf_spend_data,
        dims=['geo', 'time', 'rf_channel'],
        coords={
            'geo': ['Geo0', 'Geo1', 'Geo2'],
            'time': range(n_times),
            'rf_channel': ['Channel3']
        }
    )
    
    return mock_input_data

  def test_initialization_success(self):
    """Test successful initialization of ROIToCoefficientsConverter."""
    converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
        roi_df=self.sample_roi_data,
        parameters_df=self.sample_parameters,
        input_data_obj=self.mock_input_data,
        model_config=self.model_config
    )
    
    self.assertIsNotNone(converter)
    self.assertEqual(converter.media_channels, ['Channel0', 'Channel1', 'Channel2'])
    self.assertEqual(converter.rf_channels, ['Channel3'])
    self.assertIsNone(converter.converted_coefficients_df)

  def test_initialization_empty_roi_data(self):
    """Test initialization fails with empty ROI data."""
    empty_roi_df = pd.DataFrame()
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=empty_roi_df,
          parameters_df=self.sample_parameters,
          input_data_obj=self.mock_input_data,
          model_config=self.model_config
      )
    
    self.assertIn("ROI DataFrame is None or empty", str(cm.exception))

  def test_initialization_missing_roi_columns(self):
    """Test initialization fails when ROI data is missing required columns."""
    incomplete_roi_df = pd.DataFrame({
        'geo': ['Geo0', 'Geo1', 'Geo2'],
        'Channel0': [2.5, 3.2, 2.8],
        # Missing Channel1, Channel2, Channel3
    })
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=incomplete_roi_df,
          parameters_df=self.sample_parameters,
          input_data_obj=self.mock_input_data,
          model_config=self.model_config
      )
    
    self.assertIn("Missing required columns in ROI data", str(cm.exception))

  def test_validation_negative_roi_values(self):
    """Test validation fails with negative ROI values."""
    invalid_roi_df = self.sample_roi_data.copy()
    invalid_roi_df.loc[0, 'Channel0'] = -1.5  # Negative ROI
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=invalid_roi_df,
          parameters_df=self.sample_parameters,
          input_data_obj=self.mock_input_data,
          model_config=self.model_config
      )
    
    self.assertIn("contains negative values", str(cm.exception))

  def test_validation_nan_roi_values(self):
    """Test validation fails with NaN ROI values."""
    invalid_roi_df = self.sample_roi_data.copy()
    invalid_roi_df.loc[0, 'Channel1'] = np.nan
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=invalid_roi_df,
          parameters_df=self.sample_parameters,
          input_data_obj=self.mock_input_data,
          model_config=self.model_config
      )
    
    self.assertIn("contains NaN values", str(cm.exception))

  def test_validation_invalid_adstock_parameter(self):
    """Test validation fails with invalid adstock parameter values."""
    invalid_params_df = self.sample_parameters.copy()
    invalid_params_df.loc[0, 'Adstock'] = 1.5  # Adstock should be <= 1
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=self.sample_roi_data,
          parameters_df=invalid_params_df,
          input_data_obj=self.mock_input_data,
          model_config=self.model_config
      )
    
    self.assertIn("must be between 0 and 1", str(cm.exception))

  def test_validation_negative_inflexion_parameter(self):
    """Test validation fails with negative inflexion parameter."""
    invalid_params_df = self.sample_parameters.copy()
    invalid_params_df.loc[1, 'Inflexion'] = -0.5
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=self.sample_roi_data,
          parameters_df=invalid_params_df,
          input_data_obj=self.mock_input_data,
          model_config=self.model_config
      )
    
    self.assertIn("must be positive", str(cm.exception))

  def test_validation_missing_channels_in_parameters(self):
    """Test validation fails when parameters are missing for some channels."""
    incomplete_params_df = self.sample_parameters[
        self.sample_parameters['MediaVariable'] != 'Channel2'
    ].copy()
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=self.sample_roi_data,
          parameters_df=incomplete_params_df,
          input_data_obj=self.mock_input_data,
          model_config=self.model_config
      )
    
    self.assertIn("Missing channels in Parameters data", str(cm.exception))

  def test_extract_media_parameters(self):
    """Test media parameter extraction from parameters DataFrame."""
    converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
        roi_df=self.sample_roi_data,
        parameters_df=self.sample_parameters,
        input_data_obj=self.mock_input_data,
        model_config=self.model_config
    )
    
    media_params = converter._extract_media_parameters()
    
    # Check structure
    self.assertIn('alpha_m', media_params)
    self.assertIn('ec_m', media_params)
    self.assertIn('slope_m', media_params)
    
    # Check values (should be in channel order)
    expected_alpha = [0.51, 0.29, 0.17]  # Channel0, Channel1, Channel2
    expected_ec = [1.53, 1.23, 1.16]
    expected_slope = [1.0, 1.0, 1.0]
    
    self.assertEqual(media_params['alpha_m'], expected_alpha)
    self.assertEqual(media_params['ec_m'], expected_ec)
    self.assertEqual(media_params['slope_m'], expected_slope)

  def test_extract_rf_parameters(self):
    """Test RF parameter extraction from parameters DataFrame."""
    converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
        roi_df=self.sample_roi_data,
        parameters_df=self.sample_parameters,
        input_data_obj=self.mock_input_data,
        model_config=self.model_config
    )
    
    rf_params = converter._extract_rf_parameters()
    
    # Check structure
    self.assertIn('alpha_rf', rf_params)
    self.assertIn('ec_rf', rf_params)
    self.assertIn('slope_rf', rf_params)
    
    # Check values
    expected_alpha = [0.6]  # Channel3
    expected_ec = [1.4]
    expected_slope = [3.0]
    
    self.assertEqual(rf_params['alpha_rf'], expected_alpha)
    self.assertEqual(rf_params['ec_rf'], expected_ec)
    self.assertEqual(rf_params['slope_rf'], expected_slope)

  @mock.patch('meridian.model.model.Meridian')
  def test_create_dummy_model_success(self, mock_meridian_class):
    """Test successful creation of dummy Meridian model."""
    mock_model_instance = mock.MagicMock()
    mock_meridian_class.return_value = mock_model_instance
    
    converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
        roi_df=self.sample_roi_data,
        parameters_df=self.sample_parameters,
        input_data_obj=self.mock_input_data,
        model_config=self.model_config
    )
    
    dummy_model = converter._create_dummy_model()
    
    # Verify Meridian was called with correct arguments
    mock_meridian_class.assert_called_once()
    args, kwargs = mock_meridian_class.call_args
    self.assertEqual(kwargs['input_data'], self.mock_input_data)
    self.assertIsInstance(kwargs['model_spec'], spec.ModelSpec)
    
    self.assertEqual(dummy_model, mock_model_instance)

  def test_get_conversion_summary_not_converted(self):
    """Test conversion summary when conversion hasn't been performed."""
    converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
        roi_df=self.sample_roi_data,
        parameters_df=self.sample_parameters,
        input_data_obj=self.mock_input_data,
        model_config=self.model_config
    )
    
    summary = converter.get_conversion_summary()
    
    self.assertEqual(summary['status'], 'not_converted')
    self.assertIn('message', summary)

  def test_media_only_configuration(self):
    """Test converter with media channels only (no RF channels)."""
    media_only_config = self.model_config.copy()
    del media_only_config['reach_cols']
    del media_only_config['frequency_cols']
    del media_only_config['rf_spend_cols']
    del media_only_config['rf_channels']
    
    media_only_roi = self.sample_roi_data[['geo', 'Channel0', 'Channel1', 'Channel2']].copy()
    
    media_only_params = self.sample_parameters[
        self.sample_parameters['MediaVariable'].isin(['Channel0', 'Channel1', 'Channel2'])
    ].copy()
    
    converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
        roi_df=media_only_roi,
        parameters_df=media_only_params,
        input_data_obj=self.mock_input_data,
        model_config=media_only_config
    )
    
    self.assertEqual(converter.media_channels, ['Channel0', 'Channel1', 'Channel2'])
    self.assertEqual(converter.rf_channels, [])

  def test_rf_only_configuration(self):
    """Test converter with RF channels only (no media channels).""" 
    rf_only_config = {
      'time_col': 'week',
      'geo_col': 'geo',
      'population_col': 'population',
      'kpi_type': 'non_revenue',
      'kpi_col': 'conversions',
      'media_channels': [],  # No media channels
      'reach_cols': ['Channel3_reach'],
      'frequency_cols': ['Channel3_frequency'],
      'rf_spend_cols': ['Channel3_spend'],
      'rf_channels': ['Channel3']
    }
    
    rf_only_roi = self.sample_roi_data[['geo', 'Channel3']].copy()
    
    rf_only_params = self.sample_parameters[
        self.sample_parameters['MediaVariable'] == 'Channel3'
    ].copy()
    
    converter = roi_to_coefficients_converter.ROIToCoefficientsConverter(
        roi_df=rf_only_roi,
        parameters_df=rf_only_params,
        input_data_obj=self.mock_input_data,
        model_config=rf_only_config
    )
    
    self.assertEqual(converter.media_channels, [])
    self.assertEqual(converter.rf_channels, ['Channel3'])

  def test_validation_model_config_missing_keys(self):
    """Test validation fails when model_config is missing required keys."""
    incomplete_config = self.model_config.copy()
    del incomplete_config['media_channels']
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=self.sample_roi_data,
          parameters_df=self.sample_parameters,
          input_data_obj=self.mock_input_data,
          model_config=incomplete_config
      )
    
    self.assertIn("Missing required keys in model_config", str(cm.exception))

  def test_validation_mismatched_media_cols_length(self):
    """Test validation fails when media_cols length doesn't match media_channels."""
    invalid_config = self.model_config.copy()
    invalid_config['media_cols'] = ['Channel0_impression', 'Channel1_impression']  # Missing Channel2
    
    with self.assertRaises(ValueError) as cm:
      roi_to_coefficients_converter.ROIToCoefficientsConverter(
          roi_df=self.sample_roi_data,
          parameters_df=self.sample_parameters,
          input_data_obj=self.mock_input_data,
          model_config=invalid_config
      )
    
    self.assertIn("Length of media_cols must match media_channels", str(cm.exception))


if __name__ == '__main__':
  absltest.main()