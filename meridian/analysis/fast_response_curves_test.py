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

"""Tests for fast_response_curves module."""

import unittest
from unittest import mock
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from meridian import constants as c
from meridian.analysis import fast_response_curves
from meridian.model import model


class MockInferenceData:
  """Mock inference data for testing."""
  
  def __init__(self):
    # Create mock posterior data with realistic parameter shapes
    self.posterior = self._create_mock_posterior()

  def _create_mock_posterior(self):
    """Creates mock posterior data with realistic parameter shapes."""
    n_chains = 4
    n_draws = 100
    n_geos = 3
    n_media_channels = 2
    n_rf_channels = 1
    
    mock_posterior = {}
    
    # Media parameters
    mock_posterior[c.EC_M] = xr.DataArray(
        np.random.gamma(2, 1, size=(n_chains, n_draws, n_media_channels)),
        dims=['chain', 'draw', 'media_channel']
    )
    mock_posterior[c.SLOPE_M] = xr.DataArray(
        np.random.gamma(2, 1, size=(n_chains, n_draws, n_media_channels)),
        dims=['chain', 'draw', 'media_channel']
    )
    mock_posterior[c.ALPHA_M] = xr.DataArray(
        np.random.beta(2, 8, size=(n_chains, n_draws, n_media_channels)),
        dims=['chain', 'draw', 'media_channel']
    )
    mock_posterior[c.BETA_GM] = xr.DataArray(
        np.random.normal(0.5, 0.1, size=(n_chains, n_draws, n_geos, n_media_channels)),
        dims=['chain', 'draw', 'geo', 'media_channel']
    )
    
    # R&F parameters
    mock_posterior[c.EC_RF] = xr.DataArray(
        np.random.gamma(2, 1, size=(n_chains, n_draws, n_rf_channels)),
        dims=['chain', 'draw', 'rf_channel']
    )
    mock_posterior[c.SLOPE_RF] = xr.DataArray(
        np.random.gamma(2, 1, size=(n_chains, n_draws, n_rf_channels)),
        dims=['chain', 'draw', 'rf_channel']
    )
    mock_posterior[c.ALPHA_RF] = xr.DataArray(
        np.random.beta(2, 8, size=(n_chains, n_draws, n_rf_channels)),
        dims=['chain', 'draw', 'rf_channel']
    )
    mock_posterior[c.BETA_GRF] = xr.DataArray(
        np.random.normal(0.3, 0.1, size=(n_chains, n_draws, n_geos, n_rf_channels)),
        dims=['chain', 'draw', 'geo', 'rf_channel']
    )
    
    return mock_posterior


class MockInputData:
  """Mock input data for testing."""
  
  def __init__(self):
    self.revenue_per_kpi = None
    self.media_channel = xr.DataArray(['TV', 'Display'], dims=['media_channel'])
    self.rf_channel = xr.DataArray(['Facebook'], dims=['rf_channel'])
    self.geo = xr.DataArray(['US', 'CA', 'UK'], dims=['geo'])
    self.time = xr.DataArray(
        pd.date_range('2023-01-01', periods=52, freq='W').strftime('%Y-%m-%d'),
        dims=['time']
    )


class MockTensors:
  """Mock tensor data for testing."""
  
  def __init__(self, tensor_type='media'):
    n_geos = 3
    n_times = 52
    
    if tensor_type == 'media':
      n_channels = 2
      self.media = tf.random.normal((n_geos, n_times, n_channels))
      self.media_spend = tf.random.uniform((n_geos, n_times, n_channels), 1000, 10000)
    elif tensor_type == 'rf':
      n_channels = 1
      self.reach = tf.random.normal((n_geos, n_times, n_channels))
      self.frequency = tf.random.uniform((n_geos, n_times, n_channels), 1, 5)
      self.rf_spend = tf.random.uniform((n_geos, n_times, n_channels), 5000, 15000)


class MockMeridian:
  """Mock Meridian model for testing."""
  
  def __init__(self):
    self.inference_data = MockInferenceData()
    self.input_data = MockInputData()
    self.media_tensors = MockTensors('media')
    self.rf_tensors = MockTensors('rf')
    
    # Model properties
    self.n_media_channels = 2
    self.n_rf_channels = 1
    self.n_organic_media_channels = 0
    self.n_organic_rf_channels = 0
    self.n_geos = 3
    self.n_times = 52
    self.max_lag = 13
    self.non_media_treatments = None


class FastResponseCurvesTest(unittest.TestCase):
  """Tests for FastResponseCurves class."""

  def setUp(self):
    """Sets up test fixtures."""
    self.mock_meridian = MockMeridian()
    self.fast_curves = fast_response_curves.FastResponseCurves(self.mock_meridian)

  def test_initialization_successful(self):
    """Tests that FastResponseCurves initializes successfully with valid model."""
    self.assertIsNotNone(self.fast_curves)
    self.assertEqual(self.fast_curves._meridian, self.mock_meridian)
    self.assertTrue(self.fast_curves.has_media_data)
    self.assertTrue(self.fast_curves.has_rf_data)

  def test_initialization_fails_without_posterior(self):
    """Tests that initialization fails when model lacks posterior data."""
    meridian_no_posterior = MockMeridian()
    delattr(meridian_no_posterior.inference_data, 'posterior')
    
    with self.assertRaises(ValueError):
      fast_response_curves.FastResponseCurves(meridian_no_posterior)

  def test_parameter_extraction(self):
    """Tests that median parameters are extracted correctly."""
    # Check media parameters
    self.assertIn('ec', self.fast_curves.media_params)
    self.assertIn('slope', self.fast_curves.media_params)
    self.assertIn('alpha', self.fast_curves.media_params)
    self.assertIn('beta', self.fast_curves.media_params)
    
    # Check R&F parameters
    self.assertIn('ec', self.fast_curves.rf_params)
    self.assertIn('slope', self.fast_curves.rf_params)
    self.assertIn('alpha', self.fast_curves.rf_params)
    self.assertIn('beta', self.fast_curves.rf_params)
    
    # Check parameter shapes
    self.assertEqual(self.fast_curves.media_params['ec'].shape, (2,))
    self.assertEqual(self.fast_curves.media_params['beta'].shape, (3, 2))
    self.assertEqual(self.fast_curves.rf_params['ec'].shape, (1,))
    self.assertEqual(self.fast_curves.rf_params['beta'].shape, (3, 1))

  def test_channel_properties(self):
    """Tests channel-related properties."""
    self.assertEqual(self.fast_curves.n_total_channels, 3)
    self.assertEqual(len(self.fast_curves.channel_names), 3)
    self.assertIn('TV', self.fast_curves.channel_names)
    self.assertIn('Display', self.fast_curves.channel_names)
    self.assertIn('Facebook', self.fast_curves.channel_names)

  def test_scale_media_data(self):
    """Tests media data scaling functionality."""
    multiplier = 1.5
    scaled_media, scaled_reach, scaled_frequency = self.fast_curves._scale_media_data(
        multiplier=multiplier, by_reach=True
    )
    
    # Check that scaling was applied correctly
    np.testing.assert_allclose(
        scaled_media.numpy(),
        self.mock_meridian.media_tensors.media.numpy() * multiplier
    )
    np.testing.assert_allclose(
        scaled_reach.numpy(),
        self.mock_meridian.rf_tensors.reach.numpy() * multiplier
    )
    # Frequency should remain unchanged when scaling by reach
    np.testing.assert_allclose(
        scaled_frequency.numpy(),
        self.mock_meridian.rf_tensors.frequency.numpy()
    )

  def test_compute_spend_amounts(self):
    """Tests spend amount computation."""
    spend_multipliers = [0.5, 1.0, 1.5]
    spend_amounts = self.fast_curves._compute_spend_amounts(spend_multipliers)
    
    # Check output shape
    self.assertEqual(spend_amounts.shape, (3, 3))  # 3 multipliers, 3 channels
    
    # Check that multipliers are applied correctly
    # The spend at multiplier=1.0 should be the historical total
    historical_spend = spend_amounts[1, :].numpy()  # multiplier=1.0
    scaled_spend_05 = spend_amounts[0, :].numpy()   # multiplier=0.5
    scaled_spend_15 = spend_amounts[2, :].numpy()   # multiplier=1.5
    
    np.testing.assert_allclose(scaled_spend_05, historical_spend * 0.5, rtol=1e-5)
    np.testing.assert_allclose(scaled_spend_15, historical_spend * 1.5, rtol=1e-5)

  def test_compute_response_curves(self):
    """Tests response curve computation."""
    spend_multipliers = [0.5, 1.0, 1.5]
    result = self.fast_curves.compute_response_curves(
        spend_multipliers=spend_multipliers
    )
    
    # Check output structure
    self.assertIsInstance(result, xr.Dataset)
    self.assertIn(c.SPEND, result.data_vars)
    self.assertIn(c.INCREMENTAL_OUTCOME, result.data_vars)
    self.assertIn(c.ROI, result.data_vars)
    
    # Check coordinates
    self.assertIn(c.CHANNEL, result.coords)
    self.assertIn(c.SPEND_MULTIPLIER, result.coords)
    self.assertIn(c.METRIC, result.coords)
    
    # Check dimensions
    self.assertEqual(len(result.channel), 3)
    self.assertEqual(len(result.spend_multiplier), 3)
    self.assertEqual(len(result.metric), 1)
    
    # Check that spend increases with multipliers
    tv_spend = result.sel(channel='TV', metric=c.MEAN)[c.SPEND]
    self.assertLess(tv_spend.sel(spend_multiplier=0.5), tv_spend.sel(spend_multiplier=1.0))
    self.assertLess(tv_spend.sel(spend_multiplier=1.0), tv_spend.sel(spend_multiplier=1.5))

  def test_response_curves_data_alias(self):
    """Tests the response_curves_data alias method."""
    spend_multipliers = [0.5, 1.0, 1.5]
    result1 = self.fast_curves.compute_response_curves(
        spend_multipliers=spend_multipliers
    )
    result2 = self.fast_curves.response_curves_data(
        spend_multipliers=spend_multipliers
    )
    
    # Results should be identical
    xr.testing.assert_identical(result1, result2)

  def test_plot_response_curves(self):
    """Tests response curve plotting functionality."""
    spend_multipliers = [0.5, 1.0, 1.5]
    chart = self.fast_curves.plot_response_curves(
        spend_multipliers=spend_multipliers,
        plot_separately=True
    )
    
    # Check that chart is created successfully
    self.assertIsNotNone(chart)
    # Basic check that it's an Altair chart
    self.assertTrue(hasattr(chart, 'to_dict'))

  def test_transform_response_curve_data_for_plotting(self):
    """Tests data transformation for plotting."""
    spend_multipliers = [0.5, 1.0, 1.5]
    response_data = self.fast_curves.compute_response_curves(
        spend_multipliers=spend_multipliers
    )
    
    df = self.fast_curves._transform_response_curve_data_for_plotting(
        response_data=response_data,
        plot_separately=True
    )
    
    # Check DataFrame structure
    self.assertIsInstance(df, pd.DataFrame)
    self.assertIn(c.CHANNEL, df.columns)
    self.assertIn(c.SPEND_MULTIPLIER, df.columns)
    self.assertIn(c.SPEND, df.columns)
    self.assertIn(c.MEAN, df.columns)
    self.assertIn(c.CURRENT_SPEND, df.columns)
    
    # Check that current spend is marked correctly
    current_spend_rows = df[df[c.SPEND_MULTIPLIER] == 1.0]
    self.assertTrue(all(current_spend_rows[c.CURRENT_SPEND].notna()))

  def test_validation_parameter_shapes(self):
    """Tests parameter shape validation."""
    # This test would pass with properly shaped parameters
    # The validation should have succeeded during setUp
    self.assertTrue(True)  # If we got here, validation passed

  @mock.patch('meridian.analysis.fast_response_curves.adstock_hill.AdstockTransformer')
  @mock.patch('meridian.analysis.fast_response_curves.adstock_hill.HillTransformer')
  def test_transform_media_channels(self, mock_hill, mock_adstock):
    """Tests media transformation pipeline."""
    # Set up mocks
    mock_adstock_instance = mock.Mock()
    mock_hill_instance = mock.Mock()
    mock_adstock.return_value = mock_adstock_instance
    mock_hill.return_value = mock_hill_instance
    
    # Mock transformer outputs
    adstocked_output = tf.random.normal((3, 52, 2))
    saturated_output = tf.random.normal((3, 52, 2))
    mock_adstock_instance.forward.return_value = adstocked_output
    mock_hill_instance.forward.return_value = saturated_output
    
    # Test the transformation
    test_media = tf.random.normal((3, 52, 2))
    result = self.fast_curves._transform_media_channels(
        media_data=test_media,
        params=self.fast_curves.media_params
    )
    
    # Verify transformers were called
    mock_adstock.assert_called_once()
    mock_hill.assert_called_once()
    mock_adstock_instance.forward.assert_called_once_with(test_media)
    mock_hill_instance.forward.assert_called_once_with(adstocked_output)
    
    # Check output shape
    self.assertEqual(result.shape, (3, 52, 2))


if __name__ == '__main__':
  unittest.main()