"""Tests for ModelDataDeepDive class.

This test suite validates deep-dive analysis functionality for understanding
geo performance differences in Meridian models.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from meridian.model import model
from meridian.data import input_data
from meridian.planner import model_data_deepdive
from meridian.planner.model_data_deepdive import ModelDataDeepDive


class ModelDataDeepDiveTest(unittest.TestCase):
  """Test suite for ModelDataDeepDive class."""

  def setUp(self):
    """Set up test fixtures with mock Meridian model."""
    # Create mock input data dimensions
    self.n_geos = 3
    self.n_times = 52
    self.n_media_channels = 2

    self.geo_names = ['GEO_A', 'GEO_B', 'GEO_C']
    self.media_channel_names = ['TV', 'Display']

    # Create realistic time-series data
    np.random.seed(42)

    # Media data (impressions): shape (n_times, n_geos, n_media_channels)
    # GEO_A: High TV impressions with positive correlation to KPI
    # GEO_B: Low TV impressions with weak correlation
    # GEO_C: Medium TV impressions with negative correlation
    media_data = np.zeros((self.n_times, self.n_geos, self.n_media_channels))

    # TV channel (index 0)
    media_data[:, 0, 0] = 100000 + 50000 * np.sin(np.linspace(0, 4*np.pi, self.n_times)) + np.random.normal(0, 5000, self.n_times)  # GEO_A: high baseline
    media_data[:, 1, 0] = 30000 + 15000 * np.sin(np.linspace(0, 4*np.pi, self.n_times)) + np.random.normal(0, 2000, self.n_times)   # GEO_B: low baseline
    media_data[:, 2, 0] = 60000 + 30000 * np.sin(np.linspace(0, 4*np.pi, self.n_times)) + np.random.normal(0, 3000, self.n_times)   # GEO_C: medium baseline

    # Display channel (index 1)
    media_data[:, 0, 1] = 80000 + 40000 * np.sin(np.linspace(0, 4*np.pi, self.n_times)) + np.random.normal(0, 4000, self.n_times)
    media_data[:, 1, 1] = 25000 + 12000 * np.sin(np.linspace(0, 4*np.pi, self.n_times)) + np.random.normal(0, 1500, self.n_times)
    media_data[:, 2, 1] = 50000 + 25000 * np.sin(np.linspace(0, 4*np.pi, self.n_times)) + np.random.normal(0, 2500, self.n_times)

    # KPI data: shape (n_times, n_geos)
    # Create correlated KPI with TV for GEO_A, weakly correlated for GEO_B, negatively correlated for GEO_C
    kpi_data = np.zeros((self.n_times, self.n_geos))
    kpi_data[:, 0] = 500 + 0.003 * media_data[:, 0, 0] + np.random.normal(0, 50, self.n_times)  # GEO_A: positive correlation
    kpi_data[:, 1] = 200 + 0.0005 * media_data[:, 1, 0] + np.random.normal(0, 30, self.n_times)  # GEO_B: weak correlation
    kpi_data[:, 2] = 400 - 0.002 * media_data[:, 2, 0] + np.random.normal(0, 40, self.n_times)  # GEO_C: negative correlation

    # Population data: shape (n_geos,)
    population_data = np.array([1000000, 500000, 750000])

    # Create xarray DataArrays with proper coordinates
    self.media_array = xr.DataArray(
        media_data,
        dims=['media_time', 'geo', 'media_channel'],
        coords={
            'media_time': np.arange(self.n_times),
            'geo': self.geo_names,
            'media_channel': self.media_channel_names,
        },
        name='media'
    )

    self.kpi_array = xr.DataArray(
        kpi_data,
        dims=['time', 'geo'],
        coords={
            'time': np.arange(self.n_times),
            'geo': self.geo_names,
        },
        name='kpi'
    )

    self.population_array = xr.DataArray(
        population_data,
        dims=['geo'],
        coords={'geo': self.geo_names},
        name='population'
    )

    # Create mock InputData object
    self.mock_input_data = MagicMock(spec=input_data.InputData)
    self.mock_input_data.media = self.media_array
    self.mock_input_data.kpi = self.kpi_array
    self.mock_input_data.population = self.population_array
    self.mock_input_data.geo = self.media_array.geo  # Add geo attribute
    self.mock_input_data.get_all_paid_channels = MagicMock(return_value=self.media_channel_names)

    # Create mock Meridian model
    self.mock_meridian = MagicMock(spec=model.Meridian)
    self.mock_meridian.input_data = self.mock_input_data

    # Create ModelDataDeepDive instance
    self.deepdive = ModelDataDeepDive(self.mock_meridian)

  def test_initialization_success(self):
    """Test successful initialization with valid Meridian model."""
    self.assertIsInstance(self.deepdive, ModelDataDeepDive)
    self.assertEqual(self.deepdive.meridian, self.mock_meridian)
    self.assertEqual(self.deepdive.input_data, self.mock_input_data)

  def test_initialization_with_none_model(self):
    """Test initialization fails with None model."""
    with self.assertRaises(ValueError) as context:
      ModelDataDeepDive(None)
    self.assertIn('meridian_model cannot be None', str(context.exception))

  def test_initialization_with_invalid_model(self):
    """Test initialization fails with invalid model type."""
    with self.assertRaises(ValueError) as context:
      ModelDataDeepDive('not_a_model')
    self.assertIn('must have input_data attribute', str(context.exception))

  def test_get_channel_timeseries_success(self):
    """Test extracting channel time-series data successfully."""
    ts = self.deepdive._get_channel_timeseries('GEO_A', 'TV')

    # Check return type and length
    self.assertIsInstance(ts, pd.Series)
    self.assertEqual(len(ts), self.n_times)

    # Check values match original data
    expected_values = self.media_array.sel(geo='GEO_A', media_channel='TV').values
    np.testing.assert_array_almost_equal(ts.values, expected_values)

    # Check series name
    self.assertEqual(ts.name, 'GEO_A_TV')

  def test_get_channel_timeseries_invalid_geo(self):
    """Test error handling for invalid geo."""
    with self.assertRaises(ValueError):
      self.deepdive._get_channel_timeseries('INVALID_GEO', 'TV')

  def test_get_channel_timeseries_invalid_channel(self):
    """Test error handling for invalid channel."""
    with self.assertRaises(ValueError):
      self.deepdive._get_channel_timeseries('GEO_A', 'INVALID_CHANNEL')

  def test_get_kpi_timeseries_success(self):
    """Test extracting KPI time-series data successfully."""
    ts = self.deepdive._get_kpi_timeseries('GEO_A')

    # Check return type and length
    self.assertIsInstance(ts, pd.Series)
    self.assertEqual(len(ts), self.n_times)

    # Check values match original data
    expected_values = self.kpi_array.sel(geo='GEO_A').values
    np.testing.assert_array_almost_equal(ts.values, expected_values)

    # Check series name
    self.assertEqual(ts.name, 'GEO_A_KPI')

  def test_get_kpi_timeseries_invalid_geo(self):
    """Test error handling for invalid geo in KPI extraction."""
    with self.assertRaises(ValueError):
      self.deepdive._get_kpi_timeseries('INVALID_GEO')

  def test_calculate_correlation_positive(self):
    """Test correlation calculation with positively correlated data."""
    # Create perfectly correlated series
    x = pd.Series(np.arange(100))
    y = pd.Series(2 * np.arange(100) + 5)

    corr = self.deepdive._calculate_correlation(x, y)

    # Should be approximately 1.0
    self.assertAlmostEqual(corr, 1.0, places=5)

  def test_calculate_correlation_negative(self):
    """Test correlation calculation with negatively correlated data."""
    # Create negatively correlated series
    x = pd.Series(np.arange(100))
    y = pd.Series(-3 * np.arange(100) + 100)

    corr = self.deepdive._calculate_correlation(x, y)

    # Should be approximately -1.0
    self.assertAlmostEqual(corr, -1.0, places=5)

  def test_calculate_correlation_zero(self):
    """Test correlation calculation with uncorrelated data."""
    # Create uncorrelated series (random)
    np.random.seed(123)
    x = pd.Series(np.random.normal(0, 1, 1000))
    y = pd.Series(np.random.normal(0, 1, 1000))

    corr = self.deepdive._calculate_correlation(x, y)

    # Should be close to 0 (within statistical noise)
    self.assertLess(abs(corr), 0.1)

  def test_calculate_correlation_constant_series(self):
    """Test correlation with constant series (returns NaN)."""
    x = pd.Series([5, 5, 5, 5, 5])
    y = pd.Series([1, 2, 3, 4, 5])

    corr = self.deepdive._calculate_correlation(x, y)

    # Should return NaN for constant series
    self.assertTrue(np.isnan(corr))

  def test_compare_geo_performance_structure(self):
    """Test compare_geo_performance returns correct structure."""
    result = self.deepdive.compare_geo_performance(
        inspect_geo='GEO_A',
        compare_geo='GEO_B',
        channel='TV'
    )

    # Check top-level keys
    self.assertIn('inspect_geo_stats', result)
    self.assertIn('compare_geo_stats', result)
    self.assertIn('comparison_metrics', result)
    self.assertIn('insights', result)

    # Check inspect_geo_stats structure
    inspect_stats = result['inspect_geo_stats']
    required_stats = [
        'geo', 'channel', 'mean_impressions', 'median_impressions', 'std_impressions',
        'cv_impressions', 'min_impressions', 'max_impressions',
        'correlation_with_kpi', 'zero_weeks', 'total_weeks', 'mean_kpi'
    ]
    for stat in required_stats:
      self.assertIn(stat, inspect_stats)

    # Check comparison_metrics structure
    comparison = result['comparison_metrics']
    required_comparisons = [
        'mean_impressions_ratio', 'mean_impressions_diff_pct',
        'correlation_diff', 'zero_weeks_diff', 'cv_ratio'
    ]
    for comp in required_comparisons:
      self.assertIn(comp, comparison)

    # Check insights is a list
    self.assertIsInstance(result['insights'], list)

  def test_compare_geo_performance_calculations(self):
    """Test compare_geo_performance calculates metrics correctly."""
    result = self.deepdive.compare_geo_performance(
        inspect_geo='GEO_A',
        compare_geo='GEO_B',
        channel='TV'
    )

    inspect_stats = result['inspect_geo_stats']
    compare_stats = result['compare_geo_stats']

    # Verify mean impressions are reasonable
    # GEO_A should have ~100k baseline, GEO_B should have ~30k baseline
    self.assertGreater(inspect_stats['mean_impressions'], 80000)
    self.assertLess(inspect_stats['mean_impressions'], 150000)
    self.assertGreater(compare_stats['mean_impressions'], 20000)
    self.assertLess(compare_stats['mean_impressions'], 50000)

    # Verify total weeks matches
    self.assertEqual(inspect_stats['total_weeks'], self.n_times)
    self.assertEqual(compare_stats['total_weeks'], self.n_times)

    # Verify comparison ratio is positive
    comparison = result['comparison_metrics']
    self.assertGreater(comparison['mean_impressions_ratio'], 0)

  def test_compare_geo_performance_insights_generation(self):
    """Test that insights are generated."""
    result = self.deepdive.compare_geo_performance(
        inspect_geo='GEO_A',
        compare_geo='GEO_C',
        channel='TV'
    )

    insights = result['insights']

    # Should have at least one insight
    self.assertGreater(len(insights), 0)

    # Each insight should be a non-empty string
    for insight in insights:
      self.assertIsInstance(insight, str)
      self.assertGreater(len(insight), 0)

  def test_compare_geo_performance_different_channels(self):
    """Test comparison works for different channels."""
    result_tv = self.deepdive.compare_geo_performance('GEO_A', 'GEO_B', 'TV')
    result_display = self.deepdive.compare_geo_performance('GEO_A', 'GEO_B', 'Display')

    # Results should differ between channels
    self.assertNotEqual(
        result_tv['inspect_geo_stats']['mean_impressions'],
        result_display['inspect_geo_stats']['mean_impressions']
    )

  def test_compare_geo_performance_same_geo(self):
    """Test comparison with same geo (edge case)."""
    result = self.deepdive.compare_geo_performance('GEO_A', 'GEO_A', 'TV')

    # Ratio should be 1.0 when comparing same geo
    self.assertAlmostEqual(result['comparison_metrics']['mean_impressions_ratio'], 1.0, places=5)
    self.assertAlmostEqual(result['comparison_metrics']['correlation_diff'], 0.0, places=5)
    self.assertAlmostEqual(result['comparison_metrics']['zero_weeks_diff'], 0.0, places=5)

  @patch('matplotlib.pyplot.show')
  def test_plot_time_series_comparison_returns_figure(self, mock_show):
    """Test that plot_time_series_comparison returns a matplotlib Figure."""
    fig = self.deepdive.plot_time_series_comparison(
        inspect_geo='GEO_A',
        compare_geo='GEO_B',
        channel='TV'
    )

    # Check return type
    self.assertIsInstance(fig, plt.Figure)

    # Check figure has correct number of axes (2x2 grid = 4 subplots)
    axes = fig.get_axes()
    self.assertEqual(len(axes), 4)

    plt.close(fig)

  @patch('matplotlib.pyplot.show')
  def test_plot_time_series_comparison_subplot_titles(self, mock_show):
    """Test that plot has correct subplot titles."""
    fig = self.deepdive.plot_time_series_comparison(
        inspect_geo='GEO_A',
        compare_geo='GEO_C',
        channel='TV'
    )

    axes = fig.get_axes()

    # Check titles contain expected text
    titles = [ax.get_title() for ax in axes]

    # Plot 1: Impressions comparison
    self.assertIn('1.', titles[0])
    self.assertIn('TV', titles[0])
    self.assertIn('GEO_A', titles[0])
    self.assertIn('GEO_C', titles[0])

    # Plot 2: Inspect geo scatter
    self.assertIn('2.', titles[1])
    self.assertIn('GEO_A', titles[1])

    # Plot 3: Compare geo scatter
    self.assertIn('3.', titles[2])
    self.assertIn('GEO_C', titles[2])

    # Plot 4: KPI comparison
    self.assertIn('4.', titles[3])
    self.assertIn('GEO_A', titles[3])
    self.assertIn('GEO_C', titles[3])

    plt.close(fig)

  @patch('matplotlib.pyplot.show')
  def test_plot_time_series_comparison_data_plotted(self, mock_show):
    """Test that actual data is plotted in subplots."""
    fig = self.deepdive.plot_time_series_comparison(
        inspect_geo='GEO_A',
        compare_geo='GEO_B',
        channel='TV'
    )

    axes = fig.get_axes()

    # Plot 1 (time series): Should have 2 lines (both geos)
    lines_plot1 = axes[0].get_lines()
    self.assertGreaterEqual(len(lines_plot1), 2)

    # Plot 2 (scatter): Should have scatter points
    collections_plot2 = axes[1].collections
    self.assertGreater(len(collections_plot2), 0)

    # Plot 3 (scatter): Should have scatter points
    collections_plot3 = axes[2].collections
    self.assertGreater(len(collections_plot3), 0)

    # Plot 4 (time series): Should have 2 lines (both geos KPI)
    lines_plot4 = axes[3].get_lines()
    self.assertGreaterEqual(len(lines_plot4), 2)

    plt.close(fig)

  @patch('matplotlib.pyplot.show')
  def test_plot_time_series_comparison_can_be_saved(self, mock_show):
    """Test that plot figure can be saved manually."""
    fig = self.deepdive.plot_time_series_comparison(
        inspect_geo='GEO_A',
        compare_geo='GEO_B',
        channel='TV'
    )

    # Verify figure can be saved (though we won't actually save it)
    self.assertIsInstance(fig, plt.Figure)
    # User can call fig.savefig('/path/to/file.png') manually

    plt.close(fig)

  def test_compare_geo_performance_with_zero_impressions(self):
    """Test handling of geos with zero impressions in some weeks."""
    # Modify data to have zero weeks
    modified_media = self.media_array.copy()
    modified_media.loc[{'geo': 'GEO_B', 'media_channel': 'TV'}][:5] = 0

    self.mock_input_data.media = modified_media
    self.mock_input_data.geo = modified_media.geo  # Update geo attribute
    deepdive = ModelDataDeepDive(self.mock_meridian)

    result = deepdive.compare_geo_performance('GEO_B', 'GEO_A', 'TV')

    # Should handle zero weeks gracefully
    self.assertGreaterEqual(result['compare_geo_stats']['zero_weeks'], 0)
    self.assertIsInstance(result['compare_geo_stats']['correlation_with_kpi'], (float, int))

  def test_compare_geo_performance_correlation_ranges(self):
    """Test that correlations are within valid range [-1, 1]."""
    result = self.deepdive.compare_geo_performance('GEO_A', 'GEO_C', 'TV')

    inspect_corr = result['inspect_geo_stats']['correlation_with_kpi']
    compare_corr = result['compare_geo_stats']['correlation_with_kpi']

    # Check correlation ranges (allowing NaN)
    if not np.isnan(inspect_corr):
      self.assertGreaterEqual(inspect_corr, -1.0)
      self.assertLessEqual(inspect_corr, 1.0)

    if not np.isnan(compare_corr):
      self.assertGreaterEqual(compare_corr, -1.0)
      self.assertLessEqual(compare_corr, 1.0)

  def test_get_available_geos(self):
    """Test retrieving available geo names from model."""
    geos = list(self.deepdive.input_data.geo.values)

    self.assertEqual(geos, self.geo_names)
    self.assertEqual(len(geos), self.n_geos)

  def test_get_available_channels(self):
    """Test retrieving available channel names from model."""
    channels = self.deepdive.input_data.get_all_paid_channels()

    self.assertEqual(channels, self.media_channel_names)
    self.assertEqual(len(channels), self.n_media_channels)

  def test_compare_geo_performance_invalid_inputs(self):
    """Test error handling for invalid input combinations."""
    # Invalid geo for inspect_geo
    with self.assertRaises(ValueError):
      self.deepdive.compare_geo_performance('INVALID', 'GEO_B', 'TV')

    # Invalid geo for compare_geo
    with self.assertRaises(ValueError):
      self.deepdive.compare_geo_performance('GEO_A', 'INVALID', 'TV')

    # Invalid channel
    with self.assertRaises(ValueError):
      self.deepdive.compare_geo_performance('GEO_A', 'GEO_B', 'INVALID')


class ModelDataDeepDiveIntegrationTest(unittest.TestCase):
  """Integration tests using real Meridian model structure (if available)."""

  def setUp(self):
    """Set up integration test fixtures."""
    # Check if sample data is available
    sample_pkl = './../inputs/sample_optimizer_input_coeff.pkl'

    if os.path.exists(sample_pkl):
      self.has_sample_data = True
      # Note: This would require loading actual model, skipping for unit tests
      self.has_sample_data = False
    else:
      self.has_sample_data = False

  @unittest.skipUnless(
      os.path.exists('./../inputs/sample_optimizer_input_coeff.pkl'),
      'Sample data not available for integration test'
  )
  def test_integration_with_real_model(self):
    """Test ModelDataDeepDive with real Meridian model."""
    # This test would load actual model and run full analysis
    # Skipped by default for unit testing
    pass


if __name__ == '__main__':
  unittest.main()
