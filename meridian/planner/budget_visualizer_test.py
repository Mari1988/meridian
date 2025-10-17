"""Tests for BudgetVisualizer MROI functionality."""

import os
import tempfile
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf
import openpyxl

from meridian.planner import budget_visualizer
from meridian.planner.budget_visualizer import BudgetVisualizer
from meridian.analysis.optimizer import OptimizationResults, OptimizationGrid
from meridian import constants


class BudgetVisualizerMROITest(parameterized.TestCase):
  """Test suite for MROI calculation and export functionality in BudgetVisualizer."""

  def setUp(self):
    """Set up test fixtures with mock optimization results."""
    super().setUp()

    # Define test dimensions
    self.n_geos = 3
    self.n_channels = 2
    self.n_grid_points = 5

    self.geo_names = ['Texas', 'California', 'New_York']
    self.channel_names = ['TV', 'Display']

    # Create mock grid dataset with spend and outcome grids
    self.spend_grid_data = self._create_test_spend_grid()
    self.outcome_grid_data = self._create_test_outcome_grid()

    # Create xarray Dataset for grid_dataset
    self.grid_dataset = self._create_grid_dataset(
        self.spend_grid_data,
        self.outcome_grid_data
    )

    # Create mock OptimizationGrid
    self.optimization_grid = self._create_mock_optimization_grid()

    # Create mock OptimizationResults with all required attributes
    self.opt_results = self._create_mock_optimization_results()

    # Create BudgetVisualizer instance
    self.visualizer = BudgetVisualizer(self.opt_results)

  def _create_test_spend_grid(self):
    """Create test spend grid with increasing values."""
    # Shape: [grid_points, geos, channels]
    # Each channel and geo has different spend patterns
    spend_grid = np.zeros((self.n_grid_points, self.n_geos, self.n_channels))

    for g in range(self.n_geos):
      for c in range(self.n_channels):
        # Create increasing spend with different rates per channel/geo
        base_spend = 1000 * (g + 1) * (c + 1)
        increment = 500 * (c + 1)
        spend_grid[:, g, c] = base_spend + np.arange(self.n_grid_points) * increment

    return spend_grid

  def _create_test_outcome_grid(self):
    """Create test outcome grid with diminishing returns."""
    # Shape: [grid_points, geos, channels]
    # Outcomes show diminishing marginal returns (concave function)
    outcome_grid = np.zeros((self.n_grid_points, self.n_geos, self.n_channels))

    for g in range(self.n_geos):
      for c in range(self.n_channels):
        # Create outcome with diminishing returns using sqrt-like curve
        base_outcome = 100 * (g + 1) * (c + 1)
        # Use a concave function: outcome = base * sqrt(spend_multiplier)
        for i in range(self.n_grid_points):
          outcome_grid[i, g, c] = base_outcome * np.sqrt(i + 1) * 50

    return outcome_grid

  def _create_grid_dataset(self, spend_grid, outcome_grid):
    """Create xarray Dataset mimicking OptimizationGrid structure."""
    dataset = xr.Dataset(
        {
            constants.SPEND_GRID: (
                [constants.GRID_SPEND_INDEX, constants.GEO, constants.CHANNEL],
                spend_grid
            ),
            constants.INCREMENTAL_OUTCOME_GRID: (
                [constants.GRID_SPEND_INDEX, constants.GEO, constants.CHANNEL],
                outcome_grid
            ),
        },
        coords={
            constants.GRID_SPEND_INDEX: np.arange(self.n_grid_points),
            constants.GEO: self.geo_names,
            constants.CHANNEL: self.channel_names,
        },
        attrs={constants.SPEND_STEP_SIZE: 100.0}
    )
    return dataset

  def _create_mock_optimization_grid(self):
    """Create mock OptimizationGrid object."""
    mock_grid = mock.MagicMock(spec=OptimizationGrid)
    mock_grid.grid_dataset = self.grid_dataset
    mock_grid.channels = self.channel_names
    mock_grid.geos = self.geo_names
    return mock_grid

  def _create_mock_optimization_results(self):
    """Create mock OptimizationResults with required attributes."""
    # Create minimal mock objects for required attributes
    mock_meridian = mock.MagicMock()
    mock_meridian.input_data.geo.values = self.geo_names
    mock_meridian.input_data.get_all_paid_channels.return_value = self.channel_names

    # Create mock geo_level_data with required DataArrays
    mock_geo_data = xr.Dataset({
        'optimized_spend_gm': xr.DataArray(
            np.random.rand(len(self.geo_names), len(self.channel_names)),
            dims=['geo', 'channel'],
            coords={'geo': self.geo_names, 'channel': self.channel_names}
        ),
        'nonoptimized_spend_gm': xr.DataArray(
            np.random.rand(len(self.geo_names), len(self.channel_names)),
            dims=['geo', 'channel'],
            coords={'geo': self.geo_names, 'channel': self.channel_names}
        ),
        'optimized_incremental_outcome_gm': xr.DataArray(
            np.random.rand(len(self.geo_names), len(self.channel_names)),
            dims=['geo', 'channel'],
            coords={'geo': self.geo_names, 'channel': self.channel_names}
        ),
        'nonoptimized_incremental_outcome_gm': xr.DataArray(
            np.random.rand(len(self.geo_names), len(self.channel_names)),
            dims=['geo', 'channel'],
            coords={'geo': self.geo_names, 'channel': self.channel_names}
        ),
    })

    # Create mock optimized/nonoptimized data
    mock_opt_data = xr.Dataset({
        'spend': xr.DataArray(
            np.random.rand(len(self.channel_names)),
            dims=['channel'],
            coords={'channel': self.channel_names}
        ),
        'incremental_outcome': xr.DataArray(
            np.random.rand(len(self.channel_names)),
            dims=['channel'],
            coords={'channel': self.channel_names}
        ),
    }, attrs={'budget': 10000, 'total_incremental_outcome': 5000, 'total_cpik': 2.0, 'total_roi': 0.5})

    mock_nonopt_data = mock_opt_data.copy(deep=True)

    # Create mock OptimizationResults
    mock_results = mock.MagicMock(spec=OptimizationResults)
    mock_results.meridian = mock_meridian
    mock_results.optimization_grid = self.optimization_grid
    mock_results.geo_level_optimized_data = mock_geo_data
    mock_results.optimized_data = mock_opt_data.expand_dims(metric=['mean'])
    mock_results.nonoptimized_data = mock_nonopt_data.expand_dims(metric=['mean'])
    mock_results.analyzer = mock.MagicMock()

    return mock_results

  # =============================================================================
  # Tests for calculate_mroi_by_geo()
  # =============================================================================

  def test_calculate_mroi_by_geo_returns_dataframe(self):
    """Test that calculate_mroi_by_geo returns a pandas DataFrame."""
    result = self.visualizer.calculate_mroi_by_geo()
    self.assertIsInstance(result, pd.DataFrame)

  def test_calculate_mroi_by_geo_has_correct_columns(self):
    """Test that returned DataFrame has correct columns."""
    result = self.visualizer.calculate_mroi_by_geo()
    expected_columns = ['geo', 'grid_idx'] + self.channel_names
    self.assertListEqual(list(result.columns), expected_columns)

  def test_calculate_mroi_by_geo_has_correct_shape(self):
    """Test that returned DataFrame has correct number of rows."""
    result = self.visualizer.calculate_mroi_by_geo()
    # Each geo should have (n_grid_points - 1) rows due to delta calculation
    expected_rows = self.n_geos * (self.n_grid_points - 1)
    self.assertEqual(len(result), expected_rows)

  def test_calculate_mroi_by_geo_all_geos_included(self):
    """Test that all geos are included in the result."""
    result = self.visualizer.calculate_mroi_by_geo()
    unique_geos = result['geo'].unique().tolist()
    self.assertListEqual(sorted(unique_geos), sorted(self.geo_names))

  def test_calculate_mroi_by_geo_grid_idx_sequential(self):
    """Test that grid_idx is sequential for each geo."""
    result = self.visualizer.calculate_mroi_by_geo()
    for geo in self.geo_names:
      geo_data = result[result['geo'] == geo]
      grid_indices = geo_data['grid_idx'].tolist()
      expected_indices = list(range(self.n_grid_points - 1))
      self.assertListEqual(grid_indices, expected_indices)

  def test_calculate_mroi_by_geo_with_selected_geos(self):
    """Test calculate_mroi_by_geo with specific geos selected."""
    selected_geos = ['Texas', 'California']
    result = self.visualizer.calculate_mroi_by_geo(geos=selected_geos)

    unique_geos = result['geo'].unique().tolist()
    self.assertListEqual(sorted(unique_geos), sorted(selected_geos))

    # Should have fewer rows
    expected_rows = len(selected_geos) * (self.n_grid_points - 1)
    self.assertEqual(len(result), expected_rows)

  def test_calculate_mroi_by_geo_with_single_geo(self):
    """Test calculate_mroi_by_geo with a single geo."""
    selected_geos = ['Texas']
    result = self.visualizer.calculate_mroi_by_geo(geos=selected_geos)

    self.assertEqual(result['geo'].nunique(), 1)
    self.assertEqual(result['geo'].iloc[0], 'Texas')
    self.assertEqual(len(result), self.n_grid_points - 1)

  def test_calculate_mroi_by_geo_raises_error_for_invalid_geo(self):
    """Test that invalid geo raises ValueError."""
    with self.assertRaises(ValueError) as context:
      self.visualizer.calculate_mroi_by_geo(geos=['InvalidGeo'])

    self.assertIn('not found in the optimization grid', str(context.exception))

  def test_calculate_mroi_by_geo_values_are_numeric(self):
    """Test that MROI values are numeric (not NaN or Inf)."""
    result = self.visualizer.calculate_mroi_by_geo()

    for channel in self.channel_names:
      # All values should be finite (not NaN or Inf)
      self.assertTrue(np.all(np.isfinite(result[channel].values)))

  def test_calculate_mroi_by_geo_diminishing_returns(self):
    """Test that MROI values show diminishing returns (decreasing trend)."""
    result = self.visualizer.calculate_mroi_by_geo()

    # For each geo and channel, MROI should generally decrease
    # (because we created outcome grid with diminishing returns)
    for geo in self.geo_names:
      geo_data = result[result['geo'] == geo].sort_values('grid_idx')
      for channel in self.channel_names:
        mroi_values = geo_data[channel].values
        # Check that most consecutive pairs show decreasing MROI
        decreasing_count = sum(mroi_values[i] > mroi_values[i+1]
                               for i in range(len(mroi_values)-1))
        # At least half should be decreasing
        self.assertGreater(decreasing_count, len(mroi_values) / 3)

  def test_calculate_mroi_by_geo_calculation_correctness(self):
    """Test that MROI calculation matches expected formula."""
    # Test for one specific geo and channel
    geo = 'Texas'
    channel = 'TV'

    result = self.visualizer.calculate_mroi_by_geo(geos=[geo])

    # Manually calculate expected MROI
    grid_data = self.grid_dataset.sel(geo=geo)
    spend_grid = grid_data.spend_grid.values
    outcome_grid = grid_data.incremental_outcome_grid.values

    channel_idx = self.channel_names.index(channel)
    spend_deltas = spend_grid[1:, channel_idx] - spend_grid[:-1, channel_idx]
    outcome_deltas = outcome_grid[1:, channel_idx] - outcome_grid[:-1, channel_idx]
    expected_mroi = outcome_deltas / spend_deltas

    actual_mroi = result[channel].values

    np.testing.assert_array_almost_equal(actual_mroi, expected_mroi, decimal=5)

  # =============================================================================
  # Tests for _get_top_geos_by_spend()
  # =============================================================================

  def test_get_top_geos_by_spend_returns_list(self):
    """Test that _get_top_geos_by_spend returns a list."""
    result = self.visualizer._get_top_geos_by_spend(2)
    self.assertIsInstance(result, list)

  def test_get_top_geos_by_spend_returns_correct_count(self):
    """Test that correct number of geos is returned."""
    top_n = 2
    result = self.visualizer._get_top_geos_by_spend(top_n)
    self.assertEqual(len(result), top_n)

  def test_get_top_geos_by_spend_all_geos(self):
    """Test requesting more geos than available."""
    result = self.visualizer._get_top_geos_by_spend(10)
    self.assertEqual(len(result), self.n_geos)

  def test_get_top_geos_by_spend_returns_geo_names(self):
    """Test that returned values are valid geo names."""
    result = self.visualizer._get_top_geos_by_spend(2)
    for geo in result:
      self.assertIn(geo, self.geo_names)

  # =============================================================================
  # Tests for export_mroi_to_excel()
  # =============================================================================

  def test_export_mroi_to_excel_creates_file(self):
    """Test that export creates an Excel file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi.xlsx')
      result_path = self.visualizer.export_mroi_to_excel(file_path)

      self.assertEqual(result_path, file_path)
      self.assertTrue(os.path.exists(file_path))

  def test_export_mroi_to_excel_adds_extension(self):
    """Test that .xlsx extension is added if missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi')  # No extension
      result_path = self.visualizer.export_mroi_to_excel(file_path)

      self.assertTrue(result_path.endswith('.xlsx'))
      self.assertTrue(os.path.exists(result_path))

  def test_export_mroi_to_excel_all_geos_creates_all_sheets(self):
    """Test that all geos create separate sheets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi_all.xlsx')
      self.visualizer.export_mroi_to_excel(file_path)

      # Read Excel file and check sheets
      workbook = openpyxl.load_workbook(file_path)
      sheet_names = workbook.sheetnames

      # Should have n_geos + 1 for the 'mroi bounds' sheet
      self.assertEqual(len(sheet_names), self.n_geos + 1)
      for geo in self.geo_names:
        self.assertIn(geo, sheet_names)
      self.assertIn('mroi bounds', sheet_names)

  def test_export_mroi_to_excel_sheet_structure(self):
    """Test that each sheet has correct structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi_structure.xlsx')
      self.visualizer.export_mroi_to_excel(file_path)

      # Read one sheet and verify structure
      df = pd.read_excel(file_path, sheet_name='Texas')

      # Check columns
      expected_columns = ['grid_idx'] + self.channel_names
      self.assertListEqual(list(df.columns), expected_columns)

      # Check number of rows
      self.assertEqual(len(df), self.n_grid_points - 1)

  def test_export_mroi_to_excel_top_n_geos(self):
    """Test export with top_n parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi_top2.xlsx')
      top_n = 2
      self.visualizer.export_mroi_to_excel(file_path, top_n=top_n)

      workbook = openpyxl.load_workbook(file_path)
      sheet_names = workbook.sheetnames

      # Should have top_n + 1 for the 'mroi bounds' sheet
      self.assertEqual(len(sheet_names), top_n + 1)
      self.assertIn('mroi bounds', sheet_names)

  def test_export_mroi_to_excel_selected_geos(self):
    """Test export with selected_geos parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi_selected.xlsx')
      selected_geos = ['Texas', 'California']
      self.visualizer.export_mroi_to_excel(file_path, selected_geos=selected_geos)

      workbook = openpyxl.load_workbook(file_path)
      sheet_names = workbook.sheetnames

      # Should have len(selected_geos) + 1 for the 'mroi bounds' sheet
      self.assertEqual(len(sheet_names), len(selected_geos) + 1)
      for geo in selected_geos:
        self.assertIn(geo, sheet_names)
      self.assertIn('mroi bounds', sheet_names)

  def test_export_mroi_to_excel_mutually_exclusive_params(self):
    """Test that top_n and selected_geos cannot be used together."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi_error.xlsx')

      with self.assertRaises(ValueError) as context:
        self.visualizer.export_mroi_to_excel(
            file_path,
            top_n=2,
            selected_geos=['Texas']
        )

      self.assertIn('Cannot specify both', str(context.exception))

  def test_export_mroi_to_excel_sheet_data_matches_calculation(self):
    """Test that exported data matches calculate_mroi_by_geo output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi_match.xlsx')
      selected_geo = 'Texas'
      self.visualizer.export_mroi_to_excel(file_path, selected_geos=[selected_geo])

      # Get data from calculation
      calculated_df = self.visualizer.calculate_mroi_by_geo(geos=[selected_geo])
      calculated_df = calculated_df.drop(columns=['geo'])

      # Get data from Excel
      excel_df = pd.read_excel(file_path, sheet_name=selected_geo)

      # Compare
      pd.testing.assert_frame_equal(
          calculated_df.reset_index(drop=True),
          excel_df.reset_index(drop=True)
      )

  def test_export_mroi_to_excel_long_geo_name_truncation(self):
    """Test that geo names longer than 31 chars are truncated (Excel limit)."""
    # Create a visualizer with a long geo name
    long_geo_name = 'A' * 40  # 40 characters (exceeds Excel's 31 limit)

    # Create new geo coordinates with long name
    new_geo_coords = [long_geo_name if geo == self.geo_names[0] else geo
                      for geo in self.geo_names]

    # Create modified grid dataset with new coordinates
    modified_grid = self.grid_dataset.assign_coords(
        {constants.GEO: new_geo_coords}
    )
    self.optimization_grid.grid_dataset = modified_grid

    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_mroi_long_name.xlsx')
      self.visualizer.export_mroi_to_excel(file_path, selected_geos=[long_geo_name])

      workbook = openpyxl.load_workbook(file_path)
      sheet_names = workbook.sheetnames

      # Sheet name should be truncated to 31 characters
      self.assertEqual(len(sheet_names[0]), 31)
      self.assertEqual(sheet_names[0], long_geo_name[:31])

  # =============================================================================
  # Tests for _calculate_mroi_bounds_summary()
  # =============================================================================

  def test_calculate_mroi_bounds_summary_structure(self):
    """Test that bounds summary has correct structure."""
    mroi_df = self.visualizer.calculate_mroi_by_geo()
    bounds_summary = self.visualizer._calculate_mroi_bounds_summary(mroi_df)

    # Check columns
    expected_cols = ['geo', 'bound_type'] + self.channel_names
    self.assertListEqual(list(bounds_summary.columns), expected_cols)

    # Check that we have 2 rows per geo (lower + upper)
    self.assertEqual(len(bounds_summary), self.n_geos * 2)

  def test_calculate_mroi_bounds_summary_bound_types(self):
    """Test that both bound types exist for each geo."""
    mroi_df = self.visualizer.calculate_mroi_by_geo()
    bounds_summary = self.visualizer._calculate_mroi_bounds_summary(mroi_df)

    for geo in self.geo_names:
      geo_bounds = bounds_summary[bounds_summary['geo'] == geo]
      bound_types = set(geo_bounds['bound_type'].values)
      self.assertEqual(bound_types, {'lower_bound', 'upper_bound'})

  def test_calculate_mroi_bounds_summary_lower_bound_values(self):
    """Test that lower_bound matches grid_idx=0."""
    mroi_df = self.visualizer.calculate_mroi_by_geo()
    bounds_summary = self.visualizer._calculate_mroi_bounds_summary(mroi_df)

    # Get lower bounds from summary
    lower_bounds = bounds_summary[bounds_summary['bound_type'] == 'lower_bound']

    # Compare with grid_idx=0 from original
    for geo in self.geo_names:
      lb_row = lower_bounds[lower_bounds['geo'] == geo].iloc[0]
      original_row = mroi_df[(mroi_df['geo'] == geo) & (mroi_df['grid_idx'] == 0)].iloc[0]

      for channel in self.channel_names:
        self.assertAlmostEqual(lb_row[channel], original_row[channel], places=5)

  def test_calculate_mroi_bounds_summary_upper_bound_values(self):
    """Test that upper_bound is last non-NaN value."""
    mroi_df = self.visualizer.calculate_mroi_by_geo()
    bounds_summary = self.visualizer._calculate_mroi_bounds_summary(mroi_df)

    upper_bounds = bounds_summary[bounds_summary['bound_type'] == 'upper_bound']

    # Verify upper bound is last valid value
    for geo in self.geo_names:
      ub_row = upper_bounds[upper_bounds['geo'] == geo].iloc[0]
      geo_data = mroi_df[mroi_df['geo'] == geo]

      for channel in self.channel_names:
        # Get last non-NaN value
        valid_values = geo_data[channel].dropna()
        if len(valid_values) > 0:
          last_valid = valid_values.iloc[-1]
          self.assertAlmostEqual(ub_row[channel], last_valid, places=5)
        else:
          # If all NaN, upper bound should also be NaN
          self.assertTrue(np.isnan(ub_row[channel]))

  def test_calculate_mroi_bounds_summary_returns_dataframe(self):
    """Test that method returns a pandas DataFrame."""
    mroi_df = self.visualizer.calculate_mroi_by_geo()
    bounds_summary = self.visualizer._calculate_mroi_bounds_summary(mroi_df)
    self.assertIsInstance(bounds_summary, pd.DataFrame)

  # =============================================================================
  # Tests for MROI Bounds in Excel Export
  # =============================================================================

  def test_export_mroi_to_excel_includes_bounds_sheet(self):
    """Test that Excel export includes 'mroi bounds' sheet."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_with_bounds.xlsx')
      self.visualizer.export_mroi_to_excel(file_path)

      workbook = openpyxl.load_workbook(file_path)
      sheet_names = workbook.sheetnames

      self.assertIn('mroi bounds', sheet_names)

  def test_export_mroi_bounds_sheet_content(self):
    """Test that bounds sheet has correct content."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_bounds_content.xlsx')
      self.visualizer.export_mroi_to_excel(file_path)

      # Read the bounds sheet
      bounds_df = pd.read_excel(file_path, sheet_name='mroi bounds')

      # Check structure
      self.assertIn('geo', bounds_df.columns)
      self.assertIn('bound_type', bounds_df.columns)

      # Check we have both bound types for each geo
      for geo in self.geo_names:
        geo_bounds = bounds_df[bounds_df['geo'] == geo]
        bound_types = set(geo_bounds['bound_type'].values)
        self.assertEqual(bound_types, {'lower_bound', 'upper_bound'})

  def test_export_mroi_bounds_sheet_matches_calculation(self):
    """Test that exported bounds match direct calculation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_bounds_match.xlsx')
      self.visualizer.export_mroi_to_excel(file_path)

      # Calculate expected bounds
      mroi_df = self.visualizer.calculate_mroi_by_geo()
      expected_bounds = self.visualizer._calculate_mroi_bounds_summary(mroi_df)

      # Read from Excel
      excel_bounds = pd.read_excel(file_path, sheet_name='mroi bounds')

      # Compare DataFrames
      pd.testing.assert_frame_equal(
          expected_bounds.reset_index(drop=True),
          excel_bounds.reset_index(drop=True)
      )

  def test_export_mroi_bounds_sheet_with_filtered_geos(self):
    """Test that bounds sheet only includes filtered geos."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      file_path = os.path.join(tmp_dir, 'test_filtered_bounds.xlsx')
      selected_geos = ['Texas', 'California']
      self.visualizer.export_mroi_to_excel(file_path, selected_geos=selected_geos)

      # Read the bounds sheet
      bounds_df = pd.read_excel(file_path, sheet_name='mroi bounds')

      # Check that only selected geos are in bounds
      unique_geos = bounds_df['geo'].unique().tolist()
      self.assertEqual(set(unique_geos), set(selected_geos))

      # Check that we have 2 rows per selected geo
      self.assertEqual(len(bounds_df), len(selected_geos) * 2)

  # =============================================================================
  # Integration Tests
  # =============================================================================

  def test_end_to_end_workflow(self):
    """Test complete workflow from calculation to export."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      # Step 1: Calculate MROI
      mroi_df = self.visualizer.calculate_mroi_by_geo()
      self.assertIsInstance(mroi_df, pd.DataFrame)
      self.assertGreater(len(mroi_df), 0)

      # Step 2: Export to Excel
      file_path = os.path.join(tmp_dir, 'workflow_test.xlsx')
      result_path = self.visualizer.export_mroi_to_excel(file_path)
      self.assertTrue(os.path.exists(result_path))

      # Step 3: Verify exported geo sheets
      for geo in self.geo_names:
        df = pd.read_excel(result_path, sheet_name=geo)
        self.assertGreater(len(df), 0)
        self.assertIn('grid_idx', df.columns)
        for channel in self.channel_names:
          self.assertIn(channel, df.columns)

      # Step 4: Verify bounds sheet exists and has content
      bounds_df = pd.read_excel(result_path, sheet_name='mroi bounds')
      self.assertGreater(len(bounds_df), 0)
      self.assertIn('geo', bounds_df.columns)
      self.assertIn('bound_type', bounds_df.columns)
      for channel in self.channel_names:
        self.assertIn(channel, bounds_df.columns)


if __name__ == '__main__':
  absltest.main()
