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

"""Tests for prior_feasibility module."""

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.model import model
from meridian.model import model_test_data
from meridian.model import prior_distribution
from meridian.model import prior_feasibility
from meridian.model import spec
import numpy as np
import tensorflow_probability as tfp


class PriorFeasibilityCheckerTest(
  parameterized.TestCase,
  model_test_data.WithInputDataSamples,
):

  def setUp(self):
    super().setUp()
    model_test_data.WithInputDataSamples.setup(self)

    # Create contribution priors for testing
    self.contribution_m = tfp.distributions.Beta(
      concentration1=[10.0, 20.0],  # Display, TV
      concentration0=[90.0, 80.0],
      name=constants.CONTRIBUTION_M
    )

    # Build prior with contribution priors
    self.prior = prior_distribution.PriorDistribution(
      contribution_m=self.contribution_m,
    )

    # Create model spec with contribution priors
    self.model_spec = spec.ModelSpec(
      prior=self.prior,
      media_prior_type='contribution',
    )

  def test_init_with_contribution_priors_succeeds(self):
    """Tests that initialization succeeds with contribution priors."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)
    self.assertIsNotNone(checker)

  def test_init_without_contribution_priors_raises_error(self):
    """Tests that initialization fails without contribution priors."""
    # Create model with coefficient priors instead
    model_spec_no_contrib = spec.ModelSpec(
      media_prior_type='coefficient',
    )

    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=model_spec_no_contrib,
    )

    with self.assertRaises(ValueError):
      prior_feasibility.PriorFeasibilityChecker(meridian)

  def test_check_contribution_feasibility_returns_report(self):
    """Tests that check_contribution_feasibility returns a valid report."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)
    report = checker.check_contribution_feasibility(
      n_draws=100,  # Small number for fast testing
      include_sensitivity=False,
      seed=42,
    )

    # Verify report structure
    self.assertIsInstance(report, prior_feasibility.PriorFeasibilityReport)
    self.assertEqual(report.n_draws, 100)
    self.assertIsNotNone(report.comparison_df)

    # Verify comparison DataFrame has expected columns
    expected_columns = [
      'channel',
      'target_mean_pct',
      'target_std_pct',
      'realized_mean_pct',
      'realized_std_pct',
      'deviation_pct',
    ]
    for col in expected_columns:
      self.assertIn(col, report.comparison_df.columns)

    # Verify we have rows for each media channel
    self.assertEqual(len(report.comparison_df), 2)  # Display, TV

  def test_check_contribution_feasibility_with_sensitivity(self):
    """Tests that sensitivity analysis is included when requested."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)
    report = checker.check_contribution_feasibility(
      n_draws=100,
      include_sensitivity=True,
      seed=42,
    )

    # Verify sensitivity DataFrame exists
    self.assertIsNotNone(report.sensitivity_df)

    # Verify sensitivity DataFrame has expected columns
    expected_columns = [
      'channel',
      'parameter',
      'correlation',
      'variance_explained_pct',
    ]
    for col in expected_columns:
      self.assertIn(col, report.sensitivity_df.columns)

  def test_analyze_parameter_sensitivity_returns_dataframe(self):
    """Tests that analyze_parameter_sensitivity returns valid DataFrame."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)
    sensitivity_df = checker.analyze_parameter_sensitivity(
      n_draws=100,
      seed=42,
    )

    # Verify DataFrame structure
    self.assertIsNotNone(sensitivity_df)
    self.assertGreater(len(sensitivity_df), 0)

    # Verify we have entries for each channel and parameter
    channels = sensitivity_df['channel'].unique()
    self.assertEqual(len(channels), 2)  # Display, TV

    parameters = sensitivity_df['parameter'].unique()
    expected_params = ['alpha_m', 'ec_m', 'slope_m', 'eta_m']
    for param in expected_params:
      self.assertIn(param, parameters)

  def test_identify_compatible_regions_returns_ranges(self):
    """Tests that identify_compatible_regions returns valid parameter ranges."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)

    # Use a wide tolerance to ensure some draws match
    compatible_regions = checker.identify_compatible_regions(
      channel='TV',
      target_contribution_pct=20.0,
      tolerance_pct=10.0,  # ±10%
      n_draws=100,
      seed=42,
    )

    # Verify we get ranges for transformation parameters
    expected_params = ['alpha_m', 'ec_m', 'slope_m', 'eta_m']
    for param in expected_params:
      self.assertIn(param, compatible_regions)

      # Verify ranges are tuples of (min, max)
      param_range = compatible_regions[param]
      self.assertIsInstance(param_range, tuple)
      self.assertEqual(len(param_range), 2)
      self.assertLessEqual(param_range[0], param_range[1])

  def test_identify_compatible_regions_invalid_channel_raises_error(self):
    """Tests that invalid channel raises ValueError."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)

    with self.assertRaises(ValueError):
      checker.identify_compatible_regions(
        channel='InvalidChannel',
        target_contribution_pct=20.0,
      )

  def test_extract_target_contributions(self):
    """Tests that target contributions are correctly extracted from priors."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)
    target_contributions = checker._extract_target_contributions()

    # Verify we have entries for each channel
    self.assertEqual(len(target_contributions), 2)

    # Verify structure
    for channel, stats in target_contributions.items():
      self.assertIn('mean', stats)
      self.assertIn('std', stats)
      self.assertIsInstance(stats['mean'], float)
      self.assertIsInstance(stats['std'], float)

      # Verify values are in reasonable range (0-100%)
      self.assertGreaterEqual(stats['mean'], 0.0)
      self.assertLessEqual(stats['mean'], 100.0)
      self.assertGreaterEqual(stats['std'], 0.0)

  def test_prior_feasibility_report_repr(self):
    """Tests that PriorFeasibilityReport has readable string representation."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)
    report = checker.check_contribution_feasibility(
      n_draws=100,
      seed=42,
    )

    # Verify __repr__ returns string
    report_str = repr(report)
    self.assertIsInstance(report_str, str)

    # Verify report contains expected sections
    self.assertIn('PRIOR FEASIBILITY REPORT', report_str)
    self.assertIn('CONTRIBUTION % COMPARISON', report_str)
    self.assertIn('Based on 100 prior draws', report_str)

  def test_suggest_prior_adjustments(self):
    """Tests that suggest_prior_adjustments returns recommendations."""
    meridian = model.Meridian(
      input_data=self.short_input_data_media_only,
      model_spec=self.model_spec,
    )

    checker = prior_feasibility.PriorFeasibilityChecker(meridian)
    report = checker.check_contribution_feasibility(
      n_draws=100,
      seed=42,
    )

    # Verify recommendations are generated
    self.assertIsInstance(report.recommendations, list)

    # Recommendations should be non-empty if there are mismatches
    # (We can't assert they're non-empty without knowing the specific results)


if __name__ == '__main__':
  absltest.main()
