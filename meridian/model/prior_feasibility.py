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

"""Prior feasibility analysis for contribution priors in Meridian MMM.

This module provides tools to validate contribution priors by checking if the
specified contribution percentages are achievable given the transformation
priors (adstock, hill saturation, hierarchical variation, baseline).

Example usage:
  ```python
  # Create model with contribution priors
  mmm = model.Meridian(input_data=data, model_spec=model_spec)

  # Check prior feasibility
  checker = PriorFeasibilityChecker(mmm)
  report = checker.check_contribution_feasibility(n_draws=1000)

  print(report)
  # Output shows target vs realized contribution % from prior
  ```
"""

from typing import TYPE_CHECKING, Any, Optional
import warnings

from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import visualizer
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from meridian.model import model  # pylint: disable=g-bad-import-order


__all__ = [
  'PriorFeasibilityChecker',
  'PriorFeasibilityReport',
]


class PriorFeasibilityReport:
  """Container for prior feasibility analysis results.

  Attributes:
    comparison_df: DataFrame comparing target vs realized contribution %
    sensitivity_df: DataFrame showing parameter sensitivity (variance contribution)
    recommendations: List of recommended prior adjustments
    n_draws: Number of prior draws used in analysis
  """

  def __init__(
      self,
      comparison_df: pd.DataFrame,
      sensitivity_df: Optional[pd.DataFrame] = None,
      recommendations: Optional[list[str]] = None,
      n_draws: int = 1000,
  ):
    self.comparison_df = comparison_df
    self.sensitivity_df = sensitivity_df
    self.recommendations = recommendations or []
    self.n_draws = n_draws

  def __repr__(self) -> str:
    """Returns a formatted string representation of the report."""
    lines = ["=" * 80]
    lines.append("PRIOR FEASIBILITY REPORT")
    lines.append("=" * 80)
    lines.append(f"\nBased on {self.n_draws} prior draws\n")

    lines.append("CONTRIBUTION % COMPARISON: Target vs Realized Prior")
    lines.append("-" * 80)
    lines.append(self.comparison_df.to_string(index=False))

    if self.sensitivity_df is not None:
      lines.append("\n" + "=" * 80)
      lines.append("PARAMETER SENSITIVITY ANALYSIS")
      lines.append("-" * 80)
      lines.append(self.sensitivity_df.to_string(index=False))

    if self.recommendations:
      lines.append("\n" + "=" * 80)
      lines.append("RECOMMENDATIONS")
      lines.append("-" * 80)
      for i, rec in enumerate(self.recommendations, 1):
        lines.append(f"{i}. {rec}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


class PriorFeasibilityChecker:
  """Validates contribution priors and identifies compatible parameter regions.

  This class helps diagnose why prior contribution percentages may not match
  the specified contribution_m priors, and provides guidance on adjusting
  transformation priors to achieve the desired contribution targets.

  Example:
    ```python
    checker = PriorFeasibilityChecker(mmm)
    report = checker.check_contribution_feasibility(n_draws=1000)
    print(report)
    ```
  """

  def __init__(self, meridian: 'model.Meridian'):
    """Initializes the prior feasibility checker.

    Args:
      meridian: A Meridian model with contribution priors specified

    Raises:
      ValueError: If the model does not use contribution priors
    """
    self._meridian = meridian

    # Validate that model uses contribution priors
    prior_type = meridian.model_spec.effective_media_prior_type
    if prior_type != constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION:
      raise ValueError(
        f"PriorFeasibilityChecker requires contribution priors. "
        f"Model uses prior type: {prior_type}"
      )

  def check_contribution_feasibility(
      self,
      n_draws: int = 1000,
      include_sensitivity: bool = False,
      seed: Optional[int] = None,
  ) -> PriorFeasibilityReport:
    """Checks if contribution priors are feasible given transformation priors.

    This method samples from the prior predictive distribution and computes
    the realized contribution percentages. It compares these to the target
    contribution percentages specified in contribution_m priors.

    Args:
      n_draws: Number of draws from the prior distribution (default: 1000)
      include_sensitivity: Whether to include parameter sensitivity analysis
      seed: Random seed for reproducibility

    Returns:
      PriorFeasibilityReport containing comparison table, sensitivity analysis,
      and recommendations for prior adjustments
    """
    mmm = self._meridian

    # Sample from prior if not already done
    try:
      if mmm.inference_data.prior is None or len(mmm.inference_data.prior.draw) != n_draws:
        mmm.sample_prior(n_draws, seed=seed)
    except AttributeError:
      # inference_data doesn't have prior yet
      mmm.sample_prior(n_draws, seed=seed)

    # Get target contribution percentages from contribution_m prior
    target_contributions = self._extract_target_contributions()

    # Compute realized contribution percentages from prior samples
    # We need to manually compute this since summary_table_without_ci expects posterior
    realized_contributions = self._compute_realized_contributions_from_prior(mmm)

    # Create comparison DataFrame
    comparison_df = self._create_comparison_table(
      target_contributions, realized_contributions
    )

    # Optionally compute sensitivity analysis
    sensitivity_df = None
    if include_sensitivity:
      sensitivity_df = self.analyze_parameter_sensitivity(n_draws, seed)

    # Generate recommendations
    recommendations = self._generate_recommendations(comparison_df)

    return PriorFeasibilityReport(
      comparison_df=comparison_df,
      sensitivity_df=sensitivity_df,
      recommendations=recommendations,
      n_draws=n_draws,
    )

  def analyze_parameter_sensitivity(
      self,
      n_draws: int = 1000,
      seed: Optional[int] = None,
  ) -> pd.DataFrame:
    """Analyzes which transformation parameters drive contribution % variance.

    This method computes the correlation between each transformation parameter
    (alpha_m, ec_m, slope_m, eta_m) and the realized contribution percentage.
    High correlation indicates that parameter has strong influence on the
    contribution % mismatch.

    Args:
      n_draws: Number of draws from the prior distribution
      seed: Random seed for reproducibility

    Returns:
      DataFrame with columns: channel, parameter, correlation, variance_explained
    """
    mmm = self._meridian

    # Ensure prior samples exist
    try:
      if mmm.inference_data.prior is None or len(mmm.inference_data.prior.draw) != n_draws:
        mmm.sample_prior(n_draws, seed=seed)
    except AttributeError:
      # inference_data doesn't have prior yet
      mmm.sample_prior(n_draws, seed=seed)

    # Get prior samples for transformation parameters
    prior_data = mmm.inference_data.prior

    # Extract parameter arrays
    # media_channel is a DataArray, convert to list
    channels = list(mmm.input_data.media_channel.values) if hasattr(mmm.input_data.media_channel, 'values') else list(mmm.input_data.media_channel)
    results = []

    for channel_idx, channel in enumerate(channels):
      channel = str(channel)  # Ensure it's a string
      # Get contribution % for this channel from all prior draws
      # This requires re-computing from incremental_outcome
      mmm_analyzer = analyzer.Analyzer(mmm)

      # Get incremental outcome for each draw
      incremental_outcome = mmm_analyzer.incremental_outcome(
        new_data=None,  # Use original data
        use_posterior=False,  # Use prior samples
        aggregate_geos=False,
        aggregate_times=False,
      ).numpy()  # Shape: (chain, draw, geo, time, channel)

      # Sum over geo and time to get total incremental outcome per channel
      incremental_outcome_total = incremental_outcome.sum(axis=(2, 3))
      # Shape: (chain, draw, channel)

      # Get total outcome (baseline + all media)
      total_outcome = mmm_analyzer.expected_outcome(
        new_data=None,
        use_posterior=False,  # Use prior samples
        aggregate_geos=False,
        aggregate_times=False,
      ).numpy().sum(axis=(2, 3))  # Shape: (chain, draw)

      # Calculate contribution % for this channel
      contribution_pct = (
        incremental_outcome_total[:, :, channel_idx] / total_outcome * 100
      )
      # Flatten across chains and draws
      contribution_pct_flat = contribution_pct.flatten()

      # Extract transformation parameters for this channel
      parameters = {
        'alpha_m': prior_data[constants.ALPHA_M].values[:, :, channel_idx].flatten(),
        'ec_m': prior_data[constants.EC_M].values[:, :, channel_idx].flatten(),
        'slope_m': prior_data[constants.SLOPE_M].values[:, :, channel_idx].flatten(),
        'eta_m': prior_data[constants.ETA_M].values[:, :, channel_idx].flatten(),
      }

      # Calculate correlation between each parameter and contribution %
      for param_name, param_values in parameters.items():
        correlation = np.corrcoef(param_values, contribution_pct_flat)[0, 1]
        variance_explained = correlation ** 2 * 100  # R²

        results.append({
          'channel': channel,
          'parameter': param_name,
          'correlation': correlation,
          'variance_explained_pct': variance_explained,
        })

    sensitivity_df = pd.DataFrame(results)
    sensitivity_df = sensitivity_df.sort_values(
      ['channel', 'variance_explained_pct'],
      ascending=[True, False]
    )

    return sensitivity_df

  def identify_compatible_regions(
      self,
      channel: str,
      target_contribution_pct: float,
      tolerance_pct: float = 2.0,
      n_draws: int = 1000,
      seed: Optional[int] = None,
  ) -> dict[str, tuple[float, float]]:
    """Identifies transformation parameter ranges compatible with target contribution.

    This method finds the ranges of transformation parameters (alpha_m, ec_m,
    slope_m) that result in contribution percentages within the specified
    tolerance of the target.

    Args:
      channel: Media channel name (e.g., 'TV', 'Display')
      target_contribution_pct: Target contribution percentage (e.g., 30.0)
      tolerance_pct: Tolerance in percentage points (default: 2.0)
      n_draws: Number of draws from the prior distribution
      seed: Random seed for reproducibility

    Returns:
      Dictionary mapping parameter names to (min, max) ranges that achieve
      the target contribution within tolerance

    Raises:
      ValueError: If channel is not found in the model
    """
    mmm = self._meridian

    # Validate channel exists
    if channel not in mmm.input_data.media_channel:
      raise ValueError(
        f"Channel '{channel}' not found. Available channels: {mmm.input_data.media_channel}"
      )

    # Ensure prior samples exist
    try:
      if mmm.inference_data.prior is None or len(mmm.inference_data.prior.draw) != n_draws:
        mmm.sample_prior(n_draws, seed=seed)
    except AttributeError:
      # inference_data doesn't have prior yet
      mmm.sample_prior(n_draws, seed=seed)

    # Get channel index
    channel_idx = list(mmm.input_data.media_channel).index(channel)

    # Get incremental outcome and total outcome (same as in sensitivity analysis)
    mmm_analyzer = analyzer.Analyzer(mmm)
    incremental_outcome = mmm_analyzer.incremental_outcome(
      new_data=None,
      use_posterior=False,  # Use prior samples
      aggregate_geos=False,
      aggregate_times=False,
    ).numpy()
    incremental_outcome_total = incremental_outcome.sum(axis=(2, 3))

    total_outcome = mmm_analyzer.expected_outcome(
      new_data=None,
      use_posterior=False,  # Use prior samples
      aggregate_geos=False,
      aggregate_times=False,
    ).numpy().sum(axis=(2, 3))

    contribution_pct = (
      incremental_outcome_total[:, :, channel_idx] / total_outcome * 100
    ).flatten()

    # Find draws where contribution % is within tolerance
    lower_bound = target_contribution_pct - tolerance_pct
    upper_bound = target_contribution_pct + tolerance_pct
    compatible_mask = (contribution_pct >= lower_bound) & (contribution_pct <= upper_bound)

    if compatible_mask.sum() == 0:
      warnings.warn(
        f"No prior draws achieved {target_contribution_pct}% ± {tolerance_pct}% "
        f"for channel '{channel}'. Try increasing tolerance or n_draws."
      )
      return {}

    # Extract transformation parameters for compatible draws
    prior_data = mmm.inference_data.prior
    parameters = {
      'alpha_m': prior_data[constants.ALPHA_M].values[:, :, channel_idx].flatten(),
      'ec_m': prior_data[constants.EC_M].values[:, :, channel_idx].flatten(),
      'slope_m': prior_data[constants.SLOPE_M].values[:, :, channel_idx].flatten(),
      'eta_m': prior_data[constants.ETA_M].values[:, :, channel_idx].flatten(),
    }

    # Find min/max ranges for compatible draws
    compatible_regions = {}
    for param_name, param_values in parameters.items():
      compatible_values = param_values[compatible_mask]
      compatible_regions[param_name] = (
        float(compatible_values.min()),
        float(compatible_values.max()),
      )

    return compatible_regions

  def suggest_prior_adjustments(
      self,
      comparison_df: pd.DataFrame,
      max_deviation_pct: float = 5.0,
  ) -> list[str]:
    """Suggests prior adjustments to improve contribution % match.

    Args:
      comparison_df: DataFrame from check_contribution_feasibility()
      max_deviation_pct: Maximum acceptable deviation in percentage points

    Returns:
      List of recommendation strings
    """
    recommendations = []

    # Check for large mismatches
    large_mismatches = comparison_df[
      comparison_df['deviation_pct'].abs() > max_deviation_pct
    ]

    if len(large_mismatches) > 0:
      recommendations.append(
        "Large contribution % mismatch detected. Consider these adjustments:"
      )
      recommendations.append(
        "  1. Tighten baseline priors: knot_values=N(0,1), tau_g=N(0,1), knots_per_year=3"
      )
      recommendations.append(
        "  2. Narrow transformation priors: Use tighter ranges for alpha_m, ec_m, slope_m"
      )
      recommendations.append(
        "  3. Reduce geo heterogeneity: Use smaller eta_m (e.g., HalfNormal(0.2))"
      )

    # Check for high variance
    high_variance = comparison_df[
      comparison_df['realized_std_pct'] > comparison_df['target_mean_pct'] * 0.2
    ]

    if len(high_variance) > 0:
      recommendations.append(
        "\nHigh variance in realized contribution % detected:"
      )
      for _, row in high_variance.iterrows():
        recommendations.append(
          f"  - {row['channel']}: std={row['realized_std_pct']:.1f}% "
          f"(>{row['target_mean_pct'] * 0.2:.1f}%)"
        )
      recommendations.append(
        "  → Run analyze_parameter_sensitivity() to identify which priors drive variance"
      )

    return recommendations

  def _extract_target_contributions(self) -> dict[str, dict[str, float]]:
    """Extracts target contribution % from contribution_m prior distribution.

    Returns:
      Dict mapping channel name to {'mean': float, 'std': float}
    """
    mmm = self._meridian
    prior = mmm.model_spec.prior

    # contribution_m is a Beta distribution with shape (n_media_channels,)
    # Extract mean and std for each channel
    contribution_dist = prior.contribution_m

    # For Beta distribution: mean = alpha / (alpha + beta)
    # std = sqrt(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)))
    alpha = contribution_dist.concentration1
    beta = contribution_dist.concentration0

    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = np.sqrt(variance)

    # Convert to percentages
    target_contributions = {}
    # media_channel is a DataArray, convert to list
    channels = list(mmm.input_data.media_channel.values) if hasattr(mmm.input_data.media_channel, 'values') else list(mmm.input_data.media_channel)
    for i, channel in enumerate(channels):
      target_contributions[str(channel)] = {
        'mean': float(mean[i] * 100),
        'std': float(std[i] * 100),
      }

    return target_contributions

  def _compute_realized_contributions_from_prior(
      self, mmm: 'model.Meridian'
  ) -> dict[str, dict[str, float]]:
    """Computes realized contribution % from prior samples.

    Args:
      mmm: Meridian model with prior samples

    Returns:
      Dict mapping channel name to {'mean': float, 'std': float}
    """
    # Create analyzer instance
    mmm_analyzer = analyzer.Analyzer(mmm)

    # Get incremental outcome for each channel from prior
    incremental_outcome = mmm_analyzer.incremental_outcome(
      new_data=None,
      use_posterior=False,  # Use prior samples
      aggregate_geos=False,  # Keep geo dimension
      aggregate_times=False,  # Keep time dimension
    ).numpy()  # Shape: (chain, draw, geo, time, channel)

    # Sum over geo and time
    incremental_outcome_total = incremental_outcome.sum(axis=(2, 3))
    # Shape: (chain, draw, channel)

    # Get total outcome
    total_outcome = mmm_analyzer.expected_outcome(
      new_data=None,
      use_posterior=False,  # Use prior samples
      aggregate_geos=False,
      aggregate_times=False,
    ).numpy().sum(axis=(2, 3))  # Shape: (chain, draw)

    # Calculate contribution % for each channel
    # Broadcast total_outcome to match incremental_outcome shape
    contribution_pct = (
      incremental_outcome_total / total_outcome[..., np.newaxis] * 100
    )  # Shape: (chain, draw, channel)

    # Clip extreme values to avoid outliers from near-zero total_outcome
    # These occur when baseline priors are very loose
    contribution_pct = np.clip(contribution_pct, -500, 500)  # Cap at ±500%

    # Flatten across chains and draws for each channel
    # Compute mean and std across all samples
    channels = list(mmm.input_data.media_channel.values) if hasattr(mmm.input_data.media_channel, 'values') else list(mmm.input_data.media_channel)
    realized_contributions = {}

    for i, channel in enumerate(channels):
      channel_str = str(channel)
      channel_samples = contribution_pct[:, :, i].flatten()

      realized_contributions[channel_str] = {
        'mean': float(np.mean(channel_samples)),
        'std': float(np.std(channel_samples)),
      }

    return realized_contributions

  def _extract_realized_contributions(
      self, summary_df: pd.DataFrame
  ) -> dict[str, dict[str, float]]:
    """Extracts realized contribution % from media summary table.

    Args:
      summary_df: Output from MediaSummary.summary_table_without_ci()

    Returns:
      Dict mapping channel name to {'mean': float, 'std': float}
    """
    # Filter for prior distribution rows
    prior_df = summary_df[summary_df['distribution'] == 'prior']

    realized_contributions = {}

    # Look for contribution percentage columns
    # media_channel is a DataArray, convert to list
    channels = list(self._meridian.input_data.media_channel.values) if hasattr(self._meridian.input_data.media_channel, 'values') else list(self._meridian.input_data.media_channel)
    for channel in channels:
      channel = str(channel)  # Ensure it's a string
      # Find row for this channel's contribution
      channel_row = prior_df[
        prior_df['metric'].str.contains(channel, case=False, na=False) &
        prior_df['metric'].str.contains('contribution', case=False, na=False)
      ]

      if len(channel_row) == 0:
        warnings.warn(f"Could not find contribution metric for channel '{channel}'")
        continue

      # Extract mean and std
      # The summary table should have 'mean' and 'std' columns
      if 'mean' in channel_row.columns and 'std' in channel_row.columns:
        mean_val = channel_row['mean'].values[0]
        std_val = channel_row['std'].values[0]

        realized_contributions[channel] = {
          'mean': float(mean_val * 100),  # Convert to percentage
          'std': float(std_val * 100),
        }
      else:
        warnings.warn(
          f"Could not extract mean/std for channel '{channel}' from summary table"
        )

    return realized_contributions

  def _create_comparison_table(
      self,
      target_contributions: dict[str, dict[str, float]],
      realized_contributions: dict[str, dict[str, float]],
  ) -> pd.DataFrame:
    """Creates a comparison table of target vs realized contributions.

    Args:
      target_contributions: Dict from _extract_target_contributions()
      realized_contributions: Dict from _extract_realized_contributions()

    Returns:
      DataFrame with columns: channel, target_mean_pct, target_std_pct,
        realized_mean_pct, realized_std_pct, deviation_pct
    """
    rows = []

    # media_channel is a DataArray, convert to list
    channels = list(self._meridian.input_data.media_channel.values) if hasattr(self._meridian.input_data.media_channel, 'values') else list(self._meridian.input_data.media_channel)
    for channel in channels:
      channel = str(channel)  # Ensure it's a string
      if channel not in target_contributions or channel not in realized_contributions:
        continue

      target = target_contributions[channel]
      realized = realized_contributions[channel]

      deviation = realized['mean'] - target['mean']

      rows.append({
        'channel': channel,
        'target_mean_pct': target['mean'],
        'target_std_pct': target['std'],
        'realized_mean_pct': realized['mean'],
        'realized_std_pct': realized['std'],
        'deviation_pct': deviation,
      })

    return pd.DataFrame(rows)

  def _generate_recommendations(
      self, comparison_df: pd.DataFrame
  ) -> list[str]:
    """Generates recommendations based on comparison results.

    Args:
      comparison_df: Output from _create_comparison_table()

    Returns:
      List of recommendation strings
    """
    return self.suggest_prior_adjustments(comparison_df)
