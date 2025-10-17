"""ModelDataDeepDive for analyzing model input data and geo performance differences.

This module provides deep-dive analysis capabilities for understanding why certain
geos perform differently by examining time-series patterns, correlations, and
input data characteristics.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from meridian.model import model


class ModelDataDeepDive:
  """Deep-dive analysis of model data to understand geo performance differences.

  This class analyzes input data (impressions, conversions) to explain why
  certain geos perform differently. It focuses on time-series patterns,
  correlations, and data quality indicators.

  Attributes:
    meridian: Fitted Meridian model with input_data
    input_data: InputData object from the Meridian model
  """

  def __init__(self, meridian_model: model.Meridian):
    """Initialize ModelDataDeepDive with a fitted Meridian model.

    Args:
      meridian_model: Fitted Meridian model object containing input_data.

    Raises:
      ValueError: If meridian_model is None or doesn't have input_data.
    """
    if meridian_model is None:
      raise ValueError("meridian_model cannot be None")
    if not hasattr(meridian_model, 'input_data'):
      raise ValueError("meridian_model must have input_data attribute")

    self.meridian = meridian_model
    self.input_data = meridian_model.input_data

  # =============================================================================
  # Helper Methods for Data Extraction
  # =============================================================================

  def _get_channel_timeseries(self, geo: str, channel: str) -> pd.Series:
    """Extract impression time-series for a specific geo and channel.

    Args:
      geo: Geographic identifier (e.g., 'NEW_YORK')
      channel: Media channel name (e.g., 'TV')

    Returns:
      pd.Series with time index and impression values.

    Raises:
      ValueError: If geo or channel not found in input_data.
    """
    # Validate geo
    if geo not in self.input_data.geo.values:
      raise ValueError(f"Geo '{geo}' not found in input data")

    # Validate channel
    all_channels = self.input_data.get_all_paid_channels()
    if channel not in all_channels:
      raise ValueError(
          f"Channel '{channel}' not found. Available: {all_channels}"
      )

    # Extract media data for this geo and channel
    media_data = self.input_data.media.sel(geo=geo, media_channel=channel)

    # Create pandas Series with time index
    time_coords = media_data.media_time.values
    values = media_data.values

    series = pd.Series(values, index=time_coords, name=f"{geo}_{channel}")
    return series

  def _get_kpi_timeseries(self, geo: str) -> pd.Series:
    """Extract KPI (conversions) time-series for a specific geo.

    Args:
      geo: Geographic identifier (e.g., 'NEW_YORK')

    Returns:
      pd.Series with time index and KPI values.

    Raises:
      ValueError: If geo not found in input_data.
    """
    # Validate geo
    if geo not in self.input_data.geo.values:
      raise ValueError(f"Geo '{geo}' not found in input data")

    # Extract KPI data for this geo
    kpi_data = self.input_data.kpi.sel(geo=geo)

    # Create pandas Series with time index
    time_coords = kpi_data.time.values
    values = kpi_data.values

    series = pd.Series(values, index=time_coords, name=f"{geo}_KPI")
    return series

  def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> float:
    """Calculate Pearson correlation between two time series.

    Args:
      x: First time series
      y: Second time series

    Returns:
      Pearson correlation coefficient (-1 to 1).
    """
    # Align series by index
    aligned_x, aligned_y = x.align(y, join='inner')

    # Remove NaN values
    mask = ~(aligned_x.isna() | aligned_y.isna())
    clean_x = aligned_x[mask]
    clean_y = aligned_y[mask]

    if len(clean_x) < 2:
      return np.nan

    correlation, _ = stats.pearsonr(clean_x, clean_y)
    return correlation

  # =============================================================================
  # Main Analysis Method
  # =============================================================================

  def compare_geo_performance(
      self,
      inspect_geo: str,
      compare_geo: str,
      channel: str
  ) -> Dict[str, Any]:
    """Compare performance between two geos for a specific channel.

    This method analyzes time-series patterns, correlations with KPI, and
    generates insights explaining performance differences.

    Args:
      inspect_geo: Geo to investigate (e.g., 'NEW_YORK')
      compare_geo: Reference geo for comparison (e.g., 'OREGON')
      channel: Media channel to analyze (e.g., 'TV')

    Returns:
      Dict containing:
        - 'inspect_geo_stats': Time series statistics for inspect_geo
        - 'compare_geo_stats': Time series statistics for compare_geo
        - 'comparison_metrics': Direct comparison metrics
        - 'insights': List of actionable insight strings

    Example:
      >>> deepdive = ModelDataDeepDive(mmm)
      >>> results = deepdive.compare_geo_performance('NEW_YORK', 'OREGON', 'TV')
      >>> for insight in results['insights']:
      ...     print(insight)
    """
    # Extract time series
    inspect_impressions = self._get_channel_timeseries(inspect_geo, channel)
    compare_impressions = self._get_channel_timeseries(compare_geo, channel)
    inspect_kpi = self._get_kpi_timeseries(inspect_geo)
    compare_kpi = self._get_kpi_timeseries(compare_geo)

    # Calculate statistics for inspect_geo
    inspect_stats = {
        'geo': inspect_geo,
        'channel': channel,
        'mean_impressions': float(inspect_impressions.mean()),
        'median_impressions': float(inspect_impressions.median()),
        'std_impressions': float(inspect_impressions.std()),
        'cv_impressions': float(inspect_impressions.std() / inspect_impressions.mean())
            if inspect_impressions.mean() > 0 else np.nan,
        'min_impressions': float(inspect_impressions.min()),
        'max_impressions': float(inspect_impressions.max()),
        'zero_weeks': int((inspect_impressions == 0).sum()),
        'total_weeks': len(inspect_impressions),
        'correlation_with_kpi': self._calculate_correlation(inspect_impressions, inspect_kpi),
        'mean_kpi': float(inspect_kpi.mean()),
    }

    # Calculate statistics for compare_geo
    compare_stats = {
        'geo': compare_geo,
        'channel': channel,
        'mean_impressions': float(compare_impressions.mean()),
        'median_impressions': float(compare_impressions.median()),
        'std_impressions': float(compare_impressions.std()),
        'cv_impressions': float(compare_impressions.std() / compare_impressions.mean())
            if compare_impressions.mean() > 0 else np.nan,
        'min_impressions': float(compare_impressions.min()),
        'max_impressions': float(compare_impressions.max()),
        'zero_weeks': int((compare_impressions == 0).sum()),
        'total_weeks': len(compare_impressions),
        'correlation_with_kpi': self._calculate_correlation(compare_impressions, compare_kpi),
        'mean_kpi': float(compare_kpi.mean()),
    }

    # Calculate comparison metrics
    comparison_metrics = {
        'mean_impressions_ratio': inspect_stats['mean_impressions'] / compare_stats['mean_impressions']
            if compare_stats['mean_impressions'] > 0 else np.nan,
        'mean_impressions_diff_pct': ((inspect_stats['mean_impressions'] / compare_stats['mean_impressions']) - 1) * 100
            if compare_stats['mean_impressions'] > 0 else np.nan,
        'correlation_diff': inspect_stats['correlation_with_kpi'] - compare_stats['correlation_with_kpi'],
        'correlation_diff_pct': ((inspect_stats['correlation_with_kpi'] / compare_stats['correlation_with_kpi']) - 1) * 100
            if compare_stats['correlation_with_kpi'] != 0 else np.nan,
        'zero_weeks_diff': inspect_stats['zero_weeks'] - compare_stats['zero_weeks'],
        'cv_ratio': inspect_stats['cv_impressions'] / compare_stats['cv_impressions']
            if compare_stats['cv_impressions'] > 0 else np.nan,
    }

    # Generate insights
    insights = self._generate_insights(
        inspect_stats,
        compare_stats,
        comparison_metrics
    )

    return {
        'inspect_geo_stats': inspect_stats,
        'compare_geo_stats': compare_stats,
        'comparison_metrics': comparison_metrics,
        'insights': insights,
    }

  def _generate_insights(
      self,
      inspect_stats: Dict[str, Any],
      compare_stats: Dict[str, Any],
      comparison_metrics: Dict[str, Any]
  ) -> list[str]:
    """Generate actionable insights from comparison analysis.

    Args:
      inspect_stats: Statistics for the inspect geo
      compare_stats: Statistics for the compare geo
      comparison_metrics: Comparison metrics between the two geos

    Returns:
      List of insight strings explaining key differences.
    """
    insights = []
    inspect_geo = inspect_stats['geo']
    compare_geo = compare_stats['geo']
    channel = inspect_stats['channel']

    # Insight 1: Mean impression difference
    mean_diff_pct = comparison_metrics['mean_impressions_diff_pct']
    if not np.isnan(mean_diff_pct):
      if abs(mean_diff_pct) > 10:
        direction = "higher" if mean_diff_pct > 0 else "lower"
        insights.append(
            f"{inspect_geo} has {abs(mean_diff_pct):.1f}% {direction} average {channel} "
            f"impressions than {compare_geo} "
            f"({inspect_stats['mean_impressions']:.0f} vs {compare_stats['mean_impressions']:.0f})"
        )

    # Insight 2: Correlation difference
    inspect_corr = inspect_stats['correlation_with_kpi']
    compare_corr = compare_stats['correlation_with_kpi']
    if not (np.isnan(inspect_corr) or np.isnan(compare_corr)):
      corr_diff = comparison_metrics['correlation_diff']
      if abs(corr_diff) > 0.1:
        insights.append(
            f"Correlation with conversions: {inspect_geo} ({inspect_corr:.2f}) "
            f"vs {compare_geo} ({compare_corr:.2f}) - "
            f"{abs(corr_diff):.2f} {'weaker' if corr_diff < 0 else 'stronger'}"
        )

    # Insight 3: Zero activity periods
    zero_diff = comparison_metrics['zero_weeks_diff']
    if zero_diff > 0:
      inspect_pct = (inspect_stats['zero_weeks'] / inspect_stats['total_weeks']) * 100
      compare_pct = (compare_stats['zero_weeks'] / compare_stats['total_weeks']) * 100
      insights.append(
          f"{inspect_geo} has {inspect_stats['zero_weeks']} weeks ({inspect_pct:.1f}%) "
          f"with zero {channel} activity vs {compare_geo} "
          f"({compare_stats['zero_weeks']} weeks, {compare_pct:.1f}%)"
      )

    # Insight 4: Volatility difference
    cv_ratio = comparison_metrics['cv_ratio']
    if not np.isnan(cv_ratio) and abs(cv_ratio - 1.0) > 0.3:
      if cv_ratio > 1.3:
        insights.append(
            f"{inspect_geo} {channel} impressions are {cv_ratio:.1f}x more volatile "
            f"than {compare_geo}"
        )
      elif cv_ratio < 0.7:
        insights.append(
            f"{inspect_geo} {channel} impressions are more stable (less volatile) "
            f"than {compare_geo}"
        )

    # Insight 5: Overall assessment
    if not np.isnan(inspect_corr) and inspect_corr < 0.3:
      insights.append(
          f"⚠️  Weak correlation ({inspect_corr:.2f}) suggests {channel} "
          f"may not be effective in {inspect_geo} market"
      )
    elif not np.isnan(compare_corr) and compare_corr > 0.6:
      insights.append(
          f"✓ Strong correlation ({compare_corr:.2f}) in {compare_geo} "
          f"indicates {channel} is effective in this market"
      )

    return insights

  # =============================================================================
  # Visualization Method
  # =============================================================================

  def plot_time_series_comparison(
      self,
      inspect_geo: str,
      compare_geo: str,
      channel: str,
      figsize: tuple = (16, 12)
  ) -> plt.Figure:
    """Create time series comparison plots between two geos.

    This method generates a 4-subplot figure showing:
    1. Inspect-Geo impressions vs Compare-Geo impressions (time series overlay)
    2. Inspect-Geo impressions vs Inspect-Geo KPI (scatter with trend)
    3. Compare-Geo impressions vs Compare-Geo KPI (scatter with trend)
    4. Inspect-Geo KPI vs Compare-Geo KPI (time series overlay)

    Args:
      inspect_geo: Geo to investigate (e.g., 'NEW_YORK')
      compare_geo: Reference geo for comparison (e.g., 'OREGON')
      channel: Media channel to analyze (e.g., 'TV')
      figsize: Figure size as (width, height) tuple

    Returns:
      matplotlib Figure object

    Example:
      >>> fig = deepdive.plot_time_series_comparison('NEW_YORK', 'OREGON', 'TV')
      >>> fig.savefig('comparison.png')
      >>> plt.show()
    """
    # Extract time series
    inspect_impressions = self._get_channel_timeseries(inspect_geo, channel)
    compare_impressions = self._get_channel_timeseries(compare_geo, channel)
    inspect_kpi = self._get_kpi_timeseries(inspect_geo)
    compare_kpi = self._get_kpi_timeseries(compare_geo)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1 (Top Left): Inspect-Geo impressions vs Compare-Geo impressions
    ax1 = axes[0, 0]
    ax1.plot(inspect_impressions.index, inspect_impressions.values,
             label=f'{inspect_geo} Impressions', color='#E74C3C', linewidth=2, alpha=0.8)
    ax1.plot(compare_impressions.index, compare_impressions.values,
             label=f'{compare_geo} Impressions', color='#3498DB', linewidth=2, alpha=0.8)
    ax1.set_title(f'1. {channel} Impressions: {inspect_geo} vs {compare_geo}',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'{channel} Impressions', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, frameon=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle='--')

    # Plot 2 (Top Right): Inspect-Geo impressions vs Inspect-Geo KPI
    ax2 = axes[0, 1]

    # Align series for inspect geo
    inspect_imp_aligned, inspect_kpi_aligned = inspect_impressions.align(
        inspect_kpi, join='inner'
    )

    # Scatter plot
    ax2.scatter(inspect_imp_aligned, inspect_kpi_aligned,
                color='#E74C3C', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Add trend line
    if len(inspect_imp_aligned) > 1:
      z = np.polyfit(inspect_imp_aligned, inspect_kpi_aligned, 1)
      p = np.poly1d(z)
      x_line = np.linspace(inspect_imp_aligned.min(), inspect_imp_aligned.max(), 100)
      ax2.plot(x_line, p(x_line), color='#C0392B', linestyle='--',
               linewidth=2.5, alpha=0.9, label='Trend Line')

      # Calculate and display correlation
      corr = self._calculate_correlation(inspect_imp_aligned, inspect_kpi_aligned)
      r_squared = corr**2 if not np.isnan(corr) else 0
      ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}\nCorr = {corr:.3f}',
               transform=ax2.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#E74C3C', alpha=0.3, pad=0.5))

    ax2.set_title(f'2. {inspect_geo}: {channel} Impressions vs KPI',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel(f'{channel} Impressions', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Conversions (KPI)', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')

    # Plot 3 (Bottom Left): Compare-Geo impressions vs Compare-Geo KPI
    ax3 = axes[1, 0]

    # Align series for compare geo
    compare_imp_aligned, compare_kpi_aligned = compare_impressions.align(
        compare_kpi, join='inner'
    )

    # Scatter plot
    ax3.scatter(compare_imp_aligned, compare_kpi_aligned,
                color='#3498DB', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Add trend line
    if len(compare_imp_aligned) > 1:
      z = np.polyfit(compare_imp_aligned, compare_kpi_aligned, 1)
      p = np.poly1d(z)
      x_line = np.linspace(compare_imp_aligned.min(), compare_imp_aligned.max(), 100)
      ax3.plot(x_line, p(x_line), color='#2471A3', linestyle='--',
               linewidth=2.5, alpha=0.9, label='Trend Line')

      # Calculate and display correlation
      corr = self._calculate_correlation(compare_imp_aligned, compare_kpi_aligned)
      r_squared = corr**2 if not np.isnan(corr) else 0
      ax3.text(0.05, 0.95, f'R² = {r_squared:.3f}\nCorr = {corr:.3f}',
               transform=ax3.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#3498DB', alpha=0.3, pad=0.5))

    ax3.set_title(f'3. {compare_geo}: {channel} Impressions vs KPI',
                  fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlabel(f'{channel} Impressions', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Conversions (KPI)', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')

    # Plot 4 (Bottom Right): Inspect-Geo KPI vs Compare-Geo KPI
    ax4 = axes[1, 1]
    ax4.plot(inspect_kpi.index, inspect_kpi.values,
             label=f'{inspect_geo} KPI', color='#E74C3C', linewidth=2, alpha=0.8)
    ax4.plot(compare_kpi.index, compare_kpi.values,
             label=f'{compare_geo} KPI', color='#3498DB', linewidth=2, alpha=0.8)
    ax4.set_title(f'4. Conversions (KPI): {inspect_geo} vs {compare_geo}',
                  fontsize=13, fontweight='bold', pad=15)
    ax4.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Conversions (KPI)', fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, frameon=True, shadow=True)
    ax4.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig
