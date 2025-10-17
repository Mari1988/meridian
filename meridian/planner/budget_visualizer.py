"""BudgetVisualizer for visualizing budget optimization results.

This module provides comprehensive visualization capabilities for OptimizationResults
from the FlexibleBudgetPlanner.optimize() method.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf

from meridian.analysis.optimizer import OptimizationResults, OptimizationGrid
from meridian.model import model


class BudgetVisualizer:
  """Visualization class for budget optimization results.

  This class provides comprehensive visualization capabilities for analyzing
  budget optimization results from the Meridian FlexibleBudgetPlanner. It
  extracts key attributes from OptimizationResults and provides methods to
  visualize spend allocation, incremental outcomes, ROI, and other optimization
  metrics at both channel and geo levels.

  Attributes:
    opt_results: The OptimizationResults object from planner.optimize()
    meridian: The fitted Meridian model used for optimization
    optimization_grid: OptimizationGrid containing historical spend and grid data
    geo_level_data: xr.Dataset with geo-level optimized metrics
    optimized_data: xr.Dataset with channel-level optimized metrics
    nonoptimized_data: xr.Dataset with historical/baseline metrics
    analyzer: Analyzer bound to the Meridian model
  """

  def __init__(self, opt_results: OptimizationResults):
    """Initialize BudgetVisualizer with optimization results.

    Args:
      opt_results: OptimizationResults object from FlexibleBudgetPlanner.optimize()
        containing all optimization outputs including geo-level data, optimization
        grid, and meridian model reference.
    """
    # Store the complete opt_results object
    self.opt_results = opt_results

    # Extract key attributes from opt_results for easier access
    self.meridian: model.Meridian = opt_results.meridian
    self.optimization_grid: OptimizationGrid = opt_results.optimization_grid
    self.geo_level_data: xr.Dataset = opt_results.geo_level_optimized_data
    self.optimized_data: xr.Dataset = opt_results.optimized_data.sel(metric='mean')
    self.nonoptimized_data: xr.Dataset = opt_results.nonoptimized_data.sel(metric='mean')
    self.analyzer = opt_results.analyzer

  # ============================================================================
  # Convenience Properties
  # ============================================================================

  @property
  def channels(self) -> List[str]:
    """Get list of all paid channel names.

    Returns:
      List of channel names from the optimization results.
    """
    return self.meridian.input_data.get_all_paid_channels()

  @property
  def geos(self) -> List[str]:
    """Get list of all geo names.

    Returns:
      List of geo names from the input data.
    """
    return list(self.meridian.input_data.geo.values)

  @property
  def optimized_spend_by_geo(self) -> xr.DataArray:
    """Get optimized spend by geo and channel.

    Returns:
      xr.DataArray with dimensions (geo, channel) containing optimized spend values.
    """
    return self.geo_level_data.optimized_spend_gm

  @property
  def historical_spend_by_geo(self) -> xr.DataArray:
    """Get historical (non-optimized) spend by geo and channel.

    Returns:
      xr.DataArray with dimensions (geo, channel) containing historical spend values.
    """
    return self.geo_level_data.nonoptimized_spend_gm

  @property
  def optimized_outcome_by_geo(self) -> xr.DataArray:
    """Get optimized incremental outcome by geo and channel.

    Returns:
      xr.DataArray with dimensions (geo, channel) containing optimized incremental
      outcome values.
    """
    return self.geo_level_data.optimized_incremental_outcome_gm

  @property
  def historical_outcome_by_geo(self) -> xr.DataArray:
    """Get historical (non-optimized) incremental outcome by geo and channel.

    Returns:
      xr.DataArray with dimensions (geo, channel) containing historical incremental
      outcome values.
    """
    return self.geo_level_data.nonoptimized_incremental_outcome_gm

  @property
  def spend_change_ratio_by_geo(self) -> xr.DataArray:
    """Calculate spend change ratio (optimized / historical) by geo and channel.

    Returns:
      xr.DataArray with dimensions (geo, channel) containing spend change ratios.
      Values > 1 indicate increased spend, < 1 indicate decreased spend.
    """
    return self.optimized_spend_by_geo / self.historical_spend_by_geo

  @property
  def outcome_change_ratio_by_geo(self) -> xr.DataArray:
    """Calculate outcome change ratio (optimized / historical) by geo and channel.

    Returns:
      xr.DataArray with dimensions (geo, channel) containing outcome change ratios.
      Values > 1 indicate increased outcome, < 1 indicate decreased outcome.
    """
    return self.optimized_outcome_by_geo / self.historical_outcome_by_geo

  # ============================================================================
  # Wrapper Methods for Existing OptimizationResults Plots
  # ============================================================================

  def plot_incremental_outcome_delta(self):
    """Plot waterfall chart showing change in incremental outcome.

    Delegates to OptimizationResults.plot_incremental_outcome_delta().

    Returns:
      Altair chart object showing incremental outcome changes.
    """
    return self.opt_results.plot_incremental_outcome_delta()

  def plot_budget_allocation(self, optimized: bool = True):
    """Plot budget allocation across channels.

    Delegates to OptimizationResults.plot_budget_allocation().

    Args:
      optimized: If True, plot optimized allocation. If False, plot historical.

    Returns:
      Altair chart object showing budget allocation.
    """
    return self.opt_results.plot_budget_allocation(optimized=optimized)

  def plot_spend_delta(self):
    """Plot spend delta between optimized and historical.

    Delegates to OptimizationResults.plot_spend_delta().

    Returns:
      Altair chart object showing spend changes.
    """
    return self.opt_results.plot_spend_delta()

  def plot_response_curves(self):
    """Plot response curves for all channels.

    Delegates to OptimizationResults.plot_response_curves().

    Returns:
      Altair chart object showing response curves.
    """
    return self.opt_results.plot_response_curves()

  # ============================================================================
  # Bar Plot Visualization Methods
  # ============================================================================

  def _create_comparison_bar_plot(
      self,
      df: pd.DataFrame,
      x_col: str,
      hist_col: str,
      opt_col: str,
      title: str,
      xlabel: str,
      ylabel: str = 'Spend',
      figsize: tuple = (12, 6)
  ):
    """Internal helper to create comparison bar plots.

    Args:
      df: DataFrame containing the data to plot
      x_col: Column name for x-axis categories
      hist_col: Column name for historical values
      opt_col: Column name for optimized values
      title: Plot title
      xlabel: X-axis label
      ylabel: Y-axis label
      figsize: Figure size as (width, height)

    Returns:
      matplotlib Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set up x-axis positions
    x_pos = np.arange(len(df))
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x_pos - width/2,
        df[hist_col],
        width,
        label='Historical',
        color='#4A90E2',
        alpha=0.8
    )
    bars2 = ax.bar(
        x_pos + width/2,
        df[opt_col],
        width,
        label='Optimized',
        color='#50C878',
        alpha=0.8
    )

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df[x_col], rotation=45, ha='right')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    def add_value_labels(bars):
      for bar in bars:
        height = bar.get_height()
        if height > 0:
          # Format large numbers in millions
          if height >= 1_000_000:
            label = f'${height/1_000_000:.1f}M'
          elif height >= 1_000:
            label = f'${height/1_000:.0f}K'
          else:
            label = f'${height:.0f}'

          ax.text(
              bar.get_x() + bar.get_width() / 2,
              height,
              label,
              ha='center',
              va='bottom',
              fontsize=9,
              fontweight='bold'
          )

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Add percentage change annotations
    if 'spend_change_pct' in df.columns:
      for i, (_, row) in enumerate(df.iterrows()):
        pct_change = row['spend_change_pct']
        color = 'green' if pct_change > 0 else 'red'
        ax.text(
            x_pos[i],
            max(row[hist_col], row[opt_col]) * 1.15,
            f'{pct_change:+.1f}%',
            ha='center',
            va='bottom',
            fontsize=8,
            color=color,
            fontweight='bold'
        )

    plt.tight_layout()
    return fig

  def plot_spend_by_channel(self, figsize: tuple = (12, 6)):
    """Create grouped bar chart of optimized vs historical spend by channel.

    This method visualizes the spend comparison across all channels, showing
    both historical (non-optimized) and optimized spend values side-by-side.
    Percentage changes are annotated above the bars.

    Args:
      figsize: Figure size as (width, height) tuple. Default is (12, 6).

    Returns:
      matplotlib Figure object that can be displayed, saved, or further customized.

    Example:
      >>> visualizer = BudgetVisualizer(opt_results)
      >>> fig = visualizer.plot_spend_by_channel()
      >>> fig.savefig('channel_spend_comparison.png')
      >>> plt.show()
    """
    # Get channel summary data
    channel_df = self.get_channel_summary()

    # Remove duplicates if any (from the metric dimension)
    channel_df = channel_df.drop_duplicates(subset=['channel'])

    return self._create_comparison_bar_plot(
        df=channel_df,
        x_col='channel',
        hist_col='historical_spend',
        opt_col='optimized_spend',
        title='Spend Comparison by Channel: Historical vs Optimized',
        xlabel='Channel',
        ylabel='Spend ($)',
        figsize=figsize
    )

  def plot_spend_by_top_geos(self, top_n: int = 5, figsize: tuple = (12, 6)):
    """Create grouped bar chart of optimized vs historical spend for top N geos.

    This method identifies the top N geos by historical spend and creates a
    comparison bar chart showing both historical and optimized spend values.
    This helps focus on the geos with the highest budget allocation.

    Args:
      top_n: Number of top geos to display. Default is 5.
      figsize: Figure size as (width, height) tuple. Default is (12, 6).

    Returns:
      matplotlib Figure object that can be displayed, saved, or further customized.

    Example:
      >>> visualizer = BudgetVisualizer(opt_results)
      >>> fig = visualizer.plot_spend_by_top_geos(top_n=10)
      >>> fig.savefig('top_geos_spend_comparison.png')
      >>> plt.show()
    """
    # Get geo summary data
    geo_df = self.get_geo_summary()

    # Sort by historical spend and take top N
    top_geos_df = geo_df.sort_values('historical_spend_total', ascending=False).head(top_n)

    return self._create_comparison_bar_plot(
        df=top_geos_df,
        x_col='geo',
        hist_col='historical_spend_total',
        opt_col='optimized_spend_total',
        title=f'Spend Comparison for Top {top_n} Geos: Historical vs Optimized',
        xlabel='Geo',
        ylabel='Total Spend ($)',
        figsize=figsize
    )

  def plot_optimization_summary_card(self, figsize: tuple = (12, 4)):
    """Create a visual summary card showing total-level optimization results.

    This method creates a professional dashboard-style card displaying the three
    key metrics (Spend, Outcome, CPA) side-by-side with visual indicators showing
    the change from historical to optimized values. Each metric is color-coded:
    - Green: Improvement
    - Red: Decline
    - Gray: Minimal change

    Args:
      figsize: Figure size as (width, height) tuple. Default is (12, 4).

    Returns:
      matplotlib Figure object that can be displayed, saved, or further customized.

    Example:
      >>> visualizer = BudgetVisualizer(opt_results)
      >>> fig = visualizer.plot_optimization_summary_card()
      >>> fig.savefig('optimization_summary.png')
      >>> plt.show()
    """
    # Get summary data
    summary = self.get_optimization_summary().iloc[0]

    # Create figure with 3 columns for the metrics
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.15, 0.85], hspace=0.05, wspace=0.3)

    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.text(
        0.5, 0.5,
        f'BUDGET OPTIMIZATION SUMMARY\n{summary["start_date"]} to {summary["end_date"]}',
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=1.5)
    )

    # Metric boxes
    metrics = [
        {
            'name': 'SPEND',
            'hist': summary['total_budget_historical'],
            'opt': summary['total_budget_optimized'],
            'change_pct': summary['budget_change_pct'],
            'format': 'currency',
            'lower_is_better': False
        },
        {
            'name': 'OUTCOME',
            'hist': summary['total_outcome_historical'],
            'opt': summary['total_outcome_optimized'],
            'change_pct': summary['outcome_change_pct'],
            'format': 'number',
            'lower_is_better': False
        },
        {
            'name': 'CPA (CPIK)',
            'hist': summary['total_cpa_historical'],
            'opt': summary['total_cpa_optimized'],
            'change_pct': summary['cpa_change_pct'],
            'format': 'currency',
            'lower_is_better': True
        }
    ]

    for i, metric in enumerate(metrics):
      ax = fig.add_subplot(gs[1, i])
      ax.axis('off')

      # Determine improvement status
      is_improved = (
          (metric['change_pct'] < 0 and metric['lower_is_better']) or
          (metric['change_pct'] > 0 and not metric['lower_is_better'])
      )

      # Set colors based on improvement
      if abs(metric['change_pct']) < 0.5:
        bg_color = '#f5f5f5'
        text_color = '#666666'
        arrow = '→'
      elif is_improved:
        bg_color = '#e8f5e9'
        text_color = '#2e7d32'
        arrow = '↑' if not metric['lower_is_better'] else '↓'
      else:
        bg_color = '#ffebee'
        text_color = '#c62828'
        arrow = '↓' if not metric['lower_is_better'] else '↑'

      # Format values
      if metric['format'] == 'currency':
        if metric['hist'] >= 1_000_000:
          hist_str = f"${metric['hist']/1_000_000:.1f}M"
          opt_str = f"${metric['opt']/1_000_000:.1f}M"
        elif metric['hist'] >= 1_000:
          hist_str = f"${metric['hist']/1_000:.1f}K"
          opt_str = f"${metric['opt']/1_000:.1f}K"
        else:
          hist_str = f"${metric['hist']:.2f}"
          opt_str = f"${metric['opt']:.2f}"
      else:
        if metric['hist'] >= 1_000_000:
          hist_str = f"{metric['hist']/1_000_000:.1f}M"
          opt_str = f"{metric['opt']/1_000_000:.1f}M"
        elif metric['hist'] >= 1_000:
          hist_str = f"{metric['hist']/1_000:.1f}K"
          opt_str = f"{metric['opt']/1_000:.1f}K"
        else:
          hist_str = f"{metric['hist']:.1f}"
          opt_str = f"{metric['opt']:.1f}"

      # Create metric box
      box = plt.Rectangle((0.1, 0.1), 0.8, 0.8,
                          facecolor=bg_color,
                          edgecolor='#cccccc',
                          linewidth=2,
                          transform=ax.transAxes,
                          zorder=1)
      ax.add_patch(box)

      # Add text elements
      # Title
      ax.text(0.5, 0.85, metric['name'],
              ha='center', va='center', transform=ax.transAxes,
              fontsize=11, fontweight='bold', color='#333333')

      # Historical value (smaller, lighter)
      ax.text(0.5, 0.65, hist_str,
              ha='center', va='center', transform=ax.transAxes,
              fontsize=13, color='#888888')

      # Arrow
      ax.text(0.5, 0.50, arrow,
              ha='center', va='center', transform=ax.transAxes,
              fontsize=20, color=text_color, fontweight='bold')

      # Optimized value (larger, prominent)
      ax.text(0.5, 0.35, opt_str,
              ha='center', va='center', transform=ax.transAxes,
              fontsize=16, fontweight='bold', color='#333333')

      # Change percentage (show 0.0% instead of -0.0% for very small changes)
      if abs(metric['change_pct']) < 0.01:
        change_text = "0.0%"
      else:
        change_text = f"{metric['change_pct']:+.1f}%"
      ax.text(0.5, 0.18, change_text,
              ha='center', va='center', transform=ax.transAxes,
              fontsize=12, fontweight='bold', color=text_color)

    plt.tight_layout()
    return fig

  def plot_optimized_spend_by_geo_per_channel(
      self,
      top_n: int = 10,
      figsize: tuple = (14, 10)
  ):
    """Create bar plots showing optimized spend by geo for each channel.

    This method creates separate horizontal bar charts for each channel, showing
    the top N geos by optimized spend. Each channel gets its own subplot with
    geos ordered by spend amount (highest at top). This visualization helps
    identify which geographic markets receive the most budget allocation for
    each marketing channel.

    Args:
      top_n: Number of top geos to display per channel. Default is 10.
      figsize: Figure size as (width, height) tuple. Default is (14, 10).

    Returns:
      matplotlib Figure object that can be displayed, saved, or further customized.

    Example:
      >>> visualizer = BudgetVisualizer(opt_results)
      >>> fig = visualizer.plot_optimized_spend_by_geo_per_channel(top_n=15)
      >>> fig.savefig('geo_spend_by_channel.png')
      >>> plt.show()
    """
    # Get data
    optimized_spend = self.optimized_spend_by_geo
    channels = self.channels
    n_channels = len(channels)

    # Create subplots (one per channel)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize)
    if n_channels == 1:
      axes = [axes]

    # Define channel colors (consistent palette)
    channel_colors = {
        'Display': '#4A90E2',
        'TV': '#FF8C42',
        'Video': '#9B59B6',
        # Add more colors for additional channels
    }

    for i, channel in enumerate(channels):
      ax = axes[i]

      # Get spend for this channel
      channel_spend = optimized_spend.sel(channel=channel)

      # Sort and get top N geos
      sorted_spend = channel_spend.sortby(channel_spend, ascending=False)
      top_geos = sorted_spend.head(top_n)

      # Prepare data for plotting
      geos = top_geos.geo.values
      spend_values = top_geos.values

      # Calculate percentage of channel budget
      total_channel_spend = channel_spend.sum().values
      pct_values = (spend_values / total_channel_spend) * 100

      # Create horizontal bar chart
      y_pos = np.arange(len(geos))
      ax.barh(
          y_pos,
          spend_values,
          color=channel_colors.get(str(channel), '#888888'),
          alpha=0.8,
          edgecolor='white',
          linewidth=0.5
      )

      # Formatting
      ax.set_yticks(y_pos)
      ax.set_yticklabels(geos, fontsize=9)
      ax.invert_yaxis()  # Top geo at top
      ax.set_xlabel('Optimized Spend ($)', fontsize=10, fontweight='bold')
      ax.set_title(
          f'{channel} - Top {top_n} Geos by Optimized Spend',
          fontweight='bold',
          fontsize=11,
          pad=10
      )
      ax.grid(axis='x', alpha=0.3, linestyle='--')

      # Add value labels on bars with percentage
      for j, (value, pct) in enumerate(zip(spend_values, pct_values)):
        # Format value
        if value >= 1_000_000:
          label = f'${value/1_000_000:.1f}M ({pct:.1f}%)'
        elif value >= 1_000:
          label = f'${value/1_000:.0f}K ({pct:.1f}%)'
        else:
          label = f'${value:.0f} ({pct:.1f}%)'

        # Position label at end of bar
        ax.text(
            value,
            j,
            f' {label}',
            va='center',
            ha='left',
            fontsize=8,
            fontweight='bold'
        )

    plt.tight_layout()
    return fig

  # ============================================================================
  # MROI (Marginal ROI) Calculation Methods
  # ============================================================================

  def calculate_mroi_by_geo(
      self,
      geos: Optional[List[str]] = None
  ) -> pd.DataFrame:
    """Calculate marginal ROI (MROI) by geo and channel from optimization grid.

    This method computes the marginal return on investment for each channel at
    each point in the optimization grid. MROI represents the incremental outcome
    per incremental spend unit, calculated as the derivative of the response curve.

    The calculation follows:
      1. Extract spend_grid and incremental_outcome_grid for each geo
      2. Calculate delta_outcome = outcome_grid[1:] - outcome_grid[:-1]
      3. Calculate delta_spend = spend_grid[1:] - spend_grid[:-1]
      4. Calculate MROI = delta_outcome / delta_spend (using divide_no_nan)

    Args:
      geos: Optional list of geo names to calculate MROI for. If None, calculates
        for all geos in the optimization grid. Default is None.

    Returns:
      pd.DataFrame with columns [geo, grid_idx, channel1, channel2, ...] where
      each row represents MROI values at a specific grid point for a geo.
      The DataFrame has one fewer grid_idx per geo than the original grid
      (due to delta calculation).

    Example:
      >>> visualizer = BudgetVisualizer(opt_results)
      >>> mroi_df = visualizer.calculate_mroi_by_geo()
      >>> print(mroi_df.head())
         geo  grid_idx        TV   Display     Video
      0  TX         0  0.300000  0.200000  0.100000
      1  TX         1  0.290000  0.200000  0.090000
      2  NY         0  0.310000  0.210000  0.105000

      >>> # Calculate for specific geos only
      >>> mroi_df = visualizer.calculate_mroi_by_geo(geos=['TX', 'CA', 'NY'])

    Raises:
      ValueError: If any specified geo is not found in the optimization grid.
    """
    # Access the grid dataset
    grid_dataset = self.optimization_grid.grid_dataset

    # Get list of geos to process
    all_geos = grid_dataset.geo.values.tolist()
    if geos is None:
      geos_to_process = all_geos
    else:
      # Validate that all requested geos exist
      invalid_geos = set(geos) - set(all_geos)
      if invalid_geos:
        raise ValueError(
            f"The following geos are not found in the optimization grid: {invalid_geos}"
        )
      geos_to_process = geos

    # Get channel names
    channels = grid_dataset.channel.values.tolist()

    # Container for results across all geos
    all_results = []

    # Process each geo
    for geo in geos_to_process:
      # Select data for this geo
      grid_dataset_g = grid_dataset.sel(geo=geo)

      # Extract spend and outcome grids
      spend_grid_g = grid_dataset_g.spend_grid.values  # Shape: [grid_idx, channels]
      outcome_grid_g = grid_dataset_g.incremental_outcome_grid.values  # Shape: [grid_idx, channels]

      # Calculate deltas (differences between consecutive grid points)
      outcome_delta = outcome_grid_g[1:] - outcome_grid_g[:-1]  # Shape: [grid_idx-1, channels]
      spend_delta = spend_grid_g[1:] - spend_grid_g[:-1]  # Shape: [grid_idx-1, channels]

      # Calculate MROI using TensorFlow's safe division (handles divide-by-zero)
      mroi_g = tf.math.divide_no_nan(outcome_delta, spend_delta)

      # Convert to numpy array
      mroi_g_np = mroi_g.numpy()

      # Create DataFrame for this geo
      # Each row is a grid point, columns are channels
      num_grid_points = mroi_g_np.shape[0]
      geo_df = pd.DataFrame(
          mroi_g_np,
          columns=channels
      )

      # Add geo and grid_idx columns
      geo_df.insert(0, 'grid_idx', range(num_grid_points))
      geo_df.insert(0, 'geo', geo)

      all_results.append(geo_df)

    # Combine all geos into single DataFrame
    result_df = pd.concat(all_results, ignore_index=True)

    return result_df

  # ============================================================================
  # Summary Methods
  # ============================================================================

  def get_optimization_summary(self) -> pd.DataFrame:
    """Get high-level optimization summary comparing optimized vs historical.

    Returns:
      DataFrame with columns:
        - start_date, end_date
        - total_budget_historical, total_budget_optimized, budget_change_pct
        - total_outcome_historical, total_outcome_optimized, outcome_change_pct
        - total_cpa_historical, total_cpa_optimized, cpa_change_pct
        - total_roi_historical, total_roi_optimized
    """
    summary = {
      'start_date': self.optimized_data.attrs.get('start_date', ''),
      'end_date': self.optimized_data.attrs.get('end_date', ''),
      'total_budget_historical': float(self.nonoptimized_data.attrs.get('budget', 0)),
      'total_budget_optimized': float(self.optimized_data.attrs.get('budget', 0)),
      'total_outcome_historical': float(
        self.nonoptimized_data.attrs.get('total_incremental_outcome', 0)
      ),
      'total_outcome_optimized': float(
        self.optimized_data.attrs.get('total_incremental_outcome', 0)
      ),
      'total_cpa_historical': float(self.nonoptimized_data.attrs.get('total_cpik', 0)),
      'total_cpa_optimized': float(self.optimized_data.attrs.get('total_cpik', 0)),
      'total_roi_historical': float(self.nonoptimized_data.attrs.get('total_roi', 0)),
      'total_roi_optimized': float(self.optimized_data.attrs.get('total_roi', 0)),
    }

    # Calculate percentage changes
    summary['budget_change_pct'] = (
      (summary['total_budget_optimized'] / summary['total_budget_historical'] - 1) * 100
      if summary['total_budget_historical'] > 0 else 0
    )
    summary['outcome_change_pct'] = (
      (summary['total_outcome_optimized'] / summary['total_outcome_historical'] - 1) * 100
      if summary['total_outcome_historical'] > 0 else 0
    )
    summary['cpa_change_pct'] = (
      (summary['total_cpa_optimized'] / summary['total_cpa_historical'] - 1) * 100
      if summary['total_cpa_historical'] > 0 else 0
    )

    return pd.DataFrame([summary])

  def get_channel_summary(self) -> pd.DataFrame:
    """Get channel-level summary comparing optimized vs historical.

    Returns:
      DataFrame with columns:
        - channel
        - historical_spend
        - optimized_spend
        - spend_change_pct
        - historical_outcome
        - optimized_outcome
        - outcome_change_pct
    """
    # Convert optimized and nonoptimized datasets to dataframes
    opt_df = self.optimized_data.to_dataframe().reset_index()
    nonopt_df = self.nonoptimized_data.to_dataframe().reset_index()

    summary_df = pd.DataFrame({
      'channel': opt_df['channel'],
      'historical_spend': nonopt_df['spend'],
      'optimized_spend': opt_df['spend'],
      'historical_outcome': nonopt_df['incremental_outcome'],
      'optimized_outcome': opt_df['incremental_outcome'],
    })

    # Calculate percentage changes
    summary_df['spend_change_pct'] = (
      (summary_df['optimized_spend'] / summary_df['historical_spend'] - 1) * 100
    )
    summary_df['outcome_change_pct'] = (
      (summary_df['optimized_outcome'] / summary_df['historical_outcome'] - 1) * 100
    )

    return summary_df

  def get_geo_summary(self) -> pd.DataFrame:
    """Get geo-level summary of total spend and outcome across all channels.

    Returns:
      DataFrame with columns:
        - geo
        - historical_spend_total
        - optimized_spend_total
        - spend_change_pct
        - historical_outcome_total
        - optimized_outcome_total
        - outcome_change_pct
    """
    # Sum across channels for each geo
    geo_data = []
    for geo in self.geos:
      hist_spend = float(self.historical_spend_by_geo.sel(geo=geo).sum())
      opt_spend = float(self.optimized_spend_by_geo.sel(geo=geo).sum())
      hist_outcome = float(self.historical_outcome_by_geo.sel(geo=geo).sum())
      opt_outcome = float(self.optimized_outcome_by_geo.sel(geo=geo).sum())

      geo_data.append({
        'geo': geo,
        'historical_spend_total': hist_spend,
        'optimized_spend_total': opt_spend,
        'historical_outcome_total': hist_outcome,
        'optimized_outcome_total': opt_outcome,
        'spend_change_pct': ((opt_spend / hist_spend - 1) * 100) if hist_spend > 0 else 0,
        'outcome_change_pct': ((opt_outcome / hist_outcome - 1) * 100) if hist_outcome > 0 else 0,
      })

    return pd.DataFrame(geo_data)

  def _get_top_geos_by_spend(self, top_n: int) -> List[str]:
    """Get top N geos by historical spend (internal helper method).

    Args:
      top_n: Number of top geos to return.

    Returns:
      List of geo names ordered by historical spend (descending).

    Example:
      >>> visualizer = BudgetVisualizer(opt_results)
      >>> top_5_geos = visualizer._get_top_geos_by_spend(5)
      >>> print(top_5_geos)
      ['California', 'Texas', 'New York', 'Florida', 'Illinois']
    """
    geo_summary = self.get_geo_summary()
    top_geos = geo_summary.sort_values(
        'historical_spend_total',
        ascending=False
    ).head(top_n)
    return top_geos['geo'].tolist()

  def _calculate_mroi_bounds_summary(self, mroi_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MROI at lower and upper bounds for all geos.

    This internal method extracts MROI values at the lower bound (grid_idx = 0)
    and upper bound (highest non-NaN grid_idx) for each geo and channel.

    Args:
      mroi_df: DataFrame from calculate_mroi_by_geo() with columns:
        ['geo', 'grid_idx', channel1, channel2, ...]

    Returns:
      DataFrame with columns ['geo', 'bound_type', channel1, channel2, ...]
      where bound_type is either 'lower_bound' or 'upper_bound'.
      Each geo will have 2 rows (one for each bound type).

    Example:
      >>> mroi_df = visualizer.calculate_mroi_by_geo()
      >>> bounds = visualizer._calculate_mroi_bounds_summary(mroi_df)
      >>> print(bounds)
         geo       bound_type     TV   Display     Video
      0  Texas     lower_bound  0.30  0.20        0.10
      1  Texas     upper_bound  0.15  0.12        0.08
      2  CA        lower_bound  0.35  0.22        0.12
      3  CA        upper_bound  0.18  0.14        0.09
    """
    # Step 1: Get lower bound (grid_idx == 0)
    mroi_at_lb = mroi_df.query("grid_idx == 0").copy()
    mroi_at_lb['bound_type'] = 'lower_bound'
    mroi_at_lb = mroi_at_lb.drop(columns=['grid_idx'])

    # Step 2: Get upper bound (max non-NaN grid_idx per geo and channel)
    upper_bounds = []
    channels = [col for col in mroi_df.columns if col not in ['geo', 'grid_idx']]

    for geo in mroi_df['geo'].unique():
      geo_data = mroi_df[mroi_df['geo'] == geo]

      # For each channel, find last non-NaN value
      last_valid_row = {}
      for channel in channels:
        valid_data = geo_data[['grid_idx', channel]].dropna(subset=[channel])
        if len(valid_data) > 0:
          last_valid_row[channel] = valid_data.iloc[-1][channel]
        else:
          # If all values are NaN, use NaN
          last_valid_row[channel] = np.nan

      upper_bounds.append({
          'geo': geo,
          'bound_type': 'upper_bound',
          **last_valid_row
      })

    mroi_at_ub = pd.DataFrame(upper_bounds)

    # Step 3: Concatenate lower_bound and upper_bound DataFrames
    result = pd.concat([mroi_at_lb, mroi_at_ub], ignore_index=True)

    # Step 4: Reorder columns to match: [geo, bound_type, channel1, channel2, ...]
    cols = ['geo', 'bound_type'] + channels
    result = result[cols]

    return result

  # ============================================================================
  # MROI Excel Export Methods
  # ============================================================================

  def export_mroi_to_excel(
      self,
      file_path: str,
      top_n: Optional[int] = None,
      selected_geos: Optional[List[str]] = None
  ) -> str:
    """Export marginal ROI data to Excel with separate sheets per geo.

    This method calculates MROI for all or selected geos and exports the data
    to an Excel file where each geo gets its own sheet. The sheets contain
    MROI values at each grid point for all channels.

    Args:
      file_path: Path where the Excel file should be saved (e.g., 'mroi_data.xlsx').
        If the path doesn't end with '.xlsx', it will be appended automatically.
      top_n: Optional number of top geos (by historical spend) to export.
        If specified, only the top N geos will be included. Mutually exclusive
        with selected_geos.
      selected_geos: Optional list of specific geo names to export. Mutually
        exclusive with top_n.

    Returns:
      str: The file path where the Excel file was saved.

    Raises:
      ValueError: If both top_n and selected_geos are specified, or if specified
        geos are not found in the data.

    Example:
      >>> visualizer = BudgetVisualizer(opt_results)

      >>> # Export all geos
      >>> file_path = visualizer.export_mroi_to_excel('mroi_all_geos.xlsx')

      >>> # Export top 10 geos by spend
      >>> file_path = visualizer.export_mroi_to_excel(
      ...     'mroi_top10.xlsx',
      ...     top_n=10
      ... )

      >>> # Export specific geos
      >>> file_path = visualizer.export_mroi_to_excel(
      ...     'mroi_selected.xlsx',
      ...     selected_geos=['California', 'Texas', 'New York']
      ... )

    Excel Output Format:
      The workbook contains:

      1. Individual geo sheets (one per geo):
         - Column 'grid_idx': Grid point index (0, 1, 2, ...)
         - Columns for each channel: MROI values at each grid point

         Example sheet "California":
         | grid_idx | TV    | Display | Video |
         |----------|-------|---------|-------|
         | 0        | 0.300 | 0.200   | 0.100 |
         | 1        | 0.290 | 0.200   | 0.090 |
         | 2        | 0.280 | 0.195   | 0.088 |

      2. Summary sheet "mroi bounds":
         - Column 'geo': Geographic identifier
         - Column 'bound_type': Either 'lower_bound' or 'upper_bound'
         - Columns for each channel: MROI values at the respective bound

         This sheet contains MROI at the lower bound (grid_idx=0) and upper
         bound (highest non-NaN grid_idx) for each geo and channel.

         Example:
         | geo       | bound_type   | Display   | TV        | Video     |
         |-----------|--------------|-----------|-----------|-----------|
         | ALABAMA   | lower_bound  | 1.889162  | 4.661135  | 6.611535  |
         | ALABAMA   | upper_bound  | 0.500000  | 1.200000  | 0.800000  |
         | ARIZONA   | lower_bound  | 13.230233 | 4.325667  | 2.563205  |
         | ARIZONA   | upper_bound  | 2.100000  | 1.500000  | 0.900000  |
    """
    # Validate arguments
    if top_n is not None and selected_geos is not None:
      raise ValueError(
          "Cannot specify both 'top_n' and 'selected_geos'. "
          "Please use only one filtering option."
      )

    # Ensure file path has .xlsx extension
    if not file_path.endswith('.xlsx'):
      file_path = file_path + '.xlsx'

    # Determine which geos to export
    if selected_geos is not None:
      geos_to_export = selected_geos
    elif top_n is not None:
      geos_to_export = self._get_top_geos_by_spend(top_n)
    else:
      # Export all geos
      geos_to_export = None

    # Calculate MROI for selected geos
    mroi_df = self.calculate_mroi_by_geo(geos=geos_to_export)

    # Create Excel writer
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
      # Write individual geo sheets
      for geo in mroi_df['geo'].unique():
        geo_data = mroi_df[mroi_df['geo'] == geo].copy()

        # Drop the geo column (it's redundant in the sheet)
        geo_data = geo_data.drop(columns=['geo'])

        # Write to sheet named after the geo
        # Excel sheet names are limited to 31 characters
        sheet_name = str(geo)[:31]
        geo_data.to_excel(writer, sheet_name=sheet_name, index=False)

      # Calculate and write MROI bounds summary sheet
      mroi_bounds_summary = self._calculate_mroi_bounds_summary(mroi_df)
      mroi_bounds_summary.to_excel(writer, sheet_name='mroi bounds', index=False)

    return file_path

  # ============================================================================
  # Placeholder Methods for Future Geo-Level Visualizations
  # ============================================================================

  def plot_geo_spend_changes(self):
    """Plot spend changes by geo (to be implemented)."""
    raise NotImplementedError("This visualization will be implemented in the next phase.")

  def plot_geo_outcome_changes(self):
    """Plot outcome changes by geo (to be implemented)."""
    raise NotImplementedError("This visualization will be implemented in the next phase.")

  def plot_geo_channel_heatmap(self):
    """Plot heatmap of spend or outcome by geo and channel (to be implemented)."""
    raise NotImplementedError("This visualization will be implemented in the next phase.")
