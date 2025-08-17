"""Visualization functions for Mini MMM.

This module provides plotting functions for model diagnostics, media effects,
response curves, and budget optimization results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
from mini_mmm.model.mini_mmm import MiniMMM
from mini_mmm.analysis.analyzer import Analyzer
from mini_mmm.analysis.response_curves import ResponseCurves
from mini_mmm.analysis.optimizer import BudgetOptimizer

# Set default style
plt.style.use('default')
sns.set_palette("husl")


def plot_model_fit(model: MiniMMM, 
                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
  """Plots model fit diagnostics.
  
  Args:
    model: Fitted MiniMMM instance
    figsize: Figure size tuple
    
  Returns:
    Matplotlib figure
  """
  if not model.is_fitted:
    raise RuntimeError("Model must be fitted before plotting")
  
  # Get predictions and actual values
  predictions = model.predict()['prediction']
  actual = model.data.get_kpi_array()
  residuals = actual - predictions
  
  # Create figure with subplots
  fig, axes = plt.subplots(2, 2, figsize=figsize)
  fig.suptitle('Model Fit Diagnostics', fontsize=16, fontweight='bold')
  
  # 1. Actual vs Predicted
  ax1 = axes[0, 0]
  ax1.scatter(actual, predictions, alpha=0.6)
  min_val = min(actual.min(), predictions.min())
  max_val = max(actual.max(), predictions.max())
  ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
  ax1.set_xlabel('Actual')
  ax1.set_ylabel('Predicted')
  ax1.set_title('Actual vs Predicted')
  
  # Add R² to the plot
  r2 = 1 - np.sum(residuals**2) / np.sum((actual - np.mean(actual))**2)
  ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
  
  # 2. Time series plot
  ax2 = axes[0, 1]
  time_index = range(len(actual))
  ax2.plot(time_index, actual, label='Actual', linewidth=2)
  ax2.plot(time_index, predictions, label='Predicted', linewidth=2, alpha=0.8)
  ax2.set_xlabel('Time Period')
  ax2.set_ylabel('KPI')
  ax2.set_title('Time Series: Actual vs Predicted')
  ax2.legend()
  ax2.grid(True, alpha=0.3)
  
  # 3. Residuals plot
  ax3 = axes[1, 0]
  ax3.scatter(predictions, residuals, alpha=0.6)
  ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
  ax3.set_xlabel('Predicted')
  ax3.set_ylabel('Residuals')
  ax3.set_title('Residuals vs Predicted')
  ax3.grid(True, alpha=0.3)
  
  # 4. Residuals histogram
  ax4 = axes[1, 1]
  ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
  ax4.axvline(x=0, color='r', linestyle='--', alpha=0.8)
  ax4.set_xlabel('Residuals')
  ax4.set_ylabel('Frequency')
  ax4.set_title('Residuals Distribution')
  ax4.grid(True, alpha=0.3)
  
  plt.tight_layout()
  return fig


def plot_media_effects(model: MiniMMM,
                      figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
  """Plots media effects summary.
  
  Args:
    model: Fitted MiniMMM instance
    figsize: Figure size tuple
    
  Returns:
    Matplotlib figure
  """
  if not model.is_fitted:
    raise RuntimeError("Model must be fitted before plotting")
  
  # Get media summary
  media_summary = model.get_media_summary()
  
  fig, axes = plt.subplots(2, 2, figsize=figsize)
  fig.suptitle('Media Effects Analysis', fontsize=16, fontweight='bold')
  
  # 1. Total spend by channel
  ax1 = axes[0, 0]
  bars1 = ax1.bar(media_summary['channel'], media_summary['total_spend'])
  ax1.set_title('Total Spend by Channel')
  ax1.set_ylabel('Spend')
  ax1.tick_params(axis='x', rotation=45)
  
  # Add value labels on bars
  for bar, value in zip(bars1, media_summary['total_spend']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.0f}', ha='center', va='bottom')
  
  # 2. ROI by channel
  ax2 = axes[0, 1]
  bars2 = ax2.bar(media_summary['channel'], media_summary['roi'])
  ax2.set_title('ROI by Channel')
  ax2.set_ylabel('ROI')
  ax2.tick_params(axis='x', rotation=45)
  
  # Add value labels
  for bar, value in zip(bars2, media_summary['roi']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.2f}x', ha='center', va='bottom')
  
  # 3. Spend vs Effect share
  ax3 = axes[1, 0]
  ax3.scatter(media_summary['spend_share'] * 100, 
             media_summary['effect_share'] * 100, s=100)
  
  # Add channel labels
  for i, channel in enumerate(media_summary['channel']):
    ax3.annotate(channel, 
                (media_summary['spend_share'].iloc[i] * 100,
                 media_summary['effect_share'].iloc[i] * 100),
                xytext=(5, 5), textcoords='offset points')
  
  # Add diagonal line
  max_share = max(media_summary['spend_share'].max(), 
                  media_summary['effect_share'].max()) * 100
  ax3.plot([0, max_share], [0, max_share], 'r--', alpha=0.5)
  ax3.set_xlabel('Spend Share (%)')
  ax3.set_ylabel('Effect Share (%)')
  ax3.set_title('Spend Share vs Effect Share')
  ax3.grid(True, alpha=0.3)
  
  # 4. Media transformation parameters
  ax4 = axes[1, 1]
  x = range(len(media_summary))
  
  ax4_twin = ax4.twinx()
  
  # Retention rate (left y-axis)
  bars_ret = ax4.bar([i - 0.2 for i in x], media_summary['retention_rate'], 
                    width=0.4, alpha=0.7, label='Retention Rate', color='skyblue')
  ax4.set_ylabel('Retention Rate', color='skyblue')
  ax4.tick_params(axis='y', labelcolor='skyblue')
  
  # EC values (right y-axis)
  bars_ec = ax4_twin.bar([i + 0.2 for i in x], media_summary['ec'], 
                        width=0.4, alpha=0.7, label='EC (Shape)', color='orange')
  ax4_twin.set_ylabel('EC (Shape Parameter)', color='orange')
  ax4_twin.tick_params(axis='y', labelcolor='orange')
  
  ax4.set_xlabel('Channel')
  ax4.set_title('Media Transformation Parameters')
  ax4.set_xticks(x)
  ax4.set_xticklabels(media_summary['channel'], rotation=45)
  
  plt.tight_layout()
  return fig


def plot_response_curves(model: MiniMMM,
                        channels: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
  """Plots media response curves.
  
  Args:
    model: Fitted MiniMMM instance  
    channels: Channels to plot. If None, plots all channels.
    figsize: Figure size tuple
    
  Returns:
    Matplotlib figure
  """
  response_curves = ResponseCurves(model)
  curves_data = response_curves.compute_response_curves(channels=channels)
  
  if channels is None:
    channels = model.data.media_channels
  
  n_channels = len(channels)
  n_cols = min(3, n_channels)
  n_rows = (n_channels + n_cols - 1) // n_cols
  
  fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
  if n_channels == 1:
    axes = [axes]
  elif n_rows == 1:
    axes = [axes]
  else:
    axes = axes.flatten()
  
  fig.suptitle('Media Response Curves', fontsize=16, fontweight='bold')
  
  for i, channel in enumerate(channels):
    ax = axes[i]
    channel_data = curves_data[curves_data['channel'] == channel]
    
    # Primary y-axis: Effect level
    line1 = ax.plot(channel_data['spend_level'], channel_data['effect_level'], 
                    'b-', linewidth=2, label='Effect Level')
    ax.set_xlabel('Spend Level')
    ax.set_ylabel('Effect Level', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # Secondary y-axis: ROI
    ax2 = ax.twinx()
    line2 = ax2.plot(channel_data['spend_level'], channel_data['marginal_roi'], 
                     'r--', linewidth=2, label='Marginal ROI')
    ax2.set_ylabel('Marginal ROI', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add current spend level as vertical line
    current_spend = np.mean(model.data.get_media_matrix()[:, 
                           model.data.media_channels.index(channel)])
    ax.axvline(x=current_spend, color='gray', linestyle=':', alpha=0.7, 
              label='Current Spend')
    
    ax.set_title(f'{channel}')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
  
  # Hide empty subplots
  for i in range(n_channels, len(axes)):
    axes[i].set_visible(False)
  
  plt.tight_layout()
  return fig


def plot_contribution(model: MiniMMM, 
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
  """Plots contribution analysis.
  
  Args:
    model: Fitted MiniMMM instance
    figsize: Figure size tuple
    
  Returns:
    Matplotlib figure
  """
  analyzer = Analyzer(model)
  contribution_data = analyzer.compute_contribution()
  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
  fig.suptitle('Contribution Analysis', fontsize=16, fontweight='bold')
  
  # Filter out residual for cleaner visualization
  main_contributions = contribution_data[contribution_data['component'] != 'residual']
  
  # 1. Contribution waterfall/bar chart
  colors = sns.color_palette("husl", len(main_contributions))
  bars = ax1.bar(range(len(main_contributions)), 
                main_contributions['contribution_pct'], 
                color=colors)
  
  ax1.set_xlabel('Component')
  ax1.set_ylabel('Contribution (%)')
  ax1.set_title('Contribution Breakdown')
  ax1.set_xticks(range(len(main_contributions)))
  ax1.set_xticklabels(main_contributions['component'], rotation=45, ha='right')
  
  # Add value labels on bars
  for bar, value in zip(bars, main_contributions['contribution_pct']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.1f}%', ha='center', va='bottom')
  
  ax1.grid(True, alpha=0.3, axis='y')
  
  # 2. Pie chart
  media_contributions = main_contributions[main_contributions['component'].str.startswith('media_')]
  if len(media_contributions) > 0:
    ax2.pie(media_contributions['contribution_pct'], 
            labels=[comp.replace('media_', '') for comp in media_contributions['component']], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Media Contribution Breakdown')
  else:
    ax2.text(0.5, 0.5, 'No media contributions found', 
            ha='center', va='center', transform=ax2.transAxes)
  
  plt.tight_layout()
  return fig


def plot_diagnostics(model: MiniMMM,
                    figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
  """Plots model diagnostics including parameter posteriors.
  
  Args:
    model: Fitted MiniMMM instance
    figsize: Figure size tuple
    
  Returns:
    Matplotlib figure
  """
  if not model.is_fitted:
    raise RuntimeError("Model must be fitted before plotting diagnostics")
  
  # Get parameter summary
  summary = model.get_model_summary()
  diagnostics = model.get_diagnostics()
  
  fig = plt.figure(figsize=figsize)
  fig.suptitle('Model Diagnostics', fontsize=16, fontweight='bold')
  
  # Create a grid layout
  gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
  
  # 1. R-hat values (convergence diagnostic)
  ax1 = fig.add_subplot(gs[0, 0])
  rhat_values = summary['r_hat'].dropna()
  ax1.bar(range(len(rhat_values)), rhat_values.values)
  ax1.axhline(y=1.1, color='r', linestyle='--', alpha=0.7, label='R-hat = 1.1')
  ax1.set_xlabel('Parameter Index')
  ax1.set_ylabel('R-hat')
  ax1.set_title('Convergence Diagnostic (R-hat)')
  ax1.legend()
  ax1.grid(True, alpha=0.3)
  
  # 2. Effective sample size
  ax2 = fig.add_subplot(gs[0, 1])
  ess_values = summary['ess_bulk'].dropna()
  ax2.bar(range(len(ess_values)), ess_values.values)
  ax2.axhline(y=400, color='r', linestyle='--', alpha=0.7, label='ESS = 400')
  ax2.set_xlabel('Parameter Index')
  ax2.set_ylabel('Effective Sample Size')
  ax2.set_title('Effective Sample Size (Bulk)')
  ax2.legend()
  ax2.grid(True, alpha=0.3)
  
  # 3. Model fit metrics
  ax3 = fig.add_subplot(gs[0, 2])
  fit_metrics = diagnostics['fit_metrics']
  metrics_names = ['R²', 'RMSE', 'MAPE']
  metrics_values = [fit_metrics['r2'], fit_metrics['rmse'], fit_metrics['mape']]
  
  bars = ax3.bar(metrics_names, metrics_values)
  ax3.set_title('Model Fit Metrics')
  ax3.set_ylabel('Value')
  
  # Add value labels
  for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.3f}', ha='center', va='bottom')
  
  # 4-6. Parameter distributions (if available)
  param_names = ['retention_rate', 'ec', 'roi']
  for i, param_name in enumerate(param_names):
    ax = fig.add_subplot(gs[1, i])
    
    if param_name in summary.index.get_level_values(0):
      param_data = summary.loc[param_name]
      if len(param_data) > 1:
        # Multiple channels
        channels = model.data.media_channels[:len(param_data)]
        ax.errorbar(range(len(param_data)), param_data['mean'], 
                   yerr=[param_data['mean'] - param_data['hdi_3%'],
                         param_data['hdi_97%'] - param_data['mean']], 
                   fmt='o', capsize=5)
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels, rotation=45, ha='right')
      else:
        # Single value
        ax.bar([0], [param_data['mean']], 
              yerr=[[param_data['mean'] - param_data['hdi_3%']],
                    [param_data['hdi_97%'] - param_data['mean']]])
        ax.set_xticks([0])
        ax.set_xticklabels([param_name])
    else:
      ax.text(0.5, 0.5, f'No {param_name} data', ha='center', va='center', 
             transform=ax.transAxes)
    
    ax.set_title(f'{param_name.title()} Estimates')
    ax.grid(True, alpha=0.3)
  
  # 7-9. Additional diagnostics
  ax7 = fig.add_subplot(gs[2, :])
  # Plot parameter correlations if available
  try:
    # Get subset of key parameters for correlation plot
    key_params = ['retention_rate', 'ec', 'slope', 'roi']
    available_params = [p for p in key_params if p in summary.index.get_level_values(0)]
    
    if len(available_params) > 1:
      # Create correlation matrix from parameter means
      corr_data = []
      for param in available_params:
        param_values = summary.loc[param]['mean'].values
        corr_data.append(param_values[:min(len(param_values), len(model.data.media_channels))])
      
      if len(corr_data) > 1 and all(len(row) == len(corr_data[0]) for row in corr_data):
        corr_matrix = np.corrcoef(corr_data)
        im = ax7.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax7.set_xticks(range(len(available_params)))
        ax7.set_yticks(range(len(available_params)))
        ax7.set_xticklabels(available_params, rotation=45, ha='right')
        ax7.set_yticklabels(available_params)
        ax7.set_title('Parameter Correlation Matrix')
        
        # Add correlation values to cells
        for i in range(len(available_params)):
          for j in range(len(available_params)):
            ax7.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center')
        
        plt.colorbar(im, ax=ax7)
      else:
        ax7.text(0.5, 0.5, 'Insufficient data for correlation matrix', 
                ha='center', va='center', transform=ax7.transAxes)
    else:
      ax7.text(0.5, 0.5, 'Insufficient parameters for correlation analysis', 
              ha='center', va='center', transform=ax7.transAxes)
  except Exception as e:
    ax7.text(0.5, 0.5, f'Error creating correlation plot: {str(e)}', 
            ha='center', va='center', transform=ax7.transAxes)
  
  return fig


def plot_budget_optimization(optimization_result: Dict[str, Any],
                           figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
  """Plots budget optimization results.
  
  Args:
    optimization_result: Result from BudgetOptimizer.optimize_budget()
    figsize: Figure size tuple
    
  Returns:
    Matplotlib figure
  """
  allocation_df = optimization_result['allocation']
  total_effect = optimization_result['total_effect']
  total_roi = optimization_result['total_roi']
  
  fig, axes = plt.subplots(2, 2, figsize=figsize)
  fig.suptitle(f'Budget Optimization Results (Total ROI: {total_roi:.2f}x)', 
              fontsize=16, fontweight='bold')
  
  # 1. Optimal spend allocation
  ax1 = axes[0, 0]
  bars1 = ax1.bar(allocation_df['channel'], allocation_df['optimal_spend'])
  ax1.set_title('Optimal Spend Allocation')
  ax1.set_ylabel('Spend')
  ax1.tick_params(axis='x', rotation=45)
  
  # Add value labels
  for bar, value in zip(bars1, allocation_df['optimal_spend']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'${value:,.0f}', ha='center', va='bottom', rotation=0)
  
  # 2. Channel ROI
  ax2 = axes[0, 1]
  bars2 = ax2.bar(allocation_df['channel'], allocation_df['channel_roi'])
  ax2.set_title('Channel ROI at Optimal Allocation')
  ax2.set_ylabel('ROI')
  ax2.tick_params(axis='x', rotation=45)
  
  # Add value labels
  for bar, value in zip(bars2, allocation_df['channel_roi']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.2f}x', ha='center', va='bottom')
  
  # 3. Spend share pie chart
  ax3 = axes[1, 0]
  ax3.pie(allocation_df['spend_share'], labels=allocation_df['channel'], 
          autopct='%1.1f%%', startangle=90)
  ax3.set_title('Spend Share Distribution')
  
  # 4. Effect share pie chart
  ax4 = axes[1, 1]
  ax4.pie(allocation_df['effect_share'], labels=allocation_df['channel'], 
          autopct='%1.1f%%', startangle=90)
  ax4.set_title('Effect Share Distribution')
  
  plt.tight_layout()
  return fig


# Utility function to save all plots
def create_full_report(model: MiniMMM, 
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> List[plt.Figure]:
  """Creates a comprehensive visual report of the Mini MMM model.
  
  Args:
    model: Fitted MiniMMM instance
    save_path: Optional path prefix to save plots (e.g., 'reports/mmm_analysis')
    figsize: Default figure size for plots
    
  Returns:
    List of matplotlib figures
  """
  figures = []
  
  # 1. Model fit
  fig1 = plot_model_fit(model, figsize=figsize)
  figures.append(fig1)
  if save_path:
    fig1.savefig(f'{save_path}_model_fit.png', dpi=150, bbox_inches='tight')
  
  # 2. Media effects
  fig2 = plot_media_effects(model, figsize=figsize)
  figures.append(fig2)
  if save_path:
    fig2.savefig(f'{save_path}_media_effects.png', dpi=150, bbox_inches='tight')
  
  # 3. Response curves
  fig3 = plot_response_curves(model, figsize=figsize)
  figures.append(fig3)
  if save_path:
    fig3.savefig(f'{save_path}_response_curves.png', dpi=150, bbox_inches='tight')
  
  # 4. Contribution analysis
  fig4 = plot_contribution(model, figsize=(12, 8))
  figures.append(fig4)
  if save_path:
    fig4.savefig(f'{save_path}_contribution.png', dpi=150, bbox_inches='tight')
  
  # 5. Diagnostics
  fig5 = plot_diagnostics(model, figsize=figsize)
  figures.append(fig5)
  if save_path:
    fig5.savefig(f'{save_path}_diagnostics.png', dpi=150, bbox_inches='tight')
  
  return figures