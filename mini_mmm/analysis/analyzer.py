"""Core analysis metrics for Mini MMM.

This module provides utilities to compute key marketing mix modeling metrics
including ROI, mROI, contribution analysis, and incremental effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple, List
from mini_mmm.model.mini_mmm import MiniMMM
from mini_mmm.data.input_data import SimpleInputData


class Analyzer:
  """Analyzer for computing MMM metrics and insights.
  
  This class provides methods to analyze a fitted Mini MMM model,
  computing key metrics like ROI, contribution, and incremental effects.
  """
  
  def __init__(self, model: MiniMMM):
    """Initializes the Analyzer with a fitted Mini MMM model.
    
    Args:
      model: Fitted MiniMMM instance
      
    Raises:
      RuntimeError: If model is not fitted
    """
    if not model.is_fitted:
      raise RuntimeError("Model must be fitted before analysis")
    
    self.model = model
    self.data = model.input_data
  
  def compute_roi(self, 
                 period_start: Optional[int] = None,
                 period_end: Optional[int] = None) -> pd.DataFrame:
    """Computes Return on Investment (ROI) for each media channel.
    
    ROI = Incremental Revenue / Media Spend
    
    Args:
      period_start: Start week for analysis (inclusive). If None, uses all data.
      period_end: End week for analysis (exclusive). If None, uses all data.
      
    Returns:
      DataFrame with ROI metrics by channel
    """
    # Get period slice
    start_idx = period_start or 0
    end_idx = period_end or self.data.n_weeks
    
    # Get media spend and effects for the period
    media_spend = self.data.get_media_matrix()[start_idx:end_idx]
    
    # Get incremental effects (prediction with vs without media)
    pred_with_media = self.model.predict(
        media_spend=media_spend,
        return_components=True
    )
    
    # Prediction without media (baseline)
    zero_media = np.zeros_like(media_spend)
    pred_without_media = self.model.predict(
        media_spend=zero_media,
        return_components=True  
    )
    
    # Incremental effects
    total_incremental = (pred_with_media['prediction'] - 
                        pred_without_media['prediction'])
    media_incremental = (pred_with_media['media_effects'] - 
                        pred_without_media['media_effects'])
    
    # Channel-level incremental effects
    saturated_with = pred_with_media['saturated_media']
    saturated_without = pred_without_media['saturated_media']
    roi_params = self.model._fitted_params['roi']
    
    channel_incremental = np.sum(
        (saturated_with - saturated_without) * roi_params[None, :] * media_spend,
        axis=0
    )
    
    # Total spend by channel
    total_spend = np.sum(media_spend, axis=0)
    
    # Calculate ROI metrics
    roi_data = []
    for i, channel in enumerate(self.data.media_channels):
      spend = total_spend[i]
      incremental = channel_incremental[i]
      
      roi_data.append({
          'channel': channel,
          'total_spend': spend,
          'incremental_effect': incremental,
          'roi': incremental / spend if spend > 0 else 0,
          'spend_share': spend / np.sum(total_spend) if np.sum(total_spend) > 0 else 0,
          'effect_share': incremental / np.sum(channel_incremental) if np.sum(channel_incremental) > 0 else 0
      })
    
    return pd.DataFrame(roi_data)
  
  def compute_mroi(self, 
                  spend_multiplier: float = 1.01) -> pd.DataFrame:
    """Computes Marginal Return on Investment (mROI) for each media channel.
    
    mROI represents the incremental return from a small increase in spend.
    
    Args:
      spend_multiplier: Factor to increase spend by (e.g., 1.01 = 1% increase)
      
    Returns:
      DataFrame with mROI metrics by channel
    """
    media_spend = self.data.get_media_matrix()
    
    # Baseline prediction
    baseline_pred = self.model.predict()['prediction']
    baseline_total = np.sum(baseline_pred)
    
    mroi_data = []
    
    for i, channel in enumerate(self.data.media_channels):
      # Create increased spend scenario
      increased_spend = media_spend.copy()
      increased_spend[:, i] *= spend_multiplier
      
      # Predict with increased spend
      increased_pred = self.model.predict(media_spend=increased_spend)['prediction']
      increased_total = np.sum(increased_pred)
      
      # Calculate marginal effects
      incremental_effect = increased_total - baseline_total
      incremental_spend = np.sum(increased_spend[:, i]) - np.sum(media_spend[:, i])
      
      mroi_data.append({
          'channel': channel,
          'baseline_spend': np.sum(media_spend[:, i]),
          'incremental_spend': incremental_spend,
          'incremental_effect': incremental_effect,
          'mroi': incremental_effect / incremental_spend if incremental_spend > 0 else 0
      })
    
    return pd.DataFrame(mroi_data)
  
  def compute_contribution(self,
                          period_start: Optional[int] = None,
                          period_end: Optional[int] = None) -> pd.DataFrame:
    """Computes contribution analysis showing each component's share of KPI.
    
    Args:
      period_start: Start week for analysis (inclusive)
      period_end: End week for analysis (exclusive)
      
    Returns:
      DataFrame with contribution breakdown
    """
    # Get period slice
    start_idx = period_start or 0
    end_idx = period_end or self.data.n_weeks
    
    kpi_actual = self.data.get_kpi_array()[start_idx:end_idx]
    
    # Get media spend for period
    media_spend = self.data.get_media_matrix()[start_idx:end_idx]
    controls = (self.data.get_controls_matrix()[start_idx:end_idx] 
               if self.data.n_controls > 0 else None)
    
    # Get prediction components
    pred_components = self.model.predict(
        media_spend=media_spend,
        controls=controls,
        return_components=True
    )
    
    total_kpi = np.sum(kpi_actual)
    
    # Base contribution
    base_contribution = np.sum(pred_components['intercept'])
    
    # Media contributions by channel
    saturated = pred_components['saturated_media']
    roi_params = self.model._fitted_params['roi']
    
    media_contributions = []
    for i, channel in enumerate(self.data.media_channels):
      channel_effect = np.sum(saturated[:, i] * roi_params[i] * media_spend[:, i])
      media_contributions.append({
          'component': f'media_{channel}',
          'total_effect': channel_effect,
          'contribution_pct': channel_effect / total_kpi * 100 if total_kpi != 0 else 0
      })
    
    # Control contributions
    control_contributions = []
    if self.data.n_controls > 0:
      control_effects = pred_components['control_effects']
      control_params = self.model._fitted_params.get('control_coef', [])
      
      for i, control in enumerate(self.data.control_names or []):
        control_effect = np.sum(controls[:, i] * control_params[i])
        control_contributions.append({
            'component': f'control_{control}',
            'total_effect': control_effect,
            'contribution_pct': control_effect / total_kpi * 100 if total_kpi != 0 else 0
        })
    
    # Combine all contributions
    all_contributions = [
        {
            'component': 'base',
            'total_effect': base_contribution,
            'contribution_pct': base_contribution / total_kpi * 100 if total_kpi != 0 else 0
        }
    ] + media_contributions + control_contributions
    
    # Add residual (unexplained)
    explained_total = sum(c['total_effect'] for c in all_contributions)
    residual = total_kpi - explained_total
    
    all_contributions.append({
        'component': 'residual',
        'total_effect': residual,
        'contribution_pct': residual / total_kpi * 100 if total_kpi != 0 else 0
    })
    
    return pd.DataFrame(all_contributions)
  
  def compute_media_effectiveness(self) -> pd.DataFrame:
    """Computes various effectiveness metrics for media channels.
    
    Returns:
      DataFrame with comprehensive effectiveness metrics
    """
    media_spend = self.data.get_media_matrix()
    params = self.model._fitted_params
    
    # Get predictions with components
    pred_components = self.model.predict(return_components=True)
    
    effectiveness_data = []
    
    for i, channel in enumerate(self.data.media_channels):
      channel_spend = media_spend[:, i]
      total_spend = np.sum(channel_spend)
      
      # Transformation parameters
      retention_rate = params['retention_rate'][i]
      ec = params['ec'][i]
      slope = params['slope'][i]
      roi = params['roi'][i]
      
      # Saturation and adstock effects
      adstocked = pred_components['adstocked_media'][:, i]
      saturated = pred_components['saturated_media'][:, i]
      
      # Effectiveness metrics
      adstock_multiplier = np.sum(adstocked) / np.sum(channel_spend) if np.sum(channel_spend) > 0 else 1
      
      # Saturation level (how close to maximum effect)
      max_spend = np.max(channel_spend) if np.max(channel_spend) > 0 else 1
      saturation_at_max = HillTransformer.transform(
          np.array([max_spend]), ec, slope
      )[0] / slope
      
      effectiveness_data.append({
          'channel': channel,
          'retention_rate': retention_rate,
          'ec_shape': ec,
          'slope_max_effect': slope,
          'roi': roi,
          'total_spend': total_spend,
          'adstock_multiplier': adstock_multiplier,
          'saturation_at_max_spend': saturation_at_max,
          'avg_weekly_spend': np.mean(channel_spend),
          'spend_efficiency': roi * adstock_multiplier * np.mean(saturated / channel_spend) if np.mean(channel_spend) > 0 else 0
      })
    
    return pd.DataFrame(effectiveness_data)
  
  def compute_incremental_effects(self,
                                 scenario_media: np.ndarray,
                                 baseline_media: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
    """Computes incremental effects for a scenario vs baseline.
    
    Args:
      scenario_media: Media spend matrix for scenario
      baseline_media: Media spend matrix for baseline. If None, uses training data.
      
    Returns:
      Dictionary with incremental effects analysis
    """
    if baseline_media is None:
      baseline_media = self.data.get_media_matrix()
    
    # Predictions for both scenarios
    scenario_pred = self.model.predict(
        media_spend=scenario_media, 
        return_components=True
    )
    baseline_pred = self.model.predict(
        media_spend=baseline_media,
        return_components=True
    )
    
    # Total incremental effects
    total_incremental_kpi = (np.sum(scenario_pred['prediction']) - 
                            np.sum(baseline_pred['prediction']))
    
    total_incremental_spend = (np.sum(scenario_media) - 
                              np.sum(baseline_media))
    
    # Channel-level incrementals
    channel_incrementals = []
    params = self.model._fitted_params
    
    for i, channel in enumerate(self.data.media_channels):
      scenario_saturated = scenario_pred['saturated_media'][:, i]
      baseline_saturated = baseline_pred['saturated_media'][:, i]
      
      incremental_effect = np.sum(
          (scenario_saturated - baseline_saturated) * 
          params['roi'][i] * scenario_media[:, i]
      )
      
      incremental_spend = np.sum(scenario_media[:, i] - baseline_media[:, i])
      
      channel_incrementals.append({
          'channel': channel,
          'incremental_spend': incremental_spend,
          'incremental_effect': incremental_effect,
          'incremental_roi': incremental_effect / incremental_spend if incremental_spend != 0 else 0
      })
    
    return {
        'total_incremental_kpi': total_incremental_kpi,
        'total_incremental_spend': total_incremental_spend,
        'total_incremental_roi': total_incremental_kpi / total_incremental_spend if total_incremental_spend != 0 else 0,
        'weekly_incremental': scenario_pred['prediction'] - baseline_pred['prediction'],
        'channel_breakdown': pd.DataFrame(channel_incrementals)
    }
  
  def get_model_fit_metrics(self) -> Dict[str, float]:
    """Returns model fit quality metrics.
    
    Returns:
      Dictionary with fit metrics
    """
    # Get predictions
    predictions = self.model.predict()['prediction']
    actual = self.data.get_kpi_array()
    
    # Basic metrics
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    mae = np.mean(np.abs(predictions - actual))
    mape = np.mean(np.abs(predictions - actual) / np.abs(actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared
    n = len(actual)
    p = self.data.n_media_channels + self.data.n_controls + 1  # +1 for intercept
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'adj_r2': adj_r2,
        'n_observations': n,
        'n_parameters': p
    }