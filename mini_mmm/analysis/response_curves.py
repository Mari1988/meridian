"""Response curve analysis for Mini MMM.

This module provides fast response curve computation inspired by Meridian's
FastResponseCurves, showing how media effectiveness varies with spend levels.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple, Dict
from mini_mmm.model.mini_mmm import MiniMMM
from mini_mmm.model.transformers import AdstockTransformer, HillTransformer


class ResponseCurves:
  """Generates media response curves showing spend vs effectiveness.
  
  This class computes response curves efficiently by using point estimates
  from the fitted model rather than full Bayesian sampling, similar to
  the FastResponseCurves implementation in Meridian.
  """
  
  def __init__(self, model: MiniMMM):
    """Initializes ResponseCurves with a fitted model.
    
    Args:
      model: Fitted MiniMMM instance
      
    Raises:
      RuntimeError: If model is not fitted
    """
    if not model.is_fitted:
      raise RuntimeError("Model must be fitted before generating response curves")
    
    self.model = model
    self.data = model.input_data
    self.params = model._fitted_params
  
  def compute_response_curves(self,
                            spend_multipliers: Optional[List[float]] = None,
                            channels: Optional[List[str]] = None) -> pd.DataFrame:
    """Computes response curves for specified channels and spend levels.
    
    Args:
      spend_multipliers: List of spend multipliers (e.g., [0.5, 1.0, 1.5, 2.0]).
                        If None, uses default range from 0 to 2.2 in 0.1 steps.
      channels: List of channel names to analyze. If None, analyzes all channels.
      
    Returns:
      DataFrame with columns: channel, spend_multiplier, spend_level, 
                             effect_level, marginal_roi, total_roi
    """
    # Default spend multipliers
    if spend_multipliers is None:
      spend_multipliers = np.arange(0.0, 2.3, 0.1).tolist()
    
    # Default to all channels
    if channels is None:
      channels = self.data.media_channels
    
    # Validate channels
    invalid_channels = [c for c in channels if c not in self.data.media_channels]
    if invalid_channels:
      raise ValueError(f"Invalid channels: {invalid_channels}")
    
    # Get baseline spend levels (mean weekly spend)
    media_spend = self.data.get_media_matrix()
    baseline_spend = np.mean(media_spend, axis=0)
    
    results = []
    
    for channel in channels:
      channel_idx = self.data.media_channels.index(channel)
      channel_baseline = baseline_spend[channel_idx]
      
      # Get channel parameters
      retention_rate = self.params['retention_rate'][channel_idx]
      ec = self.params['ec'][channel_idx]
      slope = self.params['slope'][channel_idx]
      roi = self.params['roi'][channel_idx]
      
      for multiplier in spend_multipliers:
        spend_level = channel_baseline * multiplier
        
        # Compute transformed effect at this spend level
        if spend_level <= 0:
          effect_level = 0
          marginal_roi = 0
        else:
          # Apply adstock transformation
          # For single value, adstock effect is spend * adstock_multiplier
          adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
              retention_rate, self.model.adstock_max_lag)
          adstocked_spend = spend_level * adstock_multiplier
          
          # Apply Hill transformation
          effect_level = HillTransformer.transform(
              np.array([adstocked_spend]), ec, slope
          )[0]
          
          # Convert to KPI units using ROI
          effect_level *= roi
          
          # Compute marginal ROI (derivative at this point)
          marginal_roi = HillTransformer.get_marginal_effect(
              adstocked_spend, ec, slope) * roi * adstock_multiplier
        
        # Total ROI (average from 0 to current spend)
        total_roi = effect_level / spend_level if spend_level > 0 else 0
        
        results.append({
            'channel': channel,
            'spend_multiplier': multiplier,
            'spend_level': spend_level,
            'effect_level': effect_level,
            'marginal_roi': marginal_roi,
            'total_roi': total_roi
        })
    
    return pd.DataFrame(results)
  
  def compute_saturation_summary(self, 
                               channels: Optional[List[str]] = None) -> pd.DataFrame:
    """Computes saturation metrics summary for channels.
    
    Args:
      channels: List of channels to analyze. If None, uses all channels.
      
    Returns:
      DataFrame with saturation metrics for each channel
    """
    if channels is None:
      channels = self.data.media_channels
    
    # Get current spend levels
    media_spend = self.data.get_media_matrix()
    current_spend = np.mean(media_spend, axis=0)
    
    saturation_data = []
    
    for channel in channels:
      channel_idx = self.data.media_channels.index(channel)
      
      # Get parameters
      retention_rate = self.params['retention_rate'][channel_idx]
      ec = self.params['ec'][channel_idx]
      slope = self.params['slope'][channel_idx]
      
      current_weekly_spend = current_spend[channel_idx]
      
      # Apply transformations to current spend
      adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
          retention_rate, self.model.adstock_max_lag)
      adstocked_spend = current_weekly_spend * adstock_multiplier
      
      current_effect = HillTransformer.transform(
          np.array([adstocked_spend]), ec, slope)[0]
      
      # Saturation level (effect / max_possible_effect)
      saturation_level = current_effect / slope
      
      # Spend to reach 50% saturation
      half_saturation_spend = np.median(media_spend[:, channel_idx][media_spend[:, channel_idx] > 0]) if np.sum(media_spend[:, channel_idx] > 0) > 0 else current_weekly_spend
      
      # Spend to reach 90% saturation (approximately)
      # 90% saturation: 0.9 = x^ec / (half_sat^ec + x^ec)
      # Solving: x = half_sat * (9)^(1/ec)
      ninety_pct_saturation_spend = half_saturation_spend * (9 ** (1/ec))
      
      saturation_data.append({
          'channel': channel,
          'current_weekly_spend': current_weekly_spend,
          'current_saturation_level': saturation_level,
          'half_saturation_spend': half_saturation_spend,
          'ninety_pct_saturation_spend': ninety_pct_saturation_spend,
          'adstock_multiplier': adstock_multiplier,
          'ec_shape': ec,
          'slope_max_effect': slope
      })
    
    return pd.DataFrame(saturation_data)
  
  def find_optimal_spend_allocation(self, 
                                  total_budget: float,
                                  channels: Optional[List[str]] = None,
                                  method: str = 'equal_marginal') -> pd.DataFrame:
    """Finds optimal spend allocation across channels for a given budget.
    
    Args:
      total_budget: Total budget to allocate
      channels: Channels to consider. If None, uses all channels.
      method: Optimization method ('equal_marginal' or 'hill_climbing')
      
    Returns:
      DataFrame with optimal allocation by channel
    """
    if channels is None:
      channels = self.data.media_channels
    
    n_channels = len(channels)
    
    if method == 'equal_marginal':
      # Allocate budget such that marginal ROI is equal across channels
      allocation = self._equal_marginal_allocation(total_budget, channels)
    else:
      # Simple hill climbing optimization
      allocation = self._hill_climbing_allocation(total_budget, channels)
    
    # Calculate effects for optimal allocation
    results = []
    for i, channel in enumerate(channels):
      channel_idx = self.data.media_channels.index(channel)
      spend = allocation[i]
      
      # Calculate effect
      retention_rate = self.params['retention_rate'][channel_idx]
      ec = self.params['ec'][channel_idx]
      slope = self.params['slope'][channel_idx]
      roi = self.params['roi'][channel_idx]
      
      adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
          retention_rate, self.model.adstock_max_lag)
      adstocked_spend = spend * adstock_multiplier
      
      effect = HillTransformer.transform(
          np.array([adstocked_spend]), ec, slope)[0] * roi
      
      marginal_roi = HillTransformer.get_marginal_effect(
          adstocked_spend, ec, slope) * roi * adstock_multiplier
      
      results.append({
          'channel': channel,
          'optimal_spend': spend,
          'expected_effect': effect,
          'marginal_roi': marginal_roi,
          'spend_share': spend / total_budget,
          'effect_roi': effect / spend if spend > 0 else 0
      })
    
    return pd.DataFrame(results)
  
  def _equal_marginal_allocation(self, total_budget: float, channels: List[str]) -> np.ndarray:
    """Allocates budget to equalize marginal ROI across channels."""
    from scipy.optimize import minimize
    
    n_channels = len(channels)
    
    def objective(allocation):
      # Minimize negative total effect (maximize effect)
      total_effect = 0
      for i, channel in enumerate(channels):
        spend = allocation[i]
        if spend <= 0:
          continue
          
        channel_idx = self.data.media_channels.index(channel)
        retention_rate = self.params['retention_rate'][channel_idx]
        ec = self.params['ec'][channel_idx]
        slope = self.params['slope'][channel_idx]
        roi = self.params['roi'][channel_idx]
        
        adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
            retention_rate, self.model.adstock_max_lag)
        adstocked_spend = spend * adstock_multiplier
        
        effect = HillTransformer.transform(
            np.array([adstocked_spend]), ec, slope)[0] * roi
        total_effect += effect
      
      return -total_effect  # Negative because we minimize
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}]
    bounds = [(0, total_budget) for _ in range(n_channels)]
    
    # Initial guess: equal allocation
    initial_guess = np.full(n_channels, total_budget / n_channels)
    
    result = minimize(objective, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x if result.success else initial_guess
  
  def _hill_climbing_allocation(self, total_budget: float, channels: List[str]) -> np.ndarray:
    """Simple hill-climbing budget allocation."""
    n_channels = len(channels)
    allocation = np.full(n_channels, total_budget / n_channels)
    step_size = total_budget * 0.01  # 1% of budget
    
    for _ in range(100):  # Max iterations
      # Calculate current marginal ROIs
      marginal_rois = []
      for i, channel in enumerate(channels):
        channel_idx = self.data.media_channels.index(channel)
        spend = allocation[i]
        
        retention_rate = self.params['retention_rate'][channel_idx]
        ec = self.params['ec'][channel_idx]
        slope = self.params['slope'][channel_idx]
        roi = self.params['roi'][channel_idx]
        
        adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
            retention_rate, self.model.adstock_max_lag)
        adstocked_spend = spend * adstock_multiplier
        
        marginal_roi = HillTransformer.get_marginal_effect(
            adstocked_spend, ec, slope) * roi * adstock_multiplier
        marginal_rois.append(marginal_roi)
      
      marginal_rois = np.array(marginal_rois)
      
      # Find channels with highest and lowest marginal ROI
      max_idx = np.argmax(marginal_rois)
      min_idx = np.argmin(marginal_rois)
      
      # If marginal ROIs are similar, we're done
      if marginal_rois[max_idx] - marginal_rois[min_idx] < 0.01:
        break
      
      # Move budget from min to max channel
      if allocation[min_idx] >= step_size:
        allocation[min_idx] -= step_size
        allocation[max_idx] += step_size
    
    return allocation
  
  def compute_channel_efficiency_frontier(self,
                                        channel: str,
                                        budget_range: Optional[Tuple[float, float]] = None,
                                        n_points: int = 20) -> pd.DataFrame:
    """Computes efficiency frontier for a single channel.
    
    Args:
      channel: Channel name to analyze
      budget_range: (min_budget, max_budget) tuple. If None, uses reasonable defaults.
      n_points: Number of points to compute along the frontier
      
    Returns:
      DataFrame with spend levels and corresponding efficiency metrics
    """
    if channel not in self.data.media_channels:
      raise ValueError(f"Channel {channel} not found in data")
    
    channel_idx = self.data.media_channels.index(channel)
    
    # Get current average spend for reference
    media_spend = self.data.get_media_matrix()
    current_avg_spend = np.mean(media_spend[:, channel_idx])
    
    # Default budget range: 0 to 3x current average
    if budget_range is None:
      budget_range = (0, max(current_avg_spend * 3, 1000))
    
    spend_levels = np.linspace(budget_range[0], budget_range[1], n_points)
    
    # Get channel parameters
    retention_rate = self.params['retention_rate'][channel_idx]
    ec = self.params['ec'][channel_idx]
    slope = self.params['slope'][channel_idx]
    roi = self.params['roi'][channel_idx]
    
    adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
        retention_rate, self.model.adstock_max_lag)
    
    results = []
    
    for spend in spend_levels:
      if spend <= 0:
        effect = 0
        marginal_roi = 0
        avg_roi = 0
      else:
        adstocked_spend = spend * adstock_multiplier
        effect = HillTransformer.transform(
            np.array([adstocked_spend]), ec, slope)[0] * roi
        
        marginal_roi = HillTransformer.get_marginal_effect(
            adstocked_spend, ec, slope) * roi * adstock_multiplier
        
        avg_roi = effect / spend
      
      results.append({
          'channel': channel,
          'spend_level': spend,
          'effect_level': effect,
          'marginal_roi': marginal_roi,
          'average_roi': avg_roi,
          'efficiency_score': marginal_roi / avg_roi if avg_roi > 0 else 0
      })
    
    return pd.DataFrame(results)