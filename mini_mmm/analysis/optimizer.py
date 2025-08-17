"""Budget optimization for Mini MMM.

This module provides budget optimization algorithms to maximize ROI or
total effect given budget constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize, differential_evolution
from mini_mmm.model.mini_mmm import MiniMMM
from mini_mmm.model.transformers import AdstockTransformer, HillTransformer


class BudgetOptimizer:
  """Budget optimization for marketing mix models.
  
  This class provides methods to optimize budget allocation across media
  channels to maximize either total ROI or total incremental effect.
  """
  
  def __init__(self, model: MiniMMM):
    """Initializes the optimizer with a fitted model.
    
    Args:
      model: Fitted MiniMMM instance
      
    Raises:
      RuntimeError: If model is not fitted
    """
    if not model.is_fitted:
      raise RuntimeError("Model must be fitted before optimization")
    
    self.model = model
    self.data = model.input_data
    self.params = model._fitted_params
  
  def optimize_budget(self,
                     total_budget: float,
                     objective: str = 'total_effect',
                     channels: Optional[List[str]] = None,
                     min_spend_pct: float = 0.0,
                     max_spend_pct: float = 1.0,
                     method: str = 'scipy') -> Dict[str, Union[pd.DataFrame, float, Dict]]:
    """Optimizes budget allocation across channels.
    
    Args:
      total_budget: Total budget to allocate
      objective: 'total_effect' to maximize total effect, or 'roi' to maximize ROI
      channels: Channels to optimize over. If None, uses all channels.
      min_spend_pct: Minimum spend per channel as fraction of total budget
      max_spend_pct: Maximum spend per channel as fraction of total budget  
      method: 'scipy' for gradient-based or 'differential_evolution' for global
      
    Returns:
      Dictionary containing:
        - 'allocation': DataFrame with optimal allocation
        - 'total_effect': Expected total effect
        - 'total_roi': Total ROI
        - 'optimization_info': Optimization metadata
    """
    if channels is None:
      channels = self.data.media_channels
    
    n_channels = len(channels)
    
    # Define bounds
    bounds = [(total_budget * min_spend_pct, total_budget * max_spend_pct) 
             for _ in range(n_channels)]
    
    # Define constraints (budget constraint)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
    ]
    
    # Initial guess: equal allocation within bounds
    initial_guess = np.full(n_channels, total_budget / n_channels)
    initial_guess = np.clip(initial_guess, 
                           [b[0] for b in bounds], 
                           [b[1] for b in bounds])
    
    # Adjust initial guess to satisfy budget constraint
    initial_sum = np.sum(initial_guess)
    if initial_sum != total_budget:
      initial_guess = initial_guess * (total_budget / initial_sum)
    
    # Define objective function
    if objective == 'total_effect':
      obj_func = lambda x: -self._compute_total_effect(x, channels)
    elif objective == 'roi':
      obj_func = lambda x: -self._compute_total_roi(x, channels)
    else:
      raise ValueError(f"Unknown objective: {objective}")
    
    # Optimize
    if method == 'scipy':
      result = minimize(
          obj_func, 
          initial_guess, 
          method='SLSQP',
          bounds=bounds,
          constraints=constraints,
          options={'ftol': 1e-9, 'disp': False}
      )
      optimal_allocation = result.x
      success = result.success
      optimization_info = {
          'method': 'scipy_SLSQP',
          'success': success,
          'message': result.message,
          'iterations': result.nit,
          'function_evals': result.nfev
      }
    
    elif method == 'differential_evolution':
      # For differential evolution, we need to handle constraints differently
      def constrained_objective(x):
        # Penalty for violating budget constraint
        budget_violation = abs(np.sum(x) - total_budget)
        if budget_violation > 1e-6:
          return 1e6  # Large penalty
        return obj_func(x)
      
      result = differential_evolution(
          constrained_objective,
          bounds,
          seed=42,
          maxiter=1000,
          atol=1e-8
      )
      optimal_allocation = result.x
      success = result.success
      optimization_info = {
          'method': 'differential_evolution',
          'success': success,
          'message': result.message,
          'iterations': result.nit,
          'function_evals': result.nfev
      }
    
    else:
      raise ValueError(f"Unknown optimization method: {method}")
    
    # Calculate metrics for optimal allocation
    total_effect = self._compute_total_effect(optimal_allocation, channels)
    total_roi = total_effect / total_budget if total_budget > 0 else 0
    
    # Create allocation DataFrame
    allocation_data = []
    for i, channel in enumerate(channels):
      channel_idx = self.data.media_channels.index(channel)
      spend = optimal_allocation[i]
      
      # Calculate individual channel metrics
      channel_effect = self._compute_channel_effect(spend, channel_idx)
      channel_roi = channel_effect / spend if spend > 0 else 0
      
      allocation_data.append({
          'channel': channel,
          'optimal_spend': spend,
          'spend_share': spend / total_budget,
          'expected_effect': channel_effect,
          'channel_roi': channel_roi,
          'effect_share': channel_effect / total_effect if total_effect > 0 else 0
      })
    
    allocation_df = pd.DataFrame(allocation_data)
    
    return {
        'allocation': allocation_df,
        'total_effect': total_effect,
        'total_roi': total_roi,
        'optimization_info': optimization_info
    }
  
  def compare_scenarios(self,
                       scenarios: Dict[str, np.ndarray],
                       baseline_name: str = 'current') -> pd.DataFrame:
    """Compares multiple budget allocation scenarios.
    
    Args:
      scenarios: Dictionary with scenario names as keys and spend arrays as values
      baseline_name: Name of baseline scenario for comparison
      
    Returns:
      DataFrame comparing scenarios
    """
    if baseline_name not in scenarios:
      raise ValueError(f"Baseline scenario '{baseline_name}' not found")
    
    baseline_spend = scenarios[baseline_name]
    baseline_effect = self._compute_total_effect(baseline_spend, self.data.media_channels)
    baseline_budget = np.sum(baseline_spend)
    
    comparison_data = []
    
    for scenario_name, spend_allocation in scenarios.items():
      total_spend = np.sum(spend_allocation)
      total_effect = self._compute_total_effect(spend_allocation, self.data.media_channels)
      roi = total_effect / total_spend if total_spend > 0 else 0
      
      # Comparison metrics
      effect_lift = total_effect - baseline_effect
      effect_lift_pct = (effect_lift / baseline_effect * 100) if baseline_effect > 0 else 0
      budget_change = total_spend - baseline_budget
      
      comparison_data.append({
          'scenario': scenario_name,
          'total_spend': total_spend,
          'total_effect': total_effect,
          'roi': roi,
          'vs_baseline_effect_lift': effect_lift,
          'vs_baseline_effect_lift_pct': effect_lift_pct,
          'vs_baseline_budget_change': budget_change,
          'efficiency_score': roi / scenarios[baseline_name].sum() * baseline_budget if baseline_budget > 0 else 0
      })
    
    return pd.DataFrame(comparison_data)
  
  def marginal_roi_analysis(self, 
                           current_allocation: Optional[np.ndarray] = None,
                           channels: Optional[List[str]] = None,
                           delta_pct: float = 0.01) -> pd.DataFrame:
    """Analyzes marginal ROI for each channel at current allocation.
    
    Args:
      current_allocation: Current spend allocation. If None, uses average from data.
      channels: Channels to analyze. If None, uses all channels.
      delta_pct: Percentage change to use for marginal calculation
      
    Returns:
      DataFrame with marginal ROI analysis
    """
    if channels is None:
      channels = self.data.media_channels
    
    if current_allocation is None:
      # Use average spend from training data
      media_spend = self.data.get_media_matrix()
      current_allocation = np.mean(media_spend, axis=0)
    
    # Ensure allocation matches selected channels
    if len(current_allocation) != len(self.data.media_channels):
      raise ValueError("Current allocation length doesn't match number of channels")
    
    # Filter allocation to selected channels
    channel_indices = [self.data.media_channels.index(ch) for ch in channels]
    filtered_allocation = current_allocation[channel_indices]
    
    marginal_data = []
    
    for i, channel in enumerate(channels):
      current_spend = filtered_allocation[i]
      channel_idx = channel_indices[i]
      
      # Calculate baseline effect
      baseline_effect = self._compute_channel_effect(current_spend, channel_idx)
      
      # Calculate effect with small increase
      delta_spend = current_spend * delta_pct
      increased_spend = current_spend + delta_spend
      increased_effect = self._compute_channel_effect(increased_spend, channel_idx)
      
      # Marginal metrics
      marginal_effect = increased_effect - baseline_effect
      marginal_roi = marginal_effect / delta_spend if delta_spend > 0 else 0
      
      # Average ROI for comparison
      avg_roi = baseline_effect / current_spend if current_spend > 0 else 0
      
      marginal_data.append({
          'channel': channel,
          'current_spend': current_spend,
          'baseline_effect': baseline_effect,
          'marginal_effect': marginal_effect,
          'marginal_roi': marginal_roi,
          'average_roi': avg_roi,
          'marginal_vs_average_ratio': marginal_roi / avg_roi if avg_roi > 0 else 0
      })
    
    return pd.DataFrame(marginal_data)
  
  def saturation_analysis(self,
                         channels: Optional[List[str]] = None,
                         spend_multipliers: Optional[List[float]] = None) -> pd.DataFrame:
    """Analyzes saturation levels for channels at different spend levels.
    
    Args:
      channels: Channels to analyze. If None, uses all channels.
      spend_multipliers: Spend multipliers to test. If None, uses [0.5, 1.0, 1.5, 2.0]
      
    Returns:
      DataFrame with saturation analysis
    """
    if channels is None:
      channels = self.data.media_channels
    
    if spend_multipliers is None:
      spend_multipliers = [0.5, 1.0, 1.5, 2.0]
    
    # Get baseline spend
    media_spend = self.data.get_media_matrix()
    baseline_spend = np.mean(media_spend, axis=0)
    
    saturation_data = []
    
    for channel in channels:
      channel_idx = self.data.media_channels.index(channel)
      channel_baseline = baseline_spend[channel_idx]
      
      # Get channel parameters
      retention_rate = self.params['retention_rate'][channel_idx]
      ec = self.params['ec'][channel_idx]
      slope = self.params['slope'][channel_idx]
      
      for multiplier in spend_multipliers:
        spend_level = channel_baseline * multiplier
        
        if spend_level <= 0:
          saturation_level = 0
        else:
          # Apply transformations
          adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
              retention_rate, self.model.adstock_max_lag)
          adstocked_spend = spend_level * adstock_multiplier
          
          # Hill transformation gives us the effect level
          effect = HillTransformer.transform(
              np.array([adstocked_spend]), ec, slope)[0]
          
          # Saturation level = effect / max_possible_effect
          saturation_level = effect / slope
        
        saturation_data.append({
            'channel': channel,
            'spend_multiplier': multiplier,
            'spend_level': spend_level,
            'saturation_level': saturation_level,
            'saturation_pct': saturation_level * 100
        })
    
    return pd.DataFrame(saturation_data)
  
  def _compute_total_effect(self, allocation: np.ndarray, channels: List[str]) -> float:
    """Computes total effect for a given allocation."""
    total_effect = 0
    
    for i, channel in enumerate(channels):
      channel_idx = self.data.media_channels.index(channel)
      spend = allocation[i]
      effect = self._compute_channel_effect(spend, channel_idx)
      total_effect += effect
    
    return total_effect
  
  def _compute_total_roi(self, allocation: np.ndarray, channels: List[str]) -> float:
    """Computes total ROI for a given allocation."""
    total_effect = self._compute_total_effect(allocation, channels)
    total_spend = np.sum(allocation)
    
    return total_effect / total_spend if total_spend > 0 else 0
  
  def _compute_channel_effect(self, spend: float, channel_idx: int) -> float:
    """Computes effect for a single channel at given spend level."""
    if spend <= 0:
      return 0
    
    # Get channel parameters
    retention_rate = self.params['retention_rate'][channel_idx]
    ec = self.params['ec'][channel_idx]
    slope = self.params['slope'][channel_idx]
    roi = self.params['roi'][channel_idx]
    
    # Apply transformations
    adstock_multiplier = AdstockTransformer.get_adstock_multiplier(
        retention_rate, self.model.adstock_max_lag)
    adstocked_spend = spend * adstock_multiplier
    
    # Hill transformation
    saturated_effect = HillTransformer.transform(
        np.array([adstocked_spend]), ec, slope)[0]
    
    # Convert to KPI units using ROI parameter
    total_effect = saturated_effect * roi
    
    return total_effect
  
  def generate_optimization_report(self,
                                 optimization_result: Dict,
                                 current_allocation: Optional[np.ndarray] = None) -> str:
    """Generates a text report summarizing optimization results.
    
    Args:
      optimization_result: Result from optimize_budget()
      current_allocation: Current allocation for comparison
      
    Returns:
      Formatted text report
    """
    allocation_df = optimization_result['allocation']
    total_effect = optimization_result['total_effect']
    total_roi = optimization_result['total_roi']
    opt_info = optimization_result['optimization_info']
    
    total_budget = allocation_df['optimal_spend'].sum()
    
    report_lines = [
        "=== BUDGET OPTIMIZATION REPORT ===",
        "",
        f"Optimization Method: {opt_info['method']}",
        f"Success: {opt_info['success']}",
        f"Total Budget: ${total_budget:,.0f}",
        f"Expected Total Effect: {total_effect:.2f}",
        f"Expected ROI: {total_roi:.2f}x",
        "",
        "OPTIMAL ALLOCATION:",
    ]
    
    # Channel allocation details
    for _, row in allocation_df.iterrows():
      report_lines.append(
          f"  {row['channel']:15s}: ${row['optimal_spend']:8,.0f} "
          f"({row['spend_share']:5.1%}) -> ROI: {row['channel_roi']:4.2f}x"
      )
    
    # Comparison with current if provided
    if current_allocation is not None:
      current_total = np.sum(current_allocation)
      current_effect = self._compute_total_effect(current_allocation, self.data.media_channels)
      current_roi = current_effect / current_total if current_total > 0 else 0
      
      effect_improvement = (total_effect - current_effect) / current_effect * 100 if current_effect > 0 else 0
      roi_improvement = (total_roi - current_roi) / current_roi * 100 if current_roi > 0 else 0
      
      report_lines.extend([
          "",
          "COMPARISON VS CURRENT:",
          f"  Current Total Effect: {current_effect:.2f}",
          f"  Current ROI: {current_roi:.2f}x", 
          f"  Effect Improvement: {effect_improvement:+.1f}%",
          f"  ROI Improvement: {roi_improvement:+.1f}%"
      ])
    
    return "\n".join(report_lines)