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

"""Core optimization algorithms for independent budget optimizer."""

from typing import Optional, Sequence, Tuple
import numpy as np
import tensorflow as tf
import xarray as xr
from meridian.optimizer import media_transforms


__all__ = [
    "create_spend_grid",
    "hill_climbing_search",
    "compute_response_curves",
    "validate_spend_constraints",
]


def create_spend_grid(
    historical_spend: np.ndarray,
    spend_bounds_lower: np.ndarray,
    spend_bounds_upper: np.ndarray,
    step_size: float,
) -> Tuple[np.ndarray, int]:
  """Create discrete spend grid for optimization.
  
  Args:
    historical_spend: Historical spend per channel.
    spend_bounds_lower: Lower spend bounds per channel.
    spend_bounds_upper: Upper spend bounds per channel.
    step_size: Step size for discretization.
    
  Returns:
    Tuple of (spend_grid, max_steps) where spend_grid has shape 
    (max_steps, n_channels) and max_steps is the maximum grid size.
  """
  n_channels = len(historical_spend)
  
  # Calculate grid points for each channel
  channel_grids = []
  max_steps = 0
  
  for i in range(n_channels):
    channel_min = spend_bounds_lower[i]
    channel_max = spend_bounds_upper[i]
    
    # Create spend levels for this channel
    spend_levels = np.arange(channel_min, channel_max + step_size, step_size)
    channel_grids.append(spend_levels)
    max_steps = max(max_steps, len(spend_levels))
  
  # Create unified grid with NaN padding
  spend_grid = np.full((max_steps, n_channels), np.nan)
  
  for i, channel_grid in enumerate(channel_grids):
    spend_grid[:len(channel_grid), i] = channel_grid
  
  return spend_grid, max_steps


def hill_climbing_search(
    spend_grid: np.ndarray,
    incremental_outcome_grid: np.ndarray,
    scenario_type: str,
    total_budget: Optional[float] = None,
    target_roi: Optional[float] = None,
    target_mroi: Optional[float] = None,
) -> np.ndarray:
  """Hill-climbing search algorithm for budget optimization.
  
  Args:
    spend_grid: Spend grid with shape (grid_length, n_channels).
    incremental_outcome_grid: Incremental outcome grid with shape (grid_length, n_channels).
    scenario_type: 'fixed_budget' or 'flexible_budget'.
    total_budget: Total budget constraint for fixed budget scenarios.
    target_roi: Target ROI for flexible budget scenarios.
    target_mroi: Target marginal ROI for flexible budget scenarios.
    
  Returns:
    Optimal spend allocation with shape (n_channels,).
  """
  n_channels = spend_grid.shape[1]
  
  # Initialize with first row (minimum spend levels)
  current_spend = spend_grid[0, :].copy()
  current_outcome = incremental_outcome_grid[0, :].copy()
  
  # Remove first row from grids
  remaining_spend_grid = spend_grid[1:, :]
  remaining_outcome_grid = incremental_outcome_grid[1:, :]
  
  # Calculate marginal ROI grid
  marginal_roi_grid = np.divide(
      remaining_outcome_grid - current_outcome,
      remaining_spend_grid - current_spend,
      out=np.zeros_like(remaining_spend_grid),
      where=(remaining_spend_grid - current_spend) != 0
  )
  
  while True:
    # Check if all marginal ROIs are NaN (stopping condition)
    if np.isnan(marginal_roi_grid).all():
      break
    
    # Find the highest marginal ROI
    best_point = np.unravel_index(
        np.nanargmax(marginal_roi_grid), marginal_roi_grid.shape
    )
    row_idx, channel_idx = best_point
    
    # Check constraints before updating
    new_spend = current_spend.copy()
    new_outcome = current_outcome.copy()
    new_spend[channel_idx] = remaining_spend_grid[row_idx, channel_idx]
    new_outcome[channel_idx] = remaining_outcome_grid[row_idx, channel_idx]
    
    # Apply scenario constraints
    if scenario_type == 'fixed_budget':
      if total_budget and np.sum(new_spend) > total_budget:
        break
    elif scenario_type == 'flexible_budget':
      total_spend = np.sum(new_spend)
      total_outcome = np.sum(new_outcome)
      current_roi = total_outcome / total_spend if total_spend > 0 else 0
      
      if target_roi and current_roi < target_roi:
        break
      if target_mroi:
        marginal_outcome = new_outcome[channel_idx] - current_outcome[channel_idx]
        marginal_spend = new_spend[channel_idx] - current_spend[channel_idx]
        current_mroi = marginal_outcome / marginal_spend if marginal_spend > 0 else 0
        if current_mroi < target_mroi:
          break
    
    # Update current state
    current_spend = new_spend
    current_outcome = new_outcome
    
    # Update marginal ROI grid - invalidate used options
    marginal_roi_grid[:row_idx + 1, channel_idx] = np.nan
    
    # Recalculate marginal ROIs for remaining options in this channel
    if row_idx + 1 < len(marginal_roi_grid):
      marginal_roi_grid[row_idx + 1:, channel_idx] = np.divide(
          remaining_outcome_grid[row_idx + 1:, channel_idx] - current_outcome[channel_idx],
          remaining_spend_grid[row_idx + 1:, channel_idx] - current_spend[channel_idx],
          out=np.zeros(len(remaining_outcome_grid) - row_idx - 1),
          where=(remaining_spend_grid[row_idx + 1:, channel_idx] - current_spend[channel_idx]) != 0
      )
  
  return current_spend.astype(int)


def compute_response_curves(
    impression_channels: Sequence[str],
    rf_channels: Sequence[str], 
    impression_historical_spend: np.ndarray,
    rf_historical_spend: np.ndarray,
    impression_data: Optional[np.ndarray],
    rf_reach: Optional[np.ndarray],
    rf_frequency: Optional[np.ndarray],
    impression_coefficients: Optional[np.ndarray],
    rf_coefficients: Optional[np.ndarray],
    impression_alpha: Optional[np.ndarray],
    impression_ec50: Optional[np.ndarray],
    impression_slope: Optional[np.ndarray],
    rf_alpha: Optional[np.ndarray],
    rf_ec50: Optional[np.ndarray],
    rf_slope: Optional[np.ndarray],
    baseline: np.ndarray,
    spend_multipliers: np.ndarray,
    selected_times: Optional[slice] = None,
    selected_geos: Optional[slice] = None,
    max_lag: int = 13,
) -> xr.Dataset:
  """Generate response curves for mixed impression and R&F channels.
  
  Args:
    impression_channels: Impression channel names.
    rf_channels: R&F channel names.
    impression_historical_spend: Historical spend per impression channel.
    rf_historical_spend: Historical spend per R&F channel.
    impression_data: Impression data with shape (n_geos, n_times, n_impression_channels).
    rf_reach: R&F reach data with shape (n_geos, n_times, n_rf_channels).
    rf_frequency: R&F frequency data with shape (n_geos, n_times, n_rf_channels).
    impression_coefficients: Impression coefficients with shape (n_geos, n_impression_channels).
    rf_coefficients: R&F coefficients with shape (n_geos, n_rf_channels).
    impression_alpha: Impression adstock parameters.
    impression_ec50: Impression EC50 parameters.
    impression_slope: Impression slope parameters.
    rf_alpha: R&F adstock parameters.
    rf_ec50: R&F EC50 parameters.
    rf_slope: R&F slope parameters.
    baseline: Baseline effect with shape (n_geos,).
    spend_multipliers: Array of spend multipliers to evaluate.
    selected_times: Time slice for evaluation.
    selected_geos: Geo slice for evaluation.
    max_lag: Maximum lag for adstock.
    
  Returns:
    Dataset with response curves for each channel and spend multiplier.
  """
  all_channels = list(impression_channels) + list(rf_channels)
  all_historical_spend = np.concatenate([impression_historical_spend, rf_historical_spend])
  n_impression_channels = len(impression_channels)
  n_rf_channels = len(rf_channels)
  n_total_channels = n_impression_channels + n_rf_channels
  n_multipliers = len(spend_multipliers)
  
  # Initialize output arrays
  spend_levels = np.zeros((n_multipliers, n_total_channels))
  outcomes = np.zeros((n_multipliers, n_total_channels))
  
  for i, multiplier in enumerate(spend_multipliers):
    # Calculate spend for this multiplier
    spend_levels[i, :] = all_historical_spend * multiplier
    
    # Initialize combined outcomes
    impression_outcomes = np.zeros(n_impression_channels)
    rf_outcomes = np.zeros(n_rf_channels)
    
    # Compute impression channel outcomes
    if impression_data is not None and impression_coefficients is not None:
      scaled_impression_data = impression_data * multiplier
      impression_outcomes = media_transforms.compute_incremental_outcome(
          media=tf.convert_to_tensor(scaled_impression_data, dtype=tf.float32),
          media_coefficients=tf.convert_to_tensor(impression_coefficients, dtype=tf.float32),
          alpha=tf.convert_to_tensor(impression_alpha, dtype=tf.float32),
          ec50=tf.convert_to_tensor(impression_ec50, dtype=tf.float32),
          slope=tf.convert_to_tensor(impression_slope, dtype=tf.float32),
          max_lag=max_lag,
          selected_times=selected_times,
          selected_geos=selected_geos,
      ).numpy()
    
    # Compute R&F channel outcomes  
    if rf_reach is not None and rf_frequency is not None and rf_coefficients is not None:
      scaled_rf_reach = rf_reach * multiplier
      scaled_rf_frequency = rf_frequency * multiplier
      rf_outcomes = media_transforms.compute_rf_incremental_outcome(
          reach=tf.convert_to_tensor(scaled_rf_reach, dtype=tf.float32),
          frequency=tf.convert_to_tensor(scaled_rf_frequency, dtype=tf.float32),
          rf_coefficients=tf.convert_to_tensor(rf_coefficients, dtype=tf.float32),
          alpha=tf.convert_to_tensor(rf_alpha, dtype=tf.float32),
          ec50=tf.convert_to_tensor(rf_ec50, dtype=tf.float32),
          slope=tf.convert_to_tensor(rf_slope, dtype=tf.float32),
          max_lag=max_lag,
          selected_times=selected_times,
          selected_geos=selected_geos,
      ).numpy()
    
    # Combine outcomes
    outcomes[i, :] = np.concatenate([impression_outcomes, rf_outcomes])
  
  # Create xarray dataset
  return xr.Dataset({
      'spend': (['spend_multiplier', 'channel'], spend_levels),
      'incremental_outcome': (['spend_multiplier', 'channel'], outcomes),
      'roi': (['spend_multiplier', 'channel'], 
              np.divide(outcomes, spend_levels, 
                       out=np.zeros_like(outcomes), 
                       where=spend_levels!=0)),
  }, coords={
      'spend_multiplier': spend_multipliers,
      'channel': all_channels,
  })


def validate_spend_constraints(
    spend_allocation: np.ndarray,
    historical_spend: np.ndarray,
    spend_constraint_lower: float,
    spend_constraint_upper: float,
) -> Tuple[np.ndarray, np.ndarray]:
  """Validate and compute spend constraint bounds.
  
  Args:
    spend_allocation: Proposed spend allocation.
    historical_spend: Historical spend per channel.
    spend_constraint_lower: Lower constraint (0-1).
    spend_constraint_upper: Upper constraint (0-1).
    
  Returns:
    Tuple of (lower_bounds, upper_bounds) arrays.
  """
  if not (0 <= spend_constraint_lower <= 1):
    raise ValueError("spend_constraint_lower must be between 0 and 1")
  if not (0 <= spend_constraint_upper <= 1):
    raise ValueError("spend_constraint_upper must be between 0 and 1")
  
  lower_bounds = (1 - spend_constraint_lower) * spend_allocation
  upper_bounds = (1 + spend_constraint_upper) * spend_allocation
  
  return lower_bounds, upper_bounds