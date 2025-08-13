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

"""Independent budget optimizer that works without Meridian model objects."""

import dataclasses
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr


__all__ = [
    "OptimizationInput",
    "ModelParameters", 
    "IndependentOptimizer",
    "OptimizationScenario",
    "IndependentOptimizationResults",
]


@dataclasses.dataclass(frozen=True)
class OptimizationInput:
  """Input data for independent optimization from files.
  
  Attributes:
    impression_channels: List of impression-based channel names.
    rf_channels: List of reach & frequency channel names.
    impression_historical_spend: Historical spend per impression channel.
    rf_historical_spend: Historical spend per R&F channel.
    impression_data: Media impressions data with shape (n_geos, n_times, n_impression_channels).
    rf_reach: Reach data for RF channels with shape (n_geos, n_times, n_rf_channels).
    rf_frequency: Frequency data for RF channels with shape (n_geos, n_times, n_rf_channels).
    rf_spend: Spend data for RF channels with shape (n_geos, n_times, n_rf_channels).
    dates: List of date strings in 'YYYY-MM-DD' format.
    geos: List of geo identifiers.
    population: Population data per geo, required for proper scaling.
    revenue_per_kpi: Revenue per KPI conversion factor, optional.
  """
  impression_channels: Sequence[str]
  rf_channels: Sequence[str]
  impression_historical_spend: np.ndarray
  rf_historical_spend: np.ndarray
  impression_data: Optional[np.ndarray] = None
  rf_reach: Optional[np.ndarray] = None
  rf_frequency: Optional[np.ndarray] = None
  rf_spend: Optional[np.ndarray] = None
  dates: Optional[Sequence[str]] = None
  geos: Optional[Sequence[str]] = None
  population: Optional[np.ndarray] = None
  revenue_per_kpi: Optional[np.ndarray] = None
  
  @property
  def all_channels(self) -> Sequence[str]:
    """Combined list of all channels (impression + R&F)."""
    return list(self.impression_channels) + list(self.rf_channels)
  
  @property 
  def all_historical_spend(self) -> np.ndarray:
    """Combined historical spend for all channels."""
    return np.concatenate([self.impression_historical_spend, self.rf_historical_spend])
  
  @property
  def n_impression_channels(self) -> int:
    """Number of impression channels."""
    return len(self.impression_channels)
  
  @property
  def n_rf_channels(self) -> int:
    """Number of R&F channels."""
    return len(self.rf_channels)
  
  @property
  def n_total_channels(self) -> int:
    """Total number of channels."""
    return self.n_impression_channels + self.n_rf_channels
  
  def __post_init__(self):
    if len(self.impression_channels) != len(self.impression_historical_spend):
      raise ValueError("Number of impression channels must match impression historical spend length")
    if len(self.rf_channels) != len(self.rf_historical_spend):
      raise ValueError("Number of R&F channels must match R&F historical spend length")
    
    if self.impression_data is not None and self.impression_data.shape[-1] != len(self.impression_channels):
      raise ValueError("Impression data channels must match impression channel names")
    if self.rf_reach is not None and self.rf_reach.shape[-1] != len(self.rf_channels):
      raise ValueError("R&F reach channels must match R&F channel names")
    if self.rf_frequency is not None and self.rf_frequency.shape[-1] != len(self.rf_channels):
      raise ValueError("R&F frequency channels must match R&F channel names")
    
    # Validate that at least one channel type is provided
    if len(self.impression_channels) == 0 and len(self.rf_channels) == 0:
      raise ValueError("Must provide at least one impression or R&F channel")


@dataclasses.dataclass(frozen=True)
class ModelParameters:
  """Model parameters for independent optimization with mixed channel types.
  
  Attributes:
    impression_coefficients: Media effect coefficients for impression channels, shape (n_geos, n_impression_channels).
    impression_adstock_params: Adstock decay parameters for impression channels, shape (n_impression_channels,).
    impression_ec50_params: Hill saturation EC50 parameters for impression channels, shape (n_impression_channels,).
    impression_slope_params: Hill saturation slope parameters for impression channels, shape (n_impression_channels,).
    rf_coefficients: Media effect coefficients for R&F channels, shape (n_geos, n_rf_channels).
    rf_adstock_params: Adstock decay parameters for R&F channels, shape (n_rf_channels,).
    rf_ec50_params: Hill saturation EC50 parameters for R&F channels (applied to frequency), shape (n_rf_channels,).
    rf_slope_params: Hill saturation slope parameters for R&F channels (applied to frequency), shape (n_rf_channels,).
    baseline: Baseline effect per geo, shape (n_geos,).
    control_coefficients: Control variable coefficients, optional.
  """
  baseline: np.ndarray                                   # shape: (n_geos,)
  impression_coefficients: Optional[np.ndarray] = None  # shape: (n_geos, n_impression_channels)
  impression_adstock_params: Optional[np.ndarray] = None  # shape: (n_impression_channels,)
  impression_ec50_params: Optional[np.ndarray] = None    # shape: (n_impression_channels,)
  impression_slope_params: Optional[np.ndarray] = None   # shape: (n_impression_channels,)
  rf_coefficients: Optional[np.ndarray] = None           # shape: (n_geos, n_rf_channels)
  rf_adstock_params: Optional[np.ndarray] = None         # shape: (n_rf_channels,)
  rf_ec50_params: Optional[np.ndarray] = None            # shape: (n_rf_channels,)
  rf_slope_params: Optional[np.ndarray] = None           # shape: (n_rf_channels,)
  control_coefficients: Optional[np.ndarray] = None
  
  @property
  def has_impression_channels(self) -> bool:
    """Whether impression channel parameters are provided."""
    return (self.impression_coefficients is not None and 
            self.impression_adstock_params is not None and
            self.impression_ec50_params is not None and 
            self.impression_slope_params is not None)
  
  @property
  def has_rf_channels(self) -> bool:
    """Whether R&F channel parameters are provided."""
    return (self.rf_coefficients is not None and 
            self.rf_adstock_params is not None and
            self.rf_ec50_params is not None and 
            self.rf_slope_params is not None)
  
  def __post_init__(self):
    if not self.has_impression_channels and not self.has_rf_channels:
      raise ValueError("Must provide parameters for at least one channel type (impression or R&F)")
    
    # Validate impression parameters if provided
    if self.has_impression_channels:
      n_impression_channels = len(self.impression_adstock_params)
      if self.impression_coefficients.shape[1] != n_impression_channels:
        raise ValueError("Impression coefficients must match number of impression channels")
      if len(self.impression_ec50_params) != n_impression_channels:
        raise ValueError("Impression EC50 params must match number of impression channels")
      if len(self.impression_slope_params) != n_impression_channels:
        raise ValueError("Impression slope params must match number of impression channels")
    
    # Validate R&F parameters if provided
    if self.has_rf_channels:
      n_rf_channels = len(self.rf_adstock_params)
      if self.rf_coefficients.shape[1] != n_rf_channels:
        raise ValueError("R&F coefficients must match number of R&F channels")
      if len(self.rf_ec50_params) != n_rf_channels:
        raise ValueError("R&F EC50 params must match number of R&F channels")
      if len(self.rf_slope_params) != n_rf_channels:
        raise ValueError("R&F slope params must match number of R&F channels")


@dataclasses.dataclass(frozen=True)
class OptimizationScenario:
  """Budget optimization scenario configuration.
  
  Attributes:
    scenario_type: 'fixed_budget' or 'flexible_budget'.
    total_budget: Total budget for fixed budget scenarios.
    target_roi: Target ROI for flexible budget scenarios.
    target_mroi: Target marginal ROI for flexible budget scenarios.
    spend_constraint_lower: Lower bound constraints per channel (0-1).
    spend_constraint_upper: Upper bound constraints per channel (0-1).
    start_date: Start date for optimization period.
    end_date: End date for optimization period.
  """
  scenario_type: str
  total_budget: Optional[float] = None
  target_roi: Optional[float] = None
  target_mroi: Optional[float] = None
  spend_constraint_lower: float = 0.3
  spend_constraint_upper: float = 0.3
  start_date: Optional[str] = None
  end_date: Optional[str] = None
  
  def __post_init__(self):
    if self.scenario_type not in ['fixed_budget', 'flexible_budget']:
      raise ValueError("Scenario type must be 'fixed_budget' or 'flexible_budget'")
    if self.scenario_type == 'fixed_budget' and self.total_budget is None:
      raise ValueError("Fixed budget scenarios require total_budget")
    if self.scenario_type == 'flexible_budget' and not (self.target_roi or self.target_mroi):
      raise ValueError("Flexible budget scenarios require target_roi or target_mroi")


@dataclasses.dataclass(frozen=True)
class IndependentOptimizationResults:
  """Results from independent budget optimization.
  
  Attributes:
    optimized_spend: Optimized spend allocation per channel.
    historical_spend: Historical spend per channel.
    incremental_outcome: Predicted incremental outcome per channel.
    roi_by_channel: ROI per channel.
    total_roi: Total ROI across all channels.
    scenario: The optimization scenario used.
    channels: Channel names.
  """
  optimized_spend: np.ndarray
  historical_spend: np.ndarray
  incremental_outcome: np.ndarray
  roi_by_channel: np.ndarray
  total_roi: float
  scenario: OptimizationScenario
  channels: Sequence[str]
  
  def to_dataframe(self) -> pd.DataFrame:
    """Convert results to pandas DataFrame."""
    return pd.DataFrame({
        'channel': self.channels,
        'historical_spend': self.historical_spend,
        'optimized_spend': self.optimized_spend,
        'incremental_outcome': self.incremental_outcome,
        'roi': self.roi_by_channel,
        'spend_change': self.optimized_spend - self.historical_spend,
        'spend_change_pct': (self.optimized_spend - self.historical_spend) / self.historical_spend * 100
    })


class IndependentOptimizer:
  """Independent budget optimizer that works without Meridian model objects."""
  
  def __init__(self, input_data: OptimizationInput, model_params: ModelParameters):
    """Initialize the independent optimizer.
    
    Args:
      input_data: OptimizationInput containing all required data.
      model_params: ModelParameters containing model coefficients and parameters.
    """
    self.input_data = input_data
    self.model_params = model_params
    self._validate_inputs()
  
  def _validate_inputs(self):
    """Validate that input data and model parameters are consistent."""
    # Validate impression channel consistency
    if self.input_data.n_impression_channels > 0:
      if not self.model_params.has_impression_channels:
        raise ValueError("Impression channels provided but impression parameters missing")
      if self.input_data.impression_data is None:
        raise ValueError("Impression channels specified but impression_data is None")
      if self.model_params.impression_coefficients.shape[1] != self.input_data.n_impression_channels:
        raise ValueError("Impression coefficients must match number of impression channels")
    
    # Validate R&F channel consistency  
    if self.input_data.n_rf_channels > 0:
      if not self.model_params.has_rf_channels:
        raise ValueError("R&F channels provided but R&F parameters missing")
      if self.input_data.rf_reach is None or self.input_data.rf_frequency is None:
        raise ValueError("R&F channels specified but reach/frequency data missing")
      if self.model_params.rf_coefficients.shape[1] != self.input_data.n_rf_channels:
        raise ValueError("R&F coefficients must match number of R&F channels")
    
    # Validate population data if required for scaling
    if self.input_data.population is None:
      raise ValueError("Population data is required for proper media scaling")
    
    # Validate data shape consistency
    if self.input_data.impression_data is not None and self.input_data.population is not None:
      if self.input_data.impression_data.shape[0] != len(self.input_data.population):
        raise ValueError("Impression data geos must match population geos")
    
    if self.input_data.rf_reach is not None and self.input_data.population is not None:
      if self.input_data.rf_reach.shape[0] != len(self.input_data.population):
        raise ValueError("R&F reach data geos must match population geos")
  
  def optimize(self, scenario: OptimizationScenario) -> IndependentOptimizationResults:
    """Run budget optimization for the given scenario.
    
    Args:
      scenario: OptimizationScenario defining the optimization constraints.
      
    Returns:
      IndependentOptimizationResults containing the optimized budget allocation.
    """
    from meridian.optimizer import optimization_core
    
    # Determine optimization period
    selected_times = self._get_time_slice(scenario.start_date, scenario.end_date)
    
    # Set up spend constraints using combined historical spend
    all_historical_spend = self.input_data.all_historical_spend
    budget = scenario.total_budget or np.sum(all_historical_spend)
    spend_allocation = budget * (all_historical_spend / np.sum(all_historical_spend))
    
    lower_bounds, upper_bounds = optimization_core.validate_spend_constraints(
        spend_allocation=spend_allocation,
        historical_spend=all_historical_spend,
        spend_constraint_lower=scenario.spend_constraint_lower,
        spend_constraint_upper=scenario.spend_constraint_upper,
    )
    
    # Create optimization grid
    step_size = budget * 0.001  # 0.1% of budget as step size
    spend_grid, max_steps = optimization_core.create_spend_grid(
        historical_spend=all_historical_spend,
        spend_bounds_lower=lower_bounds,
        spend_bounds_upper=upper_bounds,
        step_size=step_size,
    )
    
    # Compute incremental outcomes for grid
    incremental_outcome_grid = np.zeros_like(spend_grid)
    for i in range(max_steps):
      if not np.isnan(spend_grid[i, :]).any():
        outcomes = self._predict_outcomes_for_spend(spend_grid[i, :], selected_times)
        incremental_outcome_grid[i, :] = outcomes
      else:
        incremental_outcome_grid[i, :] = np.nan
    
    # Run hill-climbing search
    optimal_spend = optimization_core.hill_climbing_search(
        spend_grid=spend_grid,
        incremental_outcome_grid=incremental_outcome_grid,
        scenario_type=scenario.scenario_type,
        total_budget=scenario.total_budget,
        target_roi=scenario.target_roi,
        target_mroi=scenario.target_mroi,
    )
    
    # Calculate final metrics
    final_outcomes = self._predict_outcomes_for_spend(optimal_spend, selected_times)
    roi_by_channel = np.divide(
        final_outcomes, optimal_spend,
        out=np.zeros_like(final_outcomes),
        where=optimal_spend != 0
    )
    total_roi = np.sum(final_outcomes) / np.sum(optimal_spend) if np.sum(optimal_spend) > 0 else 0
    
    return IndependentOptimizationResults(
        optimized_spend=optimal_spend,
        historical_spend=all_historical_spend,
        incremental_outcome=final_outcomes,
        roi_by_channel=roi_by_channel,
        total_roi=total_roi,
        scenario=scenario,
        channels=self.input_data.all_channels,
    )
  
  def predict_incremental_outcome(
      self, 
      spend_allocation: np.ndarray,
      start_date: Optional[str] = None,
      end_date: Optional[str] = None
  ) -> np.ndarray:
    """Predict incremental outcome for given spend allocation.
    
    Args:
      spend_allocation: Spend per channel.
      start_date: Start date for prediction period.
      end_date: End date for prediction period.
      
    Returns:
      Predicted incremental outcome per channel.
    """
    selected_times = self._get_time_slice(start_date, end_date)
    return self._predict_outcomes_for_spend(spend_allocation, selected_times)
  
  def get_response_curves(
      self,
      spend_multipliers: np.ndarray,
      start_date: Optional[str] = None,
      end_date: Optional[str] = None
  ) -> xr.Dataset:
    """Generate response curves for visualization.
    
    Args:
      spend_multipliers: Array of spend multipliers to evaluate.
      start_date: Start date for response curve period.
      end_date: End date for response curve period.
      
    Returns:
      Dataset containing response curves for each channel.
    """
    from meridian.optimizer import optimization_core
    
    selected_times = self._get_time_slice(start_date, end_date)
    
    return optimization_core.compute_response_curves(
        impression_channels=self.input_data.impression_channels,
        rf_channels=self.input_data.rf_channels,
        impression_historical_spend=self.input_data.impression_historical_spend,
        rf_historical_spend=self.input_data.rf_historical_spend,
        impression_data=self.input_data.impression_data,
        rf_reach=self.input_data.rf_reach,
        rf_frequency=self.input_data.rf_frequency,
        impression_coefficients=self.model_params.impression_coefficients,
        rf_coefficients=self.model_params.rf_coefficients,
        impression_alpha=self.model_params.impression_adstock_params,
        impression_ec50=self.model_params.impression_ec50_params,
        impression_slope=self.model_params.impression_slope_params,
        rf_alpha=self.model_params.rf_adstock_params,
        rf_ec50=self.model_params.rf_ec50_params,
        rf_slope=self.model_params.rf_slope_params,
        baseline=self.model_params.baseline,
        spend_multipliers=spend_multipliers,
        selected_times=selected_times,
    )
  
  def _get_time_slice(self, start_date: Optional[str], end_date: Optional[str]) -> Optional[slice]:
    """Convert date strings to time slice."""
    if not start_date and not end_date:
      return None
    
    if not self.input_data.dates:
      return None
    
    dates = list(self.input_data.dates)
    start_idx = 0
    end_idx = len(dates)
    
    if start_date and start_date in dates:
      start_idx = dates.index(start_date)
    if end_date and end_date in dates:
      end_idx = dates.index(end_date) + 1
    
    return slice(start_idx, end_idx)
  
  def _predict_outcomes_for_spend(
      self, 
      spend_allocation: np.ndarray, 
      selected_times: Optional[slice]
  ) -> np.ndarray:
    """Predict incremental outcomes for mixed impression and R&F channels."""
    from meridian.optimizer import media_transforms
    
    # Split spend allocation into impression and R&F portions
    impression_spend = spend_allocation[:self.input_data.n_impression_channels]
    rf_spend = spend_allocation[self.input_data.n_impression_channels:]
    
    # Initialize combined outcomes
    impression_outcomes = np.zeros(self.input_data.n_impression_channels)
    rf_outcomes = np.zeros(self.input_data.n_rf_channels)
    
    # Handle impression channels
    if self.input_data.n_impression_channels > 0 and self.input_data.impression_data is not None:
      # Calculate impression spend multipliers
      impression_multipliers = np.divide(
          impression_spend.astype(np.float64),
          self.input_data.impression_historical_spend.astype(np.float64),
          out=np.ones_like(impression_spend, dtype=np.float64),
          where=self.input_data.impression_historical_spend != 0
      )
      
      # Apply media scaling using MediaScaler
      population_tf = tf.convert_to_tensor(self.input_data.population, dtype=tf.float32)
      impression_data_tf = tf.convert_to_tensor(self.input_data.impression_data, dtype=tf.float32)
      impression_scaler = media_transforms.MediaScaler(impression_data_tf, population_tf)
      scaled_impression_data = impression_scaler.forward(impression_data_tf)
      
      # Scale by spend multipliers
      scaled_impression_data = scaled_impression_data * impression_multipliers[tf.newaxis, tf.newaxis, :]
      
      # Compute impression outcomes
      impression_outcomes = media_transforms.compute_incremental_outcome(
          media=scaled_impression_data,
          media_coefficients=tf.convert_to_tensor(self.model_params.impression_coefficients, dtype=tf.float32),
          alpha=tf.convert_to_tensor(self.model_params.impression_adstock_params, dtype=tf.float32),
          ec50=tf.convert_to_tensor(self.model_params.impression_ec50_params, dtype=tf.float32),
          slope=tf.convert_to_tensor(self.model_params.impression_slope_params, dtype=tf.float32),
          selected_times=selected_times,
      ).numpy()
    
    # Handle R&F channels
    if self.input_data.n_rf_channels > 0 and self.input_data.rf_reach is not None:
      # Calculate R&F spend multipliers
      rf_multipliers = np.divide(
          rf_spend.astype(np.float64),
          self.input_data.rf_historical_spend.astype(np.float64),
          out=np.ones_like(rf_spend, dtype=np.float64),
          where=self.input_data.rf_historical_spend != 0
      )
      
      # Apply media scaling to reach only (frequency stays unscaled)
      population_tf = tf.convert_to_tensor(self.input_data.population, dtype=tf.float32)
      rf_reach_tf = tf.convert_to_tensor(self.input_data.rf_reach, dtype=tf.float32)
      reach_scaler = media_transforms.MediaScaler(rf_reach_tf, population_tf)
      scaled_rf_reach = reach_scaler.forward(rf_reach_tf)
      
      # Scale reach by spend multipliers (frequency scaling is handled differently)
      scaled_rf_reach = scaled_rf_reach * rf_multipliers[tf.newaxis, tf.newaxis, :]
      rf_frequency_tf = tf.convert_to_tensor(self.input_data.rf_frequency, dtype=tf.float32)
      
      # Compute R&F outcomes
      rf_outcomes = media_transforms.compute_rf_incremental_outcome(
          reach=scaled_rf_reach,
          frequency=rf_frequency_tf,
          rf_coefficients=tf.convert_to_tensor(self.model_params.rf_coefficients, dtype=tf.float32),
          alpha=tf.convert_to_tensor(self.model_params.rf_adstock_params, dtype=tf.float32),
          ec50=tf.convert_to_tensor(self.model_params.rf_ec50_params, dtype=tf.float32),
          slope=tf.convert_to_tensor(self.model_params.rf_slope_params, dtype=tf.float32),
          selected_times=selected_times,
      ).numpy()
    
    # Combine outcomes from both channel types
    return np.concatenate([impression_outcomes, rf_outcomes])