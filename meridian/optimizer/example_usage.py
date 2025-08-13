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

"""Example usage of the independent budget optimizer."""

import numpy as np
from meridian.optimizer.independent_optimizer import (
    OptimizationInput, 
    ModelParameters, 
    OptimizationScenario,
    IndependentOptimizer
)
from meridian.optimizer import io_utils


def create_example_mixed_data():
  """Create example data with both impression and R&F channels."""
  # Example channels
  impression_channels = ['tv', 'digital', 'radio']
  rf_channels = ['social_media', 'display_video']
  
  # Example dimensions
  n_geos = 3
  n_times = 52  # 52 weeks
  n_impression_channels = len(impression_channels)
  n_rf_channels = len(rf_channels)
  
  # Generate synthetic historical spend
  impression_historical_spend = np.array([100000, 80000, 50000])  # $230k total
  rf_historical_spend = np.array([40000, 30000])  # $70k total
  
  # Generate synthetic population data
  geos = ['US_West', 'US_Central', 'US_East']
  population = np.array([50000000, 65000000, 75000000])  # Population per geo
  
  # Generate synthetic impression data (n_geos, n_times, n_impression_channels)
  np.random.seed(42)
  impression_data = np.random.lognormal(
      mean=8, sigma=1, size=(n_geos, n_times, n_impression_channels)
  )
  
  # Generate synthetic R&F data
  # Reach: number of unique people reached
  rf_reach = np.random.lognormal(
      mean=6, sigma=0.8, size=(n_geos, n_times, n_rf_channels)
  )
  
  # Frequency: average exposures per person (typically 1-10)
  rf_frequency = np.random.lognormal(
      mean=1, sigma=0.5, size=(n_geos, n_times, n_rf_channels)
  )
  rf_frequency = np.clip(rf_frequency, 1.0, 10.0)  # Realistic frequency range
  
  # R&F spend data
  rf_spend = np.random.lognormal(
      mean=7, sigma=0.8, size=(n_geos, n_times, n_rf_channels)
  )
  
  # Generate dates
  import pandas as pd
  dates = pd.date_range('2024-01-01', periods=n_times, freq='W').strftime('%Y-%m-%d').tolist()
  
  # Create optimization input with mixed channels
  input_data = OptimizationInput(
      impression_channels=impression_channels,
      rf_channels=rf_channels,
      impression_historical_spend=impression_historical_spend,
      rf_historical_spend=rf_historical_spend,
      impression_data=impression_data,
      rf_reach=rf_reach,
      rf_frequency=rf_frequency,
      rf_spend=rf_spend,
      dates=dates,
      geos=geos,
      population=population,
  )
  
  # Generate synthetic model parameters for mixed channels
  model_params = ModelParameters(
      # Impression channel parameters
      impression_coefficients=np.random.uniform(0.5, 2.0, size=(n_geos, n_impression_channels)),
      impression_adstock_params=np.random.uniform(0.1, 0.7, size=n_impression_channels),
      impression_ec50_params=np.random.uniform(0.3, 0.8, size=n_impression_channels),
      impression_slope_params=np.random.uniform(0.8, 2.0, size=n_impression_channels),
      # R&F channel parameters  
      rf_coefficients=np.random.uniform(0.3, 1.5, size=(n_geos, n_rf_channels)),
      rf_adstock_params=np.random.uniform(0.2, 0.6, size=n_rf_channels),
      rf_ec50_params=np.random.uniform(2.0, 6.0, size=n_rf_channels),  # Higher EC50 for frequency
      rf_slope_params=np.random.uniform(1.0, 2.5, size=n_rf_channels),
      # Common parameters
      baseline=np.random.uniform(1000, 5000, size=n_geos),
  )
  
  return input_data, model_params


def create_impression_only_data():
  """Create example data with impression channels only (no R&F)."""
  # Example channels
  impression_channels = ['tv', 'digital', 'radio', 'print']
  rf_channels = []  # No R&F channels
  
  # Example dimensions
  n_geos = 3
  n_times = 52  # 52 weeks
  n_impression_channels = len(impression_channels)
  
  # Generate synthetic historical spend
  impression_historical_spend = np.array([100000, 80000, 50000, 30000])  # $260k total
  rf_historical_spend = np.array([])  # Empty for R&F
  
  # Generate synthetic population data
  geos = ['US_West', 'US_Central', 'US_East']
  population = np.array([50000000, 65000000, 75000000])  # Population per geo
  
  # Generate synthetic impression data (n_geos, n_times, n_impression_channels)
  np.random.seed(42)
  impression_data = np.random.lognormal(
      mean=8, sigma=1, size=(n_geos, n_times, n_impression_channels)
  )
  
  # Generate dates
  import pandas as pd
  dates = pd.date_range('2024-01-01', periods=n_times, freq='W').strftime('%Y-%m-%d').tolist()
  
  # Create optimization input with impression channels only
  input_data = OptimizationInput(
      impression_channels=impression_channels,
      rf_channels=rf_channels,
      impression_historical_spend=impression_historical_spend,
      rf_historical_spend=rf_historical_spend,
      impression_data=impression_data,
      rf_reach=None,
      rf_frequency=None,
      rf_spend=None,
      dates=dates,
      geos=geos,
      population=population,
  )
  
  # Generate synthetic model parameters for impression channels only
  model_params = ModelParameters(
      baseline=np.random.uniform(1000, 5000, size=n_geos),
      # Impression channel parameters only
      impression_coefficients=np.random.uniform(0.5, 2.0, size=(n_geos, n_impression_channels)),
      impression_adstock_params=np.random.uniform(0.1, 0.7, size=n_impression_channels),
      impression_ec50_params=np.random.uniform(0.3, 0.8, size=n_impression_channels),
      impression_slope_params=np.random.uniform(0.8, 2.0, size=n_impression_channels),
      # No R&F parameters
      rf_coefficients=None,
      rf_adstock_params=None,
      rf_ec50_params=None,
      rf_slope_params=None,
  )
  
  return input_data, model_params


def create_rf_only_data():
  """Create example data with R&F channels only (no impression channels)."""
  # Example channels
  impression_channels = []  # No impression channels
  rf_channels = ['social_media', 'display_video', 'connected_tv']
  
  # Example dimensions
  n_geos = 3
  n_times = 52  # 52 weeks
  n_rf_channels = len(rf_channels)
  
  # Generate synthetic historical spend
  impression_historical_spend = np.array([])  # Empty for impression
  rf_historical_spend = np.array([60000, 40000, 50000])  # $150k total
  
  # Generate synthetic population data
  geos = ['US_West', 'US_Central', 'US_East']
  population = np.array([50000000, 65000000, 75000000])  # Population per geo
  
  # Generate synthetic R&F data only
  np.random.seed(42)
  # Reach: number of unique people reached
  rf_reach = np.random.lognormal(
      mean=6, sigma=0.8, size=(n_geos, n_times, n_rf_channels)
  )
  
  # Frequency: average exposures per person (typically 1-10)
  rf_frequency = np.random.lognormal(
      mean=1, sigma=0.5, size=(n_geos, n_times, n_rf_channels)
  )
  rf_frequency = np.clip(rf_frequency, 1.0, 10.0)  # Realistic frequency range
  
  # R&F spend data
  rf_spend = np.random.lognormal(
      mean=7, sigma=0.8, size=(n_geos, n_times, n_rf_channels)
  )
  
  # Generate dates
  import pandas as pd
  dates = pd.date_range('2024-01-01', periods=n_times, freq='W').strftime('%Y-%m-%d').tolist()
  
  # Create optimization input with R&F channels only
  input_data = OptimizationInput(
      impression_channels=impression_channels,
      rf_channels=rf_channels,
      impression_historical_spend=impression_historical_spend,
      rf_historical_spend=rf_historical_spend,
      impression_data=None,
      rf_reach=rf_reach,
      rf_frequency=rf_frequency,
      rf_spend=rf_spend,
      dates=dates,
      geos=geos,
      population=population,
  )
  
  # Generate synthetic model parameters for R&F channels only
  model_params = ModelParameters(
      baseline=np.random.uniform(1000, 5000, size=n_geos),
      # No impression channel parameters
      impression_coefficients=None,
      impression_adstock_params=None,
      impression_ec50_params=None,
      impression_slope_params=None,
      # R&F channel parameters only
      rf_coefficients=np.random.uniform(0.3, 1.5, size=(n_geos, n_rf_channels)),
      rf_adstock_params=np.random.uniform(0.2, 0.6, size=n_rf_channels),
      rf_ec50_params=np.random.uniform(2.0, 6.0, size=n_rf_channels),  # Higher EC50 for frequency
      rf_slope_params=np.random.uniform(1.0, 2.5, size=n_rf_channels),
  )
  
  return input_data, model_params


def example_fixed_budget_optimization():
  """Example of fixed budget optimization with mixed channels."""
  print("=== Fixed Budget Optimization Example (Mixed Channels) ===")
  
  # Create example data with both impression and R&F channels
  input_data, model_params = create_example_mixed_data()
  
  # Initialize optimizer
  optimizer = IndependentOptimizer(input_data, model_params)
  
  # Define optimization scenario
  scenario = OptimizationScenario(
      scenario_type='fixed_budget',
      total_budget=320000,  # 6.7% increase from $300k historical (230k impression + 70k R&F)
      spend_constraint_lower=0.2,  # Allow 20% decrease
      spend_constraint_upper=0.4,  # Allow 40% increase
  )
  
  # Run optimization
  results = optimizer.optimize(scenario)
  
  # Display results
  print(f"Total ROI: {results.total_roi:.2f}")
  print("\nOptimization Results:")
  print(results.to_dataframe().round(2))
  
  # Generate visualizations (demonstrating chart creation)
  print("\nVisualization charts available:")
  print("- visualizer.plot_spend_allocation(results, optimized=True)")
  print("- visualizer.plot_spend_delta(results)")
  print("- visualizer.plot_roi_comparison(results)")
  
  return results, optimizer


def example_impression_only_optimization():
  """Example of optimization with impression channels only."""
  print("=== Impression-Only Optimization Example ===")
  
  # Create example data with impression channels only
  input_data, model_params = create_impression_only_data()
  
  # Initialize optimizer
  optimizer = IndependentOptimizer(input_data, model_params)
  
  # Define optimization scenario
  scenario = OptimizationScenario(
      scenario_type='fixed_budget',
      total_budget=280000,  # 7.7% increase from $260k historical
      spend_constraint_lower=0.2,  # Allow 20% decrease
      spend_constraint_upper=0.4,  # Allow 40% increase
  )
  
  # Run optimization
  results = optimizer.optimize(scenario)
  
  print(f"Total ROI: {results.total_roi:.2f}")
  print(f"\nOptimization Results:")
  print(results.to_dataframe())
  
  print(f"\nChannel Types:")
  print(f"- Impression channels: {len(input_data.impression_channels)}")
  print(f"- R&F channels: {len(input_data.rf_channels)}")
  print(f"- Has impression params: {model_params.has_impression_channels}")
  print(f"- Has R&F params: {model_params.has_rf_channels}")


def example_rf_only_optimization():
  """Example of optimization with R&F channels only."""
  print("=== R&F-Only Optimization Example ===")
  
  # Create example data with R&F channels only
  input_data, model_params = create_rf_only_data()
  
  # Initialize optimizer
  optimizer = IndependentOptimizer(input_data, model_params)
  
  # Define optimization scenario
  scenario = OptimizationScenario(
      scenario_type='fixed_budget',
      total_budget=160000,  # 6.7% increase from $150k historical
      spend_constraint_lower=0.2,  # Allow 20% decrease
      spend_constraint_upper=0.4,  # Allow 40% increase
  )
  
  # Run optimization
  results = optimizer.optimize(scenario)
  
  print(f"Total ROI: {results.total_roi:.2f}")
  print(f"\nOptimization Results:")
  print(results.to_dataframe())
  
  print(f"\nChannel Types:")
  print(f"- Impression channels: {len(input_data.impression_channels)}")
  print(f"- R&F channels: {len(input_data.rf_channels)}")
  print(f"- Has impression params: {model_params.has_impression_channels}")
  print(f"- Has R&F params: {model_params.has_rf_channels}")


def example_flexible_budget_optimization():
  """Example of flexible budget optimization with target ROI."""
  print("\n=== Flexible Budget Optimization Example ===")
  
  # Create example data
  input_data, model_params = create_example_mixed_data()
  
  # Initialize optimizer
  optimizer = IndependentOptimizer(input_data, model_params)
  
  # Define optimization scenario
  scenario = OptimizationScenario(
      scenario_type='flexible_budget',
      target_roi=2.5,  # Target 2.5x ROI
      spend_constraint_lower=0.3,
      spend_constraint_upper=0.5,
  )
  
  # Run optimization
  results = optimizer.optimize(scenario)
  
  # Display results
  print(f"Target ROI: {scenario.target_roi}")
  print(f"Achieved ROI: {results.total_roi:.2f}")
  print(f"Total Budget: ${np.sum(results.optimized_spend):,.0f}")
  print("\nOptimization Results:")
  print(results.to_dataframe().round(2))
  
  return results, optimizer


def example_file_based_workflow():
  """Example of loading data and parameters from files."""
  print("\n=== File-Based Workflow Example ===")
  
  # Create example data
  input_data, model_params = create_example_mixed_data()
  
  # Save model parameters to JSON
  io_utils.save_model_parameters_to_json(model_params, '/tmp/model_params.json')
  print("Model parameters saved to /tmp/model_params.json")
  
  # Load model parameters from JSON
  loaded_params = io_utils.load_model_parameters_from_json('/tmp/model_params.json')
  print("Model parameters loaded from JSON")
  
  # Run optimization with loaded parameters
  optimizer = IndependentOptimizer(input_data, loaded_params)
  scenario = OptimizationScenario(scenario_type='fixed_budget', total_budget=300000)
  results = optimizer.optimize(scenario)
  
  # Save results to CSV
  io_utils.save_optimization_results_to_csv(results, '/tmp/optimization_results.csv')
  print("Results saved to /tmp/optimization_results.csv")
  
  return results


def test_all_channel_combinations():
  """Test all possible channel combinations."""
  print("Testing Independent Optimizer with Different Channel Combinations")
  print("=" * 65)
  
  # Test 1: Mixed channels (impression + R&F)
  try:
    example_fixed_budget_optimization()
    print("✅ Mixed channels test PASSED\n")
  except Exception as e:
    print(f"❌ Mixed channels test FAILED: {e}\n")
  
  # Test 2: Impression channels only
  try:
    example_impression_only_optimization()
    print("✅ Impression-only channels test PASSED\n")
  except Exception as e:
    print(f"❌ Impression-only channels test FAILED: {e}\n")
  
  # Test 3: R&F channels only
  try:
    example_rf_only_optimization()
    print("✅ R&F-only channels test PASSED\n")
  except Exception as e:
    print(f"❌ R&F-only channels test FAILED: {e}\n")


if __name__ == '__main__':
  test_all_channel_combinations()