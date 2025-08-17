"""Mini MMM Quickstart Example.

This example demonstrates the basic usage of the Mini MMM framework,
including data loading, model fitting, analysis, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Mini MMM imports
from mini_mmm import MiniMMM, SimpleInputData
from mini_mmm.data.validators import validate_input_data, check_data_quality
from mini_mmm.model.priors import DefaultPriors
from mini_mmm.analysis import Analyzer, ResponseCurves, BudgetOptimizer
from mini_mmm.viz.plots import create_full_report

# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(n_weeks: int = 104, 
                           n_channels: int = 4) -> pd.DataFrame:
  """Generates synthetic MMM data for demonstration."""
  
  # Channel names
  channels = [f'channel_{i+1}' for i in range(n_channels)]
  
  # Base parameters for data generation
  base_kpi = 1000
  seasonality = 200 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
  trend = 5 * np.arange(n_weeks)
  noise = np.random.normal(0, 50, n_weeks)
  
  # Generate media spend with some realistic patterns
  media_data = {}
  total_media_effect = 0
  
  for i, channel in enumerate(channels):
    # Different spend patterns per channel
    if i == 0:  # High spend, seasonal
      base_spend = 50 + 30 * np.sin(2 * np.pi * np.arange(n_weeks) / 26)
    elif i == 1:  # Medium spend, campaigns
      base_spend = 30 + 20 * (np.random.random(n_weeks) > 0.7)
    elif i == 2:  # Low but consistent
      base_spend = 15 + 5 * np.random.random(n_weeks)
    else:  # Variable spend
      base_spend = 25 * np.random.exponential(1, n_weeks)
    
    # Add some noise and ensure non-negative
    spend = np.maximum(0, base_spend + np.random.normal(0, 5, n_weeks))
    media_data[f'{channel}_spend'] = spend
    
    # Generate media effect (simplified adstock + hill)
    adstock_rate = 0.3 + 0.4 * np.random.random()
    saturation_point = np.percentile(spend, 70)
    
    # Simple adstock
    adstocked = np.zeros_like(spend)
    for t in range(n_weeks):
      adstocked[t] = spend[t]
      if t > 0:
        adstocked[t] += adstock_rate * adstocked[t-1]
    
    # Simple hill saturation
    hill_effect = 2 + 3 * np.random.random()  # Random ROI
    saturated = hill_effect * adstocked / (saturation_point + adstocked)
    
    total_media_effect += saturated
  
  # Generate KPI as combination of base, media effects, and noise
  kpi = base_kpi + seasonality + trend + total_media_effect + noise
  
  # Create DataFrame
  data = {'kpi': kpi}
  data.update(media_data)
  
  # Add some control variables
  data['price_index'] = 100 + 10 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + np.random.normal(0, 2, n_weeks)
  data['competitor_activity'] = 50 + 20 * np.random.random(n_weeks)
  
  # Add dates
  data['date'] = pd.date_range(start='2022-01-01', periods=n_weeks, freq='W')
  
  return pd.DataFrame(data)


def main():
  """Main quickstart example."""
  
  print("=== Mini MMM Quickstart Example ===\n")
  
  # 1. Generate synthetic data
  print("1. Generating synthetic data...")
  df = generate_synthetic_data(n_weeks=104, n_channels=4)
  
  media_channels = ['channel_1', 'channel_2', 'channel_3', 'channel_4'] 
  media_cols = [f'{ch}_spend' for ch in media_channels]
  control_cols = ['price_index', 'competitor_activity']
  
  print(f"   Generated {len(df)} weeks of data")
  print(f"   Media channels: {media_channels}")
  print(f"   Control variables: {control_cols}")
  
  # 2. Create SimpleInputData
  print("\n2. Creating SimpleInputData...")
  input_data = SimpleInputData.from_dataframe(
      df=df,
      kpi_col='kpi',
      media_cols=media_cols,
      control_cols=control_cols,
      date_col='date'
  )
  
  print(input_data.summary())
  
  # 3. Validate data quality
  print("\n3. Validating data quality...")
  is_valid = check_data_quality(input_data)
  if is_valid:
    print("   Data quality check: PASSED")
  else:
    print("   Data quality check: FAILED - continuing anyway for demo")
  
  # 4. Set up model with conservative priors
  print("\n4. Setting up Mini MMM model...")
  priors = DefaultPriors.conservative()
  print("   Using conservative priors")
  
  model = MiniMMM(
      prior_config=priors,
      random_seed=42
  )
  
  # 5. Fit the model
  print("\n5. Fitting the model...")
  print("   This may take a few minutes...")
  
  try:
    model.fit(
        data=input_data,
        draws=1000,  # Fewer draws for faster demo
        tune=500,
        chains=2,
        cores=1,
        target_accept=0.8
    )
    print("   Model fitting completed successfully!")
  except Exception as e:
    print(f"   Model fitting failed: {e}")
    print("   This might be due to missing PyMC installation")
    print("   Install with: pip install pymc")
    return
  
  # 6. Model diagnostics
  print("\n6. Model diagnostics...")
  diagnostics = model.get_diagnostics()
  print(f"   Max R-hat: {diagnostics['convergence']['max_rhat']:.3f}")
  print(f"   Min ESS: {diagnostics['convergence']['min_ess_bulk']}")
  print(f"   RMSE: {diagnostics['fit_metrics']['rmse']:.2f}")
  print(f"   R²: {diagnostics['fit_metrics']['r2']:.3f}")
  
  # 7. Media effects analysis
  print("\n7. Media effects analysis...")
  analyzer = Analyzer(model)
  
  # ROI analysis
  roi_results = analyzer.compute_roi()
  print("\n   ROI by channel:")
  for _, row in roi_results.iterrows():
    print(f"     {row['channel']}: {row['roi']:.2f}x (spend: ${row['total_spend']:,.0f})")
  
  # Contribution analysis
  contribution_results = analyzer.compute_contribution()
  print("\n   Contribution breakdown:")
  for _, row in contribution_results.iterrows():
    if row['contribution_pct'] > 1:  # Only show significant contributions
      print(f"     {row['component']}: {row['contribution_pct']:.1f}%")
  
  # 8. Response curves analysis
  print("\n8. Response curves analysis...")
  response_curves = ResponseCurves(model)
  
  # Saturation summary
  saturation_summary = response_curves.compute_saturation_summary()
  print("\n   Saturation analysis:")
  for _, row in saturation_summary.iterrows():
    print(f"     {row['channel']}: {row['current_saturation_level']:.1%} saturated")
  
  # 9. Budget optimization
  print("\n9. Budget optimization...")
  optimizer = BudgetOptimizer(model)
  
  # Current budget
  current_budget = np.sum(input_data.get_media_matrix())
  print(f"   Current total budget: ${current_budget:,.0f}")
  
  # Optimize budget
  opt_results = optimizer.optimize_budget(
      total_budget=current_budget,
      objective='total_effect',
      method='scipy'
  )
  
  if opt_results['optimization_info']['success']:
    print(f"   Optimization successful!")
    print(f"   Expected total ROI: {opt_results['total_roi']:.2f}x")
    print("\n   Optimal allocation:")
    for _, row in opt_results['allocation'].iterrows():
      print(f"     {row['channel']}: ${row['optimal_spend']:,.0f} "
            f"({row['spend_share']:.1%}) -> ROI: {row['channel_roi']:.2f}x")
  else:
    print(f"   Optimization failed: {opt_results['optimization_info']['message']}")
  
  # 10. Visualizations
  print("\n10. Creating visualizations...")
  try:
    figures = create_full_report(model, save_path='mini_mmm_report')
    print(f"   Created {len(figures)} visualization plots")
    print("   Saved plots with prefix 'mini_mmm_report'")
    
    # Show the plots
    plt.show()
  except Exception as e:
    print(f"   Visualization error: {e}")
    print("   This might be due to missing matplotlib/seaborn")
  
  print("\n=== Mini MMM Quickstart Complete ===")
  print("\nNext steps:")
  print("- Experiment with different prior configurations")
  print("- Try budget optimization with different objectives")
  print("- Analyze response curves for individual channels")
  print("- Use your own data with SimpleInputData.from_dataframe()")


if __name__ == "__main__":
  main()