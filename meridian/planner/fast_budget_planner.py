import logging
import os
import sys

from meridian.model import model
from meridian.model import spec
from meridian.analysis import optimizer

from meridian.planner import FlexibleBudgetPlanner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# step 1: -------------------------------------------- load the actual model with complete posterior estimates in it ------------------------------- #
demo_model_path = '/Users/mariappan.subramanian/Documents/repo/forked/meridian/demo/saved_models'
demo_model_file = f"{demo_model_path}/demo_model_geo_all_channels.pkl"
temp_optimization_file = '/tmp/full_optimization_results.pkl'

# Check if we already have cached optimization results
if os.path.exists(temp_optimization_file):
  print("Loading cached full optimization results...")
  try:
    full_optimization_results = model.load_mmm(temp_optimization_file)
    full_optimized_data = full_optimization_results.optimized_data
  except Exception as e:
    print(f"Failed to load cached results: {e}")
    print("Running fresh optimization...")
    full_model = model.load_mmm(demo_model_file)
    full_budget_optimizer = optimizer.BudgetOptimizer(full_model)
    full_optimization_results = full_budget_optimizer.optimize()
    full_optimized_data = full_optimization_results.optimized_data
else:
  print("Running full model optimization (this may take a while)...")
  full_model = model.load_mmm(demo_model_file)
  full_budget_optimizer = optimizer.BudgetOptimizer(full_model)
  full_optimization_results = full_budget_optimizer.optimize()
  full_optimized_data = full_optimization_results.optimized_data
  
  # Cache the results
  print("Saving optimization results to cache...")
  try:
    model.save_mmm(full_optimization_results, temp_optimization_file)
    print("Successfully cached optimization results!")
  except Exception as e:
    print(f"Could not save optimization results: {e}")
    print("Continuing without caching...")

# step 2: -------------------------------------------- Excel-based point-estimates MMM model ------------------------------- #
excel_file_path = (
  '/Users/mariappan.subramanian/Library/CloudStorage/'
  'OneDrive-TheTradeDesk/MMM/BudgetOptimizer/mmm_input_artifacts.xlsx'
)

# Model configuration based on actual Excel file structure
model_config = {

  # time and geo inputs
  'time_col': 'week',
  'geo_col': 'geo',
  'population_col': 'population',

  # kpi inputs
  'kpi_type': 'non_revenue',
  'kpi_col': 'conversions',
  'revenue_per_kpi_col': 'revenue_per_conversion',

  # impression based media inputs
  'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
  'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
  'media_channels': ['Channel0', 'Channel1', 'Channel2'],

  # reach based media inputs
  'reach_cols': ['Channel3_reach'],
  'frequency_cols': ['Channel3_frequency'],
  'rf_spend_cols': ['Channel3_spend'],
  'rf_channels': ['Channel3']
  }

# create data loader & point inference data
flexible_budget_planner = FlexibleBudgetPlanner(file_name=excel_file_path, model_config=model_config)
data = flexible_budget_planner.build_input_data()
inference_data = flexible_budget_planner.get_inference_data()

# Test different n_draws values to test hypothesis
print("\n" + "="*80)
print("TESTING HYPOTHESIS: Effect of n_draws on optimization results")
print("="*80)

n_draws_values = [1, 10, 100, 1000]
optimization_results = {}

for n_draws in n_draws_values:
  print(f"\nTesting with n_draws={n_draws}...")
  
  # create a dummy model object
  dummy_model = model.Meridian(input_data=data, model_spec=spec.ModelSpec(), inference_data=inference_data)
  dummy_model.sample_prior(n_draws=n_draws, seed=42)
  fast_budget_optimizer = optimizer.BudgetOptimizer(dummy_model)
  fast_optimizer_results = fast_budget_optimizer.optimize()
  
  # Store results for comparison
  optimization_results[n_draws] = fast_optimizer_results.optimized_data
  
  # Extract key metrics for quick comparison
  roi_mean = fast_optimizer_results.optimized_data['roi'].sel(metric='mean').values
  spend_values = fast_optimizer_results.optimized_data['spend'].values
  
  print(f"  ROI means: {roi_mean}")
  print(f"  Spend values: {spend_values}")

# Use n_draws=100 as the reference for detailed comparison (as in original)
fast_optimized_data = optimization_results[100]

print(f"\nCOMPARING RESULTS ACROSS DIFFERENT n_draws VALUES:")
print("-" * 60)

# Compare key metrics across all n_draws
metrics_to_compare = ['roi', 'mroi', 'incremental_outcome', 'spend']

for metric in metrics_to_compare:
  print(f"\n{metric.upper()} Comparison (mean values):")
  print("-" * 40)
  
  for n_draws in n_draws_values:
    if metric == 'spend':
      # Spend doesn't have metric dimension
      values = optimization_results[n_draws][metric].values
    else:
      # Other metrics have metric dimension, extract 'mean'
      values = optimization_results[n_draws][metric].sel(metric='mean').values
    
    print(f"n_draws={n_draws:4d}: {values}")
    
  # Calculate max difference across n_draws for this metric
  all_values = []
  for n_draws in n_draws_values:
    if metric == 'spend':
      vals = optimization_results[n_draws][metric].values
    else:
      vals = optimization_results[n_draws][metric].sel(metric='mean').values
    all_values.append(vals)
  
  all_values = np.array(all_values)
  max_diff = np.max(all_values, axis=0) - np.min(all_values, axis=0)
  max_pct_diff = (max_diff / np.mean(all_values, axis=0)) * 100
  
  print(f"Max abs difference: {max_diff}")
  print(f"Max % difference: {max_pct_diff}")

print("\n" + "="*80)
print("HYPOTHESIS TEST CONCLUSION:")
print("If results are identical/very similar across n_draws, then optimization")
print("is using our Excel posterior point estimates (as expected).")
print("If results vary significantly, then prior samples are affecting optimization.")
print("="*80)

# step 3: -------------------------------------------- Comparison between full and fast optimization results ------------------------------- #
import pandas as pd

def calculate_percent_difference(full_value, fast_value):
  """Calculate percent difference: (fast - full) / full * 100"""
  if full_value == 0:
    return 0 if fast_value == 0 else np.inf
  return ((fast_value - full_value) / full_value) * 100

# Debug: Check data structure first
print(f"\nDEBUG: Full optimized data type: {type(full_optimized_data)}")
print(f"DEBUG: Fast optimized data type: {type(fast_optimized_data)}")

# Check if it's a pandas DataFrame or xarray Dataset
if hasattr(full_optimized_data, 'data_vars'):
  # It's an xarray Dataset
  print(f"DEBUG: Full data variables: {list(full_optimized_data.data_vars.keys())}")
  print(f"DEBUG: Fast data variables: {list(fast_optimized_data.data_vars.keys())}")
  print(f"DEBUG: Full data coords: {list(full_optimized_data.coords.keys())}")
  print(f"DEBUG: Fast data coords: {list(fast_optimized_data.coords.keys())}")
  
  # Extract channel names from coordinates
  channels = None
  if 'channel' in full_optimized_data.coords:
    channels = full_optimized_data.coords['channel'].values
    channel_dim = 'channel'
  elif 'media_channel' in full_optimized_data.coords:
    channels = full_optimized_data.coords['media_channel'].values
    channel_dim = 'media_channel'
  elif 'rf_channel' in full_optimized_data.coords:
    channels = full_optimized_data.coords['rf_channel'].values  
    channel_dim = 'rf_channel'
  else:
    print("DEBUG: No channel dimension found, using aggregated values")
    channels = ['Total']
    channel_dim = None
    
  print(f"DEBUG: Channels found: {channels}")
  
  comparison_data = []
  
  # Compare key metrics
  common_vars = set(full_optimized_data.data_vars.keys()) & set(fast_optimized_data.data_vars.keys())
  print(f"DEBUG: Common variables: {common_vars}")
  
  for var in common_vars:
    full_val = full_optimized_data[var]
    fast_val = fast_optimized_data[var]
    
    print(f"DEBUG: {var} - Full shape: {full_val.shape}, Fast shape: {fast_val.shape}")
    print(f"DEBUG: {var} - Full dims: {full_val.dims}, Fast dims: {fast_val.dims}")
    
    # If there's a channel dimension, iterate over channels
    if channel_dim and channel_dim in full_val.dims:
      for i, channel in enumerate(channels):
        # Get mean value across other dimensions for this channel
        full_channel_val = full_val.isel({channel_dim: i}).mean().values
        fast_channel_val = fast_val.isel({channel_dim: i}).mean().values
        
        pct_diff = calculate_percent_difference(full_channel_val, fast_channel_val)
        comparison_data.append({
          'channel': channel,
          'metric': var,
          'full_value': float(full_channel_val),
          'fast_value': float(fast_channel_val),
          'percent_difference': pct_diff
        })
    else:
      # No channel dimension, use overall mean
      full_mean_val = full_val.mean().values
      fast_mean_val = fast_val.mean().values
      
      pct_diff = calculate_percent_difference(full_mean_val, fast_mean_val)
      comparison_data.append({
        'channel': 'Total',
        'metric': var,
        'full_value': float(full_mean_val),
        'fast_value': float(fast_mean_val),
        'percent_difference': pct_diff
      })

elif hasattr(full_optimized_data, 'keys'):
  # It's a dictionary or similar
  channels = list(full_optimized_data.keys())
  print(f"DEBUG: Channels from keys: {channels}")
  
  comparison_data = []
  
  for channel in channels:
    full_data = full_optimized_data[channel]
    fast_data = fast_optimized_data[channel]
    
    print(f"DEBUG: {channel} - Full data type: {type(full_data)}")
    
    # Compare key attributes/metrics
    if hasattr(full_data, '__dict__'):
      attrs = [attr for attr in vars(full_data).keys() if not attr.startswith('_')]
      for attr in attrs:
        full_val = getattr(full_data, attr)
        fast_val = getattr(fast_data, attr)
        
        if np.isscalar(full_val) and np.isscalar(fast_val):
          pct_diff = calculate_percent_difference(full_val, fast_val)
          comparison_data.append({
            'channel': channel,
            'metric': attr,
            'full_value': full_val,
            'fast_value': fast_val,
            'percent_difference': pct_diff
          })

# Create and display comparison table
if comparison_data:
  comparison_df = pd.DataFrame(comparison_data)
  
  # Create pivot table for better formatting
  if 'channel' in comparison_df.columns:
    pivot_df = comparison_df.pivot_table(
      index='channel', 
      columns='metric', 
      values=['full_value', 'fast_value', 'percent_difference'],
      aggfunc='first'
    )
  else:
    # Fallback for data without channel column
    pivot_df = comparison_df.set_index('metric')[['full_value', 'fast_value', 'percent_difference']]
  
  print("\n" + "="*120)
  print("BUDGET OPTIMIZATION COMPARISON: Full Model vs Excel Point Estimates")
  print("="*120)
  
  # Display table by metric
  if 'channel' in comparison_df.columns and hasattr(pivot_df, 'columns') and hasattr(pivot_df.columns, 'levels'):
    # Multi-level column structure
    metrics = ['spend', 'roi', 'mroi', 'incremental_outcome', 'effectiveness', 'cpik']
    
    for metric in metrics:
      if metric in pivot_df.columns.levels[1]:
        print(f"\n{metric.upper()} COMPARISON:")
        print("-" * 80)
        print(f"{'Channel':<12} {'Full_Opt':>12} {'Point_Opt':>12} {'Pct_Diff':>10}")
        print("-" * 80)
        
        for channel in pivot_df.index:
          full_val = pivot_df.loc[channel, ('full_value', metric)]
          point_val = pivot_df.loc[channel, ('fast_value', metric)]
          pct_diff = pivot_df.loc[channel, ('percent_difference', metric)]
          
          # Skip rows with NaN values
          if pd.isna(full_val) or pd.isna(point_val):
            continue
          
          # Format numbers based on metric type
          if metric in ['spend', 'incremental_outcome']:
            print(f"{channel:<12} {full_val:>12,.0f} {point_val:>12,.0f} {pct_diff:>9.1f}%")
          else:
            print(f"{channel:<12} {full_val:>12.3f} {point_val:>12.3f} {pct_diff:>9.1f}%")
  else:
    # Simple table structure - display each channel and metric combination
    print(f"\n{'Channel':<15} {'Metric':<20} {'Full_Opt':>12} {'Point_Opt':>12} {'Pct_Diff':>10}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
      channel = row['channel'] if 'channel' in row else 'Total'
      metric = row['metric']
      full_val = row['full_value']
      point_val = row['fast_value']
      pct_diff = row['percent_difference']
      
      # Format numbers based on metric type
      if metric in ['spend', 'incremental_outcome']:
        print(f"{channel:<15} {metric:<20} {full_val:>12,.0f} {point_val:>12,.0f} {pct_diff:>9.1f}%")
      else:
        print(f"{channel:<15} {metric:<20} {full_val:>12.3f} {point_val:>12.3f} {pct_diff:>9.1f}%")
  
  # Summary statistics by metric
  print(f"\n{'SUMMARY BY METRIC':<20}:")
  print("-" * 60)
  print(f"{'Metric':<20} {'Avg_Abs_Diff':>15} {'Max_Abs_Diff':>15}")
  print("-" * 60)
  
  for metric in comparison_df['metric'].unique():
    metric_data = comparison_df[comparison_df['metric'] == metric]
    avg_abs_diff = metric_data['percent_difference'].abs().mean()
    max_abs_diff = metric_data['percent_difference'].abs().max()
    print(f"{metric:<20} {avg_abs_diff:>14.1f}% {max_abs_diff:>14.1f}%")
  
  # Overall summary
  print(f"\n{'OVERALL SUMMARY':<20}:")
  print("-" * 40)
  avg_abs_diff = comparison_df['percent_difference'].abs().mean()
  max_abs_diff = comparison_df['percent_difference'].abs().max()
  print(f"{'Average Abs Diff':<20}: {avg_abs_diff:7.1f}%")
  print(f"{'Max Abs Diff':<20}: {max_abs_diff:7.1f}%")
else:
  print("No comparable data found")
