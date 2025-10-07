import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf # type: ignore

from meridian.model import model
from meridian.model import spec
from meridian.analysis import optimizer
from meridian.analysis import analyzer

from meridian.planner.flex_budget_planner import FlexibleBudgetPlanner, CompareOptimizedVsNonOptimized
from meridian.analysis.optimizer import OptimizationResults
import logging

tmp_dir = '/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/BudgetOptimizer/trash'

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')


# ---------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- Define Configuration ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------- #
home_dir = '/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/BudgetOptimizer'

opt_period = {
    'Mazda': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Live_Nation_MasterAdvertiser': {'start_date': '2024-07-06','end_date': '2025-06-28'},
    'Huntington_National_Bank': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Hyundai': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Burger_King': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'IBM_-_US': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Meijer': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Audi': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'MRG_Chevy_LMA': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Chumba_Casino': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Allergan': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Popeyes': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Samsung_US_Starcom': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Intuit_-_Quickbooks': {'start_date': '2024-07-06', 'end_date': '2025-06-28'}
}

opt_period = {k:opt_period[k] for k in sorted(opt_period)}
advertisers = list(opt_period.keys())


# Configuration for input excel file
input_config = {

  # time and geo inputs
  'time_col': 'week',
  'geo_col': 'geo',
  'population_col': 'population',

  # kpi inputs
  'kpi_col': 'conversions',  #
  'kpi_type': 'non_revenue',
  'revenue_per_kpi_col': 'revenue_per_conversion',  # needed if kpi_type is non_revenue

  # impression based media inputs
  'media_cols': ['Display_impression', 'TV_impression', 'Video_impression'],
  'media_spend_cols': ['Display_spend', 'TV_spend', 'Video_spend'],
  'media_channels': ['Display', 'TV', 'Video']

}

# optimization config
optimization_config = {
  'fixed_budget': True,
  'use_kpi': True,

  # spend constraints
  'spend_constraint_lower': {  # (1 - value)% of historical
    'Display': 0.3,
    'TV': 0.3,
    'Video': 0.3,
  },

  'spend_constraint_upper': {  # (1 + value)% of historical
  'Display': 0.3,
  'TV': 0.3,
  'Video': 0.3,
  }
}

# ---------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- Run Optimization ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------- #
client = 'Huntington_National_Bank'
model_type = 'Oct_1_WSInf'

input_file_path = f'{home_dir}/optimizer_inputs/{model_type}/{client}_optimizer_input_{model_type}.xlsx'
if not os.path.exists(input_file_path):
  raise FileNotFoundError(f'File not found: {input_file_path}')

# optimizer config
opt_config = optimization_config.copy()
opt_config['start_date'] = opt_period[client]['start_date']
opt_config['end_date'] = opt_period[client]['end_date']
# opt_config['gtol'] = 0.00001  # Use default 0.0001 for faster execution

# Add debugging to understand why optimization stops early
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# call the planner
planner = FlexibleBudgetPlanner(file_name=input_file_path, model_config=input_config)
opt_results = planner.optimize(opt_config)

optimized_data = opt_results.optimized_data.sel(metric='mean')
nonoptimized_data = opt_results.nonoptimized_data.sel(metric='mean')
opt_attrs = optimized_data.attrs
nonopt_attrs = nonoptimized_data.attrs

# Debug: Check actual spend values
print("\n=== Budget Discrepancy Analysis ===")
print(f"Optimized budget: {opt_attrs['budget']:,.2f}")
print(f"Non-optimized budget: {nonopt_attrs['budget']:,.2f}")
print(f"Difference: {opt_attrs['budget'] - nonopt_attrs['budget']:,.2f}")
print(f"% Difference: {(opt_attrs['budget'] / nonopt_attrs['budget'] - 1) * 100:.2f}%")

# Check individual channel spends
print("\n=== Channel-Level Spend ===")
opt_df = optimized_data.to_dataframe().reset_index()
nonopt_df = nonoptimized_data.to_dataframe().reset_index()
for i, channel in enumerate(opt_df['channel']):
    opt_spend = opt_df.loc[i, 'spend']
    nonopt_spend = nonopt_df.loc[i, 'spend']
    pct_change = (opt_spend / nonopt_spend - 1) * 100 if nonopt_spend > 0 else 0

    # Calculate expected bounds (±30%)
    lower_bound = nonopt_spend * 0.7
    upper_bound = nonopt_spend * 1.3
    within_bounds = "✓" if lower_bound <= opt_spend <= upper_bound else "✗ VIOLATION"

    print(f"{channel}: Opt={opt_spend:,.2f}, NonOpt={nonopt_spend:,.2f}, Change={pct_change:+.2f}%, Bounds=[{lower_bound:,.0f}, {upper_bound:,.0f}] {within_bounds}")

# Check historical spend from optimization grid
print("\n=== Historical Spend from Grid ===")
hist_spend = opt_results._optimization_grid.historical_spend
print(f"Historical spend: {hist_spend}")
print(f"Sum of historical: {np.sum(hist_spend):,.2f}")
print(f"Round factor: {opt_results._optimization_grid.round_factor}")
print(f"Grid shape: {opt_results._optimization_grid.spend_grid.shape}")

# Check spend bounds
print("\n=== Spend Bounds ===")
print(f"Spend bounds from optimizer: {opt_results.spend_bounds}")

# Inspect the optimization grid to understand why we stopped
print("\n=== Grid Analysis ===")
grid = opt_results._optimization_grid
print(f"Grid dimensions: {grid.spend_grid.shape}")
print(f"\nFirst 5 rows of spend grid:")
print(grid.spend_grid[:5, :].values)
print(f"\nLast 5 rows of spend grid:")
print(grid.spend_grid[-5:, :].values)

# Calculate what the bounds should allow
print("\n=== Expected Spend Range ===")
total_min = 0
total_max = 0
for i, channel in enumerate(opt_df['channel']):
    nonopt_spend = nonopt_df.loc[i, 'spend']
    lower = nonopt_spend * 0.7
    upper = nonopt_spend * 1.3
    total_min += lower
    total_max += upper
    print(f"{channel}: Min={lower:,.0f}, Max={upper:,.0f}, Range={upper-lower:,.0f}")
print(f"\nTotal feasible range: [{total_min:,.0f}, {total_max:,.0f}]")
print(f"Target budget: {nonopt_attrs['budget']:,.0f}")
print(f"Current optimized: {opt_attrs['budget']:,.0f}")
print(f"Gap to target: {nonopt_attrs['budget'] - opt_attrs['budget']:,.0f}")

# Check if target is reachable - what's the max we can achieve?
print("\n=== Maximum Achievable Budget (greedy) ===")
# Start with minimum on all channels, then max out highest ROI
print("If we max out Display and TV (at upper bounds) and minimize Video:")
max_scenario = (7_616_700 + 3_589_170 + 416_920)
print(f"Max Display + Max TV + Min Video = {max_scenario:,.0f}")
print("If we minimize Display and max out TV and Video:")
min_display_scenario = (4_101_300 + 3_589_170 + 774_280)
print(f"Min Display + Max TV + Max Video = {min_display_scenario:,.0f}")
print(f"\nTarget budget {nonopt_attrs['budget']:,.0f} is {'achievable' if min_display_scenario >= nonopt_attrs['budget'] else 'NOT achievable'}!")

print(f"\n{opt_attrs}")
print(f"{nonopt_attrs}")
