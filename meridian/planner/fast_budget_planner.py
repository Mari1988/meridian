import logging
import os
import sys

from meridian.model import model
from meridian.model import spec

from meridian.planner import adhoc_data_loader
from meridian.planner.adhoc_data_loader import AdhocDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Excel file path (update this to match the actual location)
excel_file_path = (
  '/Users/mariappan.subramanian/Library/CloudStorage/'
  'OneDrive-TheTradeDesk/MMM/BudgetOptimizer/mmm_input_artifacts.xlsx'
)

# Check if file exists
if not os.path.exists(excel_file_path):
  print(f"ERROR: Excel file not found at {excel_file_path}")
  print("Please update the excel_file_path variable to point to the correct location.")
  sys.exit(1)

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

# create data loader
adhoc_data_loader = AdhocDataLoader(file_name=excel_file_path, model_config=model_config)
data = adhoc_data_loader.build_input_data()

# create inference data
inference_data = adhoc_data_loader.get_inference_data()

# create model spec
model_spec = spec.ModelSpec()

# create model
mmm = model.Meridian(input_data=data, model_spec=model_spec, inference_data=inference_data)

print("="*60)
print("SUCCESS: Meridian Model Created Successfully!")
print("="*60)
print(f"Model has inference data: {mmm.inference_data is not None}")
print(f"Number of data variables: {len(mmm.inference_data.posterior.data_vars)}")

# Sample prior to make model ready for optimization
print("\nSampling prior distributions...")
mmm.sample_prior(n_draws=100, seed=42)
print("✓ Prior sampling completed")

# create budget optimizer
print("\nInitializing budget optimizer...")
from meridian.analysis import optimizer
budget_optimizer = optimizer.BudgetOptimizer(mmm)

print("Starting budget optimization...")
optimization_results = budget_optimizer.optimize()
print("✓ Budget optimization completed!")

print(f"\nOptimization results type: {type(optimization_results)}")
print(f"Optimization results attributes: {[attr for attr in dir(optimization_results) if not attr.startswith('_')]}")

# Try to display results if it's a dataframe-like object
if hasattr(optimization_results, 'head'):
  print("\nFirst few optimization results:")
  print(optimization_results.head())
elif hasattr(optimization_results, 'data'):
  print("\nOptimization results data:")
  print(optimization_results.data)
else:
  print(f"\nOptimization results: {optimization_results}")
  
print("\n" + "="*60)
print("SUCCESS: Budget optimization completed with Excel parameters!")
print("="*60)
