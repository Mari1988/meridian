import logging
import os
import sys

import numpy as np
import tensorflow as tf # type: ignore

from meridian.model import model
from meridian.model import spec
from meridian.analysis import optimizer
from meridian.analysis import analyzer

from meridian.planner import flex_budget_planner
from meridian.planner.flex_budget_planner import FlexibleBudgetPlanner

tmp_dir = '/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/BudgetOptimizer/trash'

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')


# ---------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- Define Configuration ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------- #
# Excel file path (update this to match the actual location)
excel_file_path = (
  '/Users/mariappan.subramanian/Library/CloudStorage/'
  'OneDrive-TheTradeDesk/MMM/BudgetOptimizer/optimizer_input_case_coeff_rf_only.xlsx'
)

# sample files
# 1. optimizer_input_case_coeff.xlsx --> geo + cf + media + rf inputs
# 2. optimizer_input_case_coeff_media_only.xlsx --> geo + cf + media inputs
# 3. optimizer_input_case_coeff_rf_only.xlsx --> geo + cf + rf inputs


# Check if file exists
if not os.path.exists(excel_file_path):
  print(f"ERROR: Excel file not found at {excel_file_path}")
  print("Please update the excel_file_path variable to point to the correct location.")
  sys.exit(1)

# Model configuration based on actual Excel file structure
model_config = {

  # time and geo inputs
  'time_col': 'week',
  'geo_col': 'geo',  # assumed to be national model if not given
  'population_col': 'population', # mandatory if geo_col is given

  # kpi inputs
  'kpi_col': 'conversions',  #
  'kpi_type': 'non_revenue',
  'revenue_per_kpi_col': 'revenue_per_conversion',  # needed if kpi_type is non_revenue

  # impression based media inputs
  # 'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
  # 'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
  # 'media_channels': ['Channel0', 'Channel1', 'Channel2'],

  # reach based media inputs
  'reach_cols': ['Channel3_reach'],
  'frequency_cols': ['Channel3_frequency'],
  'rf_spend_cols': ['Channel3_spend'],
  'rf_channels': ['Channel3'],

  }

optimization_config = {
  'use_optimal_frequency': False
}

# ---------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- Run Optimization ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------- #
planner = FlexibleBudgetPlanner(file_name=excel_file_path, model_config=model_config)
opt_results = planner.optimize(optimization_config)
optimized_data = opt_results.optimized_data.sel(metric='mean')
print(f"{optimized_data.spend.values}")
print(f"{optimized_data.incremental_outcome.values}")
