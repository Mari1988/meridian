import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration even if already configured
)

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import tensorflow as tf # type: ignore

import matplotlib.pyplot as plt
import seaborn as sns

from meridian.model import model
from meridian.model import spec
from meridian.analysis import optimizer
from meridian.analysis import analyzer

from meridian.planner.flex_budget_planner import FlexibleBudgetPlanner, CompareOptimizedVsNonOptimized
from meridian.planner.budget_visualizer import BudgetVisualizer

# optimizer input path
optimizer_input_path = "./../inputs/sample_optimizer_input_coeff.xlsx"
if not os.path.exists(optimizer_input_path):
    raise FileNotFoundError(f'File not found: {optimizer_input_path}')

# configuration
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

optimization_config = {
  'fixed_budget': True,
  'use_kpi': True,
  'gtol': 0.001,

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
  },

  # optimization period
  'start_date': '2024-07-06',
  'end_date': '2025-06-28'
}

# run the optimizer
sample_optimizer_output_pkl = "./../inputs/sample_optimizer_input_coeff.pkl"
# run the optimizer
if os.path.exists(sample_optimizer_output_pkl):
    opt_results = model.load_mmm(sample_optimizer_output_pkl)
else:
    planner = FlexibleBudgetPlanner(file_name=optimizer_input_path, model_config=input_config)
    opt_results = planner.optimize(optimization_config)
    model.save_mmm(opt_results, sample_optimizer_output_pkl)

# budget visualizer
budget_visualizer = BudgetVisualizer(opt_results)
mroi_df = budget_visualizer.calculate_mroi_by_geo()

# raw impressions
meridian = budget_visualizer.meridian
impressions = meridian.input_data.media