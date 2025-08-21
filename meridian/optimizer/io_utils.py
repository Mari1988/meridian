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

"""Input/output utilities for independent optimizer."""

import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from meridian.optimizer.independent_optimizer import OptimizationInput, ModelParameters

try:
  import yaml
  HAS_YAML = True
except ImportError:
  HAS_YAML = False


__all__ = [
    "load_optimization_input_from_csv",
    "load_optimization_input_from_excel", 
    "load_model_parameters_from_json",
    "load_model_parameters_from_yaml",
    "save_model_parameters_to_json",
    "save_optimization_results_to_csv",
]


def load_optimization_input_from_csv(
    impression_file: Optional[str] = None,
    rf_reach_file: Optional[str] = None,
    rf_frequency_file: Optional[str] = None,
    rf_spend_file: Optional[str] = None,
    spend_file: Optional[str] = None,
    population_file: Optional[str] = None,
    dates_column: str = 'date',
    geo_column: str = 'geo',
) -> OptimizationInput:
  """Load optimization input from CSV files with mixed impression and R&F channels.
  
  Args:
    impression_file: Path to CSV with impression data (columns: geo, date, channel1, channel2, ...).
    rf_reach_file: Path to CSV with RF reach data (columns: geo, date, rf_channel1, rf_channel2, ...).
    rf_frequency_file: Path to CSV with RF frequency data (columns: geo, date, rf_channel1, rf_channel2, ...).
    rf_spend_file: Path to CSV with RF spend data (columns: geo, date, rf_channel1, rf_channel2, ...).
    spend_file: Path to CSV with historical spend data (columns: channel, spend, channel_type).
    population_file: Path to CSV with population data (columns: geo, population).
    dates_column: Name of date column in data files.
    geo_column: Name of geo column in data files.
    
  Returns:
    OptimizationInput object with loaded data.
  """
  # Load spend data to determine channel structure
  if spend_file is None:
    raise ValueError("spend_file is required to determine channel structure")
  
  spend_df = pd.read_csv(spend_file)
  
  # Separate impression and R&F channels
  if 'channel_type' in spend_df.columns:
    impression_spend_df = spend_df[spend_df['channel_type'] == 'impression']
    rf_spend_df = spend_df[spend_df['channel_type'] == 'rf']
  else:
    # Default: assume all channels are impression type
    impression_spend_df = spend_df.copy()
    rf_spend_df = pd.DataFrame(columns=['channel', 'spend'])
  
  impression_channels = impression_spend_df['channel'].tolist()
  rf_channels = rf_spend_df['channel'].tolist()
  impression_historical_spend = impression_spend_df['spend'].values
  rf_historical_spend = rf_spend_df['spend'].values
  
  # Load population data (required for scaling)
  population = None
  geos = []
  if population_file:
    pop_df = pd.read_csv(population_file)
    geos = pop_df[geo_column].tolist()
    population = pop_df['population'].values
  
  # Load impression data
  impression_data = None
  dates = []
  if impression_file and len(impression_channels) > 0:
    impression_df = pd.read_csv(impression_file)
    if not geos:  # Extract geos from impression data if not from population
      geos = sorted(impression_df[geo_column].unique().tolist())
    dates = sorted(impression_df[dates_column].unique().tolist())
    
    # Create pivot table for impression data
    impression_pivot = impression_df.pivot_table(
        index=[geo_column, dates_column],
        values=impression_channels,
        fill_value=0
    )
    
    # Reshape to (n_geos, n_times, n_impression_channels)
    n_geos, n_times = len(geos), len(dates)
    impression_data = impression_pivot.values.reshape(n_geos, n_times, len(impression_channels))
  
  # Load R&F data
  rf_reach = None
  rf_frequency = None
  rf_spend_data = None
  
  if rf_reach_file and len(rf_channels) > 0:
    rf_reach_df = pd.read_csv(rf_reach_file)
    if not dates:  # Extract dates from R&F data if not from impression
      dates = sorted(rf_reach_df[dates_column].unique().tolist())
    if not geos:  # Extract geos from R&F data if not available
      geos = sorted(rf_reach_df[geo_column].unique().tolist())
      
    rf_reach_pivot = rf_reach_df.pivot_table(
        index=[geo_column, dates_column],
        values=rf_channels,
        fill_value=0
    )
    n_geos, n_times = len(geos), len(dates)
    rf_reach = rf_reach_pivot.values.reshape(n_geos, n_times, len(rf_channels))
  
  if rf_frequency_file and len(rf_channels) > 0:
    rf_freq_df = pd.read_csv(rf_frequency_file)
    rf_freq_pivot = rf_freq_df.pivot_table(
        index=[geo_column, dates_column],
        values=rf_channels,
        fill_value=0
    )
    rf_frequency = rf_freq_pivot.values.reshape(n_geos, n_times, len(rf_channels))
  
  if rf_spend_file and len(rf_channels) > 0:
    rf_spend_df = pd.read_csv(rf_spend_file)
    rf_spend_pivot = rf_spend_df.pivot_table(
        index=[geo_column, dates_column],
        values=rf_channels,
        fill_value=0
    )
    rf_spend_data = rf_spend_pivot.values.reshape(n_geos, n_times, len(rf_channels))
  
  return OptimizationInput(
      impression_channels=impression_channels,
      rf_channels=rf_channels,
      impression_historical_spend=impression_historical_spend,
      rf_historical_spend=rf_historical_spend,
      impression_data=impression_data,
      rf_reach=rf_reach,
      rf_frequency=rf_frequency,
      rf_spend=rf_spend_data,
      dates=dates,
      geos=geos,
      population=population,
  )


def load_optimization_input_from_excel(
    excel_file: str,
    media_sheet: str = 'media',
    spend_sheet: str = 'spend',
    rf_reach_sheet: Optional[str] = None,
    rf_frequency_sheet: Optional[str] = None,
    **kwargs
) -> OptimizationInput:
  """Load optimization input from Excel file with multiple sheets.
  
  Args:
    excel_file: Path to Excel file.
    media_sheet: Name of sheet containing media data.
    spend_sheet: Name of sheet containing spend data.
    rf_reach_sheet: Optional name of sheet containing RF reach data.
    rf_frequency_sheet: Optional name of sheet containing RF frequency data.
    **kwargs: Additional arguments passed to load_optimization_input_from_csv.
    
  Returns:
    OptimizationInput object with loaded data.
  """
  # Read sheets and save as temporary CSV files
  media_df = pd.read_excel(excel_file, sheet_name=media_sheet)
  spend_df = pd.read_excel(excel_file, sheet_name=spend_sheet)
  
  # Save to temporary files
  import tempfile
  import os
  
  with tempfile.TemporaryDirectory() as temp_dir:
    media_file = os.path.join(temp_dir, 'media.csv')
    spend_file = os.path.join(temp_dir, 'spend.csv')
    
    media_df.to_csv(media_file, index=False)
    spend_df.to_csv(spend_file, index=False)
    
    rf_reach_file = None
    rf_frequency_file = None
    
    if rf_reach_sheet:
      rf_reach_df = pd.read_excel(excel_file, sheet_name=rf_reach_sheet)
      rf_reach_file = os.path.join(temp_dir, 'rf_reach.csv')
      rf_reach_df.to_csv(rf_reach_file, index=False)
    
    if rf_frequency_sheet:
      rf_freq_df = pd.read_excel(excel_file, sheet_name=rf_frequency_sheet)
      rf_frequency_file = os.path.join(temp_dir, 'rf_frequency.csv')
      rf_freq_df.to_csv(rf_frequency_file, index=False)
    
    return load_optimization_input_from_csv(
        impression_file=media_file,
        spend_file=spend_file,
        rf_reach_file=rf_reach_file,
        rf_frequency_file=rf_frequency_file,
        **kwargs
    )


def load_model_parameters_from_json(file_path: str) -> ModelParameters:
  """Load model parameters from JSON file with mixed channel types.
  
  Expected JSON format:
  {
    "impression_coefficients": [[coeff_geo1_ch1, coeff_geo1_ch2], [coeff_geo2_ch1, coeff_geo2_ch2]], // optional
    "impression_adstock_params": [alpha_ch1, alpha_ch2], // optional
    "impression_ec50_params": [ec50_ch1, ec50_ch2], // optional
    "impression_slope_params": [slope_ch1, slope_ch2], // optional
    "rf_coefficients": [[coeff_geo1_rf1, coeff_geo1_rf2], [coeff_geo2_rf1, coeff_geo2_rf2]], // optional
    "rf_adstock_params": [alpha_rf1, alpha_rf2], // optional
    "rf_ec50_params": [ec50_rf1, ec50_rf2], // optional
    "rf_slope_params": [slope_rf1, slope_rf2], // optional
    "baseline": [baseline_geo1, baseline_geo2]
  }
  
  Args:
    file_path: Path to JSON file.
    
  Returns:
    ModelParameters object.
  """
  with open(file_path, 'r') as f:
    data = json.load(f)
  
  return ModelParameters(
      impression_coefficients=np.array(data['impression_coefficients']) if 'impression_coefficients' in data else None,
      impression_adstock_params=np.array(data['impression_adstock_params']) if 'impression_adstock_params' in data else None,
      impression_ec50_params=np.array(data['impression_ec50_params']) if 'impression_ec50_params' in data else None,
      impression_slope_params=np.array(data['impression_slope_params']) if 'impression_slope_params' in data else None,
      rf_coefficients=np.array(data['rf_coefficients']) if 'rf_coefficients' in data else None,
      rf_adstock_params=np.array(data['rf_adstock_params']) if 'rf_adstock_params' in data else None,
      rf_ec50_params=np.array(data['rf_ec50_params']) if 'rf_ec50_params' in data else None,
      rf_slope_params=np.array(data['rf_slope_params']) if 'rf_slope_params' in data else None,
      baseline=np.array(data['baseline']),
      control_coefficients=np.array(data['control_coefficients']) if 'control_coefficients' in data else None,
  )


def load_model_parameters_from_yaml(file_path: str) -> ModelParameters:
  """Load model parameters from YAML file.
  
  Args:
    file_path: Path to YAML file.
    
  Returns:
    ModelParameters object.
  """
  if not HAS_YAML:
    raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
  
  with open(file_path, 'r') as f:
    data = yaml.safe_load(f)
  
  return ModelParameters(
      media_coefficients=np.array(data['media_coefficients']),
      adstock_params=np.array(data['adstock_params']),
      ec50_params=np.array(data['ec50_params']),
      slope_params=np.array(data['slope_params']),
      baseline=np.array(data['baseline']),
      optimal_frequency=np.array(data['optimal_frequency']) if 'optimal_frequency' in data else None,
  )


def save_model_parameters_to_json(params: ModelParameters, file_path: str):
  """Save model parameters to JSON file.
  
  Args:
    params: ModelParameters object to save.
    file_path: Output file path.
  """
  data = {
      'media_coefficients': params.media_coefficients.tolist(),
      'adstock_params': params.adstock_params.tolist(),
      'ec50_params': params.ec50_params.tolist(),
      'slope_params': params.slope_params.tolist(),
      'baseline': params.baseline.tolist(),
  }
  
  if params.optimal_frequency is not None:
    data['optimal_frequency'] = params.optimal_frequency.tolist()
  
  with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)


def save_optimization_results_to_csv(
    results, 
    file_path: str,
    include_details: bool = True
):
  """Save optimization results to CSV file.
  
  Args:
    results: IndependentOptimizationResults object.
    file_path: Output CSV file path.
    include_details: Whether to include detailed metrics.
  """
  df = results.to_dataframe()
  
  if include_details:
    # Add scenario information as metadata
    scenario_info = {
        'scenario_type': results.scenario.scenario_type,
        'total_budget': results.scenario.total_budget,
        'total_roi': results.total_roi,
        'spend_constraint_lower': results.scenario.spend_constraint_lower,
        'spend_constraint_upper': results.scenario.spend_constraint_upper,
    }
    
    # Add metadata as comment rows at the top
    with open(file_path, 'w') as f:
      f.write("# Optimization Results\n")
      for key, value in scenario_info.items():
        f.write(f"# {key}: {value}\n")
      f.write("\n")
      df.to_csv(f, index=False)
  else:
    df.to_csv(file_path, index=False)


def extract_model_parameters_from_meridian(meridian_model) -> ModelParameters:
  """Extract model parameters from fitted Meridian model object.
  
  This utility function helps bridge between existing Meridian models
  and the independent optimizer.
  
  Args:
    meridian_model: Fitted Meridian model object.
    
  Returns:
    ModelParameters object with extracted parameters.
  """
  # Extract posterior means from inference data
  posterior = meridian_model.inference_data.posterior
  
  # Get media coefficients (assuming they exist in posterior)
  media_coeffs = posterior.media_coef.mean(dim=['chain', 'draw']).values
  
  # Get adstock parameters
  adstock_params = posterior.alpha.mean(dim=['chain', 'draw']).values
  
  # Get Hill parameters
  ec50_params = posterior.ec.mean(dim=['chain', 'draw']).values
  slope_params = posterior.slope.mean(dim=['chain', 'draw']).values
  
  # Get baseline
  baseline = posterior.baseline.mean(dim=['chain', 'draw']).values
  
  # Get optimal frequency if available
  optimal_freq = None
  if hasattr(posterior, 'optimal_frequency'):
    optimal_freq = posterior.optimal_frequency.mean(dim=['chain', 'draw']).values
  
  return ModelParameters(
      media_coefficients=media_coeffs,
      adstock_params=adstock_params,
      ec50_params=ec50_params,
      slope_params=slope_params,
      baseline=baseline,
      optimal_frequency=optimal_freq,
  )