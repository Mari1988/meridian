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

"""Example usage of AdhocDataLoader with the provided Excel file.

This script demonstrates how to use the AdhocDataLoader to load MMM data from
an Excel file and create a Meridian InputData object ready for model fitting.
"""

import logging
import os
import sys

from meridian.planner import adhoc_data_loader


def main():
  """Main function demonstrating AdhocDataLoader usage."""

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
    'rf_channels': ['Channel3'],

    # control inputs
    'control_cols': ['sentiment_score_control', 'competitor_activity_score_control']
  }

  try:
    print("="*60)
    print("AdhocDataLoader Example Usage")
    print("="*60)

    # Step 1: Initialize the AdhocDataLoader
    print(f"\n1. Initializing AdhocDataLoader with file: {excel_file_path}")
    loader = adhoc_data_loader.AdhocDataLoader(excel_file_path, model_config)
    print("   ✓ AdhocDataLoader initialized successfully")

    # Step 2: Load Excel data
    print("\n2. Loading data from Excel sheets...")
    loader.load_excel_data()
    print("   ✓ Excel data loaded successfully")

    # Display basic information about loaded data
    if loader.data_df is not None:
      print(f"   - Data sheet shape: {loader.data_df.shape}")
      print(f"   - Data columns: {list(loader.data_df.columns)}")
      print(f"   - Date range: {loader.data_df['week'].min()} to {loader.data_df['week'].max()}")
      print(f"   - Unique geos: {loader.data_df['geo'].unique()}")

    if loader.coefficients_df is not None:
      print(f"   - Coefficients sheet shape: {loader.coefficients_df.shape}")
    else:
      print("   - No coefficients sheet found")

    if loader.parameters_df is not None:
      print(f"   - Parameters sheet shape: {loader.parameters_df.shape}")
    else:
      print("   - No parameters sheet found")

    # Step 3: Validate data columns
    print("\n3. Validating data columns...")
    loader.validate_data_columns()
    print("   ✓ All required columns validated successfully")

    # Step 4: Build InputData object
    print("\n4. Building Meridian InputData object...")
    input_data = loader.build_input_data()
    print("   ✓ InputData object created successfully")

    # Display InputData structure
    print("\n5. InputData Structure Summary:")
    print(f"   - KPI shape: {input_data.kpi.shape}")
    print(f"   - KPI dims: {list(input_data.kpi.dims)}")
    print(f"   - Population shape: {input_data.population.shape}")

    if input_data.media is not None:
      print(f"   - Media shape: {input_data.media.shape}")
      print(f"   - Media channels: {list(input_data.media.coords['media_channel'].values)}")

    if input_data.reach is not None:
      print(f"   - Reach shape: {input_data.reach.shape}")
      print(f"   - R&F channels: {list(input_data.reach.coords['rf_channel'].values)}")

    if input_data.controls is not None:
      print(f"   - Controls shape: {input_data.controls.shape}")
      print(f"   - Control variables: {list(input_data.controls.coords['control_variable'].values)}")

    # Step 5: Access additional data if available
    print("\n6. Additional Data Access:")
    coefficients = loader.get_coefficients_data()
    parameters = loader.get_parameters_data()

    if coefficients is not None:
      print("   - Coefficients data available")
      print(f"     Shape: {coefficients.shape}")
      print(f"     Columns: {list(coefficients.columns)}")
    else:
      print("   - No coefficients data available")

    if parameters is not None:
      print("   - Parameters data available")
      print(f"     Shape: {parameters.shape}")
      print(f"     Columns: {list(parameters.columns)}")
    else:
      print("   - No parameters data available")
    
    # Step 6: Process parameters using MediaParameterLoader
    print("\n7. Media Parameter Processing:")
    processed_params = loader.get_processed_parameters()
    parameter_summary = loader.get_parameter_summary()
    
    if processed_params is not None:
      print("   - Processed parameter structure (lists):")
      for key, values in processed_params.items():
        print(f"     {key}: {values}")
    else:
      print("   - No processed parameters available")
      
    if parameter_summary is not None:
      print("   - Parameter summary by channel:")
      print("     " + parameter_summary.to_string(index=False).replace('\n', '\n     '))
    else:
      print("   - No parameter summary available")
    
    # Step 7: Process parameters as xarray.DataArrays
    print("\n8. Media Parameter DataArrays:")
    parameter_arrays = loader.get_processed_parameter_arrays()
    
    if parameter_arrays is not None:
      print("   - Parameter DataArrays structure:")
      for key, data_array in parameter_arrays.items():
        print(f"     {key}:")
        print(f"       Shape: {data_array.shape}")
        print(f"       Dims: {list(data_array.dims)}")
        print(f"       Coords: {dict(data_array.coords)}")
        print(f"       Values: {data_array.values}")
        print()
    else:
      print("   - No parameter DataArrays available")
    
    # Step 8: Process coefficients as xarray.DataArrays
    print("\n9. Media Coefficients DataArrays:")
    coefficients_arrays = loader.get_processed_coefficients_arrays()
    
    if coefficients_arrays is not None:
      print("   - Coefficients DataArrays structure:")
      for key, data_array in coefficients_arrays.items():
        print(f"     {key}:")
        print(f"       Shape: {data_array.shape}")
        print(f"       Dims: {list(data_array.dims)}")
        print(f"       Coords: {dict(data_array.coords)}")
        print(f"       Values shape: {data_array.values.shape}")
        print(f"       Sample values (first 3 geos, first channel):")
        if data_array.ndim == 2:
          print(f"         {data_array.values[:3, 0] if data_array.shape[1] > 0 else 'No channels'}")
        print()
    else:
      print("   - No coefficients DataArrays available")
    
    # Step 9: Create ArviZ InferenceData from parameters and coefficients
    print("\n10. ArviZ InferenceData Creation:")
    inference_data = loader.get_inference_data()
    
    if inference_data is not None:
      print("   ✓ ArviZ InferenceData created successfully")
      print(f"   - Object type: {type(inference_data)}")
      print(f"   - Available groups: {list(inference_data.groups())}")
      
      # Show posterior structure
      posterior = inference_data.posterior
      print(f"   - Posterior dimensions: {dict(posterior.dims)}")
      print(f"   - Posterior coordinates: {list(posterior.coords.keys())}")
      print(f"   - Data variables: {list(posterior.data_vars.keys())}")
      
      # Show sample data access
      print("   - Sample data access:")
      for var_name in list(posterior.data_vars.keys())[:3]:  # Show first 3 variables
        var_data = posterior[var_name]
        print(f"     {var_name}: shape={var_data.shape}, dims={var_data.dims}")
        print(f"       Values: {var_data.values.flatten()[:3]}...")  # First 3 values
      
      # Show sample_stats group
      if 'sample_stats' in inference_data.groups():
        sample_stats = inference_data.sample_stats
        print(f"   - Sample stats variables: {list(sample_stats.data_vars.keys())}")
        
      print("\n   Usage with ArviZ:")
      print("     import arviz as az")
      print("     az.summary(inference_data)  # Statistical summary")
      print("     az.plot_trace(inference_data)  # Trace plots") 
      print("     az.plot_posterior(inference_data)  # Posterior distributions")
      
    else:
      print("   - ArviZ InferenceData not available")
      print("     (Requires both Parameters and Coefficients sheets)")

    print("\n" + "="*60)
    print("SUCCESS: InputData object ready for Meridian model fitting!")
    print("="*60)

    # Optional: Show sample usage with Meridian model
    print("\nNext Steps:")
    print("1. Use the input_data object to initialize a Meridian model:")
    print("   from meridian.model import model")
    print("   mmm = model.Meridian(input_data=input_data)")
    print("")
    print("2. Configure model specifications and fit:")
    print("   mmm.fit(chains=4, draws=2000)")
    print("")
    print("3. Use coefficients and parameters data for custom priors if needed")

    return input_data

  except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check that the Excel file exists at the specified path")
    print("2. Ensure the Excel file has the expected sheets: 'Data', 'Coefficients', 'Parameters'")
    print("3. Verify that all required columns exist in the Data sheet")
    print("4. Check column names match the model_config exactly")
    sys.exit(1)


def inspect_excel_file_structure(excel_file_path: str):
  """Helper function to inspect Excel file structure.

  Args:
    excel_file_path: Path to the Excel file to inspect.
  """
  try:
    import pandas as pd

    print("\n" + "="*60)
    print("Excel File Structure Inspection")
    print("="*60)

    # Get sheet names
    xl_file = pd.ExcelFile(excel_file_path)
    print(f"Available sheets: {xl_file.sheet_names}")

    # Inspect each sheet
    for sheet_name in xl_file.sheet_names:
      print(f"\n--- Sheet: {sheet_name} ---")
      df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
      print(f"Shape: {df.shape}")
      print(f"Columns: {list(df.columns)}")
      if 'week' in df.columns or 'date' in df.columns:
        date_col = 'week' if 'week' in df.columns else 'date'
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
      if 'geo' in df.columns:
        print(f"Unique geos: {df['geo'].unique()}")
      print(f"Sample data:\n{df.head(3)}")

  except Exception as e:
    print(f"Error inspecting Excel file: {str(e)}")


if __name__ == '__main__':
  # Optionally inspect file structure first
  excel_path = (
    '/Users/mariappan.subramanian/Library/CloudStorage/'
    'OneDrive-TheTradeDesk/MMM/BudgetOptimizer/mmm_input_artifacts.xlsx'
  )

  if len(sys.argv) > 1 and sys.argv[1] == '--inspect':
    inspect_excel_file_structure(excel_path)
  else:
    main()
