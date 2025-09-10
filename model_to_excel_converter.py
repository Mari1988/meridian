"""Convert Meridian model file to Excel format compatible with FlexibleBudgetPlanner."""

import pandas as pd
import xarray as xr
from meridian.model import model
from meridian import constants as c

def convert_model_to_excel(model_path: str, output_path: str):
  """Convert Meridian model pickle file to Excel format.

  Args:
    model_path: Path to the input pickle file
    output_path: Path for the output Excel file
  """
  # Load the model
  mmm = model.load_mmm(model_path)

  # Extract input data and posterior
  input_data = mmm.input_data
  posterior = mmm.inference_data.posterior

  # Create Data sheet using xarray.to_dataframe() approach
  # Base DataFrame from KPI (geo × time structure)
  data_df = input_data.kpi.to_dataframe().reset_index()
  data_df.rename(columns={'time': 'week', 'kpi': 'conversions'}, inplace=True)

  # Join population data
  pop_df = input_data.population.to_dataframe().reset_index()
  data_df = data_df.merge(pop_df, on='geo')

  # Add revenue per KPI if available
  if hasattr(input_data, 'revenue_per_kpi') and input_data.revenue_per_kpi is not None:
    rev_df = input_data.revenue_per_kpi.to_dataframe().reset_index()
    rev_df.rename(columns={'time': 'week', 'revenue_per_kpi': 'revenue_per_conversion'}, inplace=True)
    data_df = data_df.merge(rev_df, on=['geo', 'week'])

  # Add media impressions and spend
  if input_data.media is not None:
    # Media impressions
    media_df = input_data.media.to_dataframe().reset_index()
    media_pivot = media_df.pivot(index=['geo', 'media_time'], columns='media_channel', values='media')
    media_pivot.columns = [f'{col}_impression' for col in media_pivot.columns]
    media_pivot = media_pivot.reset_index()

    # Media spend
    spend_df = input_data.media_spend.to_dataframe().reset_index()
    spend_pivot = spend_df.pivot(index=['geo', 'time'], columns='media_channel', values='media_spend')
    spend_pivot.columns = [f'{col}_spend' for col in spend_pivot.columns]
    spend_pivot = spend_pivot.reset_index()

    # Merge both (media_time and time should align with week)
    data_df = data_df.merge(media_pivot, left_on=['geo', 'week'], right_on=['geo', 'media_time'], how='left')
    data_df = data_df.merge(spend_pivot, left_on=['geo', 'week'], right_on=['geo', 'time'], how='left')

    # Clean up duplicate time columns
    data_df = data_df.drop(columns=[col for col in data_df.columns if col.startswith('media_time') or (col == 'time' and col != 'week')])

  # Add RF reach, frequency and spend
  if input_data.reach is not None:
    # RF reach
    reach_df = input_data.reach.to_dataframe().reset_index()
    reach_pivot = reach_df.pivot(index=['geo', 'media_time'], columns='rf_channel', values='reach')
    reach_pivot.columns = [f'{col}_reach' for col in reach_pivot.columns]
    reach_pivot = reach_pivot.reset_index()

    # RF frequency
    freq_df = input_data.frequency.to_dataframe().reset_index()
    freq_pivot = freq_df.pivot(index=['geo', 'media_time'], columns='rf_channel', values='frequency')
    freq_pivot.columns = [f'{col}_frequency' for col in freq_pivot.columns]
    freq_pivot = freq_pivot.reset_index()

    # Merge RF data
    data_df = data_df.merge(reach_pivot, left_on=['geo', 'week'], right_on=['geo', 'media_time'], how='left')
    data_df = data_df.merge(freq_pivot, left_on=['geo', 'week'], right_on=['geo', 'media_time'], how='left')

    # RF spend (handle different dimensions)
    if input_data.rf_spend is not None:
      rf_spend_df = input_data.rf_spend.to_dataframe().reset_index()
      if len(input_data.rf_spend.shape) == 3:  # (geo, time, channel)
        rf_spend_pivot = rf_spend_df.pivot(index=['geo', 'time'], columns='rf_channel', values='rf_spend')
        rf_spend_pivot.columns = [f'{col}_spend' for col in rf_spend_pivot.columns]
        rf_spend_pivot = rf_spend_pivot.reset_index()
        data_df = data_df.merge(rf_spend_pivot, left_on=['geo', 'week'], right_on=['geo', 'time'], how='left')
      else:  # (channel,) - constant spend per channel - broadcast to all rows
        for rf_channel in input_data.rf_channel.values:
          rf_spend_val = input_data.rf_spend.sel(rf_channel=rf_channel).values
          data_df[f'{rf_channel}_spend'] = rf_spend_val

    # Clean up duplicate media_time columns
    data_df = data_df.drop(columns=[col for col in data_df.columns if col.startswith('media_time')])

  # Create Coefficients sheet using xarray methods
  coeff_dfs = []

  # Add media coefficients
  if c.BETA_GM in posterior:
    beta_gm_median = posterior[c.BETA_GM].median(dim=['chain', 'draw'])  # Keep geo and media_channel dims
    beta_gm_df = beta_gm_median.to_dataframe().unstack('media_channel')  # Pivot to wide format
    beta_gm_df.columns = beta_gm_df.columns.droplevel(0)  # Remove 'beta_gm' level
    beta_gm_df = beta_gm_df.reset_index()  # geo is now a column
    coeff_dfs.append(beta_gm_df)

  # Add RF coefficients
  if c.BETA_GRF in posterior:
    beta_grf_median = posterior[c.BETA_GRF].median(dim=['chain', 'draw'])
    beta_grf_df = beta_grf_median.to_dataframe().unstack('rf_channel')
    beta_grf_df.columns = beta_grf_df.columns.droplevel(0)
    beta_grf_df = beta_grf_df.reset_index()
    coeff_dfs.append(beta_grf_df)

  # Merge all coefficient DataFrames on geo
  if coeff_dfs:
    coeff_df = coeff_dfs[0]
    for df in coeff_dfs[1:]:
      coeff_df = coeff_df.merge(df, on='geo')
  else:
    coeff_df = pd.DataFrame()

  # Create Parameters sheet using xarray methods
  param_dfs = []

  # Media parameters
  if input_data.media is not None:
    # Combine all media parameters into single DataFrame
    media_params = {}
    if c.ALPHA_M in posterior:
      media_params['Adstock'] = posterior[c.ALPHA_M].median(dim=['chain', 'draw'])
    if c.EC_M in posterior:
      media_params['Inflexion'] = posterior[c.EC_M].median(dim=['chain', 'draw'])
    if c.SLOPE_M in posterior:
      media_params['Slope'] = posterior[c.SLOPE_M].median(dim=['chain', 'draw'])

    if media_params:
      # Combine into a single xarray Dataset
      media_dataset = xr.Dataset(media_params)
      media_param_df = media_dataset.to_dataframe().reset_index()
      media_param_df.rename(columns={'media_channel': 'MediaVariable'}, inplace=True)
      param_dfs.append(media_param_df)

  # RF parameters
  if input_data.reach is not None:
    rf_params = {}
    if c.ALPHA_RF in posterior:
      rf_params['Adstock'] = posterior[c.ALPHA_RF].median(dim=['chain', 'draw'])
    if c.EC_RF in posterior:
      rf_params['Inflexion'] = posterior[c.EC_RF].median(dim=['chain', 'draw'])
    if c.SLOPE_RF in posterior:
      rf_params['Slope'] = posterior[c.SLOPE_RF].median(dim=['chain', 'draw'])

    if rf_params:
      rf_dataset = xr.Dataset(rf_params)
      rf_param_df = rf_dataset.to_dataframe().reset_index()
      rf_param_df.rename(columns={'rf_channel': 'MediaVariable'}, inplace=True)
      param_dfs.append(rf_param_df)

  # Combine all parameter DataFrames
  param_df = pd.concat(param_dfs, ignore_index=True) if param_dfs else pd.DataFrame()

  # Write to Excel
  with pd.ExcelWriter(output_path) as writer:
    data_df.to_excel(writer, sheet_name='Data', index=False)
    coeff_df.to_excel(writer, sheet_name='Coefficients', index=False)
    param_df.to_excel(writer, sheet_name='Parameters', index=False)

  print(f"Successfully converted model to Excel: {output_path}")
  print(f"Data sheet: {len(data_df)} rows")
  print(f"Coefficients sheet: {len(coeff_df)} rows")
  print(f"Parameters sheet: {len(param_df)} rows")

if __name__ == '__main__':
  model_path = '/Users/mariappan.subramanian/Documents/repo/forked/meridian/demo/saved_models/demo_model_geo_all_channels_new.pkl'
  output_path = '/Users/mariappan.subramanian/Documents/repo/forked/meridian/demo/converted_model_data.xlsx'
  convert_model_to_excel(model_path, output_path)
