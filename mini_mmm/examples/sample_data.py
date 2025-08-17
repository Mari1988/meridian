"""Sample data generation utilities for Mini MMM.

This module provides functions to generate realistic synthetic MMM data
for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def generate_realistic_mmm_data(
    n_weeks: int = 104,
    n_media_channels: int = 5,
    channel_names: Optional[List[str]] = None,
    base_kpi: float = 1000,
    seasonality_strength: float = 0.2,
    trend_strength: float = 0.1,
    noise_level: float = 0.05,
    random_seed: int = 42
) -> pd.DataFrame:
  """Generates realistic MMM synthetic data.
  
  Args:
    n_weeks: Number of weeks to generate
    n_media_channels: Number of media channels
    channel_names: Optional list of channel names
    base_kpi: Base level of the KPI
    seasonality_strength: Strength of seasonal patterns (0-1)
    trend_strength: Strength of trend (0-1)  
    noise_level: Level of noise relative to base_kpi (0-1)
    random_seed: Random seed for reproducibility
    
  Returns:
    DataFrame with synthetic MMM data
  """
  np.random.seed(random_seed)
  
  if channel_names is None:
    channel_names = [f'TV', 'Digital', 'Social', 'Radio', 'Print'][:n_media_channels]
  elif len(channel_names) != n_media_channels:
    raise ValueError("Length of channel_names must match n_media_channels")
  
  # Time index
  time_idx = np.arange(n_weeks)
  dates = pd.date_range(start='2022-01-01', periods=n_weeks, freq='W-SUN')
  
  # Base components
  base = np.full(n_weeks, base_kpi)
  
  # Seasonality (yearly + some higher frequency)
  seasonality = (
      seasonality_strength * base_kpi * np.sin(2 * np.pi * time_idx / 52.2) +  # Yearly
      0.5 * seasonality_strength * base_kpi * np.sin(2 * np.pi * time_idx / 13) # Quarterly
  )
  
  # Trend
  trend = trend_strength * base_kpi * time_idx / n_weeks
  
  # Noise
  noise = np.random.normal(0, noise_level * base_kpi, n_weeks)
  
  # Media effects
  data = {'date': dates}
  total_media_effect = np.zeros(n_weeks)
  
  # Realistic channel configurations
  channel_configs = [
      {'name': 'TV', 'base_spend': 80, 'seasonality': 0.3, 'campaign_freq': 0.1, 'roi_range': (1.5, 3.0)},
      {'name': 'Digital', 'base_spend': 60, 'seasonality': 0.1, 'campaign_freq': 0.0, 'roi_range': (2.0, 4.0)},  
      {'name': 'Social', 'base_spend': 30, 'seasonality': 0.2, 'campaign_freq': 0.3, 'roi_range': (1.0, 2.5)},
      {'name': 'Radio', 'base_spend': 25, 'seasonality': 0.4, 'campaign_freq': 0.2, 'roi_range': (0.8, 2.0)},
      {'name': 'Print', 'base_spend': 15, 'seasonality': 0.1, 'campaign_freq': 0.4, 'roi_range': (0.5, 1.5)},
      {'name': 'OOH', 'base_spend': 40, 'seasonality': 0.2, 'campaign_freq': 0.1, 'roi_range': (1.2, 2.8)},
      {'name': 'Email', 'base_spend': 10, 'seasonality': 0.05, 'campaign_freq': 0.0, 'roi_range': (3.0, 6.0)},
  ]
  
  for i, channel in enumerate(channel_names):
    # Use predefined config or create generic one
    if i < len(channel_configs):
      config = channel_configs[i]
    else:
      config = {
          'name': channel,
          'base_spend': 20 + 40 * np.random.random(),
          'seasonality': 0.1 + 0.3 * np.random.random(),
          'campaign_freq': 0.2 * np.random.random(),
          'roi_range': (0.5 + 1.5 * np.random.random(), 1.5 + 2.5 * np.random.random())
      }
    
    # Generate spend pattern
    base_spend = config['base_spend']
    
    # Seasonal component
    spend_seasonality = config['seasonality'] * base_spend * np.sin(
        2 * np.pi * time_idx / 52.2 + 2 * np.pi * np.random.random())
    
    # Campaign spikes
    campaign_spikes = np.zeros(n_weeks)
    n_campaigns = int(config['campaign_freq'] * n_weeks)
    campaign_weeks = np.random.choice(n_weeks, n_campaigns, replace=False)
    for week in campaign_weeks:
      # Campaign duration 1-4 weeks
      duration = np.random.randint(1, 5)
      end_week = min(week + duration, n_weeks)
      campaign_spikes[week:end_week] += base_spend * (1 + 2 * np.random.random())
    
    # Base spend with variation
    base_variation = base_spend * (0.8 + 0.4 * np.random.random(n_weeks))
    
    # Total spend (ensure non-negative)
    spend = np.maximum(0, base_variation + spend_seasonality + campaign_spikes + 
                      np.random.normal(0, base_spend * 0.1, n_weeks))
    
    data[f'{channel}_spend'] = spend
    
    # Generate media effects using realistic transformations
    roi_min, roi_max = config['roi_range']
    channel_roi = roi_min + (roi_max - roi_min) * np.random.random()
    
    # Adstock parameters
    retention_rate = 0.1 + 0.7 * np.random.random()  # 0.1 to 0.8
    
    # Hill saturation parameters
    ec = 0.5 + 1.5 * np.random.random()  # 0.5 to 2.0
    half_saturation = np.percentile(spend[spend > 0], 60 + 30 * np.random.random())
    
    # Apply adstock transformation
    adstocked = np.zeros_like(spend)
    for t in range(n_weeks):
      adstocked[t] = spend[t]
      if t > 0:
        adstocked[t] += retention_rate * adstocked[t-1]
    
    # Apply Hill saturation
    saturated = adstocked ** ec / (half_saturation ** ec + adstocked ** ec)
    
    # Convert to media effect
    media_effect = channel_roi * saturated
    total_media_effect += media_effect
  
  # Control variables
  data['price_index'] = 100 + 5 * np.sin(2 * np.pi * time_idx / 26) + np.random.normal(0, 2, n_weeks)
  data['competitor_spend'] = 50 + 20 * np.sin(2 * np.pi * time_idx / 39) + 10 * np.random.random(n_weeks)
  data['economic_index'] = 100 + 10 * time_idx / n_weeks + np.random.normal(0, 3, n_weeks)
  data['promotion_flag'] = (np.random.random(n_weeks) > 0.85).astype(int)
  
  # Control effects (simplified)
  price_effect = -0.5 * (data['price_index'] - 100)
  competitor_effect = -0.2 * (data['competitor_spend'] - 50)
  economic_effect = 0.3 * (data['economic_index'] - 100)
  promotion_effect = 50 * data['promotion_flag']
  
  total_control_effect = price_effect + competitor_effect + economic_effect + promotion_effect
  
  # Final KPI
  data['kpi'] = base + seasonality + trend + total_media_effect + total_control_effect + noise
  
  return pd.DataFrame(data)


def create_test_scenarios() -> Dict[str, pd.DataFrame]:
  """Creates multiple test scenarios for different modeling situations.
  
  Returns:
    Dictionary with scenario names as keys and DataFrames as values
  """
  scenarios = {}
  
  # 1. Standard scenario
  scenarios['standard'] = generate_realistic_mmm_data(
      n_weeks=104, 
      n_media_channels=4,
      channel_names=['TV', 'Digital', 'Social', 'Radio'],
      random_seed=42
  )
  
  # 2. High seasonality scenario
  scenarios['high_seasonality'] = generate_realistic_mmm_data(
      n_weeks=104,
      n_media_channels=3,
      channel_names=['Retail', 'Online', 'Mobile'],
      seasonality_strength=0.4,
      random_seed=43
  )
  
  # 3. Low signal-to-noise scenario
  scenarios['noisy'] = generate_realistic_mmm_data(
      n_weeks=78,  # Shorter time series
      n_media_channels=5,
      noise_level=0.15,  # Higher noise
      random_seed=44
  )
  
  # 4. Minimal data scenario (challenging case)
  scenarios['minimal'] = generate_realistic_mmm_data(
      n_weeks=52,  # Only 1 year
      n_media_channels=3,
      channel_names=['Display', 'Search', 'Social'],
      noise_level=0.1,
      random_seed=45
  )
  
  # 5. Large scale scenario
  scenarios['large_scale'] = generate_realistic_mmm_data(
      n_weeks=156,  # 3 years
      n_media_channels=7,
      base_kpi=10000,  # Larger business
      random_seed=46
  )
  
  return scenarios


def add_data_quality_issues(df: pd.DataFrame, 
                          issue_type: str = 'missing_data',
                          severity: float = 0.1) -> pd.DataFrame:
  """Adds realistic data quality issues for testing robustness.
  
  Args:
    df: Clean DataFrame to modify
    issue_type: Type of issue ('missing_data', 'outliers', 'zero_variance')
    severity: Severity level (0-1)
    
  Returns:
    Modified DataFrame with data quality issues
  """
  df_modified = df.copy()
  
  if issue_type == 'missing_data':
    # Randomly set some values to NaN
    media_cols = [col for col in df.columns if col.endswith('_spend')]
    for col in media_cols:
      n_missing = int(severity * len(df))
      missing_idx = np.random.choice(len(df), n_missing, replace=False)
      df_modified.loc[missing_idx, col] = np.nan
  
  elif issue_type == 'outliers':
    # Add extreme outliers
    media_cols = [col for col in df.columns if col.endswith('_spend')]
    for col in media_cols:
      n_outliers = max(1, int(severity * len(df)))
      outlier_idx = np.random.choice(len(df), n_outliers, replace=False)
      normal_max = df[col].max()
      outlier_values = normal_max * (10 + 20 * np.random.random(n_outliers))
      df_modified.loc[outlier_idx, col] = outlier_values
  
  elif issue_type == 'zero_variance':
    # Set some channels to constant spend
    media_cols = [col for col in df.columns if col.endswith('_spend')]
    n_channels_affected = max(1, int(severity * len(media_cols)))
    affected_channels = np.random.choice(media_cols, n_channels_affected, replace=False)
    
    for col in affected_channels:
      constant_value = df[col].mean()
      df_modified[col] = constant_value
  
  return df_modified


def generate_comparative_dataset(
    scenario_a_params: Dict,
    scenario_b_params: Dict,
    n_weeks: int = 104
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Generates two datasets for A/B testing or comparative analysis.
  
  Args:
    scenario_a_params: Parameters for scenario A
    scenario_b_params: Parameters for scenario B  
    n_weeks: Number of weeks for both scenarios
    
  Returns:
    Tuple of (scenario_a_df, scenario_b_df)
  """
  # Generate scenario A
  df_a = generate_realistic_mmm_data(n_weeks=n_weeks, **scenario_a_params)
  
  # Generate scenario B
  df_b = generate_realistic_mmm_data(n_weeks=n_weeks, **scenario_b_params)
  
  return df_a, df_b


# Example usage
if __name__ == "__main__":
  # Generate and display sample data
  print("Generating sample MMM data...")
  
  df = generate_realistic_mmm_data(
      n_weeks=104,
      n_media_channels=4,
      channel_names=['TV', 'Digital', 'Social', 'Radio']
  )
  
  print(f"Generated data shape: {df.shape}")
  print(f"Columns: {list(df.columns)}")
  print("\nFirst 5 rows:")
  print(df.head())
  
  print("\nData summary:")
  print(df.describe())
  
  # Create test scenarios
  scenarios = create_test_scenarios()
  print(f"\nCreated {len(scenarios)} test scenarios:")
  for name, data in scenarios.items():
    print(f"  {name}: {data.shape}")
  
  # Example with data quality issues
  print("\nTesting data quality issues:")
  df_with_issues = add_data_quality_issues(df, 'outliers', severity=0.05)
  print("Added outliers to 5% of observations")