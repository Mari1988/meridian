#!/usr/bin/env python3
"""Check available time coordinates in the Excel data."""

import logging
from meridian.planner.adhoc_data_loader import AdhocDataLoader

logging.basicConfig(level=logging.WARNING)

# Load Excel data
excel_file_path = (
    '/Users/mariappan.subramanian/Library/CloudStorage/'
    'OneDrive-TheTradeDesk/MMM/BudgetOptimizer/mmm_input_artifacts.xlsx'
)

model_config = {
    'time_col': 'week',
    'geo_col': 'geo',
    'population_col': 'population',
    'kpi_type': 'non_revenue',
    'kpi_col': 'conversions',
    'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
    'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
    'media_channels': ['Channel0', 'Channel1', 'Channel2'],
    'reach_cols': ['Channel3_reach'],
    'frequency_cols': ['Channel3_frequency'],
    'rf_spend_cols': ['Channel3_spend'],
    'rf_channels': ['Channel3']
}

print("Loading Excel data to check time coordinates...")
loader = AdhocDataLoader(file_name=excel_file_path, model_config=model_config)
data = loader.build_input_data()

print(f"\nTotal time periods: {len(data.time)}")
print(f"Time range: {data.time.values[0]} to {data.time.values[-1]}")

# Show first and last 10 time periods
print(f"\nFirst 10 time periods:")
for i, time_val in enumerate(data.time.values[:10]):
    print(f"  {i}: {time_val}")

print(f"\nLast 10 time periods:")
for i, time_val in enumerate(data.time.values[-10:]):
    print(f"  {len(data.time)-10+i}: {time_val}")

# Show some sample periods for testing
print(f"\nSample periods for testing:")
total_periods = len(data.time)
quarter_size = total_periods // 4

print(f"First quarter (periods 0-{quarter_size-1}):")
print(f"  Start: {data.time.values[0]}")
print(f"  End: {data.time.values[quarter_size-1]}")

print(f"Middle year (periods {quarter_size}-{3*quarter_size-1}):")
print(f"  Start: {data.time.values[quarter_size]}")
print(f"  End: {data.time.values[3*quarter_size-1]}")

print(f"Last 6 months (periods {total_periods-26}-{total_periods-1}):")
print(f"  Start: {data.time.values[total_periods-26]}")
print(f"  End: {data.time.values[total_periods-1]}")

print(f"Single month (periods {total_periods//2}-{total_periods//2+3}):")
print(f"  Start: {data.time.values[total_periods//2]}")
print(f"  End: {data.time.values[total_periods//2+3]}")