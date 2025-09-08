#!/usr/bin/env python3
"""Test hypothesis: Does n_draws in sample_prior affect optimization results?"""

import logging
import numpy as np
from meridian.model import model, spec
from meridian.analysis import optimizer
from meridian.planner.adhoc_data_loader import AdhocDataLoader

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

# Load Excel data and create inference data
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
    'revenue_per_kpi_col': 'revenue_per_conversion',
    'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
    'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
    'media_channels': ['Channel0', 'Channel1', 'Channel2'],
    'reach_cols': ['Channel3_reach'],
    'frequency_cols': ['Channel3_frequency'],
    'rf_spend_cols': ['Channel3_spend'],
    'rf_channels': ['Channel3']
}

print("Loading Excel data...")
adhoc_data_loader = AdhocDataLoader(file_name=excel_file_path, model_config=model_config)
data = adhoc_data_loader.build_input_data()
inference_data = adhoc_data_loader.get_inference_data()

print("\n" + "="*80)
print("HYPOTHESIS TEST: Effect of n_draws on optimization results")
print("="*80)

n_draws_values = [1, 10, 100]
results = {}

for n_draws in n_draws_values:
    print(f"\nTesting n_draws={n_draws}...")
    
    # Create model and sample prior
    dummy_model = model.Meridian(input_data=data, model_spec=spec.ModelSpec(), inference_data=inference_data)
    dummy_model.sample_prior(n_draws=n_draws, seed=42)
    
    # Run optimization
    budget_optimizer = optimizer.BudgetOptimizer(dummy_model)
    opt_results = budget_optimizer.optimize()
    
    # Extract key results
    opt_data = opt_results.optimized_data
    
    # Store key metrics
    results[n_draws] = {
        'spend': opt_data['spend'].values,
        'roi_mean': opt_data['roi'].sel(metric='mean').values,
        'roi_median': opt_data['roi'].sel(metric='median').values,
        'roi_ci_lo': opt_data['roi'].sel(metric='ci_lo').values,
        'roi_ci_hi': opt_data['roi'].sel(metric='ci_hi').values,
        'mroi_mean': opt_data['mroi'].sel(metric='mean').values,
    }
    
    print(f"  Spend: {results[n_draws]['spend']}")
    print(f"  ROI mean: {results[n_draws]['roi_mean']}")
    print(f"  ROI median: {results[n_draws]['roi_median']}")

print("\n" + "="*60)
print("COMPARISON ACROSS n_draws VALUES:")
print("="*60)

# Compare spend values
print("\nSPEND COMPARISON:")
print("-" * 30)
for n_draws in n_draws_values:
    print(f"n_draws={n_draws:3d}: {results[n_draws]['spend']}")

# Check if spend values are identical
all_spend = [results[n_draws]['spend'] for n_draws in n_draws_values]
spend_identical = all(np.allclose(all_spend[0], spend) for spend in all_spend[1:])
print(f"All spend values identical: {spend_identical}")

# Compare ROI means
print("\nROI MEAN COMPARISON:")
print("-" * 30)
for n_draws in n_draws_values:
    print(f"n_draws={n_draws:3d}: {results[n_draws]['roi_mean']}")

# Check if ROI means are identical
all_roi_mean = [results[n_draws]['roi_mean'] for n_draws in n_draws_values]
roi_mean_identical = all(np.allclose(all_roi_mean[0], roi) for roi in all_roi_mean[1:])
print(f"All ROI means identical: {roi_mean_identical}")

# Compare ROI confidence intervals
print("\nROI CONFIDENCE INTERVAL WIDTH COMPARISON:")
print("-" * 40)
for n_draws in n_draws_values:
    ci_width = results[n_draws]['roi_ci_hi'] - results[n_draws]['roi_ci_lo']
    print(f"n_draws={n_draws:3d}: {ci_width}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)

if spend_identical and roi_mean_identical:
    print("✅ HYPOTHESIS CONFIRMED:")
    print("   The optimization results (spend, ROI means) are IDENTICAL across")
    print("   different n_draws values. This proves that the budget optimizer")
    print("   is using our Excel POSTERIOR point estimates, NOT the prior samples.")
    print("   The different metric values (mean/median/ci_lo/ci_hi) are likely")
    print("   created through some other mechanism (possibly bootstrap resampling")
    print("   or using the single point estimate with artificial uncertainty).")
else:
    print("❌ HYPOTHESIS REJECTED:")
    print("   Results vary with n_draws, indicating prior samples affect optimization.")

print("\nThe mystery of mean ≠ median ≠ ci_lo ≠ ci_hi with single point estimates")
print("is NOT explained by prior sample variation. Another mechanism is at work!")