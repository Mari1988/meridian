#!/usr/bin/env python3
"""
Test 3.1: Partial Time Period Optimization

Test optimization across different time periods to ensure the framework works 
with subset date ranges.
"""

import logging
import pandas as pd
import numpy as np
from meridian.model import model, spec
from meridian.analysis import optimizer
from meridian.planner.adhoc_data_loader import AdhocDataLoader

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)

def setup_test_data():
    """Load Excel data and create inference data for testing."""
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

    print("Loading Excel data and creating inference data...")
    loader = AdhocDataLoader(file_name=excel_file_path, model_config=model_config)
    data = loader.build_input_data()
    inference_data = loader.get_inference_data()
    
    return data, inference_data

def run_optimization_for_period(data, inference_data, start_date, end_date, period_name):
    """Run optimization for a specific time period."""
    print(f"\n{'='*60}")
    print(f"Testing {period_name}")
    print(f"Period: {start_date} to {end_date}")
    print('='*60)
    
    try:
        # Create model and sample prior
        dummy_model = model.Meridian(input_data=data, model_spec=spec.ModelSpec(), inference_data=inference_data)
        dummy_model.sample_prior(n_draws=100, seed=42)
        
        # Run optimization with time period constraints
        budget_optimizer = optimizer.BudgetOptimizer(dummy_model)
        opt_results = budget_optimizer.optimize(
            start_date=start_date,
            end_date=end_date,
            fixed_budget=True  # Use fixed budget for consistent comparison
        )
        
        # Extract results
        opt_data = opt_results.optimized_data
        
        results = {
            'period_name': period_name,
            'start_date': start_date,
            'end_date': end_date,
            'spend': opt_data['spend'].values,
            'pct_of_spend': opt_data['pct_of_spend'].values,
            'roi_mean': opt_data['roi'].sel(metric='mean').values,
            'mroi_mean': opt_data['mroi'].sel(metric='mean').values,
            'incremental_outcome': opt_data['incremental_outcome'].sel(metric='mean').values,
            'effectiveness': opt_data['effectiveness'].sel(metric='mean').values,
            'total_budget': np.sum(opt_data['spend'].values),
            'total_incremental_outcome': np.sum(opt_data['incremental_outcome'].sel(metric='mean').values)
        }
        
        # Print results summary
        print(f"✅ Optimization completed successfully")
        print(f"Total Budget: ${results['total_budget']:,.0f}")
        print(f"Total Incremental Outcome: {results['total_incremental_outcome']:,.0f}")
        print(f"Spend Allocation: {results['spend']}")
        print(f"ROI by Channel: {results['roi_mean']}")
        print(f"mROI by Channel: {results['mroi_mean']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        return None

def calculate_period_length_days(start_date, end_date):
    """Calculate the number of days between two dates."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return (end - start).days + 1  # +1 to include end date

def analyze_results(results_list):
    """Analyze and compare results across different time periods."""
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS ACROSS TIME PERIODS")
    print('='*80)
    
    # Filter out failed optimizations
    successful_results = [r for r in results_list if r is not None]
    
    if not successful_results:
        print("❌ No successful optimizations to analyze")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for result in successful_results:
        period_days = calculate_period_length_days(result['start_date'], result['end_date'])
        
        comparison_data.append({
            'Period': result['period_name'],
            'Days': period_days,
            'Total_Budget': result['total_budget'],
            'Total_Outcome': result['total_incremental_outcome'],
            'Budget_per_Day': result['total_budget'] / period_days,
            'Outcome_per_Day': result['total_incremental_outcome'] / period_days,
            'Overall_ROI': result['total_incremental_outcome'] / result['total_budget'],
            'Channel0_ROI': result['roi_mean'][0],
            'Channel1_ROI': result['roi_mean'][1], 
            'Channel2_ROI': result['roi_mean'][2],
            'Channel3_ROI': result['roi_mean'][3],
            'Channel0_Spend_Pct': result['pct_of_spend'][0] * 100,
            'Channel1_Spend_Pct': result['pct_of_spend'][1] * 100,
            'Channel2_Spend_Pct': result['pct_of_spend'][2] * 100,
            'Channel3_Spend_Pct': result['pct_of_spend'][3] * 100,
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Display detailed comparison
    print("\n1. PERIOD OVERVIEW:")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['Period']:<20}: {row['Days']:>3} days, Budget: ${row['Total_Budget']:>10,.0f}")
    
    print("\n2. SPEND EFFICIENCY ANALYSIS:")
    print("-" * 50)
    print(f"{'Period':<20} {'Budget/Day':<12} {'Outcome/Day':<14} {'Overall ROI':<12}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['Period']:<20} ${row['Budget_per_Day']:<11,.0f} {row['Outcome_per_Day']:<13,.0f} {row['Overall_ROI']:<11.3f}")
    
    print("\n3. CHANNEL ROI COMPARISON:")
    print("-" * 50)
    print(f"{'Period':<20} {'Ch0 ROI':<8} {'Ch1 ROI':<8} {'Ch2 ROI':<8} {'Ch3 ROI':<8}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['Period']:<20} {row['Channel0_ROI']:<8.3f} {row['Channel1_ROI']:<8.3f} {row['Channel2_ROI']:<8.3f} {row['Channel3_ROI']:<8.3f}")
    
    print("\n4. SPEND ALLOCATION COMPARISON:")
    print("-" * 50)
    print(f"{'Period':<20} {'Ch0 %':<8} {'Ch1 %':<8} {'Ch2 %':<8} {'Ch3 %':<8}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['Period']:<20} {row['Channel0_Spend_Pct']:<8.1f} {row['Channel1_Spend_Pct']:<8.1f} {row['Channel2_Spend_Pct']:<8.1f} {row['Channel3_Spend_Pct']:<8.1f}")
    
    # Statistical analysis
    print("\n5. STATISTICAL ANALYSIS:")
    print("-" * 50)
    
    # Efficiency metrics
    efficiency_stats = df[['Budget_per_Day', 'Outcome_per_Day', 'Overall_ROI']].describe()
    print("\nEfficiency Statistics:")
    print(efficiency_stats.round(2))
    
    # Channel consistency analysis
    roi_cols = ['Channel0_ROI', 'Channel1_ROI', 'Channel2_ROI', 'Channel3_ROI']
    roi_std = df[roi_cols].std()
    allocation_cols = ['Channel0_Spend_Pct', 'Channel1_Spend_Pct', 'Channel2_Spend_Pct', 'Channel3_Spend_Pct']
    allocation_std = df[allocation_cols].std()
    
    print(f"\nChannel ROI Consistency (lower std = more consistent):")
    for i, col in enumerate(roi_cols):
        print(f"  Channel{i}: {roi_std[col]:.4f} std deviation")
    
    print(f"\nSpend Allocation Consistency (lower std = more consistent):")
    for i, col in enumerate(allocation_cols):
        print(f"  Channel{i}: {allocation_std[col]:.2f}% std deviation")
    
    # Seasonal effects analysis
    print("\n6. SEASONAL EFFECTS ANALYSIS:")
    print("-" * 50)
    
    # Find best and worst performing periods
    best_roi_period = df.loc[df['Overall_ROI'].idxmax()]
    worst_roi_period = df.loc[df['Overall_ROI'].idxmin()]
    
    print(f"Best ROI Period: {best_roi_period['Period']} (ROI: {best_roi_period['Overall_ROI']:.3f})")
    print(f"Worst ROI Period: {worst_roi_period['Period']} (ROI: {worst_roi_period['Overall_ROI']:.3f})")
    
    roi_range = df['Overall_ROI'].max() - df['Overall_ROI'].min()
    print(f"ROI Range: {roi_range:.3f} ({(roi_range/df['Overall_ROI'].mean()*100):.1f}% of mean)")
    
    if roi_range > 0.1:  # If more than 0.1 difference in ROI
        print("⚠️  Significant seasonal variation detected!")
    else:
        print("✅ ROI relatively consistent across periods")
    
    return df

def main():
    """Main test execution function."""
    print("="*80)
    print("TEST 3.1: PARTIAL TIME PERIOD OPTIMIZATION")
    print("Testing fast optimization framework across different time periods")
    print("="*80)
    
    # Setup test data
    data, inference_data = setup_test_data()
    
    # Define test scenarios using actual dataset dates (2021-01-25 to 2024-01-15)
    test_scenarios = [
        ('2021-01-25', '2021-10-18', 'First Quarter (39 weeks)'),
        ('2021-10-25', '2023-04-17', 'Middle Year (78 weeks)'), 
        ('2023-07-24', '2024-01-15', 'Last 6 Months (26 weeks)'),
        ('2022-07-25', '2022-08-15', 'Single Month (4 weeks)')
    ]
    
    # Run optimizations for each period
    results = []
    for start_date, end_date, period_name in test_scenarios:
        result = run_optimization_for_period(data, inference_data, start_date, end_date, period_name)
        results.append(result)
    
    # Analyze results
    comparison_df = analyze_results(results)
    
    # Final validation
    print(f"\n{'='*80}")
    print("TEST VALIDATION SUMMARY")
    print('='*80)
    
    successful_tests = sum(1 for r in results if r is not None)
    total_tests = len(results)
    
    print(f"✅ Successful optimizations: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("✅ All time period tests passed!")
        print("✅ Framework works correctly with subset date ranges")
        
        if comparison_df is not None:
            roi_consistency = comparison_df['Overall_ROI'].std() < 0.1
            allocation_consistency = all(comparison_df[[f'Channel{i}_Spend_Pct' for i in range(4)]].std() < 5.0)
            
            if roi_consistency:
                print("✅ ROI performance is consistent across periods")
            else:
                print("⚠️  ROI shows variation across periods - potential seasonal effects")
            
            if allocation_consistency:
                print("✅ Spend allocation is consistent across periods")
            else:
                print("⚠️  Spend allocation varies across periods")
    else:
        print(f"❌ {total_tests - successful_tests} tests failed - framework needs debugging")
    
    print("\nTest 3.1 Complete!")

if __name__ == "__main__":
    main()