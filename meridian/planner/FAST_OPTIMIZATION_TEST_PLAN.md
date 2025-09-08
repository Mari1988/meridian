# Fast Budget Optimization Testing Framework

## Overview

This document provides a comprehensive test plan for the Excel-based fast budget optimization framework. Each test is written as an actionable prompt that can be copy-pasted and executed to validate the framework's functionality across different `BudgetOptimizer.optimize()` parameter configurations.

## Test Categories Priority

| Priority | Category | Description |
|----------|----------|-------------|
| 🔴 **P0** | Basic Functionality | Core optimization with default parameters |
| 🟡 **P1** | Budget Configurations | Fixed vs flexible budget scenarios |  
| 🟡 **P1** | Time Period Tests | Different start/end date combinations |
| 🟢 **P2** | Constraint Tests | Spend limits and bounds |
| 🟢 **P2** | Advanced Configurations | KPI vs revenue, confidence levels |
| 🟢 **P3** | Edge Cases | Extreme values and error handling |
| 🔵 **P4** | Performance & Validation | Benchmarking and comparison tests |

---

## 🔴 P0: Basic Functionality Tests

### Test 1.1: Default Parameters Baseline

**Prompt:**
```
Test the fast budget optimization with completely default parameters. Create a test that loads the Excel data using AdhocDataLoader, creates the Meridian model with point inference data, and runs optimization with default settings. Compare the results against our known baseline to ensure basic functionality works correctly.

script to use: fast_budget_planner.py inside the planner directory.

Expected Results:
- Optimization should complete without errors
- Should return spend allocation: [28300000, 27100000, 30000000, 25500000] 
- ROI means should be: [1.83, 2.51, 2.92, 5.42] (approximately)
- All metric statistics (mean, median, ci_lo, ci_hi) should be identical for each channel
```


## 🟡 P1: Budget Configuration Tests

### Test 2.1: Fixed Budget Scenarios

**Prompt:**
```
Test fixed budget optimization with different budget amounts to ensure the framework scales properly. Test the following scenarios:

1. Historical budget (budget=None, default)
2. 50% of historical budget (budget=55_000_000)  
3. 150% of historical budget (budget=165_000_000)
4. 200% of historical budget (budget=220_000_000)

For each scenario:
- Verify total allocated spend equals the target budget
- Check that spend allocation percentages make sense
- Ensure ROI and mROI values scale appropriately
- Document any patterns in channel reallocation

Expected: Higher budgets should generally improve ROI up to saturation points
```

### Test 2.2: Flexible Budget with ROI Targets

**Prompt:**
```
Test flexible budget optimization with different target ROI values. This tests the framework's ability to find optimal budgets that achieve specific ROI targets.

Test Scenarios:
1. target_roi=2.0 (conservative target)
2. target_roi=3.0 (moderate target) 
3. target_roi=4.0 (aggressive target)
4. target_roi=6.0 (potentially unrealistic target)

For each target:
- Set fixed_budget=False
- Set target_roi to the test value
- Verify the optimization finds a budget that achieves the target ROI
- Check if the optimization fails gracefully for unrealistic targets
- Document the budget levels required for each ROI target
```

### Test 2.3: Flexible Budget with mROI Targets  

**Prompt:**
```
Test flexible budget optimization with different target marginal ROI (mROI) values. This tests saturation point optimization.

Test Scenarios:
1. target_mroi=1.5 (high efficiency target)
2. target_mroi=2.0 (moderate efficiency target)
3. target_mroi=3.0 (conservative efficiency target)

For each target:
- Set fixed_budget=False, target_roi=None
- Set target_mroi to the test value
- Verify the optimization achieves the target mROI
- Compare budget levels vs ROI targets
- Document efficiency differences between ROI vs mROI optimization
```

### Test 2.4: Custom Spend Allocation

**Prompt:**
```
Test custom percentage spend allocation to ensure the framework respects user-defined allocation preferences.

Test Scenarios:
1. Equal allocation: pct_of_spend=[0.25, 0.25, 0.25, 0.25]
2. Heavy Channel0: pct_of_spend=[0.50, 0.20, 0.20, 0.10] 
3. Heavy Channel3: pct_of_spend=[0.15, 0.15, 0.20, 0.50]
4. No Channel2: pct_of_spend=[0.40, 0.35, 0.0, 0.25]

For each allocation:
- Verify spend allocation matches the specified percentages
- Check that optimization respects the allocation constraints
- Compare ROI performance across different allocation strategies
- Document which allocations perform best/worst
```

---

## 🟡 P1: Time Period Tests

### Test 3.1: Partial Time Period Optimization

**Prompt:**
```
Test optimization across different time periods to ensure the framework works with subset date ranges.

Test Scenarios:
1. First quarter: start_date='2020-01-01', end_date='2020-03-31'
2. Middle period: start_date='2021-01-01', end_date='2021-12-31' 
3. Last 6 months: start_date='2022-07-01', end_date='2022-12-31'
4. Single month: start_date='2021-06-01', end_date='2021-06-30'

For each period:
- Verify optimization completes successfully
- Check that results scale appropriately for the time period length
- Compare spend efficiency across different periods
- Document seasonal effects on optimization results
```

### Test 3.2: Full Time Range vs Partial Periods

**Prompt:**
```
Compare optimization results between full time range and partial periods to validate consistency and understand temporal effects.

Comparison Tests:
1. Full period (no start_date/end_date) vs sum of quarterly results
2. Annual optimization vs quarterly optimization aggregated
3. Seasonal comparison: Q1 vs Q2 vs Q3 vs Q4 optimization efficiency

Analysis:
- Check if partial period optimizations are consistent with full period
- Identify any seasonal patterns in channel performance
- Verify that time-scaled results are mathematically consistent
- Document any unexpected temporal dependencies
```

---

## 🟢 P2: Constraint Tests

### Test 4.1: Lower Spend Constraints

**Prompt:**
```
Test optimization with lower spend constraints to ensure channels maintain minimum spend levels.

Test Scenarios:
1. Conservative: spend_constraint_lower=0.2 (80% minimum of historical)
2. Moderate: spend_constraint_lower=0.5 (50% minimum of historical) 
3. Aggressive: spend_constraint_lower=0.8 (20% minimum of historical)
4. Channel-specific: spend_constraint_lower=[0.2, 0.3, 0.5, 0.1] (per channel)

For each constraint:
- Verify no channel goes below the minimum constraint
- Check how constraints affect overall optimization efficiency
- Document trade-offs between constraints and ROI performance
- Test that constraints are enforced correctly for each channel
```

### Test 4.2: Upper Spend Constraints

**Prompt:**
```
Test optimization with upper spend constraints to limit maximum spend increases.

Test Scenarios:
1. Conservative: spend_constraint_upper=0.2 (120% maximum of historical)
2. Moderate: spend_constraint_upper=0.5 (150% maximum of historical)
3. Aggressive: spend_constraint_upper=1.0 (200% maximum of historical)
4. Channel-specific: spend_constraint_upper=[0.3, 0.2, 0.8, 0.5] (per channel)

For each constraint:
- Verify no channel exceeds the maximum constraint
- Check optimization efficiency within constraint bounds
- Compare constrained vs unconstrained performance
- Test asymmetric constraints across channels
```

### Test 4.3: Combined Constraint Scenarios

**Prompt:**
```
Test optimization with both lower and upper constraints simultaneously to validate complex constraint handling.

Test Scenarios:
1. Tight constraints: lower=0.3, upper=0.3 (70%-130% range)
2. Loose constraints: lower=0.5, upper=1.0 (50%-200% range)
3. Asymmetric constraints: varying ranges per channel
4. Impossible constraints: lower > upper (should error gracefully)

Analysis:
- Verify all constraints are respected simultaneously  
- Test constraint conflict resolution
- Document performance impact of tight constraints
- Validate error handling for invalid constraint combinations
```

---

## 🟢 P2: Advanced Configuration Tests

### Test 5.1: KPI vs Revenue Optimization

**Prompt:**
```
Compare optimization results when optimizing for KPI vs Revenue to understand the impact of the use_kpi parameter.

Test Scenarios:
1. Revenue optimization: use_kpi=False (default)
2. KPI optimization: use_kpi=True 
3. Side-by-side comparison of spend allocation differences
4. ROI calculation differences between KPI and revenue focus

Analysis:
- Document spend allocation differences between KPI vs revenue focus
- Compare ROI and mROI values for each approach
- Verify mathematical consistency between approaches
- Identify which channels benefit more from each optimization type
```

### Test 5.2: Optimal vs Historical Frequency

**Prompt:**
```
Test the impact of use_optimal_frequency parameter on reach & frequency channel optimization (Channel3 in our case).

Test Scenarios:  
1. Optimal frequency: use_optimal_frequency=True (default)
2. Historical frequency: use_optimal_frequency=False
3. Compare Channel3 performance between both approaches
4. Analyze frequency recommendations vs historical values

Analysis:
- Document frequency value differences between optimal and historical
- Compare Channel3 ROI under both frequency strategies
- Verify that reach channels are affected while impression channels are not
- Calculate performance lift from frequency optimization
```

### Test 5.3: Confidence Level Variations

**Prompt:**
```
Test different confidence levels to understand their impact on uncertainty quantification.

Test Scenarios:
1. Low confidence: confidence_level=0.80
2. Standard confidence: confidence_level=0.90 (default)  
3. High confidence: confidence_level=0.95
4. Very high confidence: confidence_level=0.99

Analysis:
- Since we use point estimates, confidence intervals should remain zero-width
- Verify that confidence level changes don't affect optimization results
- Confirm that mean values remain identical across confidence levels
- Document any unexpected behavior in confidence interval calculation
```

---

## 🟢 P3: Edge Case Tests

### Test 6.1: Extreme Budget Values

**Prompt:**
```
Test optimization behavior with extreme budget values to validate robustness and error handling.

Test Scenarios:
1. Very low budget: budget=1000 (unrealistically low)
2. Very high budget: budget=1_000_000_000 (unrealistically high)
3. Zero budget: budget=0 (should fail gracefully)
4. Negative budget: budget=-100000 (should fail gracefully)

Expected Behavior:
- System should handle extreme values gracefully
- Low budgets should optimize to highest ROI channels
- High budgets should show saturation effects
- Invalid budgets should raise appropriate errors
- Document warning messages and error handling
```

### Test 6.2: Invalid Parameter Combinations  

**Prompt:**
```
Test invalid parameter combinations to ensure proper error handling and user guidance.

Test Scenarios:
1. Flexible budget with no target: fixed_budget=False, target_roi=None, target_mroi=None
2. Both ROI targets: target_roi=2.0, target_mroi=1.5 (should choose one)
3. Invalid dates: start_date='2025-01-01' (future date)
4. Backwards dates: start_date='2022-01-01', end_date='2021-01-01'
5. Invalid allocations: pct_of_spend=[0.3, 0.3, 0.3] (doesn't sum to 1.0)

Expected Behavior:
- System should validate parameters before optimization
- Clear error messages should guide users to correct inputs
- No silent failures or unexpected behavior
- Document all error conditions and messages
```

### Test 6.3: Boundary Condition Tests

**Prompt:**
```
Test boundary conditions to ensure robust behavior at parameter limits.

Test Scenarios:
1. Single channel allocation: pct_of_spend=[1.0, 0.0, 0.0, 0.0]
2. Maximum constraints: spend_constraint_upper=10.0 (1000% increase allowed)
3. Minimum time period: Single day optimization
4. Maximum confidence: confidence_level=0.999
5. Zero tolerance: gtol=0.0

Analysis:
- Verify system handles boundary conditions appropriately
- Document performance at extreme parameter values
- Check for numerical stability issues
- Validate that results remain mathematically sound
```

---

## 🔵 P4: Performance & Validation Tests

### Test 7.1: Performance Benchmarking

**Prompt:**
```
Benchmark the performance of fast optimization vs full Bayesian optimization to quantify speed improvements.

Benchmark Tests:
1. Measure optimization runtime for fast vs full approach
2. Test with different complexity levels (more channels, longer periods)
3. Memory usage comparison between approaches
4. Scalability testing with larger datasets

Metrics to Measure:
- Total optimization time (seconds)
- Memory peak usage (MB)  
- CPU utilization during optimization
- Results accuracy vs runtime trade-off
- Document performance improvements and any limitations
```

### Test 7.2: Accuracy Validation Against Full Model

**Prompt:**
```
Validate accuracy of fast optimization results against full Bayesian optimization across multiple scenarios.

Validation Tests:
1. Default scenario: Compare fast vs full optimization with identical parameters
2. Multiple budget levels: Test accuracy across different budget amounts
3. Different time periods: Validate temporal consistency
4. Various constraints: Test accuracy with different constraint combinations

Accuracy Metrics:
- Spend allocation differences (absolute and percentage)
- ROI prediction accuracy
- mROI prediction accuracy  
- Incremental outcome prediction accuracy
- Document accuracy patterns and any systematic biases
```

### Test 7.3: Consistency and Reproducibility

**Prompt:**
```
Test consistency and reproducibility of fast optimization results across multiple runs and scenarios.

Consistency Tests:
1. Multiple runs with identical parameters (should be identical)
2. Different random seeds (should be identical since using point estimates)
3. Different n_draws in sample_prior (should be identical - already validated)
4. Different batch_size values (should not affect results)

Reproducibility Tests:
- Same results across different Python environments
- Same results across different TensorFlow versions (if applicable)
- Same results with different Excel file loading methods
- Document any sources of variability and how to control them
```

### Test 7.4: Integration Test Suite

**Prompt:**
```
Create an automated test suite that runs all critical tests to ensure framework reliability.

Integration Test Components:
1. Load test data and verify Excel parsing
2. Run optimization with 5 most important parameter combinations
3. Validate results against known benchmarks
4. Performance regression testing
5. Error condition testing

Test Automation:
- Create a single script that runs all P0 and P1 tests
- Generate a test report with pass/fail status
- Include performance metrics and accuracy measurements
- Set up for continuous integration if needed
- Document test coverage and any gaps
```

---

## Test Execution Guidelines

### Setup Requirements
```python
# Standard test setup for all prompts
from meridian.model import model, spec
from meridian.analysis import optimizer
from meridian.planner.adhoc_data_loader import AdhocDataLoader
import time
import numpy as np

# Excel configuration
excel_file_path = "/path/to/mmm_input_artifacts.xlsx"
model_config = {
    'time_col': 'week', 'geo_col': 'geo', 'population_col': 'population',
    'kpi_type': 'non_revenue', 'kpi_col': 'conversions',
    'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
    'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
    'media_channels': ['Channel0', 'Channel1', 'Channel2'],
    'reach_cols': ['Channel3_reach'], 'frequency_cols': ['Channel3_frequency'],
    'rf_spend_cols': ['Channel3_spend'], 'rf_channels': ['Channel3']
}

# Load data
loader = AdhocDataLoader(excel_file_path, model_config)
data = loader.build_input_data()
inference_data = loader.get_inference_data()
```

### Test Result Documentation Template
```markdown
## Test Results: [Test Name]

### Parameters Tested
- Parameter1: value1
- Parameter2: value2

### Results
- Spend Allocation: [values]
- ROI Values: [values] 
- Performance: X seconds
- Memory Usage: Y MB

### Validation
- ✅ Expected behavior confirmed
- ❌ Issue found: [description]

### Notes
[Any observations or recommendations]
```

## Success Criteria

### P0 Tests (Must Pass)
- All basic functionality tests pass without errors
- Results are consistent with known baselines
- Point estimates are used correctly (not prior samples)

### P1 Tests (Should Pass)
- Budget configurations work across different scenarios
- Time period selection functions correctly
- Results are mathematically consistent

### P2+ Tests (Nice to Have)
- Advanced configurations behave as expected
- Edge cases are handled gracefully
- Performance meets requirements

## Conclusion

This test plan provides comprehensive coverage of the fast budget optimization framework. Execute tests in priority order, documenting results and any issues discovered. The framework should demonstrate reliable performance across all tested scenarios while maintaining accuracy compared to the full Bayesian approach.