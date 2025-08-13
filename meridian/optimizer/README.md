# Independent Budget Optimizer

This module provides a standalone budget optimization system that operates independently of Meridian's main model objects. It allows users to perform budget optimization using pre-extracted model parameters without needing to run the full Meridian modeling pipeline.

## Overview

The independent optimizer enables fast budget allocation optimization for marketing mix modeling scenarios. It supports both impression-based channels (TV, digital, radio) and reach & frequency (R&F) channels (social media, display video), or mixed combinations of both.

## Key Components

### 1. Core Architecture (`independent_optimizer.py`)

**Main Classes:**
- **`OptimizationInput`**: Container for all input data including channel definitions, historical spend, media data, and population information
- **`ModelParameters`**: Stores pre-trained model coefficients (media effects, adstock, Hill saturation parameters)
- **`IndependentOptimizer`**: Main optimization engine that runs budget allocation algorithms
- **`OptimizationScenario`**: Defines optimization constraints and objectives
- **`IndependentOptimizationResults`**: Contains optimization results with spend allocation, ROI metrics, and performance data

**Key Features:**
- Mixed channel support (impression + R&F channels)
- Geographic modeling with population-based scaling
- Time period selection for optimization
- Comprehensive input validation

### 2. Optimization Algorithm (`optimization_core.py`)

**Core Functions:**
- **`hill_climbing_search()`**: Greedy algorithm that maximizes incremental ROI by iteratively selecting the highest marginal ROI opportunities
- **`create_spend_grid()`**: Creates discrete optimization search space with configurable step sizes and channel-specific bounds
- **`compute_response_curves()`**: Generates spend vs. outcome curves for visualization and analysis
- **`validate_spend_constraints()`**: Ensures spend bounds are properly configured

**Algorithm Details:**
- Discrete optimization using spend grids
- Respects budget constraints (fixed or flexible)
- Handles ROI and marginal ROI targets
- Supports both impression and R&F channel optimization simultaneously

### 3. Media Transformations (`media_transforms.py`)

**Transformation Pipeline:**
- **Adstock modeling**: Applies carryover effects using exponential decay (alpha parameters)
- **Hill saturation**: Models diminishing returns using Hill transformation (EC50, slope parameters)
- **Media scaling**: Normalizes media data by population for proper geographic scaling
- **R&F transformations**: Handles reach × frequency calculations with frequency-specific saturation

**Key Classes:**
- **`MediaScaler`**: Population-based normalization for geographic scaling
- **Functions**: `compute_incremental_outcome()`, `compute_rf_incremental_outcome()` for channel-specific outcome prediction

### 4. Data I/O (`io_utils.py`)

**File Format Support:**
- **CSV**: Multi-file loading for impression data, R&F data, spend, and population
- **Excel**: Multi-sheet loading with automatic temporary file handling
- **JSON**: Model parameter serialization with mixed channel support
- **YAML**: Alternative parameter format (requires PyYAML)

**Key Functions:**
- `load_optimization_input_from_csv()`: Load mixed channel data from multiple CSV files
- `load_optimization_input_from_excel()`: Load from Excel with multiple sheets
- `load_model_parameters_from_json()`: Load pre-trained model parameters
- `save_optimization_results_to_csv()`: Export results with metadata
- `extract_model_parameters_from_meridian()`: Bridge from existing Meridian models

### 5. Visualization (`visualizer.py`)

**Chart Types:**
- **Spend allocation**: Pie charts showing budget distribution by channel
- **Delta analysis**: Bar charts showing spend changes with increase/decrease color coding
- **ROI comparison**: Historical vs. optimized performance comparison
- **Response curves**: Spend sensitivity analysis with constraint visualization overlay

**Key Functions:**
- `plot_spend_allocation()`: Budget distribution visualization
- `plot_spend_delta()`: Spend change analysis
- `plot_roi_comparison()`: Performance comparison charts
- `plot_response_curves()`: Spend sensitivity curves with constraint indicators
- `create_optimization_summary()`: Summary metrics table

## Channel Type Support

### 1. Mixed Channels (Impression + R&F)
- Combines traditional impression-based channels with reach & frequency channels
- Proper scaling and transformation for each channel type
- Unified optimization across both channel types

### 2. Impression-Only Channels
- Traditional media mix modeling approach
- Channels like TV, digital, radio, print
- Impression-based media transformations

### 3. R&F-Only Channels  
- Specialized for reach & frequency campaigns
- Channels like social media, display video, connected TV
- Reach × frequency calculations with frequency saturation

## Optimization Scenarios

### Fixed Budget Optimization
- Optimize allocation within a specified total budget
- Maximize total incremental outcome
- Respect channel-specific spend constraints

### Flexible Budget Optimization
- Optimize based on ROI or marginal ROI targets
- Allow budget to vary based on performance thresholds
- Stop when target performance metrics are achieved

## Usage Examples

### Basic Mixed Channel Optimization
```python
from meridian.optimizer.independent_optimizer import (
    OptimizationInput, ModelParameters, OptimizationScenario, IndependentOptimizer
)

# Create input data
input_data = OptimizationInput(
    impression_channels=['tv', 'digital', 'radio'],
    rf_channels=['social_media', 'display_video'],
    impression_historical_spend=np.array([100000, 80000, 50000]),
    rf_historical_spend=np.array([40000, 30000]),
    # ... other data arrays
)

# Define model parameters
model_params = ModelParameters(
    impression_coefficients=impression_coeffs,
    impression_adstock_params=impression_alpha,
    rf_coefficients=rf_coeffs,
    rf_adstock_params=rf_alpha,
    baseline=baseline_values,
    # ... other parameters
)

# Create optimizer
optimizer = IndependentOptimizer(input_data, model_params)

# Define scenario
scenario = OptimizationScenario(
    scenario_type='fixed_budget',
    total_budget=320000,
    spend_constraint_lower=0.2,  # Allow 20% decrease
    spend_constraint_upper=0.4,  # Allow 40% increase
)

# Run optimization
results = optimizer.optimize(scenario)
print(f"Total ROI: {results.total_roi:.2f}")
print(results.to_dataframe())
```

### File-Based Workflow
```python
from meridian.optimizer import io_utils

# Load data from files
input_data = io_utils.load_optimization_input_from_csv(
    impression_file='impression_data.csv',
    rf_reach_file='rf_reach.csv',
    rf_frequency_file='rf_frequency.csv',
    spend_file='historical_spend.csv',
    population_file='population.csv'
)

# Load model parameters
model_params = io_utils.load_model_parameters_from_json('model_params.json')

# Run optimization
optimizer = IndependentOptimizer(input_data, model_params)
results = optimizer.optimize(scenario)

# Save results
io_utils.save_optimization_results_to_csv(results, 'optimization_results.csv')
```

## Data Requirements

### Input Data Structure
- **Impression data**: Shape (n_geos, n_times, n_impression_channels)
- **R&F reach**: Shape (n_geos, n_times, n_rf_channels)  
- **R&F frequency**: Shape (n_geos, n_times, n_rf_channels)
- **Population**: Shape (n_geos,) - required for proper scaling
- **Historical spend**: Per-channel historical spend values

### Model Parameters
- **Media coefficients**: Geographic and channel-specific effect sizes
- **Adstock parameters**: Carryover decay rates per channel
- **Hill parameters**: EC50 and slope for saturation modeling
- **Baseline**: Geographic baseline effects

## File Structure

```
meridian/optimizer/
├── __init__.py                 # Module exports
├── independent_optimizer.py    # Core optimization classes
├── optimization_core.py        # Hill-climbing algorithm
├── media_transforms.py         # Media transformation functions
├── io_utils.py                # File I/O utilities
├── visualizer.py              # Visualization functions
├── example_usage.py           # Usage examples and tests
├── optimizer_rough_sketch.py  # Development sketches
└── README.md                  # This documentation
```

## Performance Characteristics

- **Speed**: Optimized for fast execution without full model inference
- **Memory**: Efficient handling of large geographic and time series data
- **Scalability**: Supports multiple geos, channels, and time periods
- **Flexibility**: Works with various channel combinations and constraint scenarios

## Integration with Meridian

The independent optimizer can work with existing Meridian models:
1. **Extract parameters**: Use `extract_model_parameters_from_meridian()` to get trained parameters
2. **Run optimization**: Use extracted parameters for fast budget optimization
3. **Compare results**: Validate against Meridian's built-in optimizer

## Future Enhancements

Potential areas for expansion:
- Multi-objective optimization (ROI + reach targets)
- Seasonality-aware optimization
- Control variable integration
- Advanced constraint types (e.g., channel group constraints)
- Probabilistic optimization with uncertainty quantification

---

*This independent optimizer provides a lightweight, fast alternative to running full Meridian model inference when you already have trained model parameters and need to explore budget allocation scenarios.*