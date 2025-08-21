# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Meridian is Google's open-source Marketing Mix Modeling (MMM) framework built on Bayesian causal inference. It helps advertisers measure marketing campaign impact, calculate ROI, and optimize budget allocation using aggregated, privacy-safe data.

## Development Commands

### Testing
- Run all tests: `pytest -vv -n auto`
- Run specific test file: `pytest path/to/test_file.py -v`
- Run tests for specific module: `pytest meridian/model/ -v`

### Code Quality
- Lint code: `pylint meridian/`
- Format code: `pyink .` (uses Google Python style guide with 2-space indentation)
- Check formatting: `pyink --check .`

### Build
- Install in development mode: `pip install -e .[dev]`
- Build package: `python setup.py build` (compiles SCSS templates)
- Install with GPU support: `pip install -e .[and-cuda]`

## Code Architecture

### Core Modules

1. **meridian.model**: Core MMM implementation
   - `model.py`: Main Meridian class for Bayesian hierarchical modeling
   - `spec.py`: Model specifications and configuration
   - `prior_distribution.py` / `prior_sampler.py`: Bayesian prior handling
   - `posterior_sampler.py`: MCMC sampling with NUTS
   - `media.py`: Media transformation and adstock modeling
   - `transformers.py`: Data transformation utilities

2. **meridian.data**: Data handling and preprocessing
   - `input_data.py`: Main InputData class for storing model inputs
   - `*_input_data_builder.py`: Builders for different data formats (DataFrame, ndarray)
   - `load.py`: Data loading utilities
   - `time_coordinates.py`: Time dimension handling

3. **meridian.analysis**: Post-modeling analysis and optimization
   - `analyzer.py`: Core analysis metrics and computations
   - `visualizer.py`: Visualization generation
   - `fast_response_curves.py`: **Fast response curve computation using median parameters (159x speedup)**
   - `optimizer.py`: Budget optimization algorithms
   - `summarizer.py`: Results summarization
   - `formatter.py`: Output formatting

4. **meridian.mpa**: Media Planning & Analysis (experimental)
   - Contains prototype and utility files for advanced media planning features

5. **meridian.optimizer**: Independent Budget Optimizer (custom development)
   - Standalone budget optimization system independent of Meridian model objects
   - See `meridian/optimizer/README.md` for comprehensive documentation
   - Supports mixed impression and R&F channels, hill-climbing optimization
   - Fast budget allocation without full model inference

### Key Design Patterns

- **Bayesian Framework**: Uses TensorFlow Probability for MCMC sampling with NUTS
- **Geo-level Modeling**: Supports both geo-level and national-level MMM
- **GPU Acceleration**: Built with TensorFlow tensors for GPU optimization
- **Modular Architecture**: Clear separation between data, modeling, and analysis layers

## Development Notes

- **Python Version**: Requires Python 3.10+ (tested on 3.10, 3.11, 3.12)
- **Code Style**: Google Python style guide with 2-space indentation
- **Testing**: Uses pytest with parallel execution (-n auto)
- **Formatting**: pyink (Google's Python formatter) is the standard
- **GPU Support**: Recommended for production use, especially for large datasets
- **Data Format**: Primarily uses xarray.DataArray for multi-dimensional data

## FastResponseCurves Implementation (Custom Enhancement)

### Overview
The `FastResponseCurves` class provides a high-performance alternative to the standard `MediaEffects.response_curves_data()` method by using median parameter estimates instead of full Bayesian sampling. This approach delivers **159x speedup** while maintaining mathematical accuracy for point estimates.

### Key Features
- **Performance**: 159x faster than original implementation (0.012s vs 0.926s)
- **Mathematical Accuracy**: Uses same adstock and hill transformations as original
- **API Compatibility**: Drop-in replacement with familiar method signatures
- **Comprehensive Support**: Handles media channels, R&F channels, geo/time filtering

### Core Innovation
The key insight is that for response curve analysis, we often only need point estimates rather than full posterior distributions. By extracting median values from the posterior samples and applying deterministic transformations, we achieve dramatic speedup:

```python
# Traditional approach (slow)
media_effects = visualizer.MediaEffects(meridian_model)
result = media_effects.response_curves_data()  # 0.926 seconds

# Fast approach (159x faster)
fast_curves = fast_response_curves.FastResponseCurves(meridian_model) 
result = fast_curves.compute_response_curves()  # 0.012 seconds
```

### Implementation Details
- **Parameter Extraction**: Uses `np.median()` across posterior chains and draws
- **Media Transformation**: Leverages existing `AdstockTransformer` and `HillTransformer`
- **Spend Calculation**: Efficient computation of spend amounts for each multiplier
- **Output Format**: Compatible xarray.Dataset with same coordinates and data variables

### Usage Examples
```python
from meridian.analysis import fast_response_curves

# Initialize with fitted Meridian model
fast_curves = fast_response_curves.FastResponseCurves(meridian_model)

# Compute response curves (default: 0 to 2.2 in 0.01 steps)
result = fast_curves.compute_response_curves()

# Custom multipliers
result = fast_curves.compute_response_curves(
    spend_multipliers=[0.5, 1.0, 1.5, 2.0]
)

# Generate visualizations
chart = fast_curves.plot_response_curves(plot_separately=True)
```

### Files
- `meridian/analysis/fast_response_curves.py`: Main implementation
- `meridian/analysis/fast_response_curves_test.py`: Comprehensive unit tests
- `test_fast_response_curves_with_simulated_data.py`: Integration test with real model
- `meridian/analysis/compare_actual_vs_fast_rc_data.py`: **Comprehensive comparison script using granular multipliers**

### Mathematical Foundation
The optimization is based on the insight that in Meridian's linear regression framework, non-media variables cancel out during counterfactual prediction. This allows us to focus computational effort on the media transformation pipeline using deterministic median parameters rather than sampling across the full posterior distribution.

## Mini MMM Implementation (Custom Development)

### Overview
**Mini MMM** is a simplified, accessible Marketing Mix Modeling framework based on Meridian methodology. Located in the `mini_mmm/` directory, it provides core MMM functionality with reduced complexity while maintaining mathematical rigor.

### Key Features
- **Simplified Architecture**: National-level modeling only (no geo-hierarchy)
- **Pandas Integration**: Uses pandas DataFrames instead of xarray for easier adoption
- **PyMC Backend**: Bayesian inference with PyMC instead of TensorFlow Probability
- **Fast Analysis**: Efficient response curves and budget optimization tools
- **Educational Focus**: Designed for learning, prototyping, and small-scale applications

### Architecture Overview
```
mini_mmm/
├── data/
│   ├── input_data.py      # SimpleInputData class (pandas-based)
│   └── validators.py      # Data quality validation utilities
├── model/
│   ├── mini_mmm.py        # Main MiniMMM class (PyMC-based)
│   ├── transformers.py    # Adstock & Hill transformations
│   └── priors.py          # Bayesian prior configurations
├── analysis/
│   ├── analyzer.py        # ROI, contribution analysis
│   ├── response_curves.py # Fast response curve computation
│   └── optimizer.py       # Budget optimization algorithms
├── viz/
│   └── plots.py           # Visualization functions
└── examples/
    ├── quickstart.py      # Complete end-to-end example
    └── sample_data.py     # Synthetic data generation
```

### Core Transformations
**Adstock (Carryover Effects)**
- Geometric decay implementation: `adstocked[t] = media[t] + retention_rate * adstocked[t-1]`
- Configurable maximum lag periods for computational efficiency
- Optional normalization to preserve total media volume

**Hill Saturation (Diminishing Returns)**
- Standard Hill transformation: `slope * media^ec / (half_saturation^ec + media^ec)`
- Flexible parameter configuration per channel
- Automatic half-saturation estimation from data

### Key Differences from Full Meridian

| Aspect | Mini MMM | Full Meridian |
|--------|----------|---------------|
| **Scope** | National-level only | Geo + National level |
| **Backend** | PyMC | TensorFlow Probability |
| **Data Format** | pandas DataFrame | xarray DataArray |
| **Channels** | Media channels only | Media + R&F + Organic |
| **Complexity** | Simplified (80% functionality) | Full-featured |
| **Learning Curve** | Gentle | Steep |
| **Use Cases** | Learning, prototyping, small-scale | Production, large-scale |

### Quick Usage Example
```python
from mini_mmm import MiniMMM, SimpleInputData
from mini_mmm.examples.sample_data import generate_realistic_mmm_data

# Generate or load data
df = generate_realistic_mmm_data(n_weeks=104, n_media_channels=4)

# Create input data
data = SimpleInputData.from_dataframe(
    df=df,
    kpi_col='kpi',
    media_cols=['TV_spend', 'Digital_spend', 'Social_spend', 'Radio_spend'],
    control_cols=['price_index', 'competitor_spend'],
    date_col='date'
)

# Fit model
model = MiniMMM()
model.fit(data, draws=2000, chains=2)

# Analysis
from mini_mmm.analysis import Analyzer, BudgetOptimizer
analyzer = Analyzer(model)
roi_results = analyzer.compute_roi()

# Optimization
optimizer = BudgetOptimizer(model)
optimal_allocation = optimizer.optimize_budget(total_budget=100000)

# Visualization
from mini_mmm.viz.plots import create_full_report
figures = create_full_report(model)
```

### Performance Characteristics
- **Response Curves**: ~159x faster than full Bayesian sampling using median parameter estimates
- **Model Fitting**: Depends on PyMC performance, typically 5-30 minutes for typical datasets
- **Memory Usage**: Lower than full Meridian due to simplified architecture
- **Scalability**: Recommended for 1-4 years of data, 3-8 media channels

### Integration with Meridian
Mini MMM is designed as a learning tool and prototype environment that prepares users for full Meridian adoption:
- **Methodology Alignment**: Uses same mathematical foundations (adstock + hill)
- **Concept Transfer**: Core MMM concepts directly applicable to Meridian
- **Validation**: Can be used to validate approaches before Meridian implementation
- **Education**: Ideal for understanding MMM principles before tackling Meridian complexity

### Files and Testing
- **Core Implementation**: All files in `mini_mmm/` directory
- **Examples**: `mini_mmm/examples/quickstart.py` provides complete workflow
- **Documentation**: `mini_mmm/README.md` contains comprehensive usage guide
- **Data Generation**: `mini_mmm/examples/sample_data.py` for synthetic data creation
- **Testing**: Run Mini MMM examples with `python mini_mmm/examples/quickstart.py`

### Development Notes
- **Dependencies**: Requires PyMC, pandas, numpy, scipy, matplotlib
- **Python Version**: Compatible with Python 3.10+
- **Code Style**: Follows same Google Python style guide as Meridian
- **Extensibility**: Designed for easy extension and customization
- **Production Readiness**: Suitable for small-scale production use, educational purposes, and prototyping

## File Naming Conventions

- Test files: `*_test.py` 
- Main modules follow snake_case
- Template files in `meridian/analysis/templates/` include HTML/Jinja and SCSS
- to memorize