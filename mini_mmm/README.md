# Mini MMM - Simplified Marketing Mix Modeling

A lightweight, accessible implementation of Marketing Mix Modeling (MMM) based on Google's Meridian methodology. Mini MMM provides the core functionality of advanced MMM frameworks while maintaining simplicity and ease of use.

## 🚀 Key Features

- **Simplified Architecture**: National-level modeling focused on core MMM functionality
- **Bayesian Framework**: MCMC sampling with PyMC for robust parameter estimation
- **Media Transformations**: Adstock (carryover) and Hill saturation curves
- **Fast Analysis**: Efficient response curves and budget optimization
- **Rich Visualizations**: Comprehensive plotting functions for insights
- **Easy to Use**: Pandas-friendly data handling and intuitive API

## 📦 Installation

```bash
# Core dependencies
pip install numpy pandas scipy matplotlib seaborn

# Bayesian modeling backend
pip install pymc arviz

# Optional: for advanced optimization
pip install scikit-optimize
```

## 🎯 Quick Start

```python
import pandas as pd
from mini_mmm import MiniMMM, SimpleInputData
from mini_mmm.examples.sample_data import generate_realistic_mmm_data

# 1. Generate or load your data
df = generate_realistic_mmm_data(n_weeks=104, n_media_channels=4)

# 2. Create input data object
data = SimpleInputData.from_dataframe(
    df=df,
    kpi_col='kpi',
    media_cols=['TV_spend', 'Digital_spend', 'Social_spend', 'Radio_spend'],
    control_cols=['price_index', 'competitor_spend'],
    date_col='date'
)

# 3. Fit the model
model = MiniMMM()
model.fit(data, draws=2000, chains=2)

# 4. Analyze results
from mini_mmm.analysis import Analyzer, ResponseCurves, BudgetOptimizer

analyzer = Analyzer(model)
roi_results = analyzer.compute_roi()
print(roi_results)

# 5. Optimize budget
optimizer = BudgetOptimizer(model)
optimal_budget = optimizer.optimize_budget(total_budget=100000)
print(optimal_budget['allocation'])

# 6. Generate visualizations
from mini_mmm.viz.plots import create_full_report
figures = create_full_report(model)
```

## 🏗️ Architecture Overview

### Core Components

1. **Data Layer** (`mini_mmm.data`)
   - `SimpleInputData`: Pandas-based data container
   - `validators`: Data quality checks and validation

2. **Model Layer** (`mini_mmm.model`) 
   - `MiniMMM`: Main Bayesian MMM model
   - `transformers`: Adstock and Hill transformation functions
   - `priors`: Configurable prior distributions

3. **Analysis Layer** (`mini_mmm.analysis`)
   - `Analyzer`: ROI, contribution, and incremental analysis
   - `ResponseCurves`: Fast response curve computation  
   - `BudgetOptimizer`: Budget allocation optimization

4. **Visualization Layer** (`mini_mmm.viz`)
   - `plots`: Comprehensive plotting functions
   - Model diagnostics, media effects, response curves

## 🔧 Key Concepts

### Media Transformations

**Adstock (Carryover Effects)**
```python
from mini_mmm.model.transformers import AdstockTransformer

# Apply adstock transformation
adstocked_media = AdstockTransformer.transform(
    media=spend_data,
    retention_rate=0.5,  # 50% carryover rate
    max_lag=8           # Consider up to 8 periods
)
```

**Hill Saturation (Diminishing Returns)**
```python
from mini_mmm.model.transformers import HillTransformer

# Apply Hill saturation
saturated_media = HillTransformer.transform(
    media=adstocked_media,
    ec=1.5,     # Shape parameter
    slope=2.0,  # Maximum effect
    half_saturation=50  # Half-saturation point
)
```

### Prior Configuration

```python
from mini_mmm.model.priors import DefaultPriors, PriorConfig

# Use pre-configured priors
conservative_priors = DefaultPriors.conservative()
aggressive_priors = DefaultPriors.aggressive()

# Or create custom priors
custom_priors = PriorConfig(
    retention_rate_prior=(2.0, 2.0),  # Beta(2,2) 
    roi_prior=(0.69, 0.5),            # LogNormal
    ec_prior=(0.7, 0.3)               # LogNormal
)

model = MiniMMM(prior_config=custom_priors)
```

### Analysis & Optimization

**ROI Analysis**
```python
analyzer = Analyzer(model)

# Overall ROI by channel
roi_results = analyzer.compute_roi()

# Marginal ROI (incremental returns)
mroi_results = analyzer.compute_mroi()

# Contribution analysis
contribution = analyzer.compute_contribution()
```

**Response Curves**
```python
response_curves = ResponseCurves(model)

# Compute response curves
curves = response_curves.compute_response_curves(
    spend_multipliers=[0.5, 1.0, 1.5, 2.0]
)

# Saturation analysis
saturation = response_curves.compute_saturation_summary()
```

**Budget Optimization**
```python
optimizer = BudgetOptimizer(model)

# Optimize for maximum total effect
result = optimizer.optimize_budget(
    total_budget=100000,
    objective='total_effect',
    method='scipy'
)

# Compare scenarios
scenarios = {
    'current': current_allocation,
    'optimized': result['allocation']['optimal_spend'].values
}
comparison = optimizer.compare_scenarios(scenarios)
```

## 📊 Visualization Examples

**Model Fit Diagnostics**
```python
from mini_mmm.viz.plots import plot_model_fit
fig = plot_model_fit(model)
```

**Media Effects Analysis**
```python
from mini_mmm.viz.plots import plot_media_effects
fig = plot_media_effects(model)
```

**Response Curves**
```python
from mini_mmm.viz.plots import plot_response_curves
fig = plot_response_curves(model)
```

**Full Report Generation**
```python
from mini_mmm.viz.plots import create_full_report
figures = create_full_report(model, save_path='mmm_analysis')
```

## 🎨 Comparison with Meridian

| Feature | Mini MMM | Meridian |
|---------|----------|----------|
| **Scope** | National-level | Geo-level + National |
| **Backend** | PyMC | TensorFlow Probability |
| **Channels** | Media only | Media + R&F + Organic |
| **Data Format** | Pandas DataFrame | xarray DataArray |
| **Complexity** | Simplified | Full-featured |
| **Learning Curve** | Gentle | Steep |
| **Use Cases** | Learning, prototyping, small-scale | Production, large-scale |

## 📖 Examples & Tutorials

**Basic Examples**
- `examples/quickstart.py`: Complete end-to-end example
- `examples/sample_data.py`: Synthetic data generation utilities

**Advanced Usage**
```python
# Custom prior configuration
from mini_mmm.model.priors import PriorConfig

# Data-driven priors
data_driven_priors = DefaultPriors.data_driven(
    kpi_mean=data.get_kpi_array().mean(),
    kpi_std=data.get_kpi_array().std(), 
    media_spend_means=data.get_media_matrix().mean(axis=0)
)

# Model with custom configuration
model = MiniMMM(
    prior_config=data_driven_priors,
    adstock_max_lag=12,  # Longer memory
    random_seed=42
)
```

## 🧪 Data Requirements

**Minimum Requirements**
- Time series: ≥52 weeks (1 year recommended)
- Media channels: 2-10 channels
- KPI: Non-negative, reasonable variance
- Media spend: Non-negative, some variation

**Data Quality Checks**
```python
from mini_mmm.data.validators import validate_input_data, check_data_quality

# Comprehensive validation
warnings = validate_input_data(data)
is_valid = check_data_quality(data, min_weeks=52)

# Get preprocessing suggestions
from mini_mmm.data.validators import suggest_preprocessing
suggestions = suggest_preprocessing(data)
```

## 🚀 Performance & Scalability

**Speed Comparison (vs full Bayesian sampling)**
- Response curves: ~159x faster using point estimates
- Budget optimization: Efficient scipy-based algorithms
- Model fitting: Depends on PyMC performance

**Recommended Specifications**
- **Small scale**: 1-3 years, 3-5 channels → ~5-10 minutes
- **Medium scale**: 2-4 years, 5-8 channels → ~15-30 minutes  
- **Large scale**: 3+ years, 8+ channels → ~30+ minutes

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional media transformation functions
- More optimization algorithms
- Extended visualization options
- R&F channel support
- Performance optimizations

## 📄 License

Licensed under the Apache License 2.0, following the Meridian project.

## 🙏 Acknowledgments

This project is heavily inspired by Google's [Meridian](https://github.com/google/meridian) MMM framework. Mini MMM aims to make the core Meridian methodology more accessible while maintaining mathematical rigor.

## 📚 References

- [Meridian GitHub Repository](https://github.com/google/meridian)
- [Marketing Mix Modeling Guide](https://developers.google.com/marketing-platform/mmm)
- [Bayesian Media Mix Modeling](https://research.google/pubs/pub45998/)
- [PyMC Documentation](https://docs.pymc.io/)

---

**Happy Modeling! 📈**