# Contribution Priors Guide for Meridian MMM

> **Reference guide for using contribution priors in Meridian, covering hierarchical model mechanics, parameter interpretation, and troubleshooting.**

## Table of Contents
- [Overview](#overview)
- [How Contribution Priors Work](#how-contribution-priors-work)
- [Setting Up Contribution Priors](#setting-up-contribution-priors)
- [Hierarchical Model Structure](#hierarchical-model-structure)
- [Normal vs LogNormal Effects](#normal-vs-lognormal-effects)
- [Controlling Baseline Components](#controlling-baseline-components)
- [Prior Feasibility Analysis](#prior-feasibility-analysis)
- [Troubleshooting Prior Mismatch](#troubleshooting-prior-mismatch)
- [Best Practices](#best-practices)

---

## Overview

**Contribution priors** allow you to specify prior beliefs about what **percentage of total KPI** each media channel drives.

### When to Use
- You have expert knowledge about channel contributions (e.g., "TV drives ~30% of conversions")
- You want to specify priors in percentage terms rather than ROI or coefficients
- You're building a geo-level hierarchical model

### Key Advantage
Contribution percentages are intuitive for stakeholders and easier to elicit than ROI or coefficient values.

---

## How Contribution Priors Work

### The Calculation Chain

```
contribution_m (sampled from Beta prior)
    ↓
incremental_outcome_m = contribution_m × total_outcome
    ↓
beta_m = f⁻¹(incremental_outcome_m, transformed_media, population, scaling)
    ↓
beta_gm[g] = exp(beta_m + eta_m × N(0,1)[g])  [LogNormal case]
    ↓
Realized contribution % = incremental_outcome / total_outcome
```

### Key Point
`contribution_m` sets the **intended** contribution, but the **realized** contribution depends on:
- Media transformation parameters (alpha_m, ec_m, slope_m)
- Hierarchical variation (eta_m)
- Baseline component priors (mu_t, tau_g)

---

## Setting Up Contribution Priors

### Step 1: Calculate Beta Distribution Parameters

For desired channel contributions (e.g., Display: 11.5%, TV: 30%, Video: 15%):

```python
from meridian.model.contribution_prior_helper import create_contribution_prior

# Define expected contributions
beta_dist_params = create_contribution_prior(
    channel_names=['Display', 'TV', 'Video'],
    mean_contributions={'Display': 11.5, 'TV': 30, 'Video': 15},
    confidence='moderate'  # 'low', 'moderate', or 'high'
)

# Create Beta prior
contribution_m = tfp.distributions.Beta(
    concentration1=build_media_channel_args(**beta_dist_params['alphas']),
    concentration0=build_media_channel_args(**beta_dist_params['betas']),
    name=constants.CONTRIBUTION_M
)
```

### Step 2: Confidence Levels

| Confidence | Std Dev | Example (TV 30%) | When to Use |
|------------|---------|------------------|-------------|
| **Low** | ~15% of mean | 30% ± 4.5% | Uncertain about estimates |
| **Moderate** | ~10% of mean | 30% ± 3.0% | **Recommended** - balanced |
| **High** | ~5% of mean | 30% ± 1.5% | Very confident in estimates |

### Formula: Beta Parameters from Mean and Std

```python
# For mean μ and std σ:
variance = σ²
alpha = μ × (μ(1-μ)/σ² - 1)
beta = alpha × (1-μ)/μ

# Coefficient of Variation (for checking):
CV = sqrt(exp(eta_m²) - 1)  # For LogNormal model
```

---

## Hierarchical Model Structure

### LogNormal Model (Recommended)

```python
# Hierarchical structure:
contribution_m ~ Beta(α, β)           # National-level prior
eta_m ~ HalfNormal(scale)             # Geo variation (must be positive!)
beta_gm_dev[g] ~ Normal(0, 1)         # Non-centered parameterization

# Geo-level coefficients:
beta_gm[g,m] = exp(beta_m[m] + eta_m[m] × beta_gm_dev[g,m])
```

### Key Parameters

#### eta_m (Hierarchical Standard Deviation)
- **Must be positive** → Use `HalfNormal`, NOT `Normal`
- Controls geo-level variation
- For LogNormal: interprets as ~coefficient of variation (CV)

| eta_m | CV | Geo Variation | Recommendation |
|-------|-----|---------------|----------------|
| 0.1 | 10% | Very low | Geos very similar |
| 0.3 | 31% | Moderate | **Recommended for contribution priors** |
| 0.5 | 53% | High | Geos quite different |
| 1.0 | 131% | Very high | Too much variation |

**For contribution priors, use:**
```python
eta_m = tfp.distributions.HalfNormal(0.3, name=constants.ETA_M)
```

#### N(0,1) - Non-Centered Parameterization
- **One independent draw per geo × channel**
- Creates geo-level heterogeneity
- NOT the same as eta_m prior (no collision)
- More efficient MCMC than centered parameterization

**Shapes during MCMC:**
```python
contribution_m:  (n_channels,)              # e.g., (3,) for 3 channels
eta_m:          (n_channels,)              # e.g., (3,)
beta_m:         (n_channels,)              # e.g., (3,)
N(0,1) devs:    (n_geos, n_channels)       # e.g., (5, 3)
beta_gm:        (n_geos, n_channels)       # e.g., (5, 3)
```

---

## Normal vs LogNormal Effects

### Comparison

| Aspect | Normal | LogNormal |
|--------|--------|-----------|
| **Formula** | `beta_gm = beta_m + eta_m × N(0,1)` | `beta_gm = exp(beta_m + eta_m × N(0,1))` |
| **Range** | (-∞, ∞) | (0, ∞) - always positive |
| **eta_m meaning** | Absolute std dev | Std dev on log scale ≈ CV% |
| **Scale dependency** | ❌ Depends on beta_m | ✅ Scale-free |
| **Setting priors** | Harder (need to know scale) | Easier (think in %) |
| **For MMM** | Less common | **Recommended** |

### Why LogNormal is Better for MMM

1. **Scale-free**: eta_m ≈ percentage deviation (independent of beta_m)
2. **Always positive**: Media effects can't be negative
3. **Aligns with contribution priors**: Both think in percentages
4. **Handles geo heterogeneity**: Allows some geos to respond much more strongly

### When to Use Normal

- You have direct knowledge of absolute effect sizes
- You want simpler, symmetric geo variation
- Faster computation needed (slightly)

---

## Controlling Baseline Components

### The Problem

With **loose baseline priors** (defaults):
```
total_outcome = baseline + mu_t + tau_g + media
```
If baseline components vary widely (default: `Normal(0, 5.0)`):
- total_outcome unstable → contribution % = incremental / (varying total) ≠ prior expectation

### Solution: Tighten Baseline Priors

```python
prior = prior_distribution.PriorDistribution(
    # BASELINE CONTROLS (critical for contribution priors!)
    knot_values=tfp.distributions.Normal(0.0, 1.0, name=constants.KNOT_VALUES),
    tau_g_excl_baseline=tfp.distributions.Normal(0.0, 1.0, name=constants.TAU_G_EXCL_BASELINE),

    # HIERARCHICAL CONTROLS
    eta_m=tfp.distributions.HalfNormal(0.3, name=constants.ETA_M),

    # CONTRIBUTION PRIORS
    contribution_m=contribution_m,

    # TRANSFORMATION PARAMETERS
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)

model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type='contribution',
    media_effects_dist='log_normal',
    knots_per_year=3  # Reduce from default ~6-8
)
```

### Baseline Control Levers

| Lever | Default | Recommended | Impact |
|-------|---------|-------------|--------|
| `knots_per_year` | 6-8 | 3-4 | Reduces time effect flexibility |
| `knot_values` scale | 5.0 | 1.0 | Tightens time effect magnitude |
| `tau_g` scale | 5.0 | 1.0 | Reduces geo intercept variation |

### Expected Impact

| Baseline Priors | mu_t + tau_g Explains | Media Room | Prior Contribution Match |
|-----------------|----------------------|------------|--------------------------|
| **Loose (default)** | 50-70% | 30-50% | ±5-10 pct points off |
| **Tight (recommended)** | 30-40% | 60-70% | ±2-3 pct points |

---

## Prior Feasibility Analysis

### Overview

The **Prior Feasibility Checker** validates whether your contribution priors are achievable given your transformation priors (adstock, hill saturation, hierarchical variation).

**Key Question:** "If I specify TV contribution = 30%, what contribution % do I actually get from the prior predictive distribution?"

### Why Use It

1. **Validate Prior Compatibility**: Detect conflicting priors before fitting
2. **Identify Problem Parameters**: Find which transformation parameters drive mismatch
3. **Get Recommendations**: Receive actionable suggestions for prior adjustments
4. **Save Time**: Avoid MCMC iterations only to discover incompatible priors

### Basic Usage

```python
from meridian.model import prior_feasibility

# After creating your model with contribution priors
mmm = model.Meridian(input_data=data, model_spec=model_spec)

# Check prior feasibility
checker = prior_feasibility.PriorFeasibilityChecker(mmm)
report = checker.check_contribution_feasibility(
    n_draws=1000,
    include_sensitivity=True,
    seed=42
)

# Print comprehensive report
print(report)
```

### Example Output

```
================================================================================
PRIOR FEASIBILITY REPORT
================================================================================

Based on 1000 prior draws

CONTRIBUTION % COMPARISON: Target vs Realized Prior
--------------------------------------------------------------------------------
  channel  target_mean_pct  target_std_pct  realized_mean_pct  realized_std_pct  deviation_pct
  Display           11.5             0.6                8.2               3.1           -3.3
       TV           30.0             1.5               22.5               6.8           -7.5
    Video           15.0             0.8               11.3               4.2           -3.7

================================================================================
PARAMETER SENSITIVITY ANALYSIS
--------------------------------------------------------------------------------
  channel  parameter  correlation  variance_explained_pct
       TV    alpha_m        -0.42                   17.6
       TV       ec_m         0.38                   14.4
       TV    slope_m         0.22                    4.8
       TV      eta_m        -0.15                    2.3
  Display    alpha_m        -0.35                   12.3
  Display       ec_m         0.31                    9.6
      ...        ...          ...                     ...

================================================================================
RECOMMENDATIONS
--------------------------------------------------------------------------------
1. Large contribution % mismatch detected. Consider these adjustments:
  1. Tighten baseline priors: knot_values=N(0,1), tau_g=N(0,1), knots_per_year=3
  2. Narrow transformation priors: Use tighter ranges for alpha_m, ec_m, slope_m
  3. Reduce geo heterogeneity: Use smaller eta_m (e.g., HalfNormal(0.2))

2. High variance in realized contribution % detected:
  - TV: std=6.8% (>6.0%)
  → Run analyze_parameter_sensitivity() to identify which priors drive variance

================================================================================
```

### Advanced Usage

#### 1. Parameter Sensitivity Analysis

Identify which transformation parameters drive contribution % variance:

```python
sensitivity_df = checker.analyze_parameter_sensitivity(n_draws=1000, seed=42)
print(sensitivity_df)

# Output shows correlation between each parameter and contribution %
#   channel  parameter  correlation  variance_explained_pct
#        TV    alpha_m        -0.42                   17.6  ← adstock drives 17.6% of variance
#        TV       ec_m         0.38                   14.4  ← saturation drives 14.4%
```

**Interpretation:**
- High correlation → that parameter strongly affects realized contribution %
- High variance_explained → tightening that prior will reduce mismatch

#### 2. Identify Compatible Parameter Regions

Find which transformation parameter values achieve your target contribution:

```python
compatible_regions = checker.identify_compatible_regions(
    channel='TV',
    target_contribution_pct=30.0,
    tolerance_pct=2.0,  # ±2%
    n_draws=1000,
    seed=42
)

print(compatible_regions)
# Output:
# {
#   'alpha_m': (0.45, 0.75),  ← TV achieves 30% when adstock is in this range
#   'ec_m': (0.8, 1.4),       ← and EC50 is in this range
#   'slope_m': (0.6, 1.2),    ← and slope is in this range
#   'eta_m': (0.2, 0.4)       ← and geo variation is in this range
# }
```

**Use case:** Narrow your transformation priors to these compatible regions to improve prior match.

#### 3. Iterative Prior Tuning Workflow

```python
# Step 1: Check initial feasibility
report = checker.check_contribution_feasibility(n_draws=1000)
print(report)

# Step 2: Identify problem parameters
sensitivity_df = checker.analyze_parameter_sensitivity()
print(sensitivity_df[sensitivity_df['variance_explained_pct'] > 10])

# Step 3: Find compatible regions for worst channel
worst_channel = report.comparison_df.iloc[
    report.comparison_df['deviation_pct'].abs().argmax()
]['channel']

compatible_regions = checker.identify_compatible_regions(
    channel=worst_channel,
    target_contribution_pct=30.0,
    tolerance_pct=3.0
)

# Step 4: Update priors with compatible ranges
# (Manually adjust your prior definitions)

# Step 5: Re-check feasibility
# (Repeat until satisfied)
```

### Understanding the Report

#### Comparison Table

| Column | Meaning | Good Sign | Warning Sign |
|--------|---------|-----------|--------------|
| `target_mean_pct` | Expected from Beta prior | - | - |
| `realized_mean_pct` | Actual from prior predictive | Within ±3% of target | >5% deviation |
| `deviation_pct` | Difference | Close to 0 | >±5 |
| `realized_std_pct` | Variance in prior | <20% of mean | >30% of mean |

#### Sensitivity Analysis

| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| `correlation` | Linear relationship | ±0.3+ → strong effect |
| `variance_explained_pct` | R² contribution | >10% → major driver |

**High correlation parameters** are good candidates for tightening.

### Common Patterns

#### Pattern 1: All Channels Low Contribution
```
  channel  target_mean_pct  realized_mean_pct  deviation_pct
  Display           11.5                8.2           -3.3
       TV           30.0               22.5           -7.5
    Video           15.0               11.3           -3.7
```

**Cause:** Baseline priors too loose → baseline explains too much

**Fix:** Tighten `knot_values`, `tau_g`, reduce `knots_per_year`

#### Pattern 2: High Variance in Realized Contribution
```
  channel  realized_mean_pct  realized_std_pct
       TV               30.2              12.8  ← std is 43% of mean!
```

**Cause:** Wide transformation priors create many possible scenarios

**Fix:** Narrow alpha_m, ec_m, slope_m ranges

#### Pattern 3: One Channel Far Off
```
  channel  target_mean_pct  realized_mean_pct  deviation_pct
  Display           11.5               11.3           -0.2  ✓
       TV           30.0               18.5          -11.5  ✗
    Video           15.0               14.8           -0.2  ✓
```

**Cause:** TV transformation priors incompatible with TV contribution prior

**Fix:** Use `identify_compatible_regions(channel='TV')` to find compatible values

### Limitations

1. **Computational Cost**: Requires prior sampling (1000+ draws recommended)
2. **Prior Only**: Doesn't tell you what posterior will be (data matters!)
3. **Linear Sensitivity**: Assumes linear relationships for sensitivity analysis
4. **No Interactions**: Doesn't model joint effects of multiple parameters

### When to Use

✅ **Use Prior Feasibility Checker when:**
- Setting up new model with contribution priors
- Contribution priors are critical to your use case
- You want to validate priors before expensive MCMC
- Debugging why prior doesn't match expectations

❌ **Skip if:**
- Using coefficient or ROI priors (checker requires contribution priors)
- Posterior is what matters (prior mismatch is acceptable)
- Fast iteration needed (just fit the model)

### Integration with Workflow

```python
# Recommended workflow
# 1. Define initial priors
prior = prior_distribution.PriorDistribution(...)
model_spec = spec.ModelSpec(prior=prior, media_prior_type='contribution', ...)
mmm = model.Meridian(input_data=data, model_spec=model_spec)

# 2. Check feasibility
from meridian.model import prior_feasibility
checker = prior_feasibility.PriorFeasibilityChecker(mmm)
report = checker.check_contribution_feasibility(n_draws=1000, include_sensitivity=True)
print(report)

# 3. If needed, adjust priors and re-check
# (Iterate until satisfied)

# 4. Sample posterior
mmm.sample_posterior(n_chains=2, n_adapt=1000, n_burnin=1000, n_keep=2000)

# 5. Verify posterior matches targets
media_summary = visualizer.MediaSummary(mmm)
summary_df = media_summary.summary_table_without_ci(include_posterior=True)
```

---

## Troubleshooting Prior Mismatch

### Expected vs Actual Prior Contributions Don't Match

**Example:**
```
Expected: Display 11.5%, TV 30%, Video 15%
Actual Prior: Display 7.8%, TV 20.3%, Video 10.2%
```

### Causes & Solutions

#### 1. Loose Baseline Priors (Most Common)
**Symptom:** Prior contributions off by 5-10 percentage points

**Fix:**
```python
# Add to PriorDistribution:
knot_values=tfp.distributions.Normal(0.0, 1.0, name=constants.KNOT_VALUES),
tau_g_excl_baseline=tfp.distributions.Normal(0.0, 1.0, name=constants.TAU_G_EXCL_BASELINE),

# Add to ModelSpec:
knots_per_year=3
```

#### 2. Wide Transformation Priors
**Symptom:** High variance in prior contributions across draws

**Explanation:** Each prior draw samples different (alpha_m, ec_m, slope_m) combinations → different realized contributions

**Fix:** Tighten transformation priors if you have domain knowledge:
```python
alpha_m = tfp.distributions.TruncatedNormal(
    [0.5, 0.3, 0.4],  # Expected adstock by channel
    [0.1, 0.1, 0.1],  # Tighter std dev
    0.0, 1.0
)
```

#### 3. Large eta_m (Hierarchical Variation)
**Symptom:** Prior contributions vary due to geo heterogeneity

**Fix:**
```python
eta_m = tfp.distributions.HalfNormal(0.3)  # Instead of 1.0
```

#### 4. Normal vs LogNormal Mismatch
**Symptom:** Model behavior doesn't match expectations

**Fix:** Ensure consistency:
```python
model_spec = spec.ModelSpec(
    ...,
    media_effects_dist='log_normal'  # Match with HalfNormal eta_m
)
```

### When Mismatch is Normal (and OK!)

**Prior contributions will ALWAYS be noisier than posterior** because:
1. Wide transformation priors → many parameter combinations
2. No data to constrain parameters yet
3. Prior predictive samples all possible scenarios

**This is EXPECTED and CORRECT behavior!**

### What Matters: Posterior

**Check posterior contributions:**
```python
media_effects_summary = visualizer.MediaSummary(mmm)
summary_df = media_effects_summary.summary_table_without_ci(include_prior=True)

# Look at distribution='posterior' rows
```

**Good signs:**
- Posterior contributions close to prior targets (within 2-5%)
- Posterior tighter than prior
- R² > 0.7, MAPE < 15%

**Warning signs:**
- Posterior very far from targets (>10% off)
- Prior and posterior equally noisy
- Poor model fit (low R², high MAPE)

---

## Best Practices

### 1. Contribution Prior Specification

✅ **DO:**
- Use **moderate confidence** by default (±3% std dev)
- Set total contribution to 40-70% (leave room for baseline)
- Use domain knowledge from previous models or stakeholder input

❌ **DON'T:**
- Use overly tight priors (high confidence) unless very certain
- Expect 100% of KPI from media (baseline + time effects explain 30-50%)
- Use `Beta(1, 99)` defaults (only 1% per channel)

### 2. Hierarchical Parameters

✅ **DO:**
```python
eta_m = tfp.distributions.HalfNormal(0.3)  # Moderate geo variation
```

❌ **DON'T:**
```python
eta_m = tfp.distributions.Normal(0, 1)  # WRONG! Can be negative
eta_m = tfp.distributions.HalfNormal(1.0)  # Too loose (131% CV)
```

### 3. Baseline Controls

✅ **DO:**
```python
# Tight baseline priors
knot_values=tfp.distributions.Normal(0.0, 1.0),
tau_g_excl_baseline=tfp.distributions.Normal(0.0, 1.0),
knots_per_year=3
```

❌ **DON'T:**
```python
# Use defaults (too loose for contribution priors)
model_spec = spec.ModelSpec(prior=prior, ...)  # Missing controls
```

### 4. Model Specification

✅ **DO:**
```python
model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type='contribution',
    media_effects_dist='log_normal',  # Recommended
    knots_per_year=3
)
```

❌ **DON'T:**
```python
media_effects_dist='normal'  # Unless you have specific reasons
# (Less suitable for contribution priors)
```

### 5. Diagnostic Workflow

**After fitting:**

1. **Check prior vs posterior contributions:**
   ```python
   summary_df = media_effects_summary.summary_table_without_ci(include_prior=True)
   ```

2. **Assess model fit:**
   ```python
   diagnostics = visualizer.ModelDiagnostics(mmm)
   diagnostics.predictive_accuracy_table()
   ```

3. **Review convergence:**
   ```python
   import arviz as az
   az.summary(mmm.inference_data.posterior)
   # Check r_hat < 1.1, ess_bulk > 400
   ```

4. **If posterior far from targets:**
   - Check if data supports your prior expectations
   - Review transformation parameter posteriors
   - Consider if baseline is too flexible

---

## Quick Reference

### Complete Example

```python
from meridian.model.contribution_prior_helper import create_contribution_prior

# 1. Define contribution priors
beta_dist_params = create_contribution_prior(
    channel_names=['Display', 'TV', 'Video'],
    mean_contributions={'Display': 11.5, 'TV': 30, 'Video': 15},
    confidence='moderate'
)

contribution_m = tfp.distributions.Beta(
    concentration1=build_media_channel_args(**beta_dist_params['alphas']),
    concentration0=build_media_channel_args(**beta_dist_params['betas']),
    name=constants.CONTRIBUTION_M
)

# 2. Set up priors
prior = prior_distribution.PriorDistribution(
    # Baseline controls
    knot_values=tfp.distributions.Normal(0.0, 1.0, name=constants.KNOT_VALUES),
    tau_g_excl_baseline=tfp.distributions.Normal(0.0, 1.0, name=constants.TAU_G_EXCL_BASELINE),

    # Hierarchical controls
    eta_m=tfp.distributions.HalfNormal(0.3, name=constants.ETA_M),

    # Contribution priors
    contribution_m=contribution_m,

    # Transformation parameters
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)

# 3. Create model
model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type='contribution',
    media_effects_dist='log_normal',
    knots_per_year=3
)

mmm = model.Meridian(input_data=data, model_spec=model_spec)
mmm.sample_prior(2000)
mmm.sample_posterior(n_chains=2, n_adapt=1000, n_burnin=1000, n_keep=2000)
```

### Parameter Summary

| Parameter | Type | Support | Typical Value | Purpose |
|-----------|------|---------|---------------|---------|
| `contribution_m` | Beta | [0, 1] | Beta(α, β) per channel | Set expected contribution % |
| `eta_m` | HalfNormal | [0, ∞) | 0.3 | Control geo variation |
| `knot_values` | Normal | ℝ | N(0, 1.0) | Control time effect magnitude |
| `tau_g` | Normal | ℝ | N(0, 1.0) | Control geo intercept variation |
| `knots_per_year` | Integer | ℕ | 3-4 | Control time effect flexibility |

---

## Common Questions

### Q: Why is my prior contribution % different from what I specified?

**A:** Prior contributions are calculated, not sampled directly. Mismatch is normal due to:
- Wide transformation parameter priors (alpha, ec, slope)
- Hierarchical variation (eta_m)
- Loose baseline priors

**Solution:** Check posterior (should be closer). Tighten baseline priors if needed.

### Q: Should I use Normal or LogNormal effects distribution?

**A:** **LogNormal is recommended** for contribution priors because:
- Scale-free interpretation (eta_m ≈ CV%)
- Always positive effects
- Better aligns with percentage-based thinking

### Q: What confidence level should I use?

**A:** **Moderate** is recommended:
- Provides guidance without being too restrictive
- Std dev ≈ 10% of mean
- Allows data to update beliefs

Use **high** only if very confident. Use **low** if uncertain.

### Q: Can I use eta_m ~ Normal(0, 1)?

**A:** **NO!** eta_m is a standard deviation → must be positive.

**Always use:**
```python
eta_m ~ HalfNormal(scale)  # ✓
eta_m ~ Exponential(rate)  # ✓
eta_m ~ Gamma(α, β)        # ✓
```

**Never use:**
```python
eta_m ~ Normal(0, 1)  # ✗ Can be negative!
```

### Q: Why does posterior match better than prior?

**A:** **This is Bayesian inference working correctly!**

- **Prior:** Wide parameter ranges → noisy contributions
- **Posterior:** Data constrains parameters → tight contributions

The likelihood forces parameters to jointly optimize:
1. Respect contribution_m prior
2. Fit observed data

Result: Posterior contributions close to targets.

---

## Related Documentation

- `contribution_prior_helper.py` - Helper functions for Beta parameters
- `prior_feasibility.py` - Prior feasibility checker module
- `prior_distribution.py:243-306` - PriorDistribution class and defaults
- `prior_sampler.py:115-146` - Contribution → beta_m calculation
- `model.py:1168-1248` - calculate_beta_x function
- `media.py:78-161` - MediaTensors and prior_denominator

---

**Last Updated:** 2025-10-03
**Author:** Discussion between User and Claude Code
**Version:** 1.0
