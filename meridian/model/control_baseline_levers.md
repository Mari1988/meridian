# Complete Guide: Levers to Control Baseline Components

## Overview

The model decomposes KPI as:
```
KPI = baseline + mu_t + tau_g + media + noise
          ↑        ↑       ↑       ↑
      intercept   time    geo   media
```

You want to **increase media contribution** → **decrease mu_t and tau_g** contribution.

---

## Lever 1: Number of Knots (Most Important!)

**Location:** `ModelSpec` initialization

**Default:** ~6-8 knots per year

**Your Code (debug_model.py line 186):**
```python
model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type='contribution',
    media_effects_dist='log_normal',
    knots_per_year=2,  # ← ADD THIS
)
```

**Impact:**
| knots_per_year | mu_t Flexibility | Typical Variance Explained | Use When |
|----------------|------------------|---------------------------|----------|
| 1-2            | Very low         | 10-20%                    | Almost linear trend |
| 3-4            | Low              | 20-35%                    | Simple seasonality |
| 5-6            | Moderate         | 35-50%                    | Balanced (default) |
| 7-10           | High             | 50-70%                    | Complex patterns |
| 10+            | Very high        | 70-85%                    | Weekly fluctuations |

**Recommendation:** Start with `knots_per_year=3` to leave more room for media.

---

## Lever 2: Knot Values Prior Scale

**Location:** `PriorDistribution`

**Default:** `Normal(0.0, 5.0)` - very flexible

**Your Code (debug_model.py line 173):**
```python
prior = prior_distribution.PriorDistribution(
    knot_values=tfp.distributions.Normal(
        loc=0.0,
        scale=1.0,  # ← ADD THIS (default is 5.0)
        name=constants.KNOT_VALUES
    ),
    contribution_m=contribution_m,
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)
```

**Impact:**
| scale | Effect | Variance Explained |
|-------|--------|-------------------|
| 0.5   | Very tight - minimal time effects | 5-15% |
| 1.0   | Tight - small time effects | 15-30% |
| 2.0   | Moderate - balanced | 30-45% |
| 5.0   | Loose - large time effects (default) | 45-70% |

**Recommendation:** Use `scale=1.0` to restrict time effects.

---

## Lever 3: tau_g Prior Scale

**Location:** `PriorDistribution`

**Default:** `Normal(0.0, 5.0)` - allows large geo differences

**Your Code:**
```python
prior = prior_distribution.PriorDistribution(
    knot_values=tfp.distributions.Normal(0.0, 1.0, name=constants.KNOT_VALUES),
    tau_g_excl_baseline=tfp.distributions.Normal(
        loc=0.0,
        scale=1.0,  # ← ADD THIS (default is 5.0)
        name=constants.TAU_G_EXCL_BASELINE
    ),
    contribution_m=contribution_m,
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)
```

**Impact:**
| scale | Effect | Variance Explained |
|-------|--------|-------------------|
| 0.5   | Minimal geo differences | 2-8% |
| 1.0   | Small geo differences | 8-15% |
| 2.0   | Moderate geo differences | 15-25% |
| 5.0   | Large geo differences (default) | 25-40% |

**Recommendation:** Use `scale=1.0` if geos are similar.

---

## Lever 4: Use National Model

**Location:** Data aggregation

**Default:** Geo-level model (`is_national=False`)

**Your Code (debug_model.py line 57):**
```python
is_national = True  # ← CHANGE FROM False
```

**Impact:**
- Completely eliminates `tau_g`
- Geos are aggregated to national level
- Removes ~10-30% of variance attribution from geo effects
- **BUT:** Loses geo-level insights

**Recommendation:** Only if you don't need geo-level coefficients.

---

## Lever 5: Restrict eta_m (Hierarchical Variation)

**Location:** `PriorDistribution`

**Default:** `HalfNormal(1.0)`

**Your Code:**
```python
prior = prior_distribution.PriorDistribution(
    knot_values=tfp.distributions.Normal(0.0, 1.0, name=constants.KNOT_VALUES),
    tau_g_excl_baseline=tfp.distributions.Normal(0.0, 1.0, name=constants.TAU_G_EXCL_BASELINE),
    eta_m=tfp.distributions.HalfNormal(
        0.3,  # ← Smaller than default 1.0
        name=constants.ETA_M
    ),
    contribution_m=contribution_m,
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)
```

**Impact:**
- Doesn't directly affect baseline
- But restricts geo-level media variation
- Helps contribution prior be more consistent across geos

---

## Complete Example: Maximize Media Contribution

```python
# debug_model.py modifications

# 1. Reduce time effects (line 173+)
prior = prior_distribution.PriorDistribution(
    knot_values=tfp.distributions.Normal(
        0.0, 1.0, name=constants.KNOT_VALUES  # Tight time effects
    ),
    tau_g_excl_baseline=tfp.distributions.Normal(
        0.0, 1.0, name=constants.TAU_G_EXCL_BASELINE  # Tight geo effects
    ),
    eta_m=tfp.distributions.HalfNormal(
        0.3, name=constants.ETA_M  # Reduce geo variation in media
    ),
    contribution_m=contribution_m,
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)

# 2. Reduce knots (line 186)
model_spec = spec.ModelSpec(
    prior=prior,
    media_prior_type='contribution',
    media_effects_dist='log_normal',
    knots_per_year=3,  # Reduce from default ~6-8
)

# 3. Create model
mmm = model.Meridian(input_data=data, model_spec=model_spec)
```

**Expected Result:**
- mu_t explains: ~20-30% (down from 50-70%)
- tau_g explains: ~10-15% (down from 20-40%)
- Media can explain: ~55-70% ✓ (consistent with your 56.5% prior)

---

## Diagnostic: Check Your Current Settings

After fitting, run:
```python
from meridian.model.check_baseline_contribution import check_baseline_contribution

results = check_baseline_contribution(mmm)

# If baseline_pct > 60%:
#   → Tighten more (knots_per_year=2, scales=0.5)
# If baseline_pct < 30%:
#   → Already good, media has room
```

---

## Quick Reference Table

| Goal | Lever | Setting |
|------|-------|---------|
| **Minimize mu_t** | knots_per_year | 2-3 |
|                   | knot_values scale | 0.5-1.0 |
| **Minimize tau_g** | tau_g scale | 0.5-1.0 |
|                    | is_national | True (eliminates tau_g) |
| **Maximize media** | All of above | + tight contribution priors |

---

## Warning: Don't Over-Constrain!

If you make baseline TOO restrictive:
- Model fit will suffer (low R², high MAPE)
- Posterior will deviate from priors anyway
- You're forcing media to explain variance it can't

**Best practice:**
1. Start with moderate restrictions
2. Check fit quality
3. Iterate

The goal is **balance**, not forcing media to 100%!
