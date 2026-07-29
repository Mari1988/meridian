# Archived exploration — superseded, do not quote

Everything in this directory is **superseded**. It is kept for provenance: it
records how the study arrived at its conclusions and what was ruled out along
the way, which is what answers "how did you rule that out?" in Q&A. None of it
should be cited as a current result, and no number in it should appear on a
slide.

The current work lives one level up:

| file | role |
|---|---|
| `../final/roi-vs-mroi-metric-selection.ipynb` | **The landed result.** Prefer its numbers over anything here. |
| `../meridian_ec_prior_case_study.ipynb` | Current — the 20-seed HDI coverage / calibration evidence. |
| `../meridian_tv_underreach_case_study.ipynb` | Current — budget reallocation and noisy-prior robustness. Figures are reduced-precision smoke-test runs. |

## Why these are superseded

**Most of them fit with `media_prior_type='coefficient'`.** Meridian warns
against that parameterization on every fit, and the landed notebook deliberately
switched to the `'roi'` default — using a non-default parameterization while
critiquing default settings was a fair procedural objection. The switch also
*sharpens* the findings rather than softening them, so nothing was lost. But it
means the numbers here do not line up with the landed notebook's and must never
be quoted alongside them.

**The earlier ones predate the current data-generating process.** The
`meridian_simulation_v*` / `simulate_media_data_*` notebooks are DGP development
against `data_simulator_v1_complex.py`, not the current `../data_simulator.py`.

## What each one was for

### Real-data-scaled `coefficient` series
- `tv-test-under-real-demo-data-coefficient-prior.ipynb` — the base real-data-scaled TV/Display/Social scenario.
- `tv-test-under-real-demo-data-display-fix.ipynb` — making the third channel identifiable.
- `tv-test-under-real-demo-data-slope-mismatch.ipynb` — what happens when the true curve is an S-curve, against Meridian's fixed `slope_m=1`.
- `tv-test-under-real-demo-data-roi-shape-coupling.ipynb` — ties true `roi_m` to curve shape instead of assigning it independently. **Its contribution is now folded into the landed notebook** via the simulator's `roi_ec_elasticity` / `roi_alpha_elasticity` config fields, so it is fully superseded rather than merely old. Was previously misnamed `final-tv-test-...` despite not being the final notebook.

### Simplified-DGP `coefficient` series
- `tv-test-under-simple-dgp-coefficient-prior.ipynb` — the base simplified-DGP run.
- `tv-test-under-simple-dgp-coefficient-prior-aks.ipynb` — knots, AKS, and the limits of informative priors.
- `tv-test-under-simple-dgp-coefficient-prior-low-roi.ipynb` — the same at lower target ROI (3.0/2.0/1.0), the robustness check that the finding does not depend on assumed ROI levels.
- `tv-test-under-simple-dgp-coefficient-prior-low-roi-noisy-freq.ipynb` — a further sensitivity variant: target ROI 2.0/1.5/1.0, `frequency_noise_sd` 0.5 (10x the sibling's), tighter Display frequency range, explicit `adstock_retention_range`. Renamed from `... copy.ipynb`, which made a real variant run look like an accidental duplicate.

### Mechanism side-quests
- `over-saturated-social-both-tails.ipynb` — the default prior fails at *both* ends, over-saturation as well as under-reach.
- `social-recovery-under-default-priors.ipynb` — why Social is recoverable under default priors when TV is not.
- `tv-test-under-simple-dgp.ipynb`, `meridian_tv_underreach_simple_dgp_case_study.ipynb` — simplified-DGP precursors to the current case studies.

### Pre-current-DGP
- `meridian_priors_case_study.ipynb` — earlier combined `ec_m`/`alpha_m` prior study.
- `meridian_simulation_v2.ipynb`, `meridian_simulation_v3.ipynb`, `simulate_media_data_reach_frequency_state_geo.ipynb` — DGP development.
- `data_simulator_v1_complex.py` — the earlier, richer simulator these were built against.
- `real_world_simulated_geo_data.csv`, `health_card.html` — data and output artifacts used only by the notebooks above.

## Running these

They import `data_simulator` / `model_utils` from the parent directory, so they
will not import cleanly from here without adding the parent to `sys.path`:

```python
import sys; sys.path.insert(0, '..')
```

They were also written against **older versions of those helpers**. Re-running
one may need the module APIs as they existed at that commit — check `git log`
for `../data_simulator.py` and `../model_utils.py` rather than assuming the
current signatures apply.
