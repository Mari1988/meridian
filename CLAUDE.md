# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Meridian (`google-meridian` on PyPI) is Google's open-source Marketing Mix Modeling (MMM) library: a geo-level Bayesian hierarchical model (via TensorFlow Probability NUTS/MCMC sampling) that estimates marketing channel ROI and supports budget optimization. It is largely developed internally at Google and mirrored to GitHub; external PRs are rarely merged directly (see `CONTRIBUTING.md`).

## Setup

Requires Python 3.11–3.13. Install the local proto package first, then the core package editable with the extras you need:

```sh
pip install -e proto --config-settings editable_mode=strict
pip install -e .[dev,jax,mlflow,schema,scenarioplanner] --config-settings editable_mode=strict
```

Extras (`pyproject.toml`): `dev` (pytest, pylint, pyink), `colab`, `and-cuda` (GPU TF), `mlflow`, `jax` (JAX backend deps), `schema` (proto schema serialization), `scenarioplanner`.

## Common commands

```sh
pytest                      # run full test suite (test files are named *_test.py, not test_*.py)
pytest meridian/model/model_test.py                    # single test file
pytest meridian/model/model_test.py -k test_name        # single test
pytest -n auto               # parallel via pytest-xdist
MERIDIAN_BACKEND=jax pytest  # run the suite against the JAX backend instead of the TensorFlow default
pyink meridian scenarioplanner              # format (Google style, 2-space indent, 80 cols; see [tool.pyink])
pylint meridian scenarioplanner --rcfile=.pylintrc      # lint
```

Tests use `absl.testing.absltest`/`parameterized`, not plain `unittest`, and are run through pytest. CI (`.github/workflows/ci.yml`) runs the matrix `python={3.11,3.12,3.13} x backend={tensorflow,jax}`; a commit message containing `#skip-pytest` skips the pytest job.

## Architecture

### Backend abstraction (`meridian/backend/`)

Meridian runs on either TensorFlow (+ TFP) or JAX (+ TFP-on-JAX substrate), selected via the `MERIDIAN_BACKEND` env var (`tensorflow` default, or `jax`) and read once at import time in `backend/config.py`. Do not call `backend.set_backend()` after other Meridian modules have been imported — it will not retroactively change already-bound references.

All model/analysis code must go through `meridian.backend` (imported as `backend`) rather than importing `tensorflow`/`jax` directly, so it works under both backends. `backend/__init__.py` exposes a NumPy-like standardized API (`backend.reduce_sum`, `backend.einsum`, `backend.gather`, `backend.to_tensor`, `backend.tfd`, `backend.mcmc`, `backend.RNGHandler`, ...) that dispatches to backend-specific private implementations (`_jax_*` / `_tf_*` functions) defined in that same file. When adding an op used by model code, add it here rather than branching on backend elsewhere. `RNGHandler` abstracts JAX's stateless `PRNGKey` splitting vs. TensorFlow's stateless-seed model behind one interface.

### Core model pipeline (`meridian/model/`)

- `spec.py` — `ModelSpec`: user-facing model configuration (priors, saturation/adstock settings, etc.).
- `prior_distribution.py` — Bayesian prior definitions.
- `context.py` — `ModelContext`: derived, immutable state built from `InputData` + `ModelSpec` (transformed/scaled tensors, knots, adstock/hill parameters) that analysis code reads from.
- `adstock_hill.py`, `equations.py`, `transformers.py`, `media.py`, `knots.py` — the actual MMM math (adstock decay, Hill saturation curves, scaling/transform pipelines, time-knot handling for splines).
- `prior_sampler.py` / `posterior_sampler.py` — prior predictive sampling and MCMC posterior sampling (NUTS), including out-of-memory/error handling (`MCMCSamplingError`, `MCMCOOMError`).
- `model.py` — the `Meridian` class: the top-level entry point that owns an `InputData`, builds a `ModelContext`, runs sampling, and exposes `save_mmm`/`load_mmm` for persistence.
- `eda/` — the EDA (exploratory data analysis) engine (`eda_engine.py`, `sampling_eda_engine.py`) that runs pre-fit data-quality checks (e.g. data-to-parameter ratio) and produces `eda_outcome` results, driven by an `eda_spec`.

### Data layer (`meridian/data/`)

`input_data.py` defines `InputData`, the xarray-based container for all model inputs (KPI, media, reach/frequency, controls, population, revenue-per-KPI, non-media treatments — see the name constants in `meridian/constants.py`). `input_data_builder.py` plus `data_frame_input_data_builder.py`/`nd_array_input_data_builder.py` provide a builder API for incrementally assembling and validating an `InputData` from pandas DataFrames or raw arrays before construction. `time_coordinates.py` (`TimeCoordinates`) centralizes date/time-interval handling, including non-uniform cadences (calendar-monthly, quarterly). `validator.py` performs input validation; `load.py` handles loading from files (e.g. the simulated `.xlsx` data in `data/simulated_data/`).

### Analysis & reporting (`meridian/analysis/`)

- `analyzer.py` — computes analysis metrics (ROI, response curves, contribution, etc.) from a fitted `Meridian` model; `tensors.py` holds the `DataTensors`/`DistributionTensors` shapes it operates on.
- `visualizer.py` — Altair-based plots.
- `optimizer.py` — budget optimization scenarios.
- `summarizer.py` + `summary_text.py` + `meridian/templates/` (Jinja2) — generates the 2-page HTML model summary report.
- `review/` — the "Model Quality Checks" framework: `checks.py` defines individual checks (e.g. Bayesian posterior-predictive checks), `configs.py`/`constants.py` configure them, `reviewer.py` runs a set of checks against a model, and `results.py` defines the `CheckResult` result types consumed by `model.py` and reporting.

### Schema / interchange (`meridian/schema/`, `proto/`)

`proto/mmm/v1/mmm.proto` defines the `Mmm` proto schema (published separately to PyPI as `mmm-proto-schema`, versioned independently with `proto-v*` tags — see `proto/CHANGELOG.md`). `meridian/schema/model_consumer.py` and `mmm_proto_generator.py` convert a trained `Meridian` model (core model + analysis/processor outputs) into that proto. `schema/processors/` holds one processor per proto section (model kernel, model fit, marketing, budget optimization, reach/frequency optimization). `schema/serde/` implements serialization/deserialization of model internals (priors, distributions, inference data, marketing data) to/from the proto representation. `schema/utils/` has shared helpers (date-range bucketing, proto enum conversion, time records).

### Other integrations

- `meridian/mlflow/autolog.py` — optional MLflow experiment-tracking integration; opt-in via `autolog.autolog()`.
- `scenarioplanner/` — converters between Meridian model output and the Scenario Planner (Sheets/Looker Studio) surface: `converters/dataframe/` (DataFrame-based converters for budget/R&F optimization and marketing analyses), `converters/sheets.py`, `converters/mmm.py`/`mmm_converter.py`, and `linkingapi/` (URL generation for linking into the Scenario Planner UI).
- `demo/` — Colab notebooks (`.ipynb`) demonstrating getting-started flows for both backends, MLflow, reach & frequency, and scenario planning; these are the canonical end-to-end usage examples. See `demo/synthetic/` below for an ongoing prior-recovery investigation distinct from the getting-started notebooks.

### Synthetic ground-truth prior-recovery study (`demo/synthetic/`)

An ongoing investigation into whether Meridian's default `ec_m` (Hill half-saturation) prior — `TruncatedNormal(0.8, 0.8, [0.1, 10])`, identical for every channel (`meridian/model/prior_distribution.py`) — systematically misdiagnoses under-invested channels as already-near-saturated, and whether a reach/frequency-informed prior fixes it. Uses a from-scratch synthetic geo x time data-generating process with fully known ground truth (not real client data), so a fitted Meridian model's posterior can be checked against the exact parameters used to generate the data, rather than against an unknowable real-world truth.

#### THE LANDED STATE — read this before anything else in this section

The ARF Analytics Council talk was **delivered 2026-08-04**. Everything below
this block is background, provenance, or superseded; this is what the delivered
deck actually rests on, and where follow-up work should start.

**Configuration.** Well-specified baseline
(`realistic_baseline.RealisticBaselineConfig(mu_ar1_sd=0.0, seasonal_amplitude=0.0)`
— `mu_t` is a linear trend the 8-knot spline spans exactly, with the noise
recalibrated so oracle R² stays at 0.9), basis `r90_basis.ALPHA03` (Channel-1
`ec_m` 9.0 / `alpha_m` 0.30 / `roi_m` 6.6405; Channel-2 1.3 / 0.15 / 6.0237, all
pinned and asserted on every `build_scenario()` call), **50 draws**
(`r90_basis.SEEDS_50`, whose first ten are `SEEDS` so it is a strict superset of
the old 10-draw results and comparable draw-for-draw). Two arms: `default`
(Meridian out of the box, `max_lag=8`) vs `ec_alpha_noisy` (both anchors
perturbed ~25% per seed, `max_lag=13`). Sampler
`n_chains=2, n_adapt=500, n_burnin=500, n_keep=1000, seed=1`.

**Reproduce it** (nothing here is a notebook — see below):

```sh
demo/synthetic/run_wellspec_50.sh          # ~90 min, 100 fits, 10 seeds per process
demo/synthetic/run_wellspec_mroi_50.sh     # ~2 h, NO MCMC: rebuilds scenarios, reattaches posteriors
.venv/bin/python demo/synthetic/scratch_extract_r90_wellspec_ch2.py   # Channel-2 recovery, seconds
```

**Objects.** `fitted_models/scratch_ablation_r90_wellspec/` holds
`per_seed_all.csv` (Channel-1), `per_seed_all_ch2.csv` (Channel-2),
`mroi_both_channels.csv` and `true_params_by_seed.csv` — **these are tracked in
git**, so every figure and every assertion rebuilds without re-fitting. The 100
`.nc` posteriors under `models/` (~273MB) are **not** tracked; regenerate from
the script above.

**Figures and deck.**

```sh
.venv/bin/python demo/synthetic/scratch_plot_wellspec_recovery.py \
    --run-dir=demo/synthetic/fitted_models/scratch_ablation_r90_wellspec \
    --out-prefix=wellspec_slide7 --no-header        # -> figures/wellspec_slide7_ch{1,2}.png
.venv/bin/python demo/synthetic/scratch_plot_mroi_r90_slide8_pinned.py \
    --run-dir=demo/synthetic/fitted_models/scratch_ablation_r90_wellspec \
    --out=demo/synthetic/figures/wellspec_mroi_50.png \
    --informed=ec_alpha_noisy                       # --band=90 for the 90% variant
```

`demo/synthetic/arf_deck/` builds slides 8–12 and reads every number from those
CSVs at build time. **The delivered `.pptx` is not in this repo** — it lives in
the author's OneDrive as `ARF-Analytics-Council-Talk-v2.pptx` (12 slides). The
`arf_section1_deck.pptx` that *is* in this repo is the earlier 8-slide Section-1
deck, a different artifact.

**Headline numbers, Channel-1 at 50 draws.** `ec_m` default −65.4%
[−70.9, −59.1], low on all 50; informed −0.1% [−37, +76], closer on 48/50 —
centred, **not** precise. `roi_m` +14.2% → +0.8% (41/50). `alpha_m` +0.6% vs
−0.2%, informed winning 18/50, i.e. indistinguishable. mROI −5.8% at 1x →
−61.6% at 10x, with the default's 90% interval missing the truth on 49/50
datasets at 2x and 50/50 from 3x upward. Channel-2 is clean throughout — that
contrast is the argument.

**What is NOT the landed state, and trips people up.** *No notebook reproduces
any of this* — the deck came from scripts, and every notebook in this directory
is on a superseded basis (realistic baseline, or `alpha_m=0.8`/`oracle_r2=0.80`,
or the 10-draw pinned run). `scratch_ablation_r90_pinned/` backed the earlier
10-draw version of the deck and is kept only so those figures stay rebuildable.
Anything in `archive/` is exploratory and must not be quoted.

- `data_simulator.py` — `GeoMediaDataSimulator`/`SimulationConfig`: builds geo x time media/KPI data end-to-end (real U.S. state populations, reach x frequency media execution with seasonality/flighting/AR(1) noise, adstock + Hill transforms via Meridian's own `meridian.model.adstock_hill` classes, per-channel ROI calibration). `simulate_adstock_hill_params()` derives the ground-truth `ec_m` from a stated "half of target audience reached at the channel's mean frequency" assumption. **This is a stylized modeling choice for generating a self-consistent ground truth, not a claim about real-world half-saturation** — every "true `ec_m`" reference in the notebooks below means "true under this stated assumption," not an externally validated threshold. The `roi_ec_elasticity` / `roi_alpha_elasticity` config fields optionally tie a channel's true `roi_m` to its curve shape (faster-saturating channels convert better per exposure) rather than assigning ROI as an independent assumption; `saturation_frequency` moves `ec_m` linearly while leaving media execution untouched, which is what makes controlled single-variable comparisons possible.
- `model_utils.py` — helpers shared by the notebooks below: `build_simulated_input` / `build_real_augmented_input` (run the full simulator pipeline, the latter scaling it onto the real `geo_media_rf.csv` demo data); `build_reach_based_ec_prior` / `build_alpha_prior` / `PRIOR_VARIANTS` / `build_model_spec` (construct the `default` / `ec_only` / `ec_noisy` / `ec_alpha_only` / `ec_alpha_social_tight` prior variants — `ec_noisy`'s prior `scale` grows with its `audience_noise_scale` uncertainty rather than staying fixed, so a noisier audience/reach assumption yields an appropriately less confident prior); `build_comparison_table` (true-vs-fitted recovery table with HDI coverage marks); and Hill-curve diagnostics (`hill_value`, `ceiling_fraction_at_median`, `posterior_param_mean`).

**Which notebook to read — this ordering matters, and NONE of them backs the
delivered deck (see the landed-state block above; the deck came from scripts).
Read these for provenance and for the side-results they alone carry:**

- `realistic-baseline-noise-2ch.ipynb` + `realistic-baseline-noise-2ch-seeds.ipynb` — **the interactive precursors to the scripted r90 arms. They backed the deck at an earlier stage and no longer do** (the delivered deck is the well-specified 50-draw run in the landed-state block above; where the two disagree, that block wins). They still supersede `final/roi-vs-mroi-metric-selection.ipynb` wherever those two disagree, because that notebook's DGP flattered the model in two ways since corrected: its baseline was drawn from the model's own spline basis (`n_knots_mu_t=8` against a fitted `knots=8`, so exactly recoverable) and its residual was iid, giving an unrealistic R² of 0.996. The realistic DGP (`realistic_baseline.py`) uses a baseline the fitted spline structurally cannot represent plus persistent, cross-geo-correlated noise, calibrated to an *oracle R²* of 0.80 — the ceiling for Meridian's mean structure given true media, which is the metric to quote rather than `1 - var(eps)/var(kpi)`. **`max_lag` is now informed the same way `ec_m`/`alpha_m` are, not held fixed across variants:** `default` fits at Meridian's real out-of-the-box `max_lag` (8, `model_utils.MERIDIAN_DEFAULT_MAX_LAG`), `ec_alpha_only` (informed) fits at the DGP's own window (`config.max_lag`, 13) — see `MAX_LAG_BY_VARIANT` in `export_curve_data.py` / `seed_sweep_worker.py`. TV `ec_m` −89.8% under `default` vs −0.8% informed (seed 1320); across ten draws `default` ranges −89.8 to −65.8%, informed −4.7 to +3.1% — no overlap, 10/10 informed wins; the default's mROI credible band excludes the truth from 2x spend upward. **Three findings that only appear here:** (1) `alpha_m` recovery moves with `max_lag`, not with the `alpha_m` prior — under the old shared window both variants failed `alpha_m` alike (~−6.5% median, informed winning only 3/10 draws), but with `default` now truncated to `max_lag=8` its median `alpha_m` error moved to −0.10% (closer to truth than informed's −6.30%) while informed win-rate barely moved (4/10) — the window is a genuine, separate lever, independent of the prior, and does *not* mean `default`'s `alpha_m` handling is better (`ec_m`/`roi_m` are far worse on every draw); (2) ROI does not replicate cleanly (informed wins only 7/10 draws post-split, own error spans −27.7% to +78.6%); (3) a deeply saturated channel becomes confounded with the baseline when the baseline is misspecified (why Social was dropped; see `realistic-baseline-noise.ipynb` §6). A scratch single-seed check at true `alpha_m=0.3` (fast decay) found no `max_lag`-driven `alpha_m` movement, consistent with negligible true carryover past week 8 at that decay rate — not yet folded into a formal sweep. **Retired:** the `ec9`-vs-`ec11` invariance check (never read by `results_facts.py`) is no longer run in these two notebooks; the compute budget went to the `max_lag` split instead. The still-standing invariance result lives in `final/roi-vs-mroi-metric-selection.ipynb` §6 (below), on the superseded DGP.
- `final/roi-vs-mroi-metric-selection.ipynb` — **the landed result, and still the authority on metric *choice*; but its DGP is superseded, so prefer the realistic-baseline numbers above.** Establishes which metric exposes the default prior's failure and which conceals it: `ec_m` fails unambiguously while `roi_m` misses only narrowly, and mROI *at current spend* is the least discriminating metric of all (the ROI overstatement and elasticity understatement partly cancel). The sharp discriminator is mROI at elevated spend. Its strongest result is the **invariance check** in §6: two scenarios differing only in true `ec_m`, with bit-identical media execution, where the `default` posterior absorbs 7% of an 18.7% move in the truth and the informed prior 99% — i.e. the default is reporting its prior, not estimating the parameter.
- `meridian_ec_prior_case_study.ipynb` — the narrow base case: fits `default` vs. `ec_only` on one simulated dataset, then repeats across 20 seeds reporting 90% HDI coverage of `ec_m`/`roi_m`. Still current; it is the study's calibration evidence.
- `meridian_tv_underreach_case_study.ipynb` — a sharper scenario built on the same `ec_m` definition: a TV channel with a large target audience but deliberately low current reach *and* low frequency (two independent, compounding forms of under-delivery), so current execution sits far below the DGP's saturation threshold. Adds a "fraction of ceiling effect captured today" headline diagnostic, a response-curve overlay, a `BudgetOptimizer`-vs-true-DGP-optimal budget-reallocation comparison, and a robustness check (`ec_noisy`) showing the reach-informed prior doesn't need to be exact to outperform default. Still current, but its figures come from **reduced-precision smoke-test MCMC settings** — re-run at full settings before quoting.
- `archive/` — **the notebooks here were REMOVED on 2026-08-04** (17 files, 3.8MB; 14 of them `coefficient`-parameterized, so their numbers contradict the delivered deck's — browsable contradictions in a public repo are an accident waiting to happen). They remain in git history at `fe929e4` and restore with `git checkout fe929e4 -- demo/synthetic/archive/`. What stays is `archive/README.md`, which records what each notebook was and why it was superseded — that record is the part that answers "how did you rule that out?", and it survives without shipping the files. Also kept: `data_simulator_v1_complex.py` (cited by the current `data_simulator.py`) and `real_world_simulated_geo_data.csv`. Two traps the README still documents:
  - **Nine of them fit with `media_prior_type='coefficient'`**, which Meridian warns against on every fit and which the landed notebook deliberately abandoned in favour of the `'roi'` default. Their numbers do not line up with the landed notebook's and must not be quoted alongside them.
  - `archive/tv-test-under-real-demo-data-roi-shape-coupling.ipynb` was previously named `final-tv-test-...` despite not being final. It is `coefficient`-parameterized, and its one distinctive contribution — coupling true `roi_m` to curve shape rather than assigning it independently — is now folded into the landed notebook via the simulator's `roi_ec_elasticity` / `roi_alpha_elasticity` config fields.
  - The archived notebooks import `data_simulator` / `model_utils` from the parent directory and were written against older versions of those helpers, so they need `sys.path` set and may not run against the current APIs.
- `ARF_COUNCIL_TALK_OUTLINE.md` (repo root) — outline for presenting these findings to the ARF Analytics Council (session: Tue Aug 4, 2026). It is the authority on which numbers are current and which are superseded; update as the talk is prepared/delivered.

### The ARF deck: `demo/synthetic/arf_section1_deck.pptx`

**This is the earlier eight-slide Section-1 deck, NOT the one delivered on
2026-08-04.** The delivered deck is twelve slides, lives in the author's
OneDrive, and is built by `demo/synthetic/arf_deck/` off the well-specified
50-draw run — see the landed-state block above. Everything in this section
describes the Section-1 artifact and the 10-draw pinned basis it renders from,
which is still the authority for slides 1-6 (the defaults and the DGP).

Ten slides, all generated — **never hand-edit the `.pptx`**, it is overwritten on every
build. Slides 1–4 cover Meridian's defaults; slides 5–9 the results. The adstock slide
(4) is deliberately gentler than the other two: `alpha_m ~ Uniform(0, 1)` genuinely is
flat, so it skips the "hidden skew" framing and instead makes the flat prior's half-life
implications legible, then shows the real binding assumption, `max_lag`, which isn't a
prior at all.

**Eight slides as of 2026-08-01.** Two were cut: the seed-stability slide (10, "Is this
one lucky dataset?"), because slide 7's box plot tells the same story; and the
response-curve slide (9), because slide 8's mROI panels make the same point in the
planner's own units and make it on two channels. Watch the one thing slide 10 carried
alone — *do not present any single ROI figure as characteristic* — which now has to be
said on slide 8 or not at all.

**All results slides are on the PINNED basis (`r90_basis.py`), 10 seeds.** This replaced
an earlier sweep that varied two things at once. `data_simulator` computes
`target_roi_m = base * (ec_m/ec_gmean)**roi_ec_elasticity * (alpha_m/alpha_gmean)**roi_alpha_elasticity`
with **both geometric means taken across channels**, so Channel-2's per-seed `alpha_m`
draw was moving **Channel-1's true ROI** (6.60 / 10.88 / 8.35). The ROI calibration is
exact — the *target* was moving. A seed sweep whose estimand moves cannot separate "the
estimator is noisy" from "the estimator behaves differently at a different truth".
`r90_basis` pins `ec_m`, `alpha_m` and (consequently) `roi_m` for both channels and
asserts all of them on every `build_scenario()` call; only the noise realization varies.
Never reintroduce a local `BASE_OVERRIDES` copy in a scratch script — that duplication is
how the bug survived.

**The informed arm is `ec_alpha_noisy`, not `ec_alpha_only`.** Both anchors are perturbed
~25% per seed (`build_alpha_prior` gained `benchmark_noise_scale` for this). A prior
centred on the exact truth is unbiased by construction, so beating the default with it
proves little. The cost is visible and must be stated: informed is centred
(−4% median `ec_m`, 10/10 wins) but **wide** (−53% to +43% per dataset). Say it removes
the systematic bias; never that it "recovers" the parameter.

**The deck ships with empty speaker-notes fields.** The notes text is still in
`build_section1_deck.py`, suppressed by `EMIT_NOTES = False`, because it is the study's
do-not-overclaim record; the user writes the delivered notes by hand. Read the `_notes()`
blocks before changing any results slide, exactly as before — they just no longer reach
the `.pptx`.

**Channels are presented as Channel-1 / Channel-2**, mapped at render time by
`demo/synthetic/channel_labels.py`. `TV` / `Display` remain the real keys everywhere they
carry meaning — `SimulationConfig.channel_names`, the `rf_source_map` / `plain_source_map`
wiring, the `media_channel` coordinate inside every saved `.nc`, and every run CSV column.
Never rename those; it would invalidate the fitted models on disk. Render through
`channel_labels.label()` instead.

Build chain, and where each stage's numbers come from:

| stage | slides 1–4 | slide 6 | slides 5, 7, 8 |
|---|---|---|---|
| numbers | `prior_plots.default_prior_facts()` (analytic) | `results_facts.py` | `_r90_*_facts()` in `build_section1_deck.py` |
| figures | `build_section1_figures.py` | `build_results_figures.py` | `scratch_build_slide5_r90.py`, `scratch_plot_*_r90_*.py` |
| assertions | `prior_plots_check.py` | same, `check_results_facts()` | ditto |
| assembly | `build_section1_deck.py` (builds all eight) | ditto | ditto |

```sh
.venv/bin/python demo/synthetic/export_curve_data.py       # ~5 min: 2 fits, only if the curve CSVs are missing
.venv/bin/python demo/synthetic/build_section1_figures.py                     # slides 1-4
.venv/bin/python demo/synthetic/build_results_figures.py                      # slide 6
.venv/bin/python demo/synthetic/scratch_build_slide5_r90.py                   # slide 5 (~40s, no fit)
.venv/bin/python demo/synthetic/scratch_plot_recovery_boxplot_r90_fulldraws.py  # slide 7
.venv/bin/python demo/synthetic/scratch_plot_mroi_r90_slide8_pinned.py          # slide 8
.venv/bin/python demo/synthetic/prior_plots_check.py       # assert every quoted number
.venv/bin/python demo/synthetic/build_section1_deck.py
```

**Slides 5, 7, 8 and 9 are on the r90 basis, not the canonical one.** They use
`alpha_m=0.3`, `oracle_r2=0.9`, 3 seeds (1320, 7, 42) — *not* the
`alpha_m=0.8`/`oracle_r2=0.80` run the "Provenance" paragraph below describes. That basis,
why it was chosen over 0.80/0.99, and where its data/scripts live are recorded in Claude's
memory (`project_deck_basis_r90_alpha03`); the short version: `scratch_ablation_r90.py`
(ec_m/roi_m/alpha_m point recovery), `scratch_check_mroi_recovery_r90.py` (mROI at
elevated spend) and `scratch_build_slide5_r90.py` (slide 5's DGP facts + sales/media
series; data generation only, no fit) write summary CSVs — and, for the first two, `.nc`
`InferenceData` — under `fitted_models/scratch_ablation_r90/`,
`fitted_models/scratch_check_mroi_recovery_r90/` and `fitted_models/scratch_slide5_r90/`
(none git-tracked — regenerate from the scripts if missing). The figure scripts read those
CSVs; the response-curve one reconstructs `Meridian`/`Analyzer` objects from the saved
`InferenceData` rather than refitting (`Meridian.__init__` takes `inference_data`
directly). `build_section1_deck.py` has four local facts functions (`_r90_dgp_facts`,
`_r90_recovery_facts`, `_r90_mroi_facts`, `_r90_response_curve_facts`) that read those
CSVs directly, bypassing `results_facts.py` for those four slides.

**Which truths are pinned across draws, and which are not** — this trips people up.
Channel-1's `ec_m` (9.0) and `alpha_m` (0.3) are held fixed in every seed by construction:
a two-pass solve rescales `saturation_frequency` until `ec_m` lands on 9.0, and
`adstock_retention_range` pins `alpha_m` to a degenerate `(0.3, 0.3)`. Both are asserted in
every script that builds a scenario. **Everything else moves per draw** — `roi_m` most of
all (Channel-1: 6.60 / 10.88 / 8.35 across seeds 1320/7/42), because ROI is re-derived from
whatever data each draw produces. Channel-2's `alpha_m` swings 0.006 to 0.156.
`scratch_export_r90_truths.py` writes every channel's truth per seed to
`true_params_by_seed.csv`; slide 5 prints a single value only where a quantity is pinned
and a range otherwise (`_fmt_true`), and `prior_plots_check.py` fails if a quantity that
moves is ever rendered as one number. Slide 7's boxes score each draw against **its own
seed's** truth before pooling, which is why the zero line is clean even though the truths
differ. Anywhere `alpha_m` is compared for Channel-2, use percentage **points**, not
percent — a ratio against 0.006 is meaningless.

`check_results_facts()` in `prior_plots_check.py` now asserts against **the r90 facts
functions the slides actually render from**, so an assertion cannot pass while a slide says
something else. It also guards the one cross-basis dependency left: slide 6 still reads
`results_facts` for its ceiling fractions, which is only valid while the two bases agree
on `ec_m` (they do — 9.0 and 1.29), and the check fails loudly if they ever diverge. The
slide-8 "band clears the truth at 3x" guard survives the move, now requiring it in **every**
seed; the instruction there is still to cut the slide, not soften it.

**Provenance of the results slides.** Every number traces to a CSV under
`demo/synthetic/fitted_models/` — nothing on a slide is a typed-in constant, so changing a
number means re-running the producer, not editing text. What follows describes the
canonical run, which now backs **slide 6 only**; slides 5, 7, 8 and 9 are on the r90 basis
above. `results_facts.py` loads it, and stays deliberately scoped to it.
`realistic_baseline_2ch/` holds the single-draw run (seed 1320, from
`realistic-baseline-noise-2ch.ipynb`, plus `response_curves.csv` / `mroi_sweep.csv` from
`export_curve_data.py`); `realistic_baseline_2ch_seeds/` holds the ten-draw sweep (from
`realistic-baseline-noise-2ch-seeds.ipynb` via `seed_sweep_worker.py`, one subprocess per
draw — forty fits in one kernel exhausts memory and kills it). **These CSVs are tracked**
via a scoped `.gitignore` exception; the `.nc` inference data alongside them is not, so
`export_curve_data.py` must re-run if the curve CSVs are ever deleted.

**The well-specified-baseline ablation (`scratch_ablation_r90_wellspec.py`) — read this
before quoting any ROI number.** A second arm, run 2026-08-02 on the same pinned truths,
same seeds, same sampler, same seeded anchor perturbation; the *only* thing that moves is
the baseline. `r90_basis.build_scenario()` now takes a `realism` argument
(`RealisticBaselineConfig`) and threads it to `build_real_augmented_realistic` — do the
threading there, never by rebuilding a realism config in a caller script, for the same
reason `BASE_OVERRIDES` lives in that module. The ablation passes `mu_ar1_sd=0.0` and
`seasonal_amplitude=0.0`, leaving `mu_t` as a linear trend the 8-knot spline spans
exactly (measured unrepresentable variance 3.471 → 0.00000; Channel-1's correlation with
that unrepresentable part +0.176 → +0.000). `calibrate_to_oracle_r2` scales the
geo-idiosyncratic residual up (noise scale 1.87 → ~2.4) so oracle R² stays at 0.9 — total
noise is preserved, only its character moves from partly-national to entirely per-geo.
Outputs (not git-tracked) in `fitted_models/scratch_ablation_r90_wellspec/`: per-seed
CSVs, `mroi_*.csv`, saved `.nc`, plus `per_seed_all_ch2.csv` (Channel-2, extracted from
the saved posteriors — the script itself records index 0 only).

```sh
.venv/bin/python demo/synthetic/scratch_ablation_r90_wellspec.py --all   # ~30 min, 20 fits
.venv/bin/python demo/synthetic/scratch_extract_r90_wellspec_ch2.py      # Channel-2, no refit
.venv/bin/python demo/synthetic/scratch_plot_wellspec_recovery.py        # both channels
.venv/bin/python demo/synthetic/scratch_plot_wellspec_mroi.py            # mROI, Ch-1 only
```

`--seeds=a,b,c` tops up an existing run without refitting the rest; `--report` re-prints
against the realistic-baseline run for the same seeds. The recovery figures are on a fixed
±80% y-axis so the two channels read against each other — **do not narrow it to ±60**,
which clips nine of the ten default `ec_m` points (they run −59.6 to −68.3%). Their box
labels are **medians, not means**: the informed `ec_m` errors are right-skewed (mean
+13.5%, median +0.7%), so a mean label would contradict the median line the box draws.
The mROI figure covers Channel-1 only — the ablation records `[..., 0]` for mROI, so
there is no Channel-2 control panel without extending it.

**What it establishes, and it changes how ROI must be quoted.** Channel-1's ROI
overstatement decomposes into a saturation-driven part and a baseline-driven part:
default +40.3% = +21.9% (saturation) + 18.4% (remainder); informed +20.6% = +1.2% +
19.3%. The remainders are the same size and correlate across seeds at **+0.99**, i.e.
they are a property of the data draw, not the prior — media absorbs baseline movement the
spline cannot represent, and `beta_m >= 0` rectifies a sign-random confound into a
systematic overstatement. Removing it: **default ROI +40.3% → +15.4%, informed +20.6% →
+1.9%** (sd also collapses, 16.6 → 7.4), while **`ec_m` barely moves: default −67.7% →
−64.2%**. So `ec_m` recovery is a prior problem and ROI recovery is mostly a baseline
problem. **Roughly two-thirds of the default's ROI overstatement is baseline
misspecification, not the prior** — never quote +40% as a prior effect without saying
which baseline it is on. `alpha_m` is also mostly a baseline artifact on Channel-1 (+15.2%
→ +1.2% default): a persistent national AR(1) shock (phi 0.6, ~2.5-week correlation
length) is read as carryover.

**Two results that cut against the study, both to be stated rather than buried.** (1) With
the confound gone the informed arm's `ec_m` tracks its anchor at **r = 0.96, slope 1.19**
(up from r = 0.80, slope 0.98) — removing baseline noise made the posterior rely *more* on
the prior. That strengthens "the model relays your saturation assumption" and simultaneously
kills any claim that the informed prior *recovers* `ec_m`; its accuracy is inherited.
(2) On Channel-2 (the well-reached control) the default `ec_m` is fine, which is exactly
what makes Channel-1 an indictment of the prior rather than of the model.

**This arm is now 50 draws, and that RETRACTED a Channel-2 finding — read before quoting
any Channel-2 number.** Run 2026-08-04 via `run_wellspec_50.sh` (`SEEDS_50` in
`r90_basis.py`: the original ten first, so 50 is a strict superset). At ten draws Channel-2
appeared to miss badly on `alpha_m` (default −22.1%, informed −14.0%) and on ROI (−6.3% in
both arms), and this file recorded that miss as unexplained with the `max_lag` split as
the suspect. **Both were small-sample artifacts.** At 50 draws Channel-2 is clean across
the board: `ec_m` +0.6% / +0.9%, `roi_m` −2.2% / −0.5%, `alpha_m` −4.5% / +2.3%. The first
ten seeds reproduce the old numbers exactly, so nothing is wrong with the extraction —
those ten were simply unrepresentative.

**`max_lag` was never a viable explanation and should not be re-raised.** At Channel-2's
true `alpha_m` of 0.15, 99.999996% of the adstock mass sits in lags 0–8, so the 8-vs-13
split is worth four parts in a hundred million. (At Channel-1's 0.30 it is 99.998%; only
at 0.8 does it bite, 9.4% — which is what the `alpha08` basis exists for.) An
adstock/saturation trade-off was also tested and **not** supported: the per-seed
correlation between `ec_m` and `alpha_m` error on Channel-2 looked like −0.30 to −0.44 at
n=10 but is −0.09 (p≈0.5) at n=50.

**Channel-1 at 50 draws** (this is what the deck's slides 8/9 and its takeaways slide now
render): `ec_m` default −65.4% [−70.9, −59.1], low on all 50; informed −0.1% [−37, +76],
closer on 48/50. `roi_m` default +14.2%, informed +0.8% (41/50). `alpha_m` +0.6% vs −0.2%,
informed winning 18/50 — indistinguishable, as at ten draws.

**A glob bug was fixed at the same time, and it can silently corrupt any arm.** The
`per_seed_*.csv` glob in `report()` also matched `per_seed_all_ch2.csv`, folding ten
Channel-2 rows into the Channel-1 table — 50 seeds reported as 60, with contaminated
medians. Fixed in `scratch_ablation_r90.py`, `_wellspec.py` and `_centred.py` to exclude
every `per_seed_all*` aggregate. Never name a per-channel aggregate `per_seed_*`.

**The deck's results section IS now on this arm (2026-08-04).** Slides 8, 9, 10 and the
takeaways slide render from `scratch_ablation_r90_wellspec/` at 50 draws, via
`run_wellspec_50.sh` (fits) and `run_wellspec_mroi_50.sh` (two-channel mROI, no MCMC --
it reattaches the saved posteriors, ~2 min/seed to rebuild each scenario).
`scratch_mroi_both_channels_r90.py` gained a REQUIRED `--realism=` flag: it used to
rebuild the realistic baseline unconditionally, so pointing it at well-specified
posteriors would have paired those fits with realistic data and scored them against the
wrong truth. It also now reads its seed list from the run's own `per_seed_all.csv`
instead of the 10-entry `r90_basis.SEEDS`, and writes per-seed parts into a
`mroi_both_<realism>/` subdirectory so they escape the `per_seed_*` / `mroi_*` globs.

**mROI on this arm, 50 draws** (Channel-1 default): -5.8% at 1x, -26.0% at 2x, -37.0% at
3x, -61.6% at 10x; the 90% interval misses the truth on 49/50 datasets at 2x and
**50/50 from 3x upward**. Informed stays flat, -1.3% to -3.5%. Channel-2 is within ~2.5%
in both arms at every multiplier. This is sharper than the realistic baseline (which
excluded only from 5x), so the `band clears the truth at 3x` guard in
`prior_plots_check.py` can be tightened to 2x for this arm. NOTE the informed arm's 90%
intervals cover on only ~41/50 (82%) -- mildly under-covering, worth stating if coverage
is quoted.

**The older single-arm framing below is superseded but kept for provenance.**

**The deck is NOT rebuilt on this arm.** Every slide still renders from
`scratch_ablation_r90_pinned/`. The intended framing is a *pair* — the saturation effect
surviving both a realistic and a perfectly-specified baseline is stronger than either
alone, and the well-specified arm removes the "your simulation was adversarial" objection.
If the ablation ever does back a slide, note that on it the default's mROI credible band
excludes the truth on **0/10 datasets from 2x spend upward** (vs from 5x on the realistic
baseline), so the `band clears the truth at 3x` guard in `prior_plots_check.py` would need
tightening to 2x rather than left as-is.

Before changing any results slide, read the `_notes()` blocks in
`build_section1_deck.py` — they carry the do-not-overclaim guidance (lead on `ec_m` and
quote its across-draw range; never present a single ROI figure as characteristic). They no
longer reach the `.pptx` (see `EMIT_NOTES` above) but are still the record.

**Seeding, easy to get wrong:** `config.seed_num` (default 1320) is the *only* DGP seed.
`GeoMediaDataSimulator.__init__` calls `tf.random.set_seed(seed_num)`, overriding any
caller-set seed, and `data_simulator.py` makes no `np.random` calls — so the
`np.random.seed(SIM_SEED)` / `tf.random.set_seed(SIM_SEED)` lines still present in the
older study notebooks do nothing. Pass `seed_num` to vary a draw; pass `None` to let the
caller own seeding.

## Conventions

- Formatting: `pyink` (Google's Black fork), 2-space indentation, 80-column lines, majority-quote style (`[tool.pyink]` in `pyproject.toml`). Enforced/configured for editors via `.vscode/settings.json`.
- Linting: `pylint` with the repo's `.pylintrc` (note: 2-space `indent-string`, `max-line-length=80` — matches pyink, not PEP 8 defaults).
- Test files are named `<module>_test.py` (Google convention), using `absl.testing.absltest`/`parameterized`, executed via `pytest`.
- Constants (input data field names, coordinate names, colors, etc.) are centralized in `meridian/constants.py` — check there before hardcoding string keys used across `data`/`model`/`analysis`.
- Changes are tracked in `CHANGELOG.md` under `[Unreleased]` (Keep a Changelog style); update it for user-visible fixes/features.
