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

- `data_simulator.py` — `GeoMediaDataSimulator`/`SimulationConfig`: builds geo x time media/KPI data end-to-end (real U.S. state populations, reach x frequency media execution with seasonality/flighting/AR(1) noise, adstock + Hill transforms via Meridian's own `meridian.model.adstock_hill` classes, per-channel ROI calibration). `simulate_adstock_hill_params()` derives the ground-truth `ec_m` from a stated "half of target audience reached at the channel's mean frequency" assumption. **This is a stylized modeling choice for generating a self-consistent ground truth, not a claim about real-world half-saturation** — every "true `ec_m`" reference in the notebooks below means "true under this stated assumption," not an externally validated threshold.
- `model_utils.py` — helpers shared by both notebooks below: `build_simulated_input` (runs the full simulator pipeline); `build_reach_based_ec_prior` / `PRIOR_VARIANTS` / `build_model_spec` (construct `default` / `ec_only` / `ec_noisy` prior variants — `ec_noisy`'s prior `scale` grows with its `audience_noise_scale` uncertainty rather than staying fixed, so a noisier audience/reach assumption yields an appropriately less confident prior); and Hill-curve diagnostics (`hill_value`, `ceiling_fraction_at_median`, `posterior_param_mean`).
- `meridian_ec_prior_case_study.ipynb` — the base case study: fits `default` vs. `ec_only` on one simulated dataset, then repeats across 20 seeds reporting 90% HDI coverage of `ec_m`/`roi_m`.
- `meridian_tv_underreach_case_study.ipynb` — a sharper scenario built on the same `ec_m` definition: a TV channel with a large target audience but deliberately low current reach *and* low frequency (two independent, compounding forms of under-delivery), so current execution sits far below the DGP's saturation threshold. Adds a "fraction of ceiling effect captured today" headline diagnostic, a response-curve overlay, a `BudgetOptimizer`-vs-true-DGP-optimal budget-reallocation comparison, and a robustness check (`ec_noisy`) showing the reach-informed prior doesn't need to be exact to outperform default.
- `ARF_COUNCIL_TALK_OUTLINE.md` (repo root) — outline for presenting these findings to the ARF Analytics Council; update as the talk is prepared/delivered.

## Conventions

- Formatting: `pyink` (Google's Black fork), 2-space indentation, 80-column lines, majority-quote style (`[tool.pyink]` in `pyproject.toml`). Enforced/configured for editors via `.vscode/settings.json`.
- Linting: `pylint` with the repo's `.pylintrc` (note: 2-space `indent-string`, `max-line-length=80` — matches pyink, not PEP 8 defaults).
- Test files are named `<module>_test.py` (Google convention), using `absl.testing.absltest`/`parameterized`, executed via `pytest`.
- Constants (input data field names, coordinate names, colors, etc.) are centralized in `meridian/constants.py` — check there before hardcoding string keys used across `data`/`model`/`analysis`.
- Changes are tracked in `CHANGELOG.md` under `[Unreleased]` (Keep a Changelog style); update it for user-visible fixes/features.
