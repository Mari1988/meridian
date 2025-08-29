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

6. **meridian.planner**: Excel-to-Meridian Integration System (custom development)
   - **Excel data loading**: `AdhocDataLoader` for 3-sheet Excel files (Data, Coefficients, Parameters)
   - **Parameter processing**: `MediaParameterLoader` for adstock, hill, and coefficient organization
   - **Inference data creation**: `PointInferenceData` for complete ArviZ InferenceData compatibility
   - **Budget optimization**: End-to-end workflow from Excel estimates to Meridian optimization
   - **See `meridian/planner/CLAUDE.md` for comprehensive documentation**
   - Enables direct use of Excel-based MMM parameters with Meridian's analysis functions

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
