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

## File Naming Conventions

- Test files: `*_test.py` 
- Main modules follow snake_case
- Template files in `meridian/analysis/templates/` include HTML/Jinja and SCSS