"""Mini MMM - A simplified Marketing Mix Modeling framework.

Based on Google's Meridian methodology but designed for simplicity and accessibility.
This package provides core MMM functionality including:
- Media transformations (Adstock + Hill saturation)
- Bayesian hierarchical modeling
- Response curve analysis
- Budget optimization

Key simplifications:
- National-level modeling only (no geo-level complexity)
- Media channels only (no R&F channels initially)
- Python-native stack (PyMC + numpy/pandas)
- Focus on point estimates for fast analysis
"""

from mini_mmm.model.mini_mmm import MiniMMM
from mini_mmm.data.input_data import SimpleInputData

__version__ = "0.1.0"
__author__ = "Based on Google Meridian methodology"

__all__ = [
    "MiniMMM",
    "SimpleInputData",
]