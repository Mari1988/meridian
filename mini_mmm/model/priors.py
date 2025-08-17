"""Prior distributions for Mini MMM model parameters.

This module defines default priors and utilities for setting up
Bayesian priors for the Mini MMM model parameters.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class PriorConfig:
  """Configuration for model parameter priors.
  
  Attributes:
    # Media transformation parameters
    retention_rate_prior: (alpha, beta) for Beta distribution
    ec_prior: (mean, std) for LogNormal distribution  
    slope_prior: (mean, std) for LogNormal distribution
    
    # Media coefficients (ROI-based)
    roi_prior: (mean, std) for LogNormal distribution
    
    # Control coefficients
    control_coef_prior: (mean, std) for Normal distribution
    
    # Intercept
    intercept_prior: (mean, std) for Normal distribution
    
    # Noise
    sigma_prior: (alpha, beta) for Gamma distribution
  """
  
  # Media transformation priors
  retention_rate_prior: Tuple[float, float] = (2.0, 2.0)  # Beta(2,2) -> mean=0.5
  ec_prior: Tuple[float, float] = (0.7, 0.3)  # LogNormal -> median~2.0
  slope_prior: Tuple[float, float] = (1.0, 0.5)  # LogNormal -> median~2.7
  
  # ROI prior (revenue per unit spend)
  roi_prior: Tuple[float, float] = (0.69, 0.5)  # LogNormal -> median~2.0
  
  # Control variable coefficients
  control_coef_prior: Tuple[float, float] = (0.0, 1.0)  # Normal(0, 1)
  
  # Intercept (base level)
  intercept_prior: Tuple[float, float] = (0.0, 10.0)  # Normal(0, 10)
  
  # Observation noise
  sigma_prior: Tuple[float, float] = (1.0, 1.0)  # Gamma(1, 1)


class DefaultPriors:
  """Default prior configurations for different use cases."""
  
  @staticmethod
  def conservative() -> PriorConfig:
    """Conservative priors with tighter constraints."""
    return PriorConfig(
        retention_rate_prior=(3.0, 3.0),  # More concentrated around 0.5
        ec_prior=(0.5, 0.2),  # Tighter EC values
        slope_prior=(0.8, 0.3),  # More conservative slopes
        roi_prior=(0.5, 0.3),  # More conservative ROI
        control_coef_prior=(0.0, 0.5),  # Tighter control coefficients
        intercept_prior=(0.0, 5.0),  # Tighter intercept
        sigma_prior=(2.0, 1.0)  # Lower noise expectation
    )
  
  @staticmethod
  def aggressive() -> PriorConfig:
    """More flexible priors allowing higher effects."""
    return PriorConfig(
        retention_rate_prior=(1.5, 1.5),  # More spread in retention
        ec_prior=(0.9, 0.4),  # More flexible EC
        slope_prior=(1.2, 0.7),  # Higher slopes allowed
        roi_prior=(0.9, 0.7),  # Higher ROI allowed
        control_coef_prior=(0.0, 2.0),  # More flexible controls
        intercept_prior=(0.0, 20.0),  # More flexible intercept
        sigma_prior=(0.5, 1.0)  # Higher noise expectation
    )
  
  @staticmethod
  def data_driven(kpi_mean: float, 
                  kpi_std: float,
                  media_spend_means: np.ndarray) -> PriorConfig:
    """Creates data-driven priors based on input data characteristics.
    
    Args:
      kpi_mean: Mean of the KPI time series
      kpi_std: Standard deviation of the KPI time series  
      media_spend_means: Mean spend levels for each media channel
      
    Returns:
      PriorConfig with data-informed priors
    """
    # Scale intercept prior based on KPI
    intercept_std = max(kpi_std * 2, kpi_mean * 0.5)
    
    # Scale sigma prior based on KPI variability
    sigma_shape = 2.0
    sigma_scale = kpi_std / sigma_shape  # E[sigma] ≈ kpi_std
    
    # ROI prior scaled by typical spend levels
    typical_spend = np.median(media_spend_means[media_spend_means > 0])
    if typical_spend > 0:
      # Target ROI of 1-5x with geometric mean around 2x
      roi_mean = np.log(2.0)  # Geometric mean of 2.0
      roi_std = 0.5
    else:
      roi_mean, roi_std = 0.69, 0.5  # Default
    
    return PriorConfig(
        retention_rate_prior=(2.0, 2.0),  # Keep standard
        ec_prior=(0.7, 0.3),  # Keep standard
        slope_prior=(1.0, 0.5),  # Keep standard  
        roi_prior=(roi_mean, roi_std),
        control_coef_prior=(0.0, kpi_std / typical_spend if typical_spend > 0 else 1.0),
        intercept_prior=(kpi_mean, intercept_std),
        sigma_prior=(sigma_shape, sigma_scale)
    )


def create_prior_dict(config: PriorConfig, 
                     n_media_channels: int,
                     n_controls: int = 0) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
  """Creates a dictionary of prior parameters for PyMC model.
  
  Args:
    config: PriorConfig instance
    n_media_channels: Number of media channels
    n_controls: Number of control variables
    
  Returns:
    Dictionary with prior parameters for each model parameter
  """
  priors = {
      'retention_rate': {
          'alpha': np.full(n_media_channels, config.retention_rate_prior[0]),
          'beta': np.full(n_media_channels, config.retention_rate_prior[1])
      },
      'ec': {
          'mu': np.full(n_media_channels, config.ec_prior[0]),
          'sigma': np.full(n_media_channels, config.ec_prior[1])
      },
      'slope': {
          'mu': np.full(n_media_channels, config.slope_prior[0]),
          'sigma': np.full(n_media_channels, config.slope_prior[1])
      },
      'roi': {
          'mu': np.full(n_media_channels, config.roi_prior[0]),
          'sigma': np.full(n_media_channels, config.roi_prior[1])
      },
      'intercept': {
          'mu': config.intercept_prior[0],
          'sigma': config.intercept_prior[1]
      },
      'sigma': {
          'alpha': config.sigma_prior[0],
          'beta': config.sigma_prior[1]
      }
  }
  
  # Add control coefficient priors if needed
  if n_controls > 0:
    priors['control_coef'] = {
        'mu': np.full(n_controls, config.control_coef_prior[0]),
        'sigma': np.full(n_controls, config.control_coef_prior[1])
    }
  
  return priors


def validate_priors(config: PriorConfig) -> None:
  """Validates that prior configuration is reasonable.
  
  Args:
    config: PriorConfig to validate
    
  Raises:
    ValueError: If priors are unreasonable
  """
  # Check Beta distribution parameters for retention rate
  if config.retention_rate_prior[0] <= 0 or config.retention_rate_prior[1] <= 0:
    raise ValueError("Beta distribution parameters must be positive")
  
  # Check LogNormal parameters
  if config.ec_prior[1] <= 0 or config.slope_prior[1] <= 0 or config.roi_prior[1] <= 0:
    raise ValueError("LogNormal sigma parameters must be positive")
  
  # Check Normal parameters
  if config.control_coef_prior[1] <= 0 or config.intercept_prior[1] <= 0:
    raise ValueError("Normal sigma parameters must be positive")
  
  # Check Gamma parameters
  if config.sigma_prior[0] <= 0 or config.sigma_prior[1] <= 0:
    raise ValueError("Gamma distribution parameters must be positive")
  
  # Warn about extreme values
  if np.exp(config.roi_prior[0]) > 100:
    print("WARNING: ROI prior implies very high expected ROI (>100x)")
  
  if config.retention_rate_prior[0] < 0.5 and config.retention_rate_prior[1] < 0.5:
    print("WARNING: Retention rate prior strongly favors high retention (>0.5)")


def get_prior_summary(config: PriorConfig) -> str:
  """Returns a human-readable summary of the prior configuration.
  
  Args:
    config: PriorConfig to summarize
    
  Returns:
    Formatted string describing the priors
  """
  # Calculate expected values for informative priors
  ret_mean = config.retention_rate_prior[0] / (
      config.retention_rate_prior[0] + config.retention_rate_prior[1])
  ec_median = np.exp(config.ec_prior[0])
  slope_median = np.exp(config.slope_prior[0])
  roi_median = np.exp(config.roi_prior[0])
  
  summary = f"""
Prior Configuration Summary:
  
Media Transformation:
  - Retention Rate: Beta({config.retention_rate_prior[0]:.1f}, {config.retention_rate_prior[1]:.1f}) → E[rate] = {ret_mean:.2f}
  - EC (shape): LogNormal({config.ec_prior[0]:.1f}, {config.ec_prior[1]:.1f}) → median = {ec_median:.2f}
  - Slope: LogNormal({config.slope_prior[0]:.1f}, {config.slope_prior[1]:.1f}) → median = {slope_median:.2f}

Media Effects:
  - ROI: LogNormal({config.roi_prior[0]:.1f}, {config.roi_prior[1]:.1f}) → median = {roi_median:.2f}x

Other Parameters:
  - Controls: Normal({config.control_coef_prior[0]:.1f}, {config.control_coef_prior[1]:.1f})
  - Intercept: Normal({config.intercept_prior[0]:.1f}, {config.intercept_prior[1]:.1f})
  - Noise (sigma): Gamma({config.sigma_prior[0]:.1f}, {config.sigma_prior[1]:.1f})
"""
  
  return summary.strip()