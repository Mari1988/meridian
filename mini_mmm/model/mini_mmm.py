"""Main Mini MMM model class.

This module implements the core Mini MMM model using PyMC for Bayesian inference.
The model follows the Meridian methodology but simplified for national-level modeling.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import warnings

try:
  import pymc as pm
  import arviz as az
  PYMC_AVAILABLE = True
except ImportError:
  PYMC_AVAILABLE = False
  warnings.warn("PyMC not available. Install with: pip install pymc")

from mini_mmm.data.input_data import SimpleInputData
from mini_mmm.model.transformers import AdstockTransformer, HillTransformer
from mini_mmm.model.priors import PriorConfig, DefaultPriors, create_prior_dict


class MiniMMM:
  """Mini Marketing Mix Model for national-level attribution and optimization.
  
  This class implements a simplified version of the Meridian MMM methodology,
  focusing on the core functionality while maintaining mathematical rigor.
  
  The model structure:
  1. Media transformations: Adstock + Hill saturation
  2. Linear combination: KPI = intercept + Σ(media_effects) + Σ(control_effects) + noise
  3. Bayesian inference: MCMC sampling for parameter estimation
  
  Key simplifications:
  - National level only (no geo-hierarchy)
  - Media channels only (no R&F channels initially)
  - PyMC backend (instead of TensorFlow Probability)
  """
  
  def __init__(self, 
               prior_config: Optional[PriorConfig] = None,
               adstock_max_lag: int = 8,
               random_seed: int = 42):
    """Initializes the Mini MMM model.
    
    Args:
      prior_config: Prior configuration. If None, uses default priors.
      adstock_max_lag: Maximum lag for adstock transformation.
      random_seed: Random seed for reproducibility.
    """
    if not PYMC_AVAILABLE:
      raise ImportError("PyMC is required but not installed. Install with: pip install pymc")
    
    self.prior_config = prior_config or DefaultPriors.conservative()
    self.adstock_max_lag = adstock_max_lag
    self.random_seed = random_seed
    
    # Model state
    self.is_fitted = False
    self.input_data = None
    self.pymc_model = None
    self.trace = None
    self._posterior_summary = None
    
    # Cached transformations for prediction
    self._fitted_params = None
  
  def fit(self, 
          data: SimpleInputData,
          draws: int = 2000,
          tune: int = 1000,
          chains: int = 2,
          cores: int = None,
          target_accept: float = 0.9) -> 'MiniMMM':
    """Fits the Mini MMM model to data.
    
    Args:
      data: SimpleInputData instance containing model inputs
      draws: Number of MCMC draws per chain
      tune: Number of tuning/warmup draws per chain
      chains: Number of MCMC chains
      cores: Number of CPU cores (None for auto)
      target_accept: Target acceptance rate for NUTS sampler
      
    Returns:
      Self for method chaining
    """
    self.input_data = data
    
    # Build PyMC model
    self._build_pymc_model()
    
    # Sample from posterior
    print("Fitting Mini MMM model...")
    print(f"  Data: {data.n_weeks} weeks, {data.n_media_channels} media channels")
    print(f"  MCMC: {draws} draws × {chains} chains (+ {tune} tune)")
    
    with self.pymc_model:
      self.trace = pm.sample(
          draws=draws,
          tune=tune,
          chains=chains,
          cores=cores,
          target_accept=target_accept,
          random_seed=self.random_seed,
          return_inferencedata=True
      )
    
    # Cache model summary and fitted parameters
    self._posterior_summary = az.summary(self.trace)
    self._extract_fitted_parameters()
    
    self.is_fitted = True
    print("Model fitting completed successfully!")
    
    return self
  
  def _build_pymc_model(self):
    """Builds the PyMC model specification."""
    data = self.input_data
    
    # Create prior dictionary
    priors = create_prior_dict(
        self.prior_config, 
        data.n_media_channels,
        data.n_controls
    )
    
    with pm.Model() as model:
      # Media transformation parameters
      retention_rate = pm.Beta(
          'retention_rate', 
          alpha=priors['retention_rate']['alpha'],
          beta=priors['retention_rate']['beta'],
          shape=data.n_media_channels
      )
      
      ec = pm.Lognormal(
          'ec',
          mu=priors['ec']['mu'],
          sigma=priors['ec']['sigma'], 
          shape=data.n_media_channels
      )
      
      slope = pm.Lognormal(
          'slope',
          mu=priors['slope']['mu'],
          sigma=priors['slope']['sigma'],
          shape=data.n_media_channels
      )
      
      # Media effect coefficients (ROI-based)
      roi = pm.Lognormal(
          'roi',
          mu=priors['roi']['mu'], 
          sigma=priors['roi']['sigma'],
          shape=data.n_media_channels
      )
      
      # Base level (intercept)
      intercept = pm.Normal(
          'intercept',
          mu=priors['intercept']['mu'],
          sigma=priors['intercept']['sigma']
      )
      
      # Control coefficients (if any)
      if data.n_controls > 0:
        control_coef = pm.Normal(
            'control_coef',
            mu=priors['control_coef']['mu'],
            sigma=priors['control_coef']['sigma'],
            shape=data.n_controls
        )
      
      # Observation noise
      sigma = pm.Gamma(
          'sigma',
          alpha=priors['sigma']['alpha'],
          beta=priors['sigma']['beta']
      )
      
      # Media transformations (deterministic)
      media_data = data.get_media_matrix()
      
      # Apply adstock transformation
      adstocked_media = pm.Deterministic(
          'adstocked_media',
          self._apply_adstock_pymc(media_data, retention_rate)
      )
      
      # Apply Hill saturation
      saturated_media = pm.Deterministic(
          'saturated_media', 
          self._apply_hill_pymc(adstocked_media, ec, slope)
      )
      
      # Media effects (transformed media * ROI)
      media_spend = data.get_media_matrix()
      media_effects = pm.Deterministic(
          'media_effects',
          pm.math.sum(saturated_media * roi[None, :] * media_spend, axis=1)
      )
      
      # Control effects
      if data.n_controls > 0:
        controls_data = data.get_controls_matrix()
        control_effects = pm.Deterministic(
            'control_effects',
            pm.math.sum(controls_data * control_coef[None, :], axis=1)
        )
      else:
        control_effects = 0
      
      # Expected KPI
      mu = pm.Deterministic(
          'mu',
          intercept + media_effects + control_effects
      )
      
      # Likelihood
      kpi_obs = data.get_kpi_array()
      likelihood = pm.Normal(
          'kpi',
          mu=mu,
          sigma=sigma,
          observed=kpi_obs
      )
    
    self.pymc_model = model
  
  def _apply_adstock_pymc(self, media: np.ndarray, retention_rate):
    """Applies adstock transformation within PyMC model."""
    # For PyMC, we need to implement adstock using scan or custom Op
    # For simplicity, we'll use a vectorized approach
    n_weeks, n_channels = media.shape
    
    # Initialize with original media
    adstocked = media.copy()
    
    # Apply geometric adstock (simplified version)
    for lag in range(1, min(n_weeks, self.adstock_max_lag + 1)):
      padded_media = pm.math.concatenate([
          pm.math.zeros((lag, n_channels)),
          media[:-lag]
      ], axis=0)
      adstocked = adstocked + padded_media * (retention_rate ** lag)[None, :]
    
    return adstocked
  
  def _apply_hill_pymc(self, media, ec, slope):
    """Applies Hill transformation within PyMC model."""
    # Estimate half-saturation from data
    media_medians = []
    for c in range(self.input_data.n_media_channels):
      channel_data = self.input_data.get_media_matrix()[:, c]
      non_zero = channel_data[channel_data > 0]
      if len(non_zero) > 0:
        media_medians.append(np.median(non_zero))
      else:
        media_medians.append(1.0)
    
    half_saturation = np.array(media_medians)
    
    # Hill transformation: slope * media^ec / (half_sat^ec + media^ec)
    media_powered = pm.math.power(media, ec[None, :])
    half_sat_powered = pm.math.power(half_saturation[None, :], ec[None, :])
    
    hill_transformed = (slope[None, :] * media_powered / 
                       (half_sat_powered + media_powered))
    
    return hill_transformed
  
  def _extract_fitted_parameters(self):
    """Extracts point estimates (posterior means) for fitted parameters."""
    if self.trace is None:
      return
    
    posterior = self.trace.posterior
    
    self._fitted_params = {
        'retention_rate': posterior['retention_rate'].mean(dim=['chain', 'draw']).values,
        'ec': posterior['ec'].mean(dim=['chain', 'draw']).values,
        'slope': posterior['slope'].mean(dim=['chain', 'draw']).values,
        'roi': posterior['roi'].mean(dim=['chain', 'draw']).values,
        'intercept': float(posterior['intercept'].mean(dim=['chain', 'draw']).values),
        'sigma': float(posterior['sigma'].mean(dim=['chain', 'draw']).values),
    }
    
    if 'control_coef' in posterior:
      self._fitted_params['control_coef'] = posterior['control_coef'].mean(
          dim=['chain', 'draw']).values
  
  def predict(self, 
             media_spend: Optional[np.ndarray] = None,
             controls: Optional[np.ndarray] = None,
             return_components: bool = False) -> Dict[str, np.ndarray]:
    """Makes predictions using fitted model.
    
    Args:
      media_spend: Media spend data. If None, uses training data.
      controls: Control variables. If None, uses training data.
      return_components: Whether to return individual components.
      
    Returns:
      Dictionary with predictions and optionally components.
    """
    if not self.is_fitted:
      raise RuntimeError("Model must be fitted before prediction")
    
    # Use training data if not provided
    if media_spend is None:
      media_spend = self.input_data.get_media_matrix()
    if controls is None and self.input_data.n_controls > 0:
      controls = self.input_data.get_controls_matrix()
    
    params = self._fitted_params
    
    # Apply transformations
    adstocked = AdstockTransformer.transform(
        media_spend, 
        params['retention_rate'],
        max_lag=self.adstock_max_lag
    )
    
    saturated = HillTransformer.transform(
        adstocked,
        params['ec'],
        params['slope']
    )
    
    # Calculate effects
    media_effects = np.sum(saturated * params['roi'][None, :] * media_spend, axis=1)
    
    control_effects = 0
    if controls is not None and 'control_coef' in params:
      control_effects = np.sum(controls * params['control_coef'][None, :], axis=1)
    
    # Total prediction
    prediction = params['intercept'] + media_effects + control_effects
    
    result = {'prediction': prediction}
    
    if return_components:
      result.update({
          'intercept': np.full_like(prediction, params['intercept']),
          'media_effects': media_effects,
          'control_effects': control_effects if isinstance(control_effects, np.ndarray) else np.full_like(prediction, control_effects),
          'adstocked_media': adstocked,
          'saturated_media': saturated
      })
    
    return result
  
  def get_model_summary(self) -> pd.DataFrame:
    """Returns summary statistics of fitted parameters."""
    if not self.is_fitted:
      raise RuntimeError("Model must be fitted before getting summary")
    
    return self._posterior_summary
  
  def get_media_summary(self) -> pd.DataFrame:
    """Returns summary of media channel effects and transformations."""
    if not self.is_fitted:
      raise RuntimeError("Model must be fitted before getting summary")
    
    params = self._fitted_params
    media_channels = self.input_data.media_channels
    
    # Calculate total spend and effects
    media_spend = self.input_data.get_media_matrix()
    total_spend = np.sum(media_spend, axis=0)
    
    # Get predicted components
    pred_components = self.predict(return_components=True)
    saturated_total = np.sum(pred_components['saturated_media'], axis=0)
    
    summary_data = []
    for i, channel in enumerate(media_channels):
      summary_data.append({
          'channel': channel,
          'total_spend': total_spend[i],
          'retention_rate': params['retention_rate'][i],
          'ec': params['ec'][i], 
          'slope': params['slope'][i],
          'roi': params['roi'][i],
          'total_effect': saturated_total[i] * params['roi'][i],
          'spend_share': total_spend[i] / np.sum(total_spend),
          'effect_share': (saturated_total[i] * params['roi'][i]) / 
                         np.sum(saturated_total * params['roi'])
      })
    
    return pd.DataFrame(summary_data)
  
  def get_diagnostics(self) -> Dict[str, Any]:
    """Returns model diagnostics and convergence statistics."""
    if not self.is_fitted:
      raise RuntimeError("Model must be fitted before getting diagnostics")
    
    # Basic convergence diagnostics
    rhat = az.rhat(self.trace)
    ess_bulk = az.ess(self.trace, kind='bulk') 
    ess_tail = az.ess(self.trace, kind='tail')
    
    # Model fit statistics
    waic = az.waic(self.trace)
    loo = az.loo(self.trace)
    
    # Prediction accuracy on training data
    pred = self.predict()['prediction']
    actual = self.input_data.get_kpi_array()
    
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    mape = np.mean(np.abs(pred - actual) / np.abs(actual)) * 100
    r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    
    return {
        'convergence': {
            'max_rhat': float(rhat.max()),
            'min_ess_bulk': int(ess_bulk.min()),
            'min_ess_tail': int(ess_tail.min())
        },
        'model_comparison': {
            'waic': float(waic.waic),
            'loo': float(loo.loo)
        },
        'fit_metrics': {
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    }