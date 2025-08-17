"""Data validation utilities for Mini MMM."""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
from mini_mmm.data.input_data import SimpleInputData


def validate_input_data(data: SimpleInputData) -> List[str]:
  """Validates input data and returns list of warnings/issues.
  
  Args:
    data: SimpleInputData instance to validate
    
  Returns:
    List of validation warnings/issues (empty if all good)
  """
  warnings = []
  
  # Check for negative values
  if (data.get_kpi_array() < 0).any():
    warnings.append("KPI contains negative values")
  
  if (data.get_media_matrix() < 0).any():
    warnings.append("Media spend contains negative values")
  
  # Check for zero variance
  kpi_array = data.get_kpi_array()
  if np.var(kpi_array) == 0:
    warnings.append("KPI has zero variance")
  
  media_matrix = data.get_media_matrix()
  zero_var_channels = []
  for i, channel in enumerate(data.media_channels):
    if np.var(media_matrix[:, i]) == 0:
      zero_var_channels.append(channel)
  
  if zero_var_channels:
    warnings.append(f"Media channels with zero variance: {zero_var_channels}")
  
  # Check for excessive zeros in media spend
  high_zero_channels = []
  for i, channel in enumerate(data.media_channels):
    zero_fraction = np.mean(media_matrix[:, i] == 0)
    if zero_fraction > 0.8:  # More than 80% zeros
      high_zero_channels.append(f"{channel} ({zero_fraction:.1%} zeros)")
  
  if high_zero_channels:
    warnings.append(f"Media channels with >80% zeros: {high_zero_channels}")
  
  # Check for outliers in media spend (values > 10x median)
  outlier_channels = []
  for i, channel in enumerate(data.media_channels):
    channel_data = media_matrix[:, i]
    non_zero_data = channel_data[channel_data > 0]
    if len(non_zero_data) > 0:
      median_val = np.median(non_zero_data)
      if median_val > 0:
        max_val = np.max(channel_data)
        if max_val > 10 * median_val:
          outlier_channels.append(f"{channel} (max/median: {max_val/median_val:.1f})")
  
  if outlier_channels:
    warnings.append(f"Potential outliers in media spend: {outlier_channels}")
  
  # Check data length (recommend at least 52 weeks)
  if data.n_weeks < 52:
    warnings.append(f"Short time series: {data.n_weeks} weeks (recommend ≥52)")
  
  # Check for correlation between KPI and media
  correlation_issues = []
  for i, channel in enumerate(data.media_channels):
    correlation = np.corrcoef(kpi_array, media_matrix[:, i])[0, 1]
    if np.isnan(correlation):
      correlation_issues.append(f"{channel}: correlation undefined (zero variance?)")
    elif correlation < 0:
      correlation_issues.append(f"{channel}: negative correlation ({correlation:.3f})")
  
  if correlation_issues:
    warnings.append(f"Correlation issues: {correlation_issues}")
  
  return warnings


def check_data_quality(data: SimpleInputData, 
                      min_weeks: int = 52,
                      max_zero_fraction: float = 0.8,
                      max_outlier_ratio: float = 10.0) -> bool:
  """Performs comprehensive data quality check.
  
  Args:
    data: SimpleInputData instance to check
    min_weeks: Minimum recommended number of weeks
    max_zero_fraction: Maximum allowed fraction of zeros in media channels
    max_outlier_ratio: Maximum allowed max/median ratio for outlier detection
    
  Returns:
    True if data passes quality checks, False otherwise
  """
  warnings = validate_input_data(data)
  
  # Check for critical issues that would prevent modeling
  critical_issues = [
    "KPI has zero variance",
    "Media channels with zero variance",
    "correlation undefined"
  ]
  
  for warning in warnings:
    for issue in critical_issues:
      if issue in warning:
        print(f"CRITICAL: {warning}")
        return False
  
  # Print non-critical warnings
  for warning in warnings:
    print(f"WARNING: {warning}")
  
  return True


def suggest_preprocessing(data: SimpleInputData) -> List[str]:
  """Suggests preprocessing steps based on data characteristics.
  
  Args:
    data: SimpleInputData instance to analyze
    
  Returns:
    List of preprocessing suggestions
  """
  suggestions = []
  
  # Check if data needs scaling
  kpi_scale = np.mean(data.get_kpi_array())
  media_scales = np.mean(data.get_media_matrix(), axis=0)
  
  if kpi_scale > 1000:
    suggestions.append("Consider scaling KPI (dividing by 1000 or more)")
  
  max_media_scale = np.max(media_scales)
  if max_media_scale > 100000:
    suggestions.append("Consider scaling media spend data")
  
  # Check for seasonality
  if data.n_weeks >= 52:
    kpi_array = data.get_kpi_array()
    # Simple seasonality check using autocorrelation at lag 52
    if len(kpi_array) >= 52:
      lag_52_corr = np.corrcoef(kpi_array[:-52], kpi_array[52:])[0, 1]
      if not np.isnan(lag_52_corr) and lag_52_corr > 0.3:
        suggestions.append("Consider adding seasonal controls (yearly pattern detected)")
  
  # Check for trend
  if data.n_weeks >= 12:
    kpi_array = data.get_kpi_array()
    time_trend = np.corrcoef(np.arange(len(kpi_array)), kpi_array)[0, 1]
    if not np.isnan(time_trend) and abs(time_trend) > 0.3:
      trend_direction = "increasing" if time_trend > 0 else "decreasing"
      suggestions.append(f"Consider adding trend control ({trend_direction} pattern)")
  
  return suggestions