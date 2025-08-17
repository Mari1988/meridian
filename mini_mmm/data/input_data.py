"""Input data structure for Mini MMM.

This module provides a simplified data container that accepts pandas DataFrames
and handles basic preprocessing for the Mini MMM model.
"""

from typing import Optional, List, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class SimpleInputData:
  """Simplified input data container for Mini MMM.
  
  This class stores all input data required for the Mini MMM model in a simple,
  accessible format using pandas DataFrames.
  
  Attributes:
    kpi: Target variable (e.g., sales, conversions). Shape: (n_weeks,)
    media_spend: Media spending data. Shape: (n_weeks, n_media_channels)
    media_channels: List of media channel names
    controls: Optional control variables (e.g., price, promotions). 
              Shape: (n_weeks, n_controls)
    control_names: Optional list of control variable names
    date_col: Optional date column for time series
    population: Optional population data for normalization. If scalar,
                broadcasts to all time periods
  """
  
  kpi: pd.Series
  media_spend: pd.DataFrame
  media_channels: List[str]
  controls: Optional[pd.DataFrame] = None
  control_names: Optional[List[str]] = None
  date_col: Optional[pd.Series] = None
  population: Optional[Union[float, pd.Series]] = None
  
  def __post_init__(self):
    """Validates and processes input data after initialization."""
    self._validate_inputs()
    self._process_data()
  
  def _validate_inputs(self):
    """Validates input data consistency."""
    n_weeks = len(self.kpi)
    
    # Check media spend dimensions
    if len(self.media_spend) != n_weeks:
      raise ValueError(
          f"Media spend length {len(self.media_spend)} doesn't match "
          f"KPI length {n_weeks}")
    
    if len(self.media_channels) != self.media_spend.shape[1]:
      raise ValueError(
          f"Number of media channels {len(self.media_channels)} doesn't match "
          f"media spend columns {self.media_spend.shape[1]}")
    
    # Check controls if provided
    if self.controls is not None:
      if len(self.controls) != n_weeks:
        raise ValueError(
            f"Controls length {len(self.controls)} doesn't match "
            f"KPI length {n_weeks}")
      
      if self.control_names is not None:
        if len(self.control_names) != self.controls.shape[1]:
          raise ValueError(
              f"Number of control names {len(self.control_names)} doesn't "
              f"match controls columns {self.controls.shape[1]}")
    
    # Check date column if provided
    if self.date_col is not None and len(self.date_col) != n_weeks:
      raise ValueError(
          f"Date column length {len(self.date_col)} doesn't match "
          f"KPI length {n_weeks}")
    
    # Check population if provided as Series
    if isinstance(self.population, pd.Series):
      if len(self.population) != n_weeks:
        raise ValueError(
            f"Population length {len(self.population)} doesn't match "
            f"KPI length {n_weeks}")
  
  def _process_data(self):
    """Processes and stores data in convenient formats."""
    # Store basic dimensions
    self.n_weeks = len(self.kpi)
    self.n_media_channels = len(self.media_channels)
    self.n_controls = self.controls.shape[1] if self.controls is not None else 0
    
    # Set column names for media spend
    self.media_spend.columns = self.media_channels
    
    # Set control names if not provided
    if self.controls is not None and self.control_names is None:
      self.control_names = [f"control_{i}" for i in range(self.n_controls)]
      self.controls.columns = self.control_names
    elif self.controls is not None:
      self.controls.columns = self.control_names
    
    # Handle population
    if self.population is None:
      self.population = 1.0  # Default to no normalization
    
    if isinstance(self.population, (int, float)):
      self.population = pd.Series([self.population] * self.n_weeks)
    
    # Create time index if date column provided
    if self.date_col is not None:
      self.kpi.index = self.date_col
      self.media_spend.index = self.date_col
      if self.controls is not None:
        self.controls.index = self.date_col
      self.population.index = self.date_col
  
  def get_media_matrix(self) -> np.ndarray:
    """Returns media spend data as numpy array."""
    return self.media_spend.values
  
  def get_kpi_array(self) -> np.ndarray:
    """Returns KPI data as numpy array."""
    return self.kpi.values
  
  def get_controls_matrix(self) -> Optional[np.ndarray]:
    """Returns controls data as numpy array, or None if no controls."""
    return self.controls.values if self.controls is not None else None
  
  def get_population_array(self) -> np.ndarray:
    """Returns population data as numpy array."""
    return self.population.values
  
  def summary(self) -> str:
    """Returns a summary of the input data."""
    summary_lines = [
        f"Mini MMM Input Data Summary:",
        f"  Time periods: {self.n_weeks}",
        f"  Media channels: {self.n_media_channels}",
        f"  Control variables: {self.n_controls}",
        f"  Date range: {self.date_col.min()} to {self.date_col.max()}" 
        if self.date_col is not None else "  No date column",
        f"",
        f"Media channels: {', '.join(self.media_channels)}",
    ]
    
    if self.control_names:
      summary_lines.append(f"Control variables: {', '.join(self.control_names)}")
    
    summary_lines.extend([
        f"",
        f"Data ranges:",
        f"  KPI: {self.kpi.min():.2f} to {self.kpi.max():.2f}",
    ])
    
    for channel in self.media_channels:
      spend_data = self.media_spend[channel]
      summary_lines.append(
          f"  {channel}: {spend_data.min():.2f} to {spend_data.max():.2f}")
    
    return "\n".join(summary_lines)
  
  @classmethod
  def from_dataframe(cls, 
                     df: pd.DataFrame,
                     kpi_col: str,
                     media_cols: List[str],
                     control_cols: Optional[List[str]] = None,
                     date_col: Optional[str] = None,
                     population_col: Optional[str] = None) -> 'SimpleInputData':
    """Creates SimpleInputData from a single DataFrame.
    
    Args:
      df: Input DataFrame containing all data
      kpi_col: Column name for the KPI/target variable
      media_cols: List of column names for media spend data
      control_cols: Optional list of column names for control variables
      date_col: Optional column name for dates
      population_col: Optional column name for population data
    
    Returns:
      SimpleInputData instance
    """
    kpi = df[kpi_col].copy()
    media_spend = df[media_cols].copy()
    
    controls = df[control_cols].copy() if control_cols else None
    date_series = df[date_col].copy() if date_col else None
    population = df[population_col].copy() if population_col else None
    
    return cls(
        kpi=kpi,
        media_spend=media_spend,
        media_channels=media_cols,
        controls=controls,
        control_names=control_cols,
        date_col=date_series,
        population=population
    )