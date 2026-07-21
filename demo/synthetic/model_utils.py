"""Helpers for building a Meridian `InputData` from simulated data and for
slicing an ArviZ posterior summary table by parameter name.
"""

import pandas as pd

from meridian.data import data_frame_input_data_builder
from meridian.data import input_data as input_data_lib


def build_input_data(
    df: pd.DataFrame,
    channel_names: list[str],
    control_col_names: list[str],
) -> input_data_lib.InputData:
  """Builds a Meridian `InputData` from a simulated geo x time DataFrame."""
  builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
      kpi_type='non_revenue'
  )
  builder = (
      builder.with_kpi(df, kpi_col='conversions')
      .with_revenue_per_kpi(df, revenue_per_kpi_col='revenue_per_conversion')
      .with_population(df)
      .with_controls(df, control_cols=control_col_names)
      .with_media(
          df,
          media_cols=[f'{channel}_impression' for channel in channel_names],
          media_spend_cols=[f'{channel}_spend' for channel in channel_names],
          media_channels=channel_names,
      )
  )
  return builder.build()


def filter_param_summary(
    summary_df: pd.DataFrame, prefix: str, suffix: str | None = None
) -> pd.DataFrame:
  """Returns ArviZ summary rows whose `index` matches the prefix/suffix."""
  cond = summary_df['index'].str.startswith(prefix)
  if suffix is not None:
    cond &= summary_df['index'].str.endswith(suffix)
  return summary_df.loc[cond, :]
