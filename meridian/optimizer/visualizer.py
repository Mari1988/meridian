# Copyright 2025 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization utilities for independent optimizer results."""

from typing import Optional
import altair as alt
import pandas as pd
import numpy as np
from meridian.optimizer.independent_optimizer import IndependentOptimizationResults


__all__ = [
    "plot_spend_allocation",
    "plot_spend_delta", 
    "plot_roi_comparison",
    "plot_response_curves",
    "create_optimization_summary",
]

# Disable max row limitations in Altair
alt.data_transformers.disable_max_rows()


def plot_spend_allocation(results: IndependentOptimizationResults, optimized: bool = True) -> alt.Chart:
  """Create pie chart showing spend allocation by channel.
  
  Args:
    results: IndependentOptimizationResults object.
    optimized: If True, show optimized allocation; if False, show historical.
    
  Returns:
    Altair pie chart.
  """
  df = results.to_dataframe()
  spend_column = 'optimized_spend' if optimized else 'historical_spend'
  title = 'Optimized Spend Allocation' if optimized else 'Historical Spend Allocation'
  
  return (
      alt.Chart(df)
      .mark_arc(tooltip=True, padAngle=0.02)
      .encode(
          theta=alt.Theta(f'{spend_column}:Q'),
          color=alt.Color('channel:N', legend=alt.Legend(title=None)),
          tooltip=['channel:N', f'{spend_column}:Q']
      )
      .properties(title=title, width=400, height=400)
      .configure_view(stroke=None)
  )


def plot_spend_delta(results: IndependentOptimizationResults) -> alt.Chart:
  """Create bar chart showing spend changes by channel.
  
  Args:
    results: IndependentOptimizationResults object.
    
  Returns:
    Altair bar chart.
  """
  df = results.to_dataframe()
  
  # Sort by spend change
  df_sorted = pd.concat([
      df[df['spend_change'] < 0].sort_values('spend_change'),
      df[df['spend_change'] >= 0].sort_values('spend_change', ascending=False),
  ]).reset_index(drop=True)
  
  base = (
      alt.Chart(df_sorted)
      .encode(
          x=alt.X('channel:N', sort=None, axis=alt.Axis(title=None, labelAngle=-45)),
          y=alt.Y('spend_change:Q', axis=alt.Axis(title='Spend Change ($)')),
      )
  )
  
  bars = base.mark_bar(
      tooltip=True, size=30, cornerRadiusEnd=2
  ).encode(
      color=alt.condition(
          alt.datum.spend_change > 0,
          alt.value('#17becf'),  # Cyan for increases
          alt.value('#ff7f7f'),  # Red for decreases
      ),
  )
  
  # Add text labels
  text = base.mark_text(
      baseline='bottom', dy=-5, fontSize=10
  ).encode(
      text=alt.Text('spend_change:Q', format='.0f'),
      y=alt.Y('spend_change:Q'),
  )
  
  return (
      (bars + text)
      .properties(title='Spend Changes by Channel', width=600, height=400)
      .configure_view(stroke=None)
  )


def plot_roi_comparison(results: IndependentOptimizationResults) -> alt.Chart:
  """Create bar chart comparing ROI by channel.
  
  Args:
    results: IndependentOptimizationResults object.
    
  Returns:
    Altair bar chart.
  """
  df = results.to_dataframe()
  
  # Calculate historical ROI for comparison
  df['historical_roi'] = df['incremental_outcome'] / df['historical_spend']
  
  # Reshape for grouped bar chart
  roi_data = []
  for _, row in df.iterrows():
    roi_data.append({
        'channel': row['channel'],
        'roi_type': 'Historical',
        'roi': row['historical_roi']
    })
    roi_data.append({
        'channel': row['channel'],
        'roi_type': 'Optimized',
        'roi': row['roi']
    })
  
  roi_df = pd.DataFrame(roi_data)
  
  return (
      alt.Chart(roi_df)
      .mark_bar(tooltip=True)
      .encode(
          x=alt.X('channel:N', axis=alt.Axis(title=None, labelAngle=-45)),
          y=alt.Y('roi:Q', axis=alt.Axis(title='ROI')),
          color=alt.Color('roi_type:N', legend=alt.Legend(title='Scenario')),
          column=alt.Column('roi_type:N')
      )
      .properties(title='ROI Comparison: Historical vs Optimized', width=300, height=400)
      .resolve_scale(y='independent')
  )


def plot_response_curves(
    results: IndependentOptimizationResults,
    optimizer,
    n_top_channels: Optional[int] = None
) -> alt.Chart:
  """Create response curves plot for top channels.
  
  Args:
    results: IndependentOptimizationResults object.
    optimizer: IndependentOptimizer instance (needed to generate curves).
    n_top_channels: Number of top channels by spend to show.
    
  Returns:
    Altair faceted line chart.
  """
  # Get top channels by optimized spend
  df = results.to_dataframe().sort_values('optimized_spend', ascending=False)
  if n_top_channels:
    top_channels = df.head(n_top_channels)['channel'].tolist()
  else:
    top_channels = df['channel'].tolist()
  
  # Generate response curves
  spend_multipliers = np.arange(0, 3, 0.1)  # 0 to 3x historical spend
  curves_ds = optimizer.get_response_curves(
      spend_multipliers=spend_multipliers,
      start_date=results.scenario.start_date,
      end_date=results.scenario.end_date
  )
  
  # Convert to DataFrame and filter to top channels
  curves_df = curves_ds.to_dataframe().reset_index()
  curves_df = curves_df[curves_df['channel'].isin(top_channels)]
  
  # Add spend constraint indicators
  constraint_lower = 1 - results.scenario.spend_constraint_lower
  constraint_upper = 1 + results.scenario.spend_constraint_upper
  
  curves_df['within_constraint'] = (
      (curves_df['spend_multiplier'] >= constraint_lower) &
      (curves_df['spend_multiplier'] <= constraint_upper)
  )
  
  # Add current spend points
  current_points = []
  for channel in top_channels:
    hist_spend = df[df['channel'] == channel]['historical_spend'].iloc[0]
    opt_spend = df[df['channel'] == channel]['optimized_spend'].iloc[0]
    outcome = df[df['channel'] == channel]['incremental_outcome'].iloc[0]
    
    current_points.extend([
        {
            'channel': channel,
            'spend': hist_spend,
            'incremental_outcome': outcome,
            'spend_multiplier': 1.0,
            'point_type': 'Historical'
        },
        {
            'channel': channel,
            'spend': opt_spend,
            'incremental_outcome': outcome,
            'spend_multiplier': opt_spend / hist_spend if hist_spend > 0 else 1,
            'point_type': 'Optimized'
        }
    ])
  
  points_df = pd.DataFrame(current_points)
  
  # Create base chart
  base = alt.Chart(curves_df).encode(
      x=alt.X('spend:Q', title='Spend ($)'),
      y=alt.Y('incremental_outcome:Q', title='Incremental Outcome'),
  )
  
  # Response curves with constraint styling
  curves = base.mark_line().encode(
      color=alt.Color('channel:N', legend=None),
      strokeDash=alt.condition(
          alt.datum.within_constraint,
          alt.value([]),  # Solid line within constraints
          alt.value([5, 5])  # Dashed line outside constraints
      )
  )
  
  # Current spend points
  points = alt.Chart(points_df).mark_point(
      filled=True, size=100, tooltip=True
  ).encode(
      x='spend:Q',
      y='incremental_outcome:Q',
      color='point_type:N',
      shape='point_type:N'
  )
  
  return (
      alt.layer(curves, points)
      .facet(
          facet=alt.Facet('channel:N', title=None),
          columns=3
      )
      .resolve_scale(y='independent', x='independent')
      .properties(title='Response Curves by Channel')
  )


def create_optimization_summary(results: IndependentOptimizationResults) -> pd.DataFrame:
  """Create summary table of optimization results.
  
  Args:
    results: IndependentOptimizationResults object.
    
  Returns:
    DataFrame with summary metrics.
  """
  df = results.to_dataframe()
  
  summary_data = {
      'Metric': [
          'Total Historical Spend',
          'Total Optimized Spend', 
          'Total Spend Change',
          'Total Incremental Outcome',
          'Historical Total ROI',
          'Optimized Total ROI',
          'ROI Improvement',
          'Number of Channels',
          'Channels with Increased Spend',
          'Channels with Decreased Spend',
      ],
      'Value': [
          f"${df['historical_spend'].sum():,.0f}",
          f"${df['optimized_spend'].sum():,.0f}",
          f"${df['spend_change'].sum():,.0f}",
          f"{df['incremental_outcome'].sum():,.0f}",
          f"{df['incremental_outcome'].sum() / df['historical_spend'].sum():.2f}",
          f"{results.total_roi:.2f}",
          f"{results.total_roi - (df['incremental_outcome'].sum() / df['historical_spend'].sum()):.2f}",
          f"{len(df)}",
          f"{len(df[df['spend_change'] > 0])}",
          f"{len(df[df['spend_change'] < 0])}",
      ]
  }
  
  return pd.DataFrame(summary_data)