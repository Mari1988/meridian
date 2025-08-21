import sys
import os
home_dir = "/Users/mariappan.subramanian/Documents/"
sys.path.append(f'{home_dir}/repo/forked/meridian/meridian/')

from meridian.model import model
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian import constants as c

# load a demo model object
file_path = f'{home_dir}/repo/forked/meridian/meridian/mpa/demo_rf_model_obj.pkl'
mmm = model.load_mmm(file_path)

# create a budget optimizer object
self = optimizer.BudgetOptimizer(mmm)

new_data = analyzer.DataTensors()
required_tensors = c.PERFORMANCE_DATA + (c.TIME,)
filled_data = new_data.validate_and_fill_missing_data(
    required_tensors_names=required_tensors, meridian=self._meridian
)

# ------------------------------------------------------------
import pandas as pd

df = pd.read_excel(
    'https://github.com/google/meridian/raw/main/meridian/data/simulated_data/xlsx/geo_media.xlsx',
    engine='openpyxl',
)

from meridian.data import data_frame_input_data_builder as data_builder

builder = data_builder.DataFrameInputDataBuilder(
    kpi_type='non_revenue',
    default_kpi_column="conversions",
    default_revenue_per_kpi_column="revenue_per_conversion",
)
builder = (
    builder
        .with_kpi(df)
        .with_revenue_per_kpi(df)
        .with_population(df)
        .with_controls(df, control_cols=["GQV", "Discount", "Competitor_Sales"])
)
channels = ["Channel0", "Channel1", "Channel2", "Channel3", "Channel4", "Channel5"]
builder = builder.with_media(
    df,
    media_cols=[f"{channel}_impression" for channel in channels],
    media_spend_cols=[f"{channel}_spend" for channel in channels],
    media_channels=channels,
)

data = builder.build()

data.media.__class__
filled_data.media.__class__
