# %%
import sys
import os
home_dir = "/Users/mariappan.subramanian/Documents/"
sys.path.append(f'{home_dir}/repo/media_param_analysis/codes/')
sys.path.append(f'{home_dir}/repo/forked/meridian/meridian/')

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az
import altair as alt
import gc

from mpa_utils_meridian import MeridianMPAInput
from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.analysis import summarizer
from meridian.analysis import formatter

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

# %%
from config import client_config, ec50_dict, get_main_config
main_config = get_main_config(home_dir)

# %%
# load sample meridian model file
model_file_path = f"{home_dir}/repo/forked/test/sample_meridian_model_obj.pkl"
meridian=model.load_mmm(model_file_path)
