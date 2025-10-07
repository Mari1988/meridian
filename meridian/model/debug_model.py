# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az
import IPython
import altair as alt
from IPython.display import HTML
import gc
import matplotlib.pyplot as plt

from meridian.mpa.mpa_utils_meridian import MeridianMPAInput

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

from meridian.mpa.client_config import client_config, ec50_multiplier_config, media_parameters_config

# define home directory and client
home_dir = '/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/tmp'
client = 'Mazda'

main_config =  {
    # "file_path": f"{home_dir}/data/MDF_BY_GEO_Jul8_Channel_events_wit_population.csv",
    "file_path": f"{home_dir}/data/MDF_BY_GEO_EXPANDED_CLIENTS_Aug11.csv",

    # "holiday_file_path": f"{home_dir}/data/holidays_updated_apr21/holiday_effect_",

    "paid_media_imp": ["TV_I", "Display_I", "Video_I"],  # Actual CPM will be calculated based on this
    "paid_media_spends": ["TV_AC", "Display_AC", "Video_AC"],

    "paid_media_viewability_imp": ["TV_VCR", "Display_VCR", "Video_VCR"],  # viewability rate will be calculated based on this
    "paid_media_cols": ["TV_I", "Display_I", "Video_I"],  # not used

    "reach_variables": ["TV_RHH", "Display_RPP", "Video_RPP"],
    "frequency_variables": ["TV_FHH", "Display_FPP", "Video_FPP"],

    "spend_variables_for_cpm_calc": ["TV_AC", "Display_AC", "Video_AC"],  # Prior CPM will be calculated based on this
    "imp_variables_for_cpm_calc": ['TV_I', 'Display_I', 'Video_I'],  # Prior CPM will be calculated based on this

    "response_kpi": "conversions",
    "prior_config": {},
    "prior_type": "spend"
}

prior_type = 'working_spend'  # "cpm_weighted_by_working_spend"
is_national = False
trans_prior_type = 'informative'

# ----------------------------- CREATE MODEL INPUT FILE FOR MERIDIAN --------------------------------- #
# create input object
main_config['prior_type'] = prior_type
mpa_input = MeridianMPAInput(client=client, client_config=client_config, main_config=main_config)
mdf_mw = mpa_input.mdf_mw.copy()

# write to a temp path for meridian
metrics = [mpa_input.target] + mpa_input.paid_media_imp + mpa_input.paid_media_spends
if is_national:
  # calculate pooled (geo, time level) standard deviation
  pooled_std =(mdf_mw['conversions']/mdf_mw['Population']).std()

  # national level standard deviation
  national_level_conversions = mdf_mw.groupby('WES')[['conversions']].sum().reset_index()['conversions']
  national_level_population = mdf_mw.groupby('WES')[['Population']].sum().reset_index()['Population']
  national_std = (national_level_conversions/national_level_population).std()

  correction_factor = np.round(pooled_std/national_std, 2)
  print(f"Correction factor: {correction_factor}")
  # aggregate out 'geo' field for national models
  mdf = mdf_mw. \
    groupby(['WES'])[metrics].sum(). \
      reset_index(). \
        assign(
          Region='National',
          Population=1.0,
          revenue_per_conversion=1.0
              )
else:
  correction_factor = 1.0
  mdf = mdf_mw[['WES'] + metrics + ['Region', 'Population']]. \
    reset_index(drop=True).assign(revenue_per_conversion= 1.0).copy()
  mdf.loc[:, 'WES'] = pd.to_datetime(mdf['WES']).dt.strftime('%Y-%m-%d')

# write to a temp path (input file path for meridian)
target_file_path = f"{home_dir}/trash/meridian/{client}_national_{is_national}_mdf.csv"
mdf.to_csv(target_file_path, index=False)

# ----------------------------- InputDataLoader --------------------------------- #
coord_to_columns = load.CoordToColumns(
  time='WES',
  geo='Region',
  population='Population',
  kpi='conversions',
  revenue_per_kpi='revenue_per_conversion',
  media=mpa_input.paid_media_imp,
  media_spend=mpa_input.paid_media_spends
)

ordered_chnl_names = [col.split("_")[0] for col in mpa_input.paid_media_imp]
print(f"ordered_chnl_names: {ordered_chnl_names}")
correct_media_to_channel = {col: col.split("_")[0] for col in mpa_input.paid_media_imp}
correct_media_spend_to_channel = {col: col.split("_")[0] for col in mpa_input.paid_media_spends}

# load and format input data
loader = load.CsvDataLoader(
  csv_path=target_file_path,
  kpi_type= 'non_revenue',
  coord_to_columns=coord_to_columns,
  media_to_channel=correct_media_to_channel,
  media_spend_to_channel=correct_media_spend_to_channel,
)
data = loader.load()

# ----------------------------- Define Prior Distribution --------------------------------- #
build_media_channel_args = data.get_paid_media_channels_argument_builder()

# a. beta_m prior (mean for the hierarchical distribution)
media_cf_prior = mpa_input.coeff_prior * correction_factor

if is_national:
  beta_m_prior_sigma = build_media_channel_args(**{chnl:val for chnl, val in zip(ordered_chnl_names, media_cf_prior)})
else:
  # hierarchical model
  allowable_deviation_percent = .1
  geometric_std = np.log(1 + allowable_deviation_percent)
  mu = np.log(0.8 * media_cf_prior) - (geometric_std ** 2 / 2)  # half-normal mean to log-normal mean
  geometric_std = np.full_like(mu, geometric_std)  # log-normal std-dev
  beta_m_prior_mu = build_media_channel_args(**{chnl:val for chnl, val in zip(ordered_chnl_names, mu)})
  beta_m_prior_sigma = build_media_channel_args(**{chnl:val for chnl, val in zip(ordered_chnl_names, geometric_std)})

# a. eta_m prior (std-dev for the hierarchical distribution)
eta_m_mu = 0.0
eta_m_mu_prior = build_media_channel_args(**{chnl:val for chnl, val in zip(ordered_chnl_names, np.repeat(eta_m_mu, len(ordered_chnl_names)))})

allowable_deviation_percent = .1
eta_m_sigma = np.log(1 + allowable_deviation_percent)
eta_m_sigma_prior = build_media_channel_args(**{chnl:val for chnl, val in zip(ordered_chnl_names, np.repeat(eta_m_sigma, len(ordered_chnl_names)))})

# b. Adstock Prior
adstock_lower_bounds = build_media_channel_args(**{chnl:media_parameters_config[chnl]['adstock_range'][0] for chnl in correct_media_to_channel.values()})
adstock_upper_bounds = build_media_channel_args(**{chnl:media_parameters_config[chnl]['adstock_range'][1] for chnl in correct_media_to_channel.values()})

# c. Half saturation Prior
ec50_mu_array = build_media_channel_args(**{chnl:np.round(ec50_multiplier_config[client][chnl], 2) for chnl in correct_media_to_channel.values()})
ec50_scale_array = np.array(ec50_mu_array) * 0.1  # roughly ±90% deviation within 3 standard deviations

# d. Slope Prior
slope_lb = build_media_channel_args(**{chnl:media_parameters_config[chnl]['slope_range'][0] for chnl in correct_media_to_channel.values()})
slope_ub = build_media_channel_args(**{chnl:media_parameters_config[chnl]['slope_range'][1] for chnl in correct_media_to_channel.values()})

# ----- define the prior distribution for the media parameters -----
eta_m=tfp.distributions.Normal(
  [float(x) for x in eta_m_mu_prior], [float(x) for x in eta_m_sigma_prior], name=constants.ETA_M)
alpha_m=tfp.distributions.Uniform([float(x) for x in adstock_lower_bounds], [float(x) for x in adstock_upper_bounds], name=constants.ALPHA_M)
ec_m=tfp.distributions.TruncatedNormal(
  loc=[float(x) for x in ec50_mu_array], scale=[float(x) for x in ec50_scale_array],
  low=[float(0.5) for _ in ec50_mu_array], high=[float(x) for x in ec50_mu_array],
  name=constants.EC_M)
slope_m=tfp.distributions.Uniform([float(x) for x in slope_lb], [float(x) for x in slope_ub], name=constants.SLOPE_M)

# contribution priors
from meridian.model.contribution_prior_helper import create_contribution_prior
beta_dist_params = create_contribution_prior(
      channel_names=['Display', 'TV', 'Video'],
      mean_contributions={'Display': 11.5, 'TV': 30, 'Video': 15},
      confidence='high'  # or 'low' for more flexibility, 'high' for less
  )
contribution_m=tfp.distributions.Beta(
    concentration1=build_media_channel_args(**beta_dist_params['alphas']),
    concentration0=build_media_channel_args(**beta_dist_params['betas']),
    name=constants.CONTRIBUTION_M
)
eta_m = tfp.distributions.HalfNormal(0.3, name=constants.ETA_M)

prior = prior_distribution.PriorDistribution(
    # beta_m=tfp.distributions.Normal(
    #         loc=[float(x) for x in beta_m_prior_mu],
    #         scale=[float(x) for x in beta_m_prior_sigma], name=constants.BETA_M),
    contribution_m=contribution_m,
    eta_m=eta_m,
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)

# -------------------------- Prior Feasibility Check --------------------------------- #
from meridian.model import prior_feasibility
model_spec = spec.ModelSpec(prior=prior, media_prior_type='contribution', media_effects_dist='log_normal')
mmm = model.Meridian(input_data=data, model_spec=model_spec)

# Check feasibility
checker = prior_feasibility.PriorFeasibilityChecker(mmm)
report = checker.check_contribution_feasibility(
    n_draws=1000,
    include_sensitivity=True,
    seed=42
)

print(report)


# # ---------------------------------- Model Run ------------------------------------------- #
# model_spec = spec.ModelSpec(prior=prior, media_prior_type='contribution', media_effects_dist='log_normal')
# mmm = model.Meridian(input_data=data, model_spec=model_spec)
# mmm.sample_prior(2000)
# mmm.sample_posterior(n_chains=2, n_adapt=1000, n_burnin=1000, n_keep=2000)

# # media effects summary
# media_effects_summary = visualizer.MediaSummary(mmm)
# media_effects_summary_df = media_effects_summary.summary_table_without_ci(include_prior=True)
# media_effects_summary_df

# # MAPE & R2
# model_diagnostics = visualizer.ModelDiagnostics(mmm)
# model_diagnostics_df = model_diagnostics.predictive_accuracy_table()
# model_diagnostics_df

# # media parameters summary
# model_summary = az.summary(mmm.inference_data.posterior).reset_index()
# media_model_summary = model_summary[~(model_summary['index'].str.startswith('mu') | model_summary['index'].str.startswith('knot') | model_summary['index'].str.startswith('tau_g'))]
# media_parameters_model_summary = media_model_summary[~(model_summary['index'].str.startswith('beta_') | model_summary['index'].str.startswith('eta_'))]
# media_parameters_model_summary

# # # plot prior vs posterior
# # model_diagnostics = visualizer.ModelDiagnostics(mmm)
# # display(model_diagnostics.plot_prior_and_posterior_distribution(parameter='alpha_m'))
# # display(model_diagnostics.plot_prior_and_posterior_distribution(parameter='ec_m'))
# # display(model_diagnostics.plot_prior_and_posterior_distribution(parameter='slope_m'))
# # display(model_diagnostics.plot_prior_and_posterior_distribution(parameter='contribution_m'))
# # display(model_diagnostics.plot_prior_and_posterior_distribution(parameter='beta_m'))
# # display(model_diagnostics.plot_prior_and_posterior_distribution(parameter='eta_m'))
