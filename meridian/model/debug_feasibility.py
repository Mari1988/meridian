"""Debug script to investigate contribution % calculation issue."""
import sys
sys.path.insert(0, '/Users/mariappan.subramanian/Documents/repo/forked/meridian')

import numpy as np
import tensorflow_probability as tfp
from meridian import constants
from meridian.data import load
from meridian.model import model, spec, prior_distribution
from meridian.analysis import analyzer
from meridian.mpa.mpa_utils_meridian import MeridianMPAInput
from meridian.mpa.client_config import client_config, ec50_multiplier_config, media_parameters_config
from meridian.model.contribution_prior_helper import create_contribution_prior

# Setup (minimal version)
home_dir = '/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/tmp'
client = 'Mazda'

main_config = {
    "file_path": f"{home_dir}/data/MDF_BY_GEO_EXPANDED_CLIENTS_Aug11.csv",
    "paid_media_imp": ["TV_I", "Display_I", "Video_I"],
    "paid_media_spends": ["TV_AC", "Display_AC", "Video_AC"],
    "paid_media_viewability_imp": ["TV_VCR", "Display_VCR", "Video_VCR"],
    "paid_media_cols": ["TV_I", "Display_I", "Video_I"],
    "reach_variables": ["TV_RHH", "Display_RPP", "Video_RPP"],
    "frequency_variables": ["TV_FHH", "Display_FPP", "Video_FPP"],
    "spend_variables_for_cpm_calc": ["TV_AC", "Display_AC", "Video_AC"],
    "imp_variables_for_cpm_calc": ['TV_I', 'Display_I', 'Video_I'],
    "response_kpi": "conversions",
    "prior_config": {},
    "prior_type": "spend"
}

prior_type = 'working_spend'
is_national = False

# Create input
main_config['prior_type'] = prior_type
mpa_input = MeridianMPAInput(client=client, client_config=client_config, main_config=main_config)
mdf_mw = mpa_input.mdf_mw.copy()

metrics = [mpa_input.target] + mpa_input.paid_media_imp + mpa_input.paid_media_spends
mdf = mdf_mw[['WES'] + metrics + ['Region', 'Population']].reset_index(drop=True).assign(revenue_per_conversion=1.0).copy()
import pandas as pd
mdf.loc[:, 'WES'] = pd.to_datetime(mdf['WES']).dt.strftime('%Y-%m-%d')

target_file_path = f"{home_dir}/trash/meridian/{client}_national_{is_national}_mdf.csv"
mdf.to_csv(target_file_path, index=False)

coord_to_columns = load.CoordToColumns(
    time='WES', geo='Region', population='Population',
    kpi='conversions', revenue_per_kpi='revenue_per_conversion',
    media=mpa_input.paid_media_imp, media_spend=mpa_input.paid_media_spends
)

ordered_chnl_names = [col.split("_")[0] for col in mpa_input.paid_media_imp]
correct_media_to_channel = {col: col.split("_")[0] for col in mpa_input.paid_media_imp}
correct_media_spend_to_channel = {col: col.split("_")[0] for col in mpa_input.paid_media_spends}

loader = load.CsvDataLoader(
    csv_path=target_file_path, kpi_type='non_revenue',
    coord_to_columns=coord_to_columns,
    media_to_channel=correct_media_to_channel,
    media_spend_to_channel=correct_media_spend_to_channel,
)
data = loader.load()

build_media_channel_args = data.get_paid_media_channels_argument_builder()

# Priors
adstock_lower_bounds = build_media_channel_args(**{chnl:media_parameters_config[chnl]['adstock_range'][0] for chnl in correct_media_to_channel.values()})
adstock_upper_bounds = build_media_channel_args(**{chnl:media_parameters_config[chnl]['adstock_range'][1] for chnl in correct_media_to_channel.values()})
ec50_mu_array = build_media_channel_args(**{chnl:np.round(ec50_multiplier_config[client][chnl], 2) for chnl in correct_media_to_channel.values()})
ec50_scale_array = np.array(ec50_mu_array) * 0.1
slope_lb = build_media_channel_args(**{chnl:media_parameters_config[chnl]['slope_range'][0] for chnl in correct_media_to_channel.values()})
slope_ub = build_media_channel_args(**{chnl:media_parameters_config[chnl]['slope_range'][1] for chnl in correct_media_to_channel.values()})

alpha_m = tfp.distributions.Uniform([float(x) for x in adstock_lower_bounds], [float(x) for x in adstock_upper_bounds], name=constants.ALPHA_M)
ec_m = tfp.distributions.TruncatedNormal(loc=[float(x) for x in ec50_mu_array], scale=[float(x) for x in ec50_scale_array],
                                          low=[float(0.5) for _ in ec50_mu_array], high=[float(x) for x in ec50_mu_array], name=constants.EC_M)
slope_m = tfp.distributions.Uniform([float(x) for x in slope_lb], [float(x) for x in slope_ub], name=constants.SLOPE_M)

beta_dist_params = create_contribution_prior(
    channel_names=['Display', 'TV', 'Video'],
    mean_contributions={'Display': 11.5, 'TV': 30, 'Video': 15},
    confidence='high'
)
contribution_m = tfp.distributions.Beta(
    concentration1=build_media_channel_args(**beta_dist_params['alphas']),
    concentration0=build_media_channel_args(**beta_dist_params['betas']),
    name=constants.CONTRIBUTION_M
)
eta_m = tfp.distributions.HalfNormal(0.3, name=constants.ETA_M)

prior = prior_distribution.PriorDistribution(
    contribution_m=contribution_m,
    eta_m=eta_m,
    alpha_m=alpha_m,
    ec_m=ec_m,
    slope_m=slope_m
)

model_spec = spec.ModelSpec(prior=prior, media_prior_type='contribution', media_effects_dist='log_normal')
mmm = model.Meridian(input_data=data, model_spec=model_spec)

# Sample prior
print("Sampling from prior...")
mmm.sample_prior(100, seed=42)  # Use smaller sample for debugging

# Create analyzer
mmm_analyzer = analyzer.Analyzer(mmm)

# Get incremental outcome
print("\nGetting incremental outcome...")
incremental_outcome = mmm_analyzer.incremental_outcome(
    new_data=None,
    use_posterior=False,
    aggregate_geos=False,
    aggregate_times=False,
).numpy()

print(f"incremental_outcome shape: {incremental_outcome.shape}")
print(f"incremental_outcome min: {incremental_outcome.min()}")
print(f"incremental_outcome max: {incremental_outcome.max()}")
print(f"incremental_outcome mean: {incremental_outcome.mean()}")

# Get total outcome
print("\nGetting total outcome...")
total_outcome = mmm_analyzer.expected_outcome(
    new_data=None,
    use_posterior=False,
    aggregate_geos=False,
    aggregate_times=False,
).numpy()

print(f"total_outcome shape: {total_outcome.shape}")
print(f"total_outcome min: {total_outcome.min()}")
print(f"total_outcome max: {total_outcome.max()}")
print(f"total_outcome mean: {total_outcome.mean()}")

# Check for zeros or negatives in total_outcome
print(f"\nTotal outcome zeros: {(total_outcome == 0).sum()}")
print(f"Total outcome negatives: {(total_outcome < 0).sum()}")

# Sum over geo and time
incremental_outcome_total = incremental_outcome.sum(axis=(2, 3))
total_outcome_summed = total_outcome.sum(axis=(2, 3))

print(f"\nincremental_outcome_total shape: {incremental_outcome_total.shape}")
print(f"total_outcome_summed shape: {total_outcome_summed.shape}")
print(f"total_outcome_summed min: {total_outcome_summed.min()}")
print(f"total_outcome_summed max: {total_outcome_summed.max()}")

# Check for zeros in summed total outcome
print(f"total_outcome_summed zeros: {(total_outcome_summed == 0).sum()}")
print(f"total_outcome_summed negatives: {(total_outcome_summed < 0).sum()}")

# Calculate contribution %
contribution_pct = (
    incremental_outcome_total / total_outcome_summed[..., np.newaxis] * 100
)

print(f"\ncontribution_pct shape: {contribution_pct.shape}")
print(f"contribution_pct min: {contribution_pct.min()}")
print(f"contribution_pct max: {contribution_pct.max()}")
print(f"contribution_pct mean per channel: {contribution_pct.mean(axis=(0,1))}")
print(f"contribution_pct std per channel: {contribution_pct.std(axis=(0,1))}")

# Check for inf/nan
print(f"\ncontribution_pct has inf: {np.isinf(contribution_pct).sum()}")
print(f"contribution_pct has nan: {np.isnan(contribution_pct).sum()}")

# Compare with your calculation
print("\n" + "="*80)
print("COMPARISON WITH YOUR CALCULATION")
print("="*80)
print(f"Your mean: [20.57%, 7.87%, 10.27%]")
print(f"My mean:   {contribution_pct.mean(axis=(0,1))}")
print(f"\nYour std:  [1.65%, 0.66%, 0.86%]")
print(f"My std:    {contribution_pct.std(axis=(0,1))}")
