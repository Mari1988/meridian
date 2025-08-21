# import pandas as pd

# file_path = "/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/data/final_data/"
# df = pd.read_csv(f"{file_path}/4_mas_mpa_geo_level_raw_media_data_final_nat.csv")
# df['Unique Persons'] = df['ImpressionCount'] / df['Frequency Per Person']

# # pivot the df such that WES, AdvertiserId are index, Channel are columns, and Unique Persons are values
# pivoted_reach_df = df.pivot_table(
#     index=['WES', 'AdvertiserId'],
#     columns='Channel',
#     values='Unique Persons',
#     aggfunc='sum'  # Use 'sum' in case there are duplicate combinations
# ).reset_index()

# # Group by AdvertiserId and calculate median of each column (excluding NaN values)
# median_reach_by_advertiser = pivoted_reach_df.groupby('AdvertiserId').median(numeric_only=True)

# # prior frequency for each channel
# prior_frequency_values = {
#   "Audio": 10.0,
#   "Display": 20.0,
#   "NativeDisplay": 20.0,
#   "TV": 6.0,
#   "Video": 10.0
# }

# # Broadcast prior frequency to match the number of rows in median_reach_by_advertiser
# prior_frequency = pd.DataFrame(
#     [prior_frequency_values] * len(median_reach_by_advertiser),
#     index=median_reach_by_advertiser.index,
#     columns=prior_frequency_values.keys()
# )

# # calculate prior ec50 impressions
# prior_ec50_impressions = median_reach_by_advertiser[prior_frequency.columns] * prior_frequency

# # now, compare the prior ec50 impressions with the actual median impressions
# pivoted_impressions_df = df.pivot_table(
#     index=['WES', 'AdvertiserId'],
#     columns='Channel',
#     values='ImpressionCount',
#     aggfunc='sum'  # Use 'sum' in case there are duplicate combinations
# ).reset_index()

# median_imp_by_advertiser = pivoted_impressions_df.groupby('AdvertiserId').median(numeric_only=True)
# prior_ec50_to_median_imp_ratio = prior_ec50_impressions / median_imp_by_advertiser[prior_ec50_impressions.columns]

# # Floor values to 1 if they are less than 1 or NaN
# prior_ec50_to_median_imp_ratio = prior_ec50_to_median_imp_ratio.fillna(1.0).clip(lower=1.0).astype(int)


# # Convert to nested dictionary: advertiser_id -> {channel: value}
# ratio_dict = prior_ec50_to_median_imp_ratio.to_dict('index')


import pandas as pd

file_path = "/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/data/final_data/"
df = pd.read_csv(f"{file_path}/4_mas_mpa_geo_level_raw_media_data_final_nat.csv")
# df['Unique Persons'] = df['ImpressionCount'] / df['Frequency Per Person']

# pivot the df such that WES, AdvertiserName are index, Channel are columns, and Unique Persons are values
pivoted_freq_df = df.pivot_table(
    index=['WES', 'AdvertiserName'],
    columns='Channel',
    values='Frequency Per Person',
    aggfunc='mean'  # Use 'sum' in case there are duplicate combinations
).reset_index()

# Group by AdvertiserId and calculate median of each column (excluding NaN values)
median_freq_by_advertiser = pivoted_freq_df.groupby('AdvertiserName').median(numeric_only=True).clip(lower=1.0)

# prior frequency for each channel
prior_frequency_values = {
  "Audio": 10.0,
  "Display": 20.0,
  "NativeDisplay": 20.0,
  "TV": 6.0,
  "Video": 10.0
}

# Broadcast prior frequency to match the number of rows in median_reach_by_advertiser
prior_frequency = pd.DataFrame(
    [prior_frequency_values] * len(median_freq_by_advertiser),
    index=median_freq_by_advertiser.index,
    columns=prior_frequency_values.keys()
)

# calculate prior ec50 impressions
prior_ec50_multiplier = (prior_frequency / median_freq_by_advertiser[prior_frequency.columns]).fillna(1.0).round(2)

# Convert to nested dictionary: advertiser_id -> {channel: value}
prior_ec50_multiplier = prior_ec50_multiplier.to_dict('index')


ec50_multiplier_config = {
    'ALDI_US_Starcom': {
        'Audio': 4.55, 'Display': 5.71, 'NativeDisplay': 1.0, 'TV': 1.43, 'Video': 2.86
    },
    'Boehringer_Ingelheim_-_Animal_Health': {
        'Audio': 6.06, 'Display': 8.7, 'NativeDisplay': 1.0, 'TV': 1.54, 'Video': 5.0
    },
    'Burger_King': {
        'Audio': 6.67, 'Display': 4.65, 'NativeDisplay': 1.0, 'TV': 1.62, 'Video': 6.67
    },
    'Chick-Fil-A': {
        'Audio': 1.0, 'Display': 20.0, 'NativeDisplay': 1.0, 'TV': 1.94, 'Video': 7.69
    },
    'Chumba_Casino': {
        'Audio': 10.0, 'Display': 2.35, 'NativeDisplay': 2.04, 'TV': 0.74, 'Video': 1.04
    },
    'Ford_FDAF': {
        'Audio': 1.3, 'Display': 16.67, 'NativeDisplay': 1.0, 'TV': 0.75, 'Video': 2.27
    },
    'General_Motors_FY24_Buick_GMC': {
        'Audio': 10.0, 'Display': 6.56, 'NativeDisplay': 1.0, 'TV': 2.0, 'Video': 7.14
    },
    'Huntington_National_Bank': {
        'Audio': 10.0, 'Display': 18.18, 'NativeDisplay': 20.0, 'TV': 2.73, 'Video': 5.41
    },
    'Hyundai': {
        'Audio': 10.0, 'Display': 9.52, 'NativeDisplay': 6.56, 'TV': 2.0, 'Video': 4.35
    },
    'IBM_-_US': {
        'Audio': 4.76, 'Display': 12.5, 'NativeDisplay': 8.0, 'TV': 2.03, 'Video': 2.44
    },
    'Intuit_-_Quickbooks': {
        'Audio': 7.69, 'Display': 16.67, 'NativeDisplay': 1.0, 'TV': 2.4, 'Video': 10.0
    },
    'Little_Caesars': {
        'Audio': 1.0, 'Display': 20.0, 'NativeDisplay': 1.0, 'TV': 2.07, 'Video': 7.41
    },
    'Live_Nation_MasterAdvertiser': {
        'Audio': 10.0, 'Display': 12.5, 'NativeDisplay': 15.38, 'TV': 3.53, 'Video': 5.88
    },
    'Mattress_Firm_US_Mediavest': {
        'Audio': 1.0, 'Display': 20.0, 'NativeDisplay': 1.0, 'TV': 2.22, 'Video': 4.0
    },
    'Mazda': {
        'Audio': 6.25, 'Display': 8.7, 'NativeDisplay': 1.0, 'TV': 2.31, 'Video': 7.14
    },
    'Mazda Tier 3': {
        'Audio': 2.5, 'Display': 7.14, 'NativeDisplay': 1.0, 'TV': 2.07, 'Video': 4.76
    },
    'Meijer': {
        'Audio': 9.09, 'Display': 13.33, 'NativeDisplay': 20.0, 'TV': 1.76, 'Video': 10.0
    },
    'Metro': {
        'Audio': 10.0, 'Display': 20.0, 'NativeDisplay': 1.0, 'TV': 6.0, 'Video': 9.09
    },
    'Progressive_Insurance': {
        'Audio': 1.0, 'Display': 2.78, 'NativeDisplay': 3.39, 'TV': 1.2, 'Video': 1.89
    },
    "Sam's_Club": {
        'Audio': 10.0, 'Display': 3.77, 'NativeDisplay': 1.0, 'TV': 2.5, 'Video': 5.56
    },
    'Samsung_US_Starcom': {
        'Audio': 1.0, 'Display': 14.29, 'NativeDisplay': 20.0, 'TV': 2.4, 'Video': 5.26
    },
    'Southwest_Airlines': {
        'Audio': 3.33, 'Display': 7.41, 'NativeDisplay': 20.0, 'TV': 1.25, 'Video': 2.94
    },
    'Starbucks_US_Spark': {
        'Audio': 7.69, 'Display': 14.29, 'NativeDisplay': 10.0, 'TV': 2.86, 'Video': 5.0
    },
    'T-Mobile': {
        'Audio': 10.0, 'Display': 20.0, 'NativeDisplay': 20.0, 'TV': 6.0, 'Video': 10.0
    },
    '[DNU] Metro - Initiative': {
        'Audio': 1.0, 'Display': 1.0, 'NativeDisplay': 1.0, 'TV': 2.73, 'Video': 7.14
    },
    '[DNU] T-Mobile - Initiative': {
        'Audio': 1.0, 'Display': 1.0, 'NativeDisplay': 1.0, 'TV': 1.71, 'Video': 5.88
    },
    'MRG_Chevy_LMA': {
        'Audio': 1.0, 'Display': 4.5, 'NativeDisplay': 1.0, 'TV': 1.0, 'Video': 3.0
    },
    'Allergan': {
        'Audio': 1.0, 'Display': 7.12, 'NativeDisplay': 1.0, 'TV': 2.9, 'Video': 4.4
    },
    'Audi': {
        'Audio': 1.0, 'Display': 9.2, 'NativeDisplay': 1.0, 'TV': 2.9, 'Video': 5.9
    },
    'Popeyes': {
        'Audio': 1.0, 'Display': 4.56, 'NativeDisplay': 1.0, 'TV': 2.54, 'Video': 5.5
    }
}

import pandas as pd
ec50_multiplier_df = pd.DataFrame(ec50_multiplier_config).T
ec50_multiplier_df.to_csv("/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/data/final_data/ec50_multiplier_df.csv")
