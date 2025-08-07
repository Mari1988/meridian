import pandas as pd

file_path = "/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/data/final_data/"
df = pd.read_csv(f"{file_path}/4_mas_mpa_geo_level_raw_media_data_final_nat.csv")
df['Unique Persons'] = df['ImpressionCount'] / df['Frequency Per Person']

# pivot the df such that WES, AdvertiserId are index, Channel are columns, and Unique Persons are values
pivoted_reach_df = df.pivot_table(
    index=['WES', 'AdvertiserId'],
    columns='Channel',
    values='Unique Persons',
    aggfunc='sum'  # Use 'sum' in case there are duplicate combinations
).reset_index()

# Group by AdvertiserId and calculate median of each column (excluding NaN values)
median_reach_by_advertiser = pivoted_reach_df.groupby('AdvertiserId').median(numeric_only=True)

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
    [prior_frequency_values] * len(median_reach_by_advertiser),
    index=median_reach_by_advertiser.index,
    columns=prior_frequency_values.keys()
)

# calculate prior ec50 impressions
prior_ec50_impressions = median_reach_by_advertiser[prior_frequency.columns] * prior_frequency

# now, compare the prior ec50 impressions with the actual median impressions
pivoted_impressions_df = df.pivot_table(
    index=['WES', 'AdvertiserId'],
    columns='Channel',
    values='ImpressionCount',
    aggfunc='sum'  # Use 'sum' in case there are duplicate combinations
).reset_index()

median_imp_by_advertiser = pivoted_impressions_df.groupby('AdvertiserId').median(numeric_only=True)
prior_ec50_to_median_imp_ratio = prior_ec50_impressions / median_imp_by_advertiser[prior_ec50_impressions.columns]

# Floor values to 1 if they are less than 1 or NaN
prior_ec50_to_median_imp_ratio = prior_ec50_to_median_imp_ratio.fillna(1.0).clip(lower=1.0).astype(int)

# Convert to nested dictionary: advertiser_id -> {channel: value}
ratio_dict = prior_ec50_to_median_imp_ratio.to_dict('index')
