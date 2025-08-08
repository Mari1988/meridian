import pandas as pd

file_path = "/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/combined_data/"


new_file = pd.read_csv(f"{file_path}/MDF_BY_GEO_EXPANDED_CLIENTS_Jul30_Channel_events.csv")
old_file = pd.read_csv(f"{file_path}/MDF_BY_GEO_Jul8_Channel_events.csv")

# advertisers in old_file but not in new_file
old_file_advertisers = set(old_file['AdvertiserId'].unique())
new_file_advertisers = set(new_file['AdvertiserId'].unique())
advertisers_to_add =old_file_advertisers - new_file_advertisers

old_file_to_add = old_file[old_file['AdvertiserId'].isin(advertisers_to_add)]

# common column names
common_columns = list(set(old_file.columns) & set(new_file.columns))
common_columns = [col for col in new_file.columns if col in common_columns]

combined_data = pd.concat([new_file[common_columns], old_file_to_add[common_columns]], ignore_index=True)
# combined_data.to_csv(f"{file_path}/MDF_BY_GEO_EXPANDED_CLIENTS_Jul30_Channel_events_combined.csv", index=False)

# create dummy variables for reach and frequency with 1.0 defaulted
reach_variables = ["TV_RHH", "Display_RPP", "Video_RPP"]
frequency_variables = ["TV_FHH", "Display_FPP", "Video_FPP"]

combined_data[reach_variables] = 1.0
combined_data[frequency_variables] = 1.0

combined_data.to_csv(f"{file_path}/MDF_BY_GEO_EXPANDED_CLIENTS_Jul30_Channel_events_combined_rf_dummy_filled.csv", index=False)
