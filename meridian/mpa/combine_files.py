import pandas as pd

file_path = "/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/combined_data/"

client_config = {
  'ALDI_US_Starcom': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Allergan': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Audi': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Boehringer_Ingelheim_-_Animal_Health': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Burger_King': {
    "analysis_start_dt": "2024-02-24",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Chick-Fil-A': {
    "analysis_start_dt": "2024-01-13",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Chumba_Casino': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Ford_FDAF': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Huntington_National_Bank': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Hyundai': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'IBM_-_US': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Intuit_-_Quickbooks': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Live_Nation_MasterAdvertiser': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Mattress_Firm_US_Mediavest': {
    "analysis_start_dt": "2024-02-10",
    "analysis_end_dt": "2025-06-28" ,
    "conversion_type": "Purchase"
  },
  'Mazda': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Meijer': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Metro': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  "MRG_Chevy_LMA": {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Progressive_Insurance': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Popeyes': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Samsung_US_Starcom': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'T-Mobile': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  }
}

new_file = pd.read_csv(f"{file_path}/MDF_BY_GEO_EXPANDED_CLIENTS_Jul30_Channel_events.csv")
old_file = pd.read_csv(f"{file_path}/MDF_BY_GEO_Jul8_Channel_events.csv")

# advertisers in old_file but not in new_file
old_file_advertisers = set(old_file['AdvertiserId'].unique())
new_file_advertisers = set(new_file['AdvertiserId'].unique())
advertisers_to_add = old_file_advertisers - new_file_advertisers

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

# ============================================================================
# FILL MISSING WEEKS FOR EACH ADVERTISER'S SPECIFIC REGIONS
# ============================================================================

print("=== FILLING MISSING WEEKS FOR ADVERTISER-SPECIFIC REGIONS ===")
print(f"Original combined_data shape: {combined_data.shape}")

# Create list to store complete data for each advertiser
complete_data_list = []

# Process each advertiser individually with their specific date ranges

for advertiser_name in client_config.keys():
    print(f"Processing {advertiser_name}")
    advertiser_data = combined_data[combined_data['AdvertiserName'] == advertiser_name].copy()
    advertiser_id = advertiser_data['AdvertiserId'].iloc[0]

    # Get client-specific date range
    start_date = client_config[advertiser_name]['analysis_start_dt']
    end_date = client_config[advertiser_name]['analysis_end_dt']
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Generate all weeks between start and end date for this client
    all_client_weeks = pd.date_range(start=start_dt, end=end_dt, freq='W-SAT').strftime('%Y-%m-%d').tolist()

    # Get the specific regions this advertiser operates in
    advertiser_regions = sorted(advertiser_data['Region'].unique())

    # Get the weeks this advertiser currently has data for
    current_weeks = sorted(advertiser_data['WES'].unique())

    # Create complete combinations for this advertiser's regions and client-specific weeks
    import itertools
    advertiser_combinations = list(itertools.product([advertiser_id], [advertiser_name], all_client_weeks, advertiser_regions))
    advertiser_complete_df = pd.DataFrame(advertiser_combinations, columns=['AdvertiserId', 'AdvertiserName', 'WES', 'Region'])

    # Merge with existing data to identify missing combinations
    advertiser_merged = advertiser_complete_df.merge(
        advertiser_data,
        on=['AdvertiserId', 'AdvertiserName', 'WES', 'Region'],
        how='left'
    )

    # sample unmatched rows
    missing_combinations = advertiser_merged[advertiser_merged['Population'].isna()]
    if len(missing_combinations) > 0:
        print(f"Missing combinations for {advertiser_name}: {len(missing_combinations)}")
        print(missing_combinations.head())

    # Identify numeric columns to fill with 0
    numeric_columns = []
    for col in advertiser_merged.columns:
        if col not in ['AdvertiserId', 'WES', 'Region', 'AdvertiserName', 'ConversionType']:
            numeric_columns.append(col)

    # Fill missing numeric values with 0 (except Population)
    for col in numeric_columns:
        if col != 'Population':  # Handle Population separately
            advertiser_merged[col] = advertiser_merged[col].fillna(0)

    # Fill Population with region-specific values
    if 'Population' in advertiser_merged.columns:
        # Create region-population mapping from the original combined_data
        region_population = combined_data.groupby('Region')['Population'].first().to_dict()

        # Fill missing population values using region-specific populations
        advertiser_merged['Population'] = advertiser_merged['Population'].fillna(
            advertiser_merged['Region'].map(region_population)
        )

        # Fallback to 0 if somehow a region is not found (shouldn't happen)
        advertiser_merged['Population'] = advertiser_merged['Population'].fillna(0)

    # Fill AdvertiserName and ConversionType for missing rows
    advertiser_merged['AdvertiserName'] = advertiser_merged['AdvertiserName'].fillna(advertiser_name)

    # Get ConversionType from existing data
    conversion_type = advertiser_data['ConversionType'].iloc[0]
    advertiser_merged['ConversionType'] = advertiser_merged['ConversionType'].fillna(conversion_type)

    # Count how many rows were added
    original_rows = len(advertiser_data)
    complete_rows = len(advertiser_merged)
    added_rows = complete_rows - original_rows

    print(f"  Rows: {original_rows} → {complete_rows} (+{added_rows} filled)")

    complete_data_list.append(advertiser_merged)

# Combine all complete advertiser data
complete_combined_data = pd.concat(complete_data_list, ignore_index=True)
complete_combined_data = complete_combined_data.sort_values(['AdvertiserId', 'WES', 'Region']).reset_index(drop=True)

print(f"\n=== SUMMARY ===")
print(f"Original total rows: {len(combined_data)}")
print(f"Complete total rows: {len(complete_combined_data)}")
print(f"Total rows added: {len(complete_combined_data) - len(combined_data)}")

# Verification: Check that each advertiser now has complete week coverage for their date range
print(f"\n=== VERIFICATION ===")
verification_passed = True
for advertiser_id in sorted(complete_combined_data['AdvertiserId'].unique()):
    advertiser_complete_data = complete_combined_data[complete_combined_data['AdvertiserId'] == advertiser_id]
    advertiser_name = advertiser_complete_data['AdvertiserName'].iloc[0]

    # Get client-specific expected weeks
    start_date = client_config[advertiser_name]['analysis_start_dt']
    end_date = client_config[advertiser_name]['analysis_end_dt']
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    expected_client_weeks = len(pd.date_range(start=start_dt, end=end_dt, freq='W-SAT'))

    advertiser_regions = len(advertiser_complete_data['Region'].unique())
    advertiser_weeks = len(advertiser_complete_data['WES'].unique())
    expected_rows = advertiser_regions * expected_client_weeks
    actual_rows = len(advertiser_complete_data)

    if actual_rows != expected_rows or advertiser_weeks != expected_client_weeks:
        print(f"❌ {advertiser_id}: {actual_rows}/{expected_rows} rows, {advertiser_weeks}/{expected_client_weeks} weeks")
        verification_passed = False

if verification_passed:
    print("✅ SUCCESS: All advertisers now have complete week coverage for their specific regions!")
else:
    print("❌ WARNING: Some advertisers still have incomplete week coverage")

# Save the complete dataset
complete_combined_data.to_csv(f"{file_path}/MDF_BY_GEO_EXPANDED_CLIENTS_Aug11_Channel_events_client_filtered_complete.csv", index=False)

print(f"\nComplete dataset saved to: MDF_BY_GEO_EXPANDED_CLIENTS_Jul30_Channel_events_client_filtered_complete.csv")
print("Sample of complete data:")
combined_data.head()

# ------------------------------------------------------
fp = "/Users/mariappan.subramanian/OneDrive - The Trade Desk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/analysis/Partner_to_Advertiser.csv"
advertiser_to_parnter_map = pd.read_csv(fp)

# create a dictionary of advertiser_id to partner_id
advertiser_to_parnter_map = dict(zip(advertiser_to_parnter_map['AdvertiserId'], advertiser_to_parnter_map['PartnerId']))






# --------------------------------------------------
client_config = {
  'ALDI_US_Starcom': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Allergan': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Audi': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Boehringer_Ingelheim_-_Animal_Health': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Burger_King': {
    "analysis_start_dt": "2024-02-24",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Chick-Fil-A': {
    "analysis_start_dt": "2024-01-13",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Chumba_Casino': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Ford_FDAF': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Huntington_National_Bank': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Hyundai': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'IBM_-_US': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Intuit_-_Quickbooks': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Live_Nation_MasterAdvertiser': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Mattress_Firm_US_Mediavest': {
    "analysis_start_dt": "2024-02-10",
    "analysis_end_dt": "2025-06-28" ,
    "conversion_type": "Purchase"
  },
  'Mazda': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Meijer': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Metro': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  "MRG_Chevy_LMA": {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2024-06-28",
    "conversion_type": "Purchase"
  },
  'Progressive_Insurance': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Popeyes': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'Samsung_US_Starcom': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  },
  'T-Mobile': {
    "analysis_start_dt": "2024-01-06",
    "analysis_end_dt": "2025-06-28",
    "conversion_type": "Purchase"
  }
}

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


unique_advertisers = set(combined_data['AdvertiserName'].unique())

print("=== DEBUG OUTPUT ===")
print(f"Total unique advertisers in data: {len(unique_advertisers)}")
print("Unique advertisers in data:", sorted(unique_advertisers))
print(f"\nTotal advertisers in ec50_multiplier_config: {len(ec50_multiplier_config)}")
print("Advertisers in ec50_multiplier_config:", sorted(ec50_multiplier_config.keys()))

print("\n=== MISSING ADVERTISERS ===")
missing_advertisers = []
for advertiser in unique_advertisers:
  if advertiser not in ec50_multiplier_config:
    missing_advertisers.append(advertiser)
    print(f"Advertiser '{advertiser}' not found in ec50_multiplier_config")

if not missing_advertisers:
    print("All advertisers in data are present in ec50_multiplier_config!")

print(f"\nSummary: {len(missing_advertisers)} out of {len(unique_advertisers)} advertisers are missing from config")

unique_advertiser_ids = set(combined_data['AdvertiserId'].unique())
