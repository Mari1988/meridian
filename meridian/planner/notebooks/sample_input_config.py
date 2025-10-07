# Configuration for input excel file
input_config = {

  # time and geo inputs
  'time_col': 'week',
  'geo_col': 'geo',
  'population_col': 'population',

  # kpi inputs
  'kpi_col': 'conversions',  #
  'kpi_type': 'non_revenue',
  'revenue_per_kpi_col': 'revenue_per_conversion',  # needed if kpi_type is non_revenue

  # impression based media inputs
  'media_cols': ['Display_impression', 'TV_impression', 'Video_impression'],
  'media_spend_cols': ['Display_spend', 'TV_spend', 'Video_spend'],
  'media_channels': ['Display', 'TV', 'Video']

}

# optimization config
optimization_config = {
  'fixed_budget': True,
  'use_kpi': True,

  # spend constraints
  'spend_constraint_lower': {  # (1 - value)% of historical
    'Display': 0.3,
    'TV': 0.3,
    'Video': 0.3,
  },

  'spend_constraint_upper': {  # (1 + value)% of historical
  'Display': 0.3,
  'TV': 0.3,
  'Video': 0.3,
  }
}

opt_period = {
    'Mazda': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Live_Nation_MasterAdvertiser': {'start_date': '2024-07-06','end_date': '2025-06-28'},
    'Huntington_National_Bank': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Hyundai': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Burger_King': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'IBM_-_US': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Meijer': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Audi': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'MRG_Chevy_LMA': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Chumba_Casino': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Allergan': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Popeyes': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Samsung_US_Starcom': {'start_date': '2024-07-06', 'end_date': '2025-06-28'},
    'Intuit_-_Quickbooks': {'start_date': '2024-07-06', 'end_date': '2025-06-28'}
}
