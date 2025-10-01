import numpy as np

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
  # 'Boehringer_Ingelheim_-_Animal_Health': {
  #   "analysis_start_dt": "2024-01-06",
  #   "analysis_end_dt": "2025-06-28",
  #   "conversion_type": "Purchase"
  # },
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
  # 'Ford_FDAF': {
  #   "analysis_start_dt": "2024-01-06",
  #   "analysis_end_dt": "2025-06-28",
  #   "conversion_type": "Purchase"
  # },
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
  # 'Metro': {
  #   "analysis_start_dt": "2024-01-06",
  #   "analysis_end_dt": "2025-06-28",
  #   "conversion_type": "Purchase"
  # },
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
  }
  # 'T-Mobile': {
  #   "analysis_start_dt": "2024-01-06",
  #   "analysis_end_dt": "2025-06-28",
  #   "conversion_type": "Purchase"
  # }
}


ec50_multiplier_config = {'ALDI_US_Starcom': {'Display': 3.71, 'TV': 1.0, 'Video': 1.43},
 'Allergan': {'Display': 4.81, 'TV': 1.3, 'Video': 2.5},
 'Audi': {'Display': 5.42, 'TV': 1.5, 'Video': 2.78},
 'Burger_King': {'Display': 3.02, 'TV': 1.0, 'Video': 3.33},
 'Chick-Fil-A': {'Display': np.nan, 'TV': 1.0, 'Video': 3.85},
 'Chumba_Casino': {'Display': 1.53, 'TV': 1.0, 'Video': 1.0},
 'Huntington_National_Bank': {'Display': 11.82, 'TV': 1.36, 'Video': 2.7},
 'Hyundai': {'Display': 6.19, 'TV': 1.0, 'Video': 2.17},
 'IBM_-_US': {'Display': 8.12, 'TV': 1.02, 'Video': 1.22},
 'Intuit_-_Quickbooks': {'Display': 10.83, 'TV': 1.2, 'Video': 5.0},
 'Live_Nation_MasterAdvertiser': {'Display': 8.12, 'TV': 1.76, 'Video': 2.94},
 'MRG_Chevy_LMA': {'Display': 2.83, 'TV': 1.0, 'Video': 1.56},
 'Mattress_Firm_US_Mediavest': {'Display': np.nan, 'TV': 1.11, 'Video': 2.0},
 'Mazda': {'Display': 5.65, 'TV': 1.15, 'Video': 3.57},
 'Meijer': {'Display': 8.67, 'TV': 1.0, 'Video': 5.0},
 'Popeyes': {'Display': 3.25, 'TV': 1.5, 'Video': 3.12},
 'Progressive_Insurance': {'Display': 1.81, 'TV': 1.0, 'Video': 1.0},
 'Samsung_US_Starcom': {'Display': 9.29, 'TV': 1.2, 'Video': 2.63}}


media_parameters_config = {

  'TV': {
    'adstock_range': [0, 0.8],
    'halfsat_prior_freq': 3.0,
    'slope_range': [1.0, 3.0]
  },
  'Display': {
    'adstock_range': [0, 0.45],
    'halfsat_prior_freq': 13.0,
    'slope_range': [0.5, 1.0]
  },
  'Video': {
    'adstock_range': [0, 0.69],
    'halfsat_prior_freq': 5.0,
    'slope_range': [1.0, 2.0]
  }

}
