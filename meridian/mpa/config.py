client_config = {
    "Audi": {
      "conversion_type": "Purchase",
      "analysis_start_dt": "2024-02-01",
      "analysis_end_dt": "2025-06-28"
    },
    "Popeyes": {
      "conversion_type": "Purchase",
      "analysis_start_dt": "2024-01-06",
      "analysis_end_dt": "2025-06-28"
    },
   "Allergan": {
      "conversion_type": "Purchase",
      "analysis_start_dt": "2024-01-20",
      "analysis_end_dt": "2025-06-28"
    },
    "Burger_King": {
      "conversion_type": "Purchase",
      "analysis_start_dt": "2024-03-16",
      "analysis_end_dt": "2025-06-28"
    },
    "Mazda": {
      "conversion_type": "Purchase",
      "analysis_start_dt": "2024-02-10",
      "analysis_end_dt": "2025-06-28"
    },
    "Samsung_US_Starcom": {
      "conversion_type": "Purchase",
      "analysis_start_dt":"2024-01-01",
      "analysis_end_dt": "2025-06-28"
    }

  }

ec50_dict = {
  "Audi": [2.9631775, 9.19013014, 5.88888516],
  "Allergan": [2.90128433, 7.11940649, 4.38604941],
  "Burger_King": [1.85037867, 3.32329386, 4.7274248],
  "Chick-Fil-A": [2.07933822, 4.77448688],
  "Mazda": [1.82100474, 5.35179257, 4.64883359],
  "MRG_Chevy_LMA": [0.98180917, 4.50563904, 3.04360945],
  "Popeyes": [2.54705493, 4.56689341, 5.49848168],
  "Samsung_US_Starcom": [1.99006785, 6.8736896, 1.99493994]
}

def get_main_config(home_dir):
 return {
    "file_path": f"{home_dir}/data/MDF_BY_GEO_Jul8_Channel_events_wit_population.csv",
    # "file_path": f"{home_dir}/data/MDF_May30_Channel_events.csv",

    "holiday_file_path": f"{home_dir}/data/holidays_updated_apr21/holiday_effect_",

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
