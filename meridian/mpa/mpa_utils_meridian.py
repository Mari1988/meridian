import numpy as np
import pandas as pd
import json
import altair as alt
from pathlib import Path
from html import escape

class MeridianMPAInput:

# class level constants
  date_field = "WES"

  def __init__(self, client, client_config, main_config):

    # common configs
    self.file_path = main_config['file_path']
    self.paid_media_cols = main_config['paid_media_cols']
    self.reach_variables = main_config['reach_variables']
    self.frequency_variables = main_config['frequency_variables']
    self.paid_media_imp = main_config['paid_media_imp']
    self.paid_media_viewability_imp = main_config['paid_media_viewability_imp']
    self.paid_media_spends = main_config['paid_media_spends']
    self.spend_variables_for_cpm_calc = main_config['spend_variables_for_cpm_calc']
    self.imp_variables_for_cpm_calc = main_config['imp_variables_for_cpm_calc']
    self.adstock_range_low = main_config.get('adstock_range_low')
    self.adstock_range_high = main_config.get('adstock_range_high')
    self.prior_halfsat_frequency = main_config.get('prior_halfsat_frequency')
    self.holiday_file_path = main_config['holiday_file_path'] + f"{client}.csv"
    self.target = main_config['response_kpi']
    self.prior_type = main_config['prior_type']

    # client configs
    config = client_config[client]
    self.advertiser = client
    self.conversion_type = config['conversion_type']
    self.analysis_start_dt = pd.to_datetime(config['analysis_start_dt'])
    self.analysis_end_dt = pd.to_datetime(config['analysis_end_dt'])

    # create model input dataset
    self.mdf_mw = self.read_model_input_data()
    self.coeff_prior = self.get_prior_vector(prior_type=self.prior_type)

    if not isinstance(self.coeff_prior, np.ndarray):
      self.coeff_prior = self.coeff_prior.values

    return

  def read_model_input_data(self):

    # read data
    mdf_all = pd.read_csv(self.file_path)

    # filter down to the specific advertiser and the conversion type
    mdf = mdf_all[(mdf_all['AdvertiserName'] == self.advertiser) & (mdf_all['ConversionType'] == self.conversion_type)]

    # bound the data to the analysis window
    mdf.loc[:, self.date_field] = pd.to_datetime(mdf[self.date_field])
    mdf_mw = mdf[(mdf[self.date_field] >= self.analysis_start_dt) & (mdf[self.date_field] <= self.analysis_end_dt)].reset_index(drop=True)

    # non-zero input cols
    nz_input_cols_bool = (mdf_mw[self.paid_media_imp].sum() > 0).tolist()

    self.paid_media_imp = [col for col, flag in zip(self.paid_media_imp, nz_input_cols_bool) if flag]
    self.paid_media_cols = [col for col, flag in zip(self.paid_media_cols, nz_input_cols_bool) if flag]
    self.paid_media_spends = [col for col, flag in zip(self.paid_media_spends, nz_input_cols_bool) if flag]
    self.reach_variables = [col for col, flag in zip(self.reach_variables, nz_input_cols_bool) if flag]
    self.frequency_variables = [col for col, flag in zip(self.frequency_variables, nz_input_cols_bool) if flag]
    self.paid_media_viewability_imp = [col for col, flag in zip(self.paid_media_viewability_imp, nz_input_cols_bool) if flag]
    self.spend_variables_for_cpm_calc = [col for col, flag in zip(self.spend_variables_for_cpm_calc, nz_input_cols_bool) if flag]
    self.imp_variables_for_cpm_calc = [col for col, flag in zip(self.imp_variables_for_cpm_calc, nz_input_cols_bool) if flag]

    if self.adstock_range_low is not None:
      self.adstock_range_low = [col for col, flag in zip(self.adstock_range_low, nz_input_cols_bool) if flag]

    if self.adstock_range_high is not None:
      self.adstock_range_high = [col for col, flag in zip(self.adstock_range_high, nz_input_cols_bool) if flag]

    if self.prior_halfsat_frequency is not None:
      self.prior_halfsat_frequency = [col for col, flag in zip(self.prior_halfsat_frequency, nz_input_cols_bool) if flag]

    return mdf_mw

  def get_prior_vector(self, prior_type="cpm_weighted_by_spend"):

    # 1. cost vector
    costs_array = self.mdf_mw[self.paid_media_spends].sum(axis=0)

    # 2. get CPM vector
    cost_sum_array_for_cpm_prior = np.array(self.mdf_mw[self.spend_variables_for_cpm_calc].sum(axis=0))
    imp_sum_array_for_cpm_prior = np.array(self.mdf_mw[self.imp_variables_for_cpm_calc].sum(axis=0))
    cpm_array = (cost_sum_array_for_cpm_prior / imp_sum_array_for_cpm_prior)

    # 3. viewability rate
    total_impressions = self.mdf_mw[self.paid_media_imp].sum(axis=0)
    viewable_impressions = self.mdf_mw[self.paid_media_viewability_imp].sum(axis=0)
    viewability_array = viewable_impressions.values / total_impressions.values

    print(f"cost array: {costs_array.tolist()}")
    print(f"cpm array: {cpm_array.tolist()}")
    print(f"viewability percent {viewability_array}")

    # Get the coefficient priors
    if prior_type == "spend":
      coeff_prior = costs_array / np.max(costs_array)  # --> Spend Prior
    elif prior_type == "working_spend":
      working_cost_array = (costs_array * viewability_array)/sum(costs_array * viewability_array)
      print(f"working cost array: {working_cost_array.tolist()}")
      coeff_prior = working_cost_array / np.max(working_cost_array)
    elif prior_type == "cpm_weighted_by_spend":
      cost_weights = costs_array/sum(costs_array)
      cpm_array_cost_weighted = cpm_array * np.array(cost_weights)
      coeff_prior = cpm_array_cost_weighted / np.max(cpm_array_cost_weighted)
    elif prior_type == "cpm_weighted_by_working_spend":
      working_cost_array = (costs_array * viewability_array)/sum(costs_array * viewability_array)
      print(f"working cost array: {working_cost_array.tolist()}")
      cpm_array_working_spend_weighted = cpm_array * np.array(working_cost_array)
      coeff_prior = cpm_array_working_spend_weighted / np.max(cpm_array_working_spend_weighted)  # --> working spend scaled CPM prior
    else:
      ValueError("invalid prior type")

      # self.costs_scaled = self.cost_scaler.fit_transform(costs_train.values)
      print(f"prior type is {prior_type} and prior values are {coeff_prior}")

    return coeff_prior

  def get_informative_halfsat_prior_multipliers():
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

    return ec50_dict

  @staticmethod
  def altair_charts_to_html(
      charts,
      page_title="Meridian Charts",
      include_actions=False,
      css=None,
      minified=True,
  ):
    """
    Build a standalone HTML page that renders multiple Altair charts via Vega-Embed.

    Parameters
    ----------
    charts : sequence of tuples
        Each element: (chart_obj, title_str, element_id=None)
        - chart_obj: Altair Chart (e.g., FacetChart, LayerChart, etc.)
        - title_str: Heading shown above the chart (plain text; will be HTML-escaped)
        - element_id: Optional DOM id (if None, auto 'chart_i')
    page_title : str
        <title> tag for the HTML page.
    include_actions : bool
        Show/Hide Vega-Embed action buttons (export, editor, etc.).
    css : str or None
        Additional CSS injected into <style>. Reasonable defaults used if None.
    minified : bool
        If True, strips most whitespace from generated HTML.

    Returns
    -------
    html_str : str
        Complete HTML document as a string (UTF-8 safe).
    """
    # --- default CSS ---------------------------------------------------------
    default_css = """
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
      margin: 2rem;
      line-height: 1.4;
      color: #1b1b1b;
      background: #ffffff;
    }
    h1 {
      margin: 0 0 1.5rem 0;
      font-size: 1.75rem;
      font-weight: 600;
      color: #202124;
    }
    h2 {
      margin: 2rem 0 0.5rem 0;
      font-size: 1.25rem;
      font-weight: 500;
      color: #202124;
    }
    .chart-block {
      margin-bottom: 2.5rem;
    }
    """
    if css is None:
        css = default_css

    # --- serialize specs -----------------------------------------------------
    # We'll assign each spec to a JS variable using safe JSON dumping.
    script_lines = []
    chart_sections_html = []

    for i, (chart_obj, title_str, element_id) in enumerate(charts):
        element_id = element_id or f"chart_{i}"
        # Ensure valid JSON: Altair -> JSON string -> Python dict -> JSON dump
        # (round-trip eliminates trailing commas or non-JSON artifacts).
        chart_json_str = chart_obj.to_json()
        chart_spec = json.loads(chart_json_str)
        chart_spec_js = json.dumps(chart_spec, indent=None if minified else 2)

        # heading text escaped
        safe_title = escape(title_str)

        # section markup
        chart_sections_html.append(
            f'<section class="chart-block"><h2>{safe_title}</h2><div id="{element_id}"></div></section>'
        )
        # embed script line
        script_lines.append(
            f"vegaEmbed('#{element_id}', {chart_spec_js}, {{actions: {str(include_actions).lower()}}});"
        )

    charts_html = "\n".join(chart_sections_html)
    embed_script = "\n".join(script_lines)

    # --- full HTML doc -------------------------------------------------------
    html = f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>{escape(page_title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>{css}</style>
    <!-- Vega libs from CDN -->
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    </head>
    <body>
    <h1>{escape(page_title)}</h1>
    {charts_html}
    <script>
    {embed_script}
    </script>
    </body>
    </html>"""

    if minified:
        # light compaction (avoid full minifier to keep readable debugging)
        html = "\n".join(line.strip() for line in html.splitlines() if line.strip())

    return html

  @staticmethod
  def save_altair_charts_html(
      charts, file_path, **kwargs,
  ):
    """
    Convenience wrapper: build HTML (via altair_charts_to_html) and write to disk.

    Returns the file path.
    """
    html_str = MeridianMPAInput.altair_charts_to_html(charts, **kwargs)
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(html_str, encoding="utf-8")
    return str(file_path)
