### Create Meridian model input data and configuration from user provided excel file.

#### Excel file structure.
1. 'Data' sheet has the model input data for MMM
2. 'Coefficients' sheet has the geo level coefficient for different media variables
3. 'Parameters' sheet has the media parameters such as adstock, hill parameters (inflexion and shape)

#### Model config json
1. Below is a sample config

```python
    model_config = {

    # time and geo inputs
    'time_col': 'week',
    'geo_col': 'geo',
    'population_col': 'population',

    # kpi inputs
    'kpi_type': 'non_revenue',
    'kpi_col': 'conversions',
    'revenue_per_kpi_col': 'revenue_per_conversion',

    # impression based media inputs
    'media_cols': ['Channel0_impression', 'Channel1_impression', 'Channel2_impression'],
    'media_spend_cols': ['Channel0_spend', 'Channel1_spend', 'Channel2_spend'],
    'media_channels': ['Channel0', 'Channel1', 'Channel2'],

    # reach based media inputs
    'reach_cols': ['Channel3_reach'],
    'frequency_cols': ['Channel3_frequency'],
    'rf_spend_cols': ['Channel3_spend'],
    'rf_channels': ['Channel3'],

    # control inputs
    'control_cols': ['sentiment_score_control', 'competitor_activity_score_control']
    }
```

#### instructions
##### step1: create 'data' of type data_frame_input_data_builder.DataFrameInputDataBuilder
1. Create an input class called AdhocDataLoader inside planner folder with file_name, and model_config as arguments
2. For testing use the file located at /Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/BudgetOptimizer/mmm_input_artifacts.xlsx and the model_config above
3. Parse the input data ('Data' sheet in the excel) and the model_config to create data object.
4. Here is a sample block of code on how to do it. But go through meridian utils and figure out more.

```python
builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
    kpi_type='non_revenue'
)
builder = (
    builder.with_kpi(df, kpi_col="conversions")
    .with_revenue_per_kpi(df, revenue_per_kpi_col="revenue_per_conversion")
    .with_population(df)
    .with_controls(
        df,
        control_cols=[
            "sentiment_score_control",
            "competitor_activity_score_control",
        ],
    )
)

channels = ["Channel0", "Channel1", "Channel2"]
builder = builder.with_media(
    df,
    media_cols=[f"{channel}_impression" for channel in channels],
    media_spend_cols=[f"{channel}_spend" for channel in channels],
    media_channels=channels,
).with_reach(
    df,
    reach_cols=["Channel3_reach"],
    frequency_cols=["Channel3_frequency"],
    rf_spend_cols=["Channel3_spend"],
    rf_channels=["Channel3"],
)
```

##### step2: process 'Parameters' sheet

Follow the instructions below to process the data in 'Parameters' sheet.

##### Validation and Error handling
1. In the beginning add a validation step that values in the 'MediaVariable' must be one of those in the media_channels + rf_channels lists. Also, every one of those in the combined list should be listed under MediaVaraible column.\
2. The following columns must be in the file: Adstock, Inflexion, Slope

##### Logic
1. The order of values in MediaVariable column should be in the same order as the order in media_channels + rf_channels list. If not, rearrange to make sure they align.
2. First check if a 'MediaVariable' belongs to media_channels list or rf_channels in model_config
3. If it belongs to media_channels then \
    a) assign Adstock value to a list named based on constants.ALPHA_M \
    b) assign Inflexion value to a list named based on constants.EC_M \
    c) assign Slope value to a list named based on constants.SLOPE_M
4. If it belongs to rf_channels then, do the above but use constants.{param}_RF counterparts

You can write a class called MediaParameterLoader for this. Do the logic in a separate file under the planner folder.


##### step 3: process 'Coefficients' sheet

Make sure to add the functionality in media_parameter_loader.py.

##### Validation and Error handling
1. The columns in the file should be 'geo' and one of the values in the media_channels + rf_channels
2. The values in the geo column should be one of those in the geo_field in the Data sheet
3. The values under the other columns should be numeric and non-negative (these are MMM coefficients)
4. All of the values in media_channels + rf_channels should be listed

##### Logic
1. For those channels as part of media_channels in model_config, create an xarray.DataArray named using constants.BETA_GM of shape (n_geos, n_channels). refer to the attached image-1 for the exact structure
2. Similarily, do the above step for the rf_channels


#### step 4: combine parameter arrays (the output of get_processed_parameter_arrays method) and coefficient arrays (from get_processed_coefficients_arrays) into xarray.Dataset

lets create a class called pointInferenceData and do the following.

1. combine all the parameter arrays and coefficient arrays into a single dataset
2. the resulting dataset would have the following
    a) dimensions: media_channel, rf_channel, geo \
    b) Data variables: alpha_m, alpha_rf, beta_gm, beta_grf, ec_m, ec_rf, slope_m, slope_rf \
    c) also let's dummy coordinates called chain and draw with value 0 each (see the attached image)

Overall the resulting output should have very similar data structure as the attached image.

#### final step: test if our excel

#### ROI sheet as Input

Additional Feature: Some clients would give us 'ROI' instead of 'Coefficient' as input, and we would want to be able to process them. The input format for the ROI would have the same format as the 'Coefficient'

Sample file location:
  '/Users/mariappan.subramanian/Library/CloudStorage/'
  'OneDrive-TheTradeDesk/MMM/BudgetOptimizer/optimizer_input_case_roi.xlsx'

Goal: goal is to introduce a class that would translate the ROI into Coefficient and once we got the equivalent coefficients, we would then continue with our usual workflow.

Validation Rules:
1) Either 'Coefficient' or 'ROI' data should be given. Not both! So error out if both of them are given or none of them is given.
2) Decide if the input information is 'Coefficient' or 'ROI' based on the sheet name, and add a parameter to model_config called 'is_roi_input' with True or False

ROI to Coefficient calculation logic:
1. Create a dummy MMM model object with just input_data and default model_spec something like below. the benefit of doing this step is that, model.Meridian would do things scaling and etc.
```
data = adhoc_data_loader.build_input_data()
dummy_model = model.Meridian(input_data=data, model_spec=spec.ModelSpec())
```
2. Transform the data using the media transformation parameters we have already processed. Your code would look something like this depending on the input type (media vs rf_media). These are the inbuilt functions provided by Meridian, so let's use them.
```
        media_transformed = mmm.adstock_hill_media_fn(
            media=media_tensors.media_scaled,
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
        )
```
or \
```
        rf_transformed = mmm.adstock_hill_rf_fn(
            reach=rf_tensors.reach_scaled,
            frequency=rf_tensors.frequency,
            alpha=alpha_rf,
            ec=ec_rf,
            slope=slope_rf,
        )
```
3. Once you have the transformed data, you would want to do something like below.
```
    denominator_gx = tf.einsum(
        "...gtx,gt,g,->...gx",
        media_transformed,
        revenue_per_kpi,
        self.population,
        self.kpi_transformer.population_scaled_stdev,
    )
```

3. Calculate ROI dataarray from the ROI sheet. Similar to coefficient arrays (beta_gm and beta_grf) create roi_gm and roi_grf arrays.
4. Calculate Spend dataarray as well from the Spend data in the input data. As a result calculate spend_gm and spend_grf arrays
5. Calculate IncrementalOutcome dataarray by multiplying roi and spend arrays. This would give us numerator_gx
6. Calculate the ratio between numerator_gx / denominator_gx to get the beta_gx

Think hard and let me know the implementation plan. Do you see any problems with this approach?