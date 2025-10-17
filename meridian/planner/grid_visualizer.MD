# TASK: Create MROI data / plots by GEO
## Inputs: optimization_grid.grid_dataset from BudgetVisualizer object. 
## Instructions:
1. The grid_dataset has two xarray objects - spend_grid and incremental_outcome_grid
2. Now for each geo in the grid, we want to calculate the marginal roi. The following example is for your reference.

```
select_geo = 'NEW_YORK'  # example geo
grid_dataset_g = grid_dataset.sel(geo=select_geo)
channels = grid_dataset_g.channel.data
spend_grid_g = grid_dataset_g.spend_grid.data
outcome_grid_g = grid_dataset_g.incremental_outcome_grid.data

spend_lb = spend_grid_g[0]
outcome_lb = outcome_grid_g[0]

# calc marginal roi
outcome_delta = outcome_grid_g[1:] - outcome_grid_g[:-1]
spend_delta = spend_grid_g[1:] - spend_grid_g[:-1]
mroi_g = tf.math.divide_no_nan(outcome_delta, spend_delta)
```
make sure my calculations are correct.

3. The above step should yield marginal roi vector for each channel for the selcted geo. Save this dataset to a list of results for geo
4. Finally, row bind or append the results across geos so that the resulting dataframe would have geo, channel names as columns. For example.
```
    |geo, grid_idx, TV, Display, Video|
    |Texas, 0, 0.3, 0.2, 0.1|
    |Texas, 1, 0.29, 0.2, 0.09|
    |Texas, 2, 0.28, 0.2, 0.09|
    |NY, 0, 0.3, 0.2, 0.1|
    |NY, 1, 0.29, 0.2, 0.09|
    |NY, 2, 0.28, 0.2, 0.09|
```
5. As an output, we want to export this dataset into an excel where each sheet would have a dataset for a geo
6. There should be an option to select top 'n' geos or just export for a select 'geo' (default is to export for all)
