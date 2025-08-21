import tensorflow as tf

def create_xarray_and_tf_tensor(df, date_field, geo_field, metric_fields):
    """Convert pandas DataFrame to TensorFlow tensor with shape (G,T,M).

    Args:
        df: pandas DataFrame containing the data
        date_field: column name for dates
        geo_field: column name for geographic regions
        metric_fields: list of column names for metrics/variables

    Returns:
        tf.Tensor with shape (num_geos, num_timesteps, num_metrics)
    """
    # Sort index levels to ensure consistent ordering
    df = df.sort_values([geo_field, date_field])

    # Create xarray dataset first
    xa = (
        df.set_index([geo_field, date_field])
          .to_xarray()
          [metric_fields]
    )

    # Convert xarray dataset into dataarray
    da = xa.to_array().transpose(geo_field, date_field, "variable")

    # Convert to tf tensor
    tf_tensor = tf.convert_to_tensor(
        da.values,  # G, T, M
        dtype=tf.float32
    )

    return tf_tensor
