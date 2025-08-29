# Copyright 2025 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PointInferenceData for converting Excel estimates to ArviZ InferenceData."""

import logging
from typing import Dict, Optional
import numpy as np
import xarray as xr
import arviz as az

from meridian import constants
from meridian.data import input_data
from meridian.model import spec


__all__ = [
    'PointInferenceData',
]


class PointInferenceData:
  """Converts parameter/coefficient arrays to ArviZ InferenceData.
  
  Creates inference data structure compatible with Meridian analysis functions
  by wrapping Excel point estimates as single chain/draw results.
  
  This enables direct use of point estimates from Excel with Meridian's
  analysis functions that expect Bayesian inference results.
  """
  
  def __init__(self, parameter_arrays: Dict[str, xr.DataArray], 
               coefficient_arrays: Dict[str, xr.DataArray],
               input_data_obj: Optional[input_data.InputData] = None,
               model_spec_obj: Optional[spec.ModelSpec] = None):
    """Initialize PointInferenceData.
    
    Args:
      parameter_arrays: Dictionary of parameter DataArrays from MediaParameterLoader.
        Expected keys: ALPHA_M, EC_M, SLOPE_M, ALPHA_RF, EC_RF, SLOPE_RF
      coefficient_arrays: Dictionary of coefficient DataArrays from MediaParameterLoader.
        Expected keys: BETA_GM, BETA_GRF
      input_data_obj: InputData object for extracting dimensions and coordinates.
        Required for creating complete inference data compatible with Meridian.
      model_spec_obj: ModelSpec object for model configuration.
        If None, uses default ModelSpec.
    """
    self.parameter_arrays = parameter_arrays
    self.coefficient_arrays = coefficient_arrays
    self.input_data_obj = input_data_obj
    self.model_spec_obj = model_spec_obj if model_spec_obj else spec.ModelSpec()
    
    # Validate inputs
    self._validate_inputs()
    
    # Initialize dimensions and coordinates if input_data is provided
    if self.input_data_obj is not None:
      self._initialize_dimensions()
    
    # Create posterior dataset
    posterior_dataset = self._create_posterior_dataset()
    
    # Create ArviZ InferenceData
    self.inference_data = self._create_inference_data(posterior_dataset)
    
  def _validate_inputs(self) -> None:
    """Validate input arrays."""
    if not self.parameter_arrays:
      raise ValueError("parameter_arrays cannot be empty")
      
    if not self.coefficient_arrays:
      raise ValueError("coefficient_arrays cannot be empty")
      
    # Check for required parameter keys
    expected_param_keys = {constants.ALPHA_M, constants.EC_M, constants.SLOPE_M}
    if constants.ALPHA_RF in self.parameter_arrays:
      expected_param_keys.update({constants.ALPHA_RF, constants.EC_RF, constants.SLOPE_RF})
      
    missing_params = expected_param_keys - set(self.parameter_arrays.keys())
    if missing_params:
      raise ValueError(f"Missing parameter arrays: {missing_params}")
      
    # Check for required coefficient keys
    expected_coeff_keys = {constants.BETA_GM}
    if constants.BETA_GRF in self.coefficient_arrays:
      expected_coeff_keys.add(constants.BETA_GRF)
      
    missing_coeffs = expected_coeff_keys - set(self.coefficient_arrays.keys())
    if missing_coeffs:
      raise ValueError(f"Missing coefficient arrays: {missing_coeffs}")
      
    logging.info("Input validation passed for PointInferenceData creation")
  
  def _initialize_dimensions(self) -> None:
    """Initialize dimensions and coordinates from input_data and model_spec."""
    from meridian.model import model
    
    # Create temporary Meridian model to get dimensions
    temp_model = model.Meridian(input_data=self.input_data_obj, model_spec=self.model_spec_obj)
    
    # Store dimensions
    self.n_geos = temp_model.n_geos
    self.n_times = temp_model.n_times
    self.n_media_channels = temp_model.n_media_channels
    self.n_rf_channels = temp_model.n_rf_channels
    self.n_controls = temp_model.n_controls
    self.n_knots = temp_model.knot_info.n_knots
    self.n_organic_media_channels = temp_model.n_organic_media_channels
    self.n_organic_rf_channels = temp_model.n_organic_rf_channels
    self.n_non_media_channels = temp_model.n_non_media_channels
    
    # Store coordinate values
    self.geo_coords = list(self.input_data_obj.geo.values)
    self.time_coords = list(range(self.n_times))
    self.media_channel_coords = list(self.input_data_obj.media_channel.values) if self.input_data_obj.media is not None else []
    self.rf_channel_coords = list(self.input_data_obj.rf_channel.values) if self.input_data_obj.reach is not None else []
    self.knots_coords = list(range(self.n_knots))
    
    logging.info(f"Initialized dimensions: geos={self.n_geos}, times={self.n_times}, knots={self.n_knots}")
    
  def _create_dummy_array(self, dims: list, coords: dict, name: str, fill_value: float = 0.0) -> xr.DataArray:
    """Create dummy DataArray with chain and draw dimensions.
    
    Args:
      dims: List of dimension names (excluding chain and draw)
      coords: Dictionary of coordinates for dimensions
      name: Name for the DataArray
      fill_value: Value to fill the array with (default: 0.0)
      
    Returns:
      DataArray with (chain, draw) + dims dimensions filled with fill_value
    """
    # Add chain and draw to dimensions and coordinates
    full_dims = ['chain', 'draw'] + dims
    full_coords = {'chain': [0], 'draw': [0], **coords}
    
    # Calculate shape
    shape = tuple(len(full_coords[dim]) for dim in full_dims)
    
    # Create array filled with fill_value
    data = np.full(shape, fill_value, dtype=np.float32)
    
    return xr.DataArray(
      data=data,
      dims=full_dims,
      coords=full_coords,
      name=name
    )
    
  def _reshape_array(self, data_array: xr.DataArray) -> xr.DataArray:
    """Reshape array to include chain and draw dimensions as first two dimensions.
    
    Args:
      data_array: Original DataArray with arbitrary dimensions.
      
    Returns:
      DataArray with (chain: 1, draw: 1) added as first dimensions.
    """
    # Get original data and coordinates
    original_data = data_array.values
    original_dims = list(data_array.dims)
    original_coords = dict(data_array.coords)
    
    # Add chain and draw dimensions to the front
    new_dims = ['chain', 'draw'] + original_dims
    
    # Reshape data to include (1, 1) for chain and draw
    new_shape = (1, 1) + original_data.shape
    reshaped_data = original_data.reshape(new_shape)
    
    # Create new coordinates including chain and draw
    new_coords = {
      'chain': [0],
      'draw': [0],
      **original_coords
    }
    
    # Create new DataArray
    reshaped_array = xr.DataArray(
      data=reshaped_data,
      dims=new_dims,
      coords=new_coords,
      name=data_array.name
    )
    
    return reshaped_array
  
  def _create_dummy_arrays(self) -> Dict[str, xr.DataArray]:
    """Create all required dummy arrays for complete inference data.
    
    Returns:
      Dictionary of dummy DataArrays with appropriate dimensions and coordinates.
    """
    dummy_arrays = {}
    
    if self.input_data_obj is None:
      logging.warning("No input_data provided - skipping dummy array creation")
      return dummy_arrays
      
    # Time-related parameters
    dummy_arrays[constants.MU_T] = self._create_dummy_array(
      dims=[constants.TIME],
      coords={constants.TIME: self.time_coords},
      name=constants.MU_T
    )
    
    dummy_arrays[constants.KNOT_VALUES] = self._create_dummy_array(
      dims=[constants.KNOTS],
      coords={constants.KNOTS: self.knots_coords},
      name=constants.KNOT_VALUES
    )
    
    # Geo-related parameters
    dummy_arrays[constants.TAU_G] = self._create_dummy_array(
      dims=[constants.GEO],
      coords={constants.GEO: self.geo_coords},
      name=constants.TAU_G
    )
    
    # Media parameters (beyond what we have)
    if self.n_media_channels > 0:
      for param_name in [constants.ROI_M, constants.MROI_M, constants.CONTRIBUTION_M, 
                        constants.BETA_M, constants.ETA_M]:
        dummy_arrays[param_name] = self._create_dummy_array(
          dims=[constants.MEDIA_CHANNEL],
          coords={constants.MEDIA_CHANNEL: self.media_channel_coords},
          name=param_name
        )
    
    # R&F parameters (beyond what we have)
    if self.n_rf_channels > 0:
      for param_name in [constants.ROI_RF, constants.MROI_RF, constants.CONTRIBUTION_RF,
                        constants.BETA_RF, constants.ETA_RF]:
        dummy_arrays[param_name] = self._create_dummy_array(
          dims=[constants.RF_CHANNEL],
          coords={constants.RF_CHANNEL: self.rf_channel_coords},
          name=param_name
        )
    
    # Control parameters (if any controls exist)
    if self.n_controls > 0:
      control_coords = list(self.input_data_obj.control_variable.values)
      for param_name in [constants.GAMMA_C, constants.XI_C]:
        dummy_arrays[param_name] = self._create_dummy_array(
          dims=[constants.CONTROL_VARIABLE],
          coords={constants.CONTROL_VARIABLE: control_coords},
          name=param_name
        )
        
      # Geo-control parameters
      dummy_arrays[constants.GAMMA_GC] = self._create_dummy_array(
        dims=[constants.GEO, constants.CONTROL_VARIABLE],
        coords={constants.GEO: self.geo_coords, constants.CONTROL_VARIABLE: control_coords},
        name=constants.GAMMA_GC
      )
    
    # Non-media parameters (if any exist)
    if self.n_non_media_channels > 0:
      non_media_coords = list(self.input_data_obj.non_media_channel.values) 
      for param_name in [constants.CONTRIBUTION_N, constants.GAMMA_N, constants.XI_N]:
        dummy_arrays[param_name] = self._create_dummy_array(
          dims=[constants.NON_MEDIA_CHANNEL],
          coords={constants.NON_MEDIA_CHANNEL: non_media_coords},
          name=param_name
        )
        
      # Geo-non-media parameters
      dummy_arrays[constants.GAMMA_GN] = self._create_dummy_array(
        dims=[constants.GEO, constants.NON_MEDIA_CHANNEL],
        coords={constants.GEO: self.geo_coords, constants.NON_MEDIA_CHANNEL: non_media_coords},
        name=constants.GAMMA_GN
      )
    
    # Organic media parameters (if any exist)
    if self.n_organic_media_channels > 0:
      organic_media_coords = list(self.input_data_obj.organic_media_channel.values)
      for param_name in [constants.CONTRIBUTION_OM, constants.BETA_OM, constants.ETA_OM,
                        constants.ALPHA_OM, constants.EC_OM, constants.SLOPE_OM]:
        dummy_arrays[param_name] = self._create_dummy_array(
          dims=[constants.ORGANIC_MEDIA_CHANNEL],
          coords={constants.ORGANIC_MEDIA_CHANNEL: organic_media_coords},
          name=param_name
        )
        
      # Geo-organic-media parameters
      dummy_arrays[constants.BETA_GOM] = self._create_dummy_array(
        dims=[constants.GEO, constants.ORGANIC_MEDIA_CHANNEL],
        coords={constants.GEO: self.geo_coords, constants.ORGANIC_MEDIA_CHANNEL: organic_media_coords},
        name=constants.BETA_GOM
      )
    
    # Organic R&F parameters (if any exist)
    if self.n_organic_rf_channels > 0:
      organic_rf_coords = list(self.input_data_obj.organic_rf_channel.values)
      for param_name in [constants.CONTRIBUTION_ORF, constants.BETA_ORF, constants.ETA_ORF,
                        constants.ALPHA_ORF, constants.EC_ORF, constants.SLOPE_ORF]:
        dummy_arrays[param_name] = self._create_dummy_array(
          dims=[constants.ORGANIC_RF_CHANNEL],
          coords={constants.ORGANIC_RF_CHANNEL: organic_rf_coords},
          name=param_name
        )
        
      # Geo-organic-R&F parameters
      dummy_arrays[constants.BETA_GORF] = self._create_dummy_array(
        dims=[constants.GEO, constants.ORGANIC_RF_CHANNEL],
        coords={constants.GEO: self.geo_coords, constants.ORGANIC_RF_CHANNEL: organic_rf_coords},
        name=constants.BETA_GORF
      )
    
    logging.info(f"Created {len(dummy_arrays)} dummy arrays for complete inference data")
    return dummy_arrays
    
  def _create_posterior_dataset(self) -> xr.Dataset:
    """Create posterior Dataset with all parameter and coefficient arrays.
    
    Returns:
      xarray.Dataset with all data variables having (chain, draw) as first dimensions.
    """
    data_vars = {}
    
    # Process parameter arrays
    for key, data_array in self.parameter_arrays.items():
      reshaped_array = self._reshape_array(data_array)
      # Ensure float32 dtype
      reshaped_array = reshaped_array.astype(np.float32)
      data_vars[key] = reshaped_array
      
    # Process coefficient arrays  
    for key, data_array in self.coefficient_arrays.items():
      reshaped_array = self._reshape_array(data_array)
      # Ensure float32 dtype
      reshaped_array = reshaped_array.astype(np.float32)
      data_vars[key] = reshaped_array
    
    # Add dummy arrays for complete inference data structure
    dummy_arrays = self._create_dummy_arrays()
    for key, dummy_array in dummy_arrays.items():
      data_vars[key] = dummy_array
      
    # Create Dataset
    posterior_dataset = xr.Dataset(data_vars)
    
    logging.info(f"Created posterior dataset with {len(data_vars)} data variables")
    return posterior_dataset
    
  def _create_inference_data(self, posterior_dataset: xr.Dataset) -> az.InferenceData:
    """Create ArviZ InferenceData from posterior dataset.
    
    Args:
      posterior_dataset: Dataset with all parameter/coefficient data.
      
    Returns:
      ArviZ InferenceData object with posterior group.
    """
    # Create InferenceData with posterior group
    inference_data = az.InferenceData(posterior=posterior_dataset)
    
    # Add basic sample_stats for completeness
    sample_stats = xr.Dataset({
      'diverging': xr.DataArray(
        data=np.array([[False]], dtype=bool),
        dims=['chain', 'draw'],
        coords={'chain': [0], 'draw': [0]}
      ),
      'energy': xr.DataArray(
        data=np.array([[0.0]], dtype=np.float32),
        dims=['chain', 'draw'], 
        coords={'chain': [0], 'draw': [0]}
      ),
    })
    
    # Add sample_stats group
    inference_data.add_groups(sample_stats=sample_stats)
    
    logging.info("Successfully created ArviZ InferenceData object")
    return inference_data
    
  def get_inference_data(self) -> az.InferenceData:
    """Return the ArviZ InferenceData object.
    
    Returns:
      ArviZ InferenceData containing posterior and sample_stats groups.
    """
    return self.inference_data
    
  @property
  def posterior(self) -> xr.Dataset:
    """Direct access to posterior dataset."""
    return self.inference_data.posterior
    
  def __repr__(self) -> str:
    """String representation of PointInferenceData."""
    n_vars = len(self.inference_data.posterior.data_vars)
    groups = list(self.inference_data.groups())
    return f"PointInferenceData with {n_vars} variables in groups: {groups}"