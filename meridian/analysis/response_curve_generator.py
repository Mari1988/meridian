#!/usr/bin/env python3
"""
Geo-Level Scaled Media Simulation for FastResponseCurves

This module provides clean, simple functionality to generate geo-level scaled media scenarios
from 0 to 2x the maximum historical scaled media for each geography independently.

This is Step 1 (Revised) of the revamped FastResponseCurves implementation.
Uses mmm.media_tensors.media_scaled (the data that actually feeds into transformations).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional, Sequence, Tuple
from meridian.model import model


class ResponseCurveGenerator:
    """Generates response curves and media scenarios for MMM analysis."""

    def __init__(self, meridian_model: model.Meridian):
        """Initialize with fitted Meridian model.

        Args:
            meridian_model: Fitted Meridian model with historical scaled media data.

        Raises:
            ValueError: If model doesn't have required scaled media data.
        """
        self.model = meridian_model
        self._validate_model()
        self._extract_scaled_media_data()
        self._extract_median_parameters()

    def _validate_model(self):
        """Validate that model has required scaled media data."""
        if not hasattr(self.model, 'input_data'):
            raise ValueError("Model must have input_data attribute")

        # Check for scaled media data availability
        has_media_scaled = (hasattr(self.model, 'media_tensors') and
                           hasattr(self.model.media_tensors, 'media_scaled') and
                           self.model.media_tensors.media_scaled is not None)
        has_rf_scaled = (hasattr(self.model, 'rf_tensors') and
                        hasattr(self.model.rf_tensors, 'rf_reach_scaled') and
                        self.model.rf_tensors.rf_reach_scaled is not None)

        if not (has_media_scaled or has_rf_scaled):
            raise ValueError("Model must have either media_scaled or rf_reach_scaled data")

    def _extract_scaled_media_data(self):
        """Extract and consolidate scaled media data from model."""
        media_tensors = []
        channel_names = []

        # Extract scaled media data
        if (hasattr(self.model, 'media_tensors') and
            hasattr(self.model.media_tensors, 'media_scaled') and
            self.model.media_tensors.media_scaled is not None):
            media_scaled = self.model.media_tensors.media_scaled
            media_tensors.append(media_scaled)
            if self.model.input_data.media_channel is not None:
                channel_names.extend(self.model.input_data.media_channel.values.tolist())

        # Extract scaled R&F data (reach)
        if (hasattr(self.model, 'rf_tensors') and
            hasattr(self.model.rf_tensors, 'rf_reach_scaled') and
            self.model.rf_tensors.rf_reach_scaled is not None):
            rf_reach_scaled = self.model.rf_tensors.rf_reach_scaled
            media_tensors.append(rf_reach_scaled)
            if self.model.input_data.rf_channel is not None:
                channel_names.extend(self.model.input_data.rf_channel.values.tolist())

        if not media_tensors:
            raise ValueError("No scaled media data found in model")

        # Concatenate all scaled media data
        self.historical_scaled_media = tf.concat(media_tensors, axis=-1)  # Shape: (n_geos, n_times, n_channels)
        self.channel_names = channel_names
        self.n_geos = self.historical_scaled_media.shape[0]
        self.n_times = self.historical_scaled_media.shape[1]
        self.n_channels = self.historical_scaled_media.shape[2]

        print(f"📊 Scaled media data extracted:")
        print(f"   Shape: {self.historical_scaled_media.shape} (geos, times, channels)")
        print(f"   Channels: {self.channel_names}")
        print(f"   Value range: [{tf.reduce_min(self.historical_scaled_media, axis=[0,1])}, {tf.reduce_max(self.historical_scaled_media, axis=[0,1])}]")

    def _extract_median_parameters(self):
        """Extract median parameter values from posterior samples for transformations."""
        from meridian import constants as c

        if not hasattr(self.model, 'inference_data') or not hasattr(self.model.inference_data, 'posterior'):
            raise ValueError("Model must have fitted posterior data for parameter extraction")

        posterior = self.model.inference_data.posterior

        # Initialize parameter storage
        self.media_params = {}
        self.rf_params = {}

        # Extract media parameters if they exist
        if self.model.n_media_channels > 0:
            self.media_params = {
                'ec': np.median(posterior[c.EC_M].values, axis=(0, 1)),
                'slope': np.median(posterior[c.SLOPE_M].values, axis=(0, 1)),
                'alpha': np.median(posterior[c.ALPHA_M].values, axis=(0, 1)),
                'beta': np.median(posterior[c.BETA_GM].values, axis=(0, 1)),
            }

        # Extract R&F parameters if they exist
        if self.model.n_rf_channels > 0:
            self.rf_params = {
                'ec': np.median(posterior[c.EC_RF].values, axis=(0, 1)),
                'slope': np.median(posterior[c.SLOPE_RF].values, axis=(0, 1)),
                'alpha': np.median(posterior[c.ALPHA_RF].values, axis=(0, 1)),
                'beta': np.median(posterior[c.BETA_GRF].values, axis=(0, 1)),
            }

        # Extract max_lag for transformations
        self.max_lag = self.model.model_spec.max_lag if hasattr(self.model, 'model_spec') else 8

        print(f"📊 Median parameters extracted:")
        if self.media_params:
            print(f"   Media channels: {len(self.media_params['ec'])} params")
            print(f"   Alpha range: [{self.media_params['alpha'].min():.3f}, {self.media_params['alpha'].max():.3f}]")
            print(f"   EC range: [{self.media_params['ec'].min():.3f}, {self.media_params['ec'].max():.3f}]")
        if self.rf_params:
            print(f"   R&F channels: {len(self.rf_params['ec'])} params")
        print(f"   Max lag (adstock): {self.max_lag}")

    def reverse_transform_media_scenarios(self, media_scenarios: np.ndarray, metadata: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert scaled media scenarios back to actual impressions/reach values and spend.

        Args:
            media_scenarios: Scaled media scenarios from generate_geo_media_scenarios()
                           Shape: (n_geos, n_steps, n_channels)
            metadata: Metadata from generate_geo_media_scenarios()

        Returns:
            Tuple containing:
            - actual_impressions: numpy array of actual impression/reach values
                                Shape: (n_geos, n_steps, n_channels)
            - actual_spend: numpy array of actual spend values
                          Shape: (n_geos, n_steps, n_channels)
            - reverse_metadata: dict with reverse transformation information including spend data
        """
        print(f"🔄 Converting scaled media scenarios to actual impressions...")

        # Extract media transformer scale factors
        media_scale_factors = None
        rf_scale_factors = None

        if (hasattr(self.model, 'media_tensors') and
            hasattr(self.model.media_tensors, 'media_transformer') and
            self.model.media_tensors.media_transformer is not None):
            media_transformer = self.model.media_tensors.media_transformer
            media_scale_factors = media_transformer._scale_factors_gm.numpy()
            print(f"   Media scale factors extracted: shape {media_scale_factors.shape}")

        if (hasattr(self.model, 'rf_tensors') and
            hasattr(self.model.rf_tensors, 'reach_transformer') and
            self.model.rf_tensors.reach_transformer is not None):
            rf_transformer = self.model.rf_tensors.reach_transformer
            rf_scale_factors = rf_transformer._scale_factors_gm.numpy()
            print(f"   R&F scale factors extracted: shape {rf_scale_factors.shape}")

        # Initialize output array
        actual_impressions = np.zeros_like(media_scenarios)

        # Apply reverse transformation channel by channel
        for channel_idx in range(media_scenarios.shape[2]):
            channel_name = metadata['channel_names'][channel_idx]

            # Determine which scale factors to use
            if channel_idx < len(self.media_params.get('ec', [])):
                # Standard media channel - use media scale factors
                if media_scale_factors is not None:
                    scale_factors = media_scale_factors[:, channel_idx]
                    actual_impressions[:, :, channel_idx] = (
                        media_scenarios[:, :, channel_idx] * scale_factors[:, np.newaxis]
                    )
                    print(f"   {channel_name}: Applied media scale factors")
                else:
                    print(f"   ⚠️  {channel_name}: No media scale factors available")
                    actual_impressions[:, :, channel_idx] = media_scenarios[:, :, channel_idx]

            elif rf_scale_factors is not None:
                # R&F channel - use R&F scale factors
                rf_channel_idx = channel_idx - len(self.media_params.get('ec', []))
                if rf_channel_idx < rf_scale_factors.shape[1]:
                    scale_factors = rf_scale_factors[:, rf_channel_idx]
                    actual_impressions[:, :, channel_idx] = (
                        media_scenarios[:, :, channel_idx] * scale_factors[:, np.newaxis]
                    )
                    print(f"   {channel_name}: Applied R&F scale factors")
                else:
                    print(f"   ⚠️  {channel_name}: R&F channel index out of range")
                    actual_impressions[:, :, channel_idx] = media_scenarios[:, :, channel_idx]
            else:
                print(f"   ⚠️  {channel_name}: No scale factors available")
                actual_impressions[:, :, channel_idx] = media_scenarios[:, :, channel_idx]

        # Create reverse transformation metadata
        reverse_metadata = {
            'transformation_type': 'scaled_to_actual_impressions',
            'input_shape': media_scenarios.shape,
            'output_shape': actual_impressions.shape,
            'original_range': [float(media_scenarios.min()), float(media_scenarios.max())],
            'actual_range': [float(actual_impressions.min()), float(actual_impressions.max())],
            'scale_factors_available': {
                'media_channels': media_scale_factors is not None,
                'rf_channels': rf_scale_factors is not None
            },
            'channel_info': []
        }

        # Add per-channel transformation info
        for channel_idx, channel_name in enumerate(metadata['channel_names']):
            original_values = media_scenarios[:, :, channel_idx]
            actual_values = actual_impressions[:, :, channel_idx]

            reverse_metadata['channel_info'].append({
                'channel': channel_name,
                'original_range': [float(original_values.min()), float(original_values.max())],
                'actual_range': [float(actual_values.min()), float(actual_values.max())],
                'transformation_ratio': float(actual_values.max() / original_values.max()) if original_values.max() > 0 else 1.0
            })

        print(f"✅ Reverse transformation completed:")
        print(f"   Input range (scaled): [{media_scenarios.min():.3f}, {media_scenarios.max():.3f}]")
        print(f"   Output range (actual): [{actual_impressions.min():.0f}, {actual_impressions.max():.0f}]")

        # Convert impressions to spend
        actual_spend, spend_metadata = self.convert_impressions_to_spend(actual_impressions, metadata)
        
        # Update reverse metadata to include spend information
        reverse_metadata['transformation_type'] = 'scaled_to_actual_impressions_and_spend'
        reverse_metadata['spend_metadata'] = spend_metadata
        reverse_metadata['spend_range'] = [float(actual_spend.min()), float(actual_spend.max())]

        return actual_impressions, actual_spend, reverse_metadata

    def reverse_transform_media_effects(self, media_effects: np.ndarray, metadata: dict) -> Tuple[np.ndarray, dict]:
        """Convert scaled media effects back to actual KPI/revenue values.

        Args:
            media_effects: Scaled media effects from apply_media_transformations()
                         Shape: (n_geos, n_steps, n_channels)
            metadata: Metadata from the original transformation

        Returns:
            Tuple containing:
            - actual_kpi_effects: numpy array of actual KPI contribution values
                                Shape: (n_geos, n_steps, n_channels)
            - kpi_metadata: dict with KPI transformation information
        """
        print(f"🔄 Converting scaled media effects to actual KPI values...")

        # Extract KPI transformer
        if not hasattr(self.model, 'kpi_transformer') or self.model.kpi_transformer is None:
            print(f"   ⚠️  No KPI transformer available in model")
            return media_effects.copy(), {'transformation_applied': False}

        kpi_transformer = self.model.kpi_transformer

        # Get transformation parameters
        population = kpi_transformer._population.numpy()
        pop_scaled_mean = kpi_transformer._population_scaled_mean.numpy()
        pop_scaled_stdev = kpi_transformer._population_scaled_stdev.numpy()

        print(f"   KPI transformer parameters:")
        print(f"     Population scaled mean: {pop_scaled_mean:.6f}")
        print(f"     Population scaled stdev: {pop_scaled_stdev:.6f}")
        print(f"     Population shape: {population.shape}")

        # Apply inverse KPI transformation using correct difference method
        # IMPORTANT: KpiTransformer.inverse() expects (n_geos, n_times) - NOT (n_geos, n_times, n_channels)
        # We must process one channel at a time, following fast_response_curves.py approach
        
        # Convert to TensorFlow tensors for kpi_transformer operations
        import tensorflow as tf
        
        n_geos, n_steps, n_channels = media_effects.shape
        actual_kpi_effects = np.zeros_like(media_effects)
        
        # Process each channel separately (KpiTransformer limitation)
        for ch in range(n_channels):
            # Extract channel effects: (n_geos, n_steps)
            channel_effects = tf.constant(media_effects[:, :, ch], dtype=tf.float32)
            
            # Apply inverse KPI transformation (matching fast_response_curves.py approach)
            # The transformation is: inverse(effects) - inverse(zeros)
            t1 = kpi_transformer.inverse(channel_effects)  # (n_geos, n_steps)
            t2 = kpi_transformer.inverse(tf.zeros_like(channel_effects))  # Baseline
            
            # Take difference to get incremental effects for this channel
            channel_kpi_effects = t1 - t2
            
            # Store results for this channel
            actual_kpi_effects[:, :, ch] = channel_kpi_effects.numpy()

        # Create KPI transformation metadata
        kpi_metadata = {
            'transformation_type': 'scaled_to_actual_kpi',
            'transformation_applied': True,
            'input_shape': media_effects.shape,
            'output_shape': actual_kpi_effects.shape,
            'original_range': [float(media_effects.min()), float(media_effects.max())],
            'actual_kpi_range': [float(actual_kpi_effects.min()), float(actual_kpi_effects.max())],
            'transformation_parameters': {
                'population_scaled_mean': float(pop_scaled_mean),
                'population_scaled_stdev': float(pop_scaled_stdev),
                'population_sample': population[:5].tolist()
            },
            'channel_kpi_contributions': []
        }

        # Add per-channel KPI contribution info
        for channel_idx, channel_name in enumerate(metadata.get('channel_names', [])):
            if channel_idx < actual_kpi_effects.shape[2]:
                original_contrib = media_effects[:, :, channel_idx]
                actual_contrib = actual_kpi_effects[:, :, channel_idx]

                kpi_metadata['channel_kpi_contributions'].append({
                    'channel': channel_name,
                    'original_contrib_range': [float(original_contrib.min()), float(original_contrib.max())],
                    'actual_kpi_contrib_range': [float(actual_contrib.min()), float(actual_contrib.max())],
                    'total_kpi_contribution': float(actual_contrib.sum())
                })

        print(f"✅ KPI reverse transformation completed:")
        print(f"   Input range (scaled effects): [{media_effects.min():.6f}, {media_effects.max():.6f}]")
        print(f"   Output range (actual KPI): [{actual_kpi_effects.min():.2f}, {actual_kpi_effects.max():.2f}]")

        return actual_kpi_effects, kpi_metadata

    def convert_impressions_to_spend(self, actual_impressions: np.ndarray, metadata: dict) -> Tuple[np.ndarray, dict]:
        """Convert actual impressions to actual spend using historical CPM data.
        
        This method calculates spend using the formula: spend = (impressions / 1000) × CPM
        where CPM is calculated from historical data per geo and channel.
        
        Args:
            actual_impressions: Actual impression values
                              Shape: (n_geos, n_steps, n_channels)
            metadata: Metadata from previous transformations
            
        Returns:
            Tuple containing:
            - actual_spend: numpy array of actual spend values
                          Shape: (n_geos, n_steps, n_channels)  
            - spend_metadata: dict with spend conversion information
        """
        print(f"💰 Converting actual impressions to actual spend...")
        
        # Validate input
        if not hasattr(self.model, 'input_data'):
            raise ValueError("Model must have input_data for spend conversion")
        
        n_geos, n_steps, n_channels = actual_impressions.shape
        print(f"   Input shape: {actual_impressions.shape}")
        print(f"   Sample impressions (geo 0, step 0): {actual_impressions[0, 0, :]}")
        
        # Extract historical spend and impression data
        try:
            # Get historical spend data
            historical_spend = self.model.input_data.get_total_spend()  # (n_geos, n_times, n_channels)
            print(f"   Historical spend shape: {historical_spend.shape}")
            
            # Get historical impression data (media + RF)
            historical_impressions = self.model.input_data.get_all_media_and_rf()  # (n_geos, n_times, n_channels)
            print(f"   Historical impressions shape: {historical_impressions.shape}")
            
        except Exception as e:
            print(f"   ❌ Error extracting historical data: {e}")
            # Fallback: use dummy CPM values
            print(f"   Using fallback CPM values...")
            avg_cpm_per_geo_channel = np.full((n_geos, n_channels), 50.0)  # $50 CPM fallback
            
        else:
            # Calculate CPM per geo and channel: CPM = (spend / impressions) * 1000
            print(f"   Calculating CPM from historical data...")
            
            # Avoid division by zero
            safe_impressions = np.where(historical_impressions == 0, np.nan, historical_impressions)
            cpm_per_time = (historical_spend / safe_impressions) * 1000  # (n_geos, n_times, n_channels)
            
            # Use median CPM per geo/channel (robust to outliers)
            avg_cpm_per_geo_channel = np.nanmedian(cpm_per_time, axis=1)  # (n_geos, n_channels)
            
            # Handle any remaining NaN values with channel-wide median
            for ch in range(n_channels):
                channel_cpm = avg_cpm_per_geo_channel[:, ch]
                nan_mask = np.isnan(channel_cpm)
                if nan_mask.any():
                    channel_median = np.nanmedian(channel_cpm[~nan_mask])
                    if not np.isnan(channel_median):
                        avg_cpm_per_geo_channel[nan_mask, ch] = channel_median
                    else:
                        # Ultimate fallback
                        avg_cpm_per_geo_channel[nan_mask, ch] = 50.0
        
        print(f"   CPM range: [${avg_cpm_per_geo_channel.min():.2f}, ${avg_cpm_per_geo_channel.max():.2f}]")
        print(f"   Sample CPM (geo 0): {avg_cpm_per_geo_channel[0, :]} per channel")
        
        # Convert impressions to spend: spend = (impressions / 1000) × CPM
        actual_spend = np.zeros_like(actual_impressions)
        
        for geo_idx in range(n_geos):
            for ch in range(n_channels):
                geo_channel_cpm = avg_cpm_per_geo_channel[geo_idx, ch]
                actual_spend[geo_idx, :, ch] = (actual_impressions[geo_idx, :, ch] / 1000) * geo_channel_cpm
        
        # Create spend conversion metadata
        spend_metadata = {
            'transformation_type': 'impressions_to_spend',
            'transformation_applied': True,
            'input_shape': actual_impressions.shape,
            'output_shape': actual_spend.shape,
            'impression_range': [float(actual_impressions.min()), float(actual_impressions.max())],
            'spend_range': [float(actual_spend.min()), float(actual_spend.max())],
            'cpm_calculation': {
                'method': 'median_historical_cpm',
                'cpm_range': [float(avg_cpm_per_geo_channel.min()), float(avg_cpm_per_geo_channel.max())],
                'cpm_per_geo_channel': avg_cpm_per_geo_channel.tolist(),
                'fallback_used': not hasattr(self.model.input_data, 'get_total_spend')
            },
            'channel_spend_stats': []
        }
        
        # Add per-channel spend statistics
        channel_names = metadata.get('channel_names', [f'Channel_{i}' for i in range(n_channels)])
        for ch, channel_name in enumerate(channel_names):
            if ch < n_channels:
                channel_impressions = actual_impressions[:, :, ch]
                channel_spend = actual_spend[:, :, ch]
                channel_cpm = avg_cpm_per_geo_channel[:, ch]
                
                spend_metadata['channel_spend_stats'].append({
                    'channel': channel_name,
                    'impression_range': [float(channel_impressions.min()), float(channel_impressions.max())],
                    'spend_range': [float(channel_spend.min()), float(channel_spend.max())],
                    'avg_cpm_range': [float(channel_cpm.min()), float(channel_cpm.max())],
                    'total_spend': float(channel_spend.sum()),
                    'total_impressions': float(channel_impressions.sum())
                })
        
        print(f"✅ Spend conversion completed:")
        print(f"   Input range (impressions): [{actual_impressions.min():,.0f}, {actual_impressions.max():,.0f}]")
        print(f"   Output range (spend): [${actual_spend.min():,.2f}, ${actual_spend.max():,.2f}]")
        print(f"   Average spend per impression: ${(actual_spend.sum() / actual_impressions.sum()):.6f}")
        
        return actual_spend, spend_metadata

    def generate_response_curves(self,
                               max_multiplier: float = 2.0,
                               num_steps: int = 50,
                               aggregation_level: str = "national") -> Tuple[dict, dict]:
        """Generate response curves with actual business metrics by aggregating across geos.

        This is the main method for creating response curves that shows the relationship
        between actual media spend and actual KPI contributions.

        Args:
            max_multiplier: Maximum spend multiplier (default 2.0 = 200% of historical max)
            num_steps: Number of points in the response curve
            aggregation_level: "national" (sum across geos) or "geo" (keep geo-level)

        Returns:
            Tuple containing:
            - response_curves: Dict with actual spend levels (X-axis) and KPI contributions (Y-axis)
            - curve_metadata: Dict with response curve information
        """
        print(f"🎯 Generating response curves with actual business metrics...")
        print(f"   Aggregation level: {aggregation_level}")
        print(f"   Multiplier range: 0 to {max_multiplier}x")
        print(f"   Curve resolution: {num_steps} points")

        # Step 1: Generate scaled media scenarios
        print(f"\n📊 Step 1: Generating scaled media scenarios...")
        media_scenarios, scenario_metadata = self.generate_geo_media_scenarios(
            max_multiplier=max_multiplier,
            num_steps=num_steps
        )

        # Step 2: Apply media transformations
        print(f"\n🔄 Step 2: Applying media transformations...")
        media_effects, transform_metadata = self.apply_media_transformations(
            media_scenarios, scenario_metadata
        )

        # Step 3: Reverse transform to actual business metrics
        print(f"\n💼 Step 3: Converting to actual business metrics...")
        actual_impressions, actual_spend, impression_metadata = self.reverse_transform_media_scenarios(
            media_scenarios, scenario_metadata
        )

        actual_kpi_effects, kpi_metadata = self.reverse_transform_media_effects(
            media_effects, scenario_metadata
        )

        # Step 4: Aggregate across geographies
        print(f"\n🌍 Step 4: Aggregating across geographies...")
        if aggregation_level == "national":
            # Sum across geos (axis=0) to get national totals
            national_impressions = np.sum(actual_impressions, axis=0)  # Shape: (n_steps, n_channels)
            national_spend = np.sum(actual_spend, axis=0)  # Shape: (n_steps, n_channels)
            national_kpi_effects = np.sum(actual_kpi_effects, axis=0)  # Shape: (n_steps, n_channels)

            print(f"   ✅ Aggregated to national level")
            print(f"   National impression range: [{national_impressions.min():.0f}, {national_impressions.max():.0f}]")
            print(f"   National spend range: [${national_spend.min():,.2f}, ${national_spend.max():,.2f}]")
            print(f"   National KPI effect range: [{national_kpi_effects.min():.2f}, {national_kpi_effects.max():.2f}]")

            # Create response curves data structure
            response_curves = {}
            for channel_idx, channel_name in enumerate(scenario_metadata['channel_names']):
                # Create spend multipliers for x-axis
                spend_multipliers = np.linspace(0, max_multiplier, num_steps)

                response_curves[channel_name] = {
                    'spend_multipliers': spend_multipliers,
                    'actual_spend': national_spend[:, channel_idx],  # PRIMARY X-AXIS DATA
                    'actual_kpi_contributions': national_kpi_effects[:, channel_idx],  # Y-AXIS DATA
                    'actual_impressions': national_impressions[:, channel_idx],  # SECONDARY REFERENCE
                    'scaled_impressions': np.sum(media_scenarios, axis=0)[:, channel_idx],
                    'scaled_effects': np.sum(media_effects, axis=0)[:, channel_idx]
                }

                print(f"   📈 {channel_name}: ${national_spend[:, channel_idx].max():,.0f} max spend → ${national_kpi_effects[:, channel_idx].max():.0f} max KPI")

        else:  # geo-level
            # Keep geo dimension for geo-level analysis
            response_curves = {}
            for channel_idx, channel_name in enumerate(scenario_metadata['channel_names']):
                spend_multipliers = np.linspace(0, max_multiplier, num_steps)

                response_curves[channel_name] = {
                    'spend_multipliers': spend_multipliers,
                    'actual_spend': actual_spend[:, :, channel_idx],  # PRIMARY X-AXIS (n_geos, n_steps)
                    'actual_kpi_contributions': actual_kpi_effects[:, :, channel_idx],  # Y-AXIS (n_geos, n_steps)
                    'actual_impressions': actual_impressions[:, :, channel_idx],  # SECONDARY REFERENCE
                    'scaled_impressions': media_scenarios[:, :, channel_idx],
                    'scaled_effects': media_effects[:, :, channel_idx],
                    'geo_names': scenario_metadata['geo_names']
                }

        # Step 5: Create comprehensive metadata
        curve_metadata = {
            'aggregation_level': aggregation_level,
            'max_multiplier': max_multiplier,
            'num_steps': num_steps,
            'n_channels': len(scenario_metadata['channel_names']),
            'channel_names': scenario_metadata['channel_names'],
            'transformation_pipeline': ['scaled_scenarios', 'adstock', 'hill', 'coefficients', 'reverse_transform'],
            'business_metrics': {
                'impression_units': 'actual_impressions_per_period',
                'kpi_units': 'actual_kpi_contributions_per_period',
                'currency': 'USD' if kpi_metadata.get('transformation_applied') else 'scaled_units'
            }
        }

        if aggregation_level == "national":
            curve_metadata.update({
                'total_impression_range': [float(national_impressions.min()), float(national_impressions.max())],
                'total_kpi_range': [float(national_kpi_effects.min()), float(national_kpi_effects.max())],
                'n_geos_aggregated': scenario_metadata['n_geos']
            })
        else:
            curve_metadata.update({
                'geo_names': scenario_metadata['geo_names'],
                'n_geos': scenario_metadata['n_geos']
            })

        # Add channel summaries
        curve_metadata['channel_summaries'] = []
        for channel_name in scenario_metadata['channel_names']:
            if aggregation_level == "national":
                max_impressions = response_curves[channel_name]['actual_impressions'].max()
                max_kpi = response_curves[channel_name]['actual_kpi_contributions'].max()
            else:
                max_impressions = response_curves[channel_name]['actual_impressions'].max()
                max_kpi = response_curves[channel_name]['actual_kpi_contributions'].max()

            curve_metadata['channel_summaries'].append({
                'channel': channel_name,
                'max_impressions': float(max_impressions),
                'max_kpi_contribution': float(max_kpi),
                'efficiency_at_max': float(max_kpi / max_impressions) if max_impressions > 0 else 0.0
            })

        print(f"\n✅ Response curves generated successfully!")
        print(f"   📊 Channels: {len(response_curves)}")
        print(f"   📈 Data points per curve: {num_steps}")
        print(f"   💰 Business metrics: Actual impressions & KPI contributions")

        return response_curves, curve_metadata

    def export_response_curves(self, response_curves: dict, curve_metadata: dict, output_file: str = None):
        """Export response curves with actual business metrics to CSV.

        Args:
            response_curves: Output from generate_response_curves()
            curve_metadata: Metadata from generate_response_curves()
            output_file: CSV filename for export (default: auto-generate based on aggregation level)
        """

        export_rows = []

        if curve_metadata['aggregation_level'] == "national":
            # National-level export
            for channel_name, curve_data in response_curves.items():
                for step_idx, multiplier in enumerate(curve_data['spend_multipliers']):
                    actual_spend_val = curve_data['actual_spend'][step_idx]
                    actual_impressions_val = curve_data['actual_impressions'][step_idx] 
                    actual_kpi_val = curve_data['actual_kpi_contributions'][step_idx]
                    
                    row = {
                        'channel': channel_name,
                        'spend_multiplier': multiplier,
                        'actual_spend': actual_spend_val,  # PRIMARY X-AXIS
                        'actual_kpi_contribution': actual_kpi_val,  # Y-AXIS  
                        'actual_impressions': actual_impressions_val,  # REFERENCE
                        'scaled_impressions': curve_data['scaled_impressions'][step_idx],
                        'scaled_effects': curve_data['scaled_effects'][step_idx],
                        'kpi_per_spend': actual_kpi_val / actual_spend_val if actual_spend_val > 0 else 0,
                        'kpi_per_impression': actual_kpi_val / actual_impressions_val if actual_impressions_val > 0 else 0
                    }
                    export_rows.append(row)

        else:
            # Geo-level export
            for channel_name, curve_data in response_curves.items():
                for geo_idx, geo_name in enumerate(curve_data['geo_names']):
                    for step_idx, multiplier in enumerate(curve_data['spend_multipliers']):
                        actual_spend_val = curve_data['actual_spend'][geo_idx, step_idx]
                        actual_impressions_val = curve_data['actual_impressions'][geo_idx, step_idx]
                        actual_kpi_val = curve_data['actual_kpi_contributions'][geo_idx, step_idx]
                        
                        row = {
                            'channel': channel_name,
                            'geo': geo_name,
                            'geo_index': geo_idx,
                            'spend_multiplier': multiplier,
                            'actual_spend': actual_spend_val,  # PRIMARY X-AXIS
                            'actual_kpi_contribution': actual_kpi_val,  # Y-AXIS
                            'actual_impressions': actual_impressions_val,  # REFERENCE
                            'scaled_impressions': curve_data['scaled_impressions'][geo_idx, step_idx],
                            'scaled_effects': curve_data['scaled_effects'][geo_idx, step_idx],
                            'kpi_per_spend': actual_kpi_val / actual_spend_val if actual_spend_val > 0 else 0,
                            'kpi_per_impression': actual_kpi_val / actual_impressions_val if actual_impressions_val > 0 else 0
                        }
                        export_rows.append(row)

        # Create and export DataFrame
        curves_df = pd.DataFrame(export_rows)
        
        # Auto-generate filename if not provided
        if output_file is None:
            output_file = f"{curve_metadata['aggregation_level']}_response_curves.csv"
        
        # Save to CSV
        curves_df.to_csv(output_file, index=False)

        print(f"   ✅ Exported {len(curves_df)} rows to {output_file}")
        print(f"   📊 Columns: {list(curves_df.columns)}")
        print(f"   💰 Value ranges:")
        if 'actual_spend' in curves_df.columns:
            print(f"     Spend: [${curves_df['actual_spend'].min():.2f}, ${curves_df['actual_spend'].max():.2f}]")
        print(f"     Impressions: [{curves_df['actual_impressions'].min():.0f}, {curves_df['actual_impressions'].max():.0f}]")
        print(f"     KPI: [{curves_df['actual_kpi_contribution'].min():.2f}, {curves_df['actual_kpi_contribution'].max():.2f}]")

        return curves_df

    def _calculate_key_spending_points(self) -> dict:
        """Calculate historical average and half-saturation spending points.
        
        Returns:
            Dictionary containing:
            - historical_avg_spend: Actual historical average spend per channel
            - half_saturation_spend: Half-saturation spend per channel (ec_50 * median)
            - metadata: Additional information about calculations
        """
        try:
            # Get historical spend data
            historical_spend = self.model.input_data.get_total_spend()  # (n_geos, n_times, n_channels)
            
            # Aggregate across geos to get national spend
            national_spend = np.sum(historical_spend, axis=0)  # (n_times, n_channels)
            
            # Calculate historical average across time
            historical_avg_spend = np.mean(national_spend, axis=0)  # (n_channels,)
            
            # Calculate historical median across time for half-saturation
            historical_median_spend = np.median(national_spend, axis=0)  # (n_channels,)
            
            # Calculate half-saturation using ec_50 * historical_median
            ec_values = self.media_params.get('ec', np.ones(self.n_channels))  # Default to 1.0 if missing
            half_saturation_spend = ec_values * historical_median_spend
            
            # Create metadata
            metadata = {
                'calculation_method': 'actual_historical_data',
                'historical_spend_shape': historical_spend.shape,
                'national_spend_shape': national_spend.shape,
                'channels': self.channel_names,
                'ec_values': ec_values.tolist() if hasattr(ec_values, 'tolist') else ec_values,
                'historical_avg_range': [float(historical_avg_spend.min()), float(historical_avg_spend.max())],
                'half_saturation_range': [float(half_saturation_spend.min()), float(half_saturation_spend.max())]
            }
            
            print(f"📊 Key spending points calculated:")
            print(f"   Historical average spend: [{historical_avg_spend.min():,.0f}, {historical_avg_spend.max():,.0f}]")
            print(f"   Half-saturation spend: [{half_saturation_spend.min():,.0f}, {half_saturation_spend.max():,.0f}]")
            print(f"   EC values used: {[f'{x:.3f}' for x in ec_values]}")
            
            return {
                'historical_avg_spend': historical_avg_spend,
                'half_saturation_spend': half_saturation_spend,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"⚠️  Could not calculate key spending points: {e}")
            print(f"   Falling back to default estimates")
            
            # Fallback to reasonable defaults
            n_channels = len(self.channel_names)
            return {
                'historical_avg_spend': np.ones(n_channels) * 100000,  # Default $100k
                'half_saturation_spend': np.ones(n_channels) * 200000,  # Default $200k
                'metadata': {
                    'calculation_method': 'fallback_defaults',
                    'error': str(e)
                }
            }

    def plot_response_curves(self, response_curves: dict, curve_metadata: dict, 
                           figure_size: tuple = (12, 12), n_columns: int = 1, 
                           marker_size: int = 8, legend_fontsize: int = 8,
                           save_plots: bool = True, output_dir: str = ".") -> None:
        """Create response curve visualizations matching the MPA style.

        Args:
            response_curves: Output from generate_response_curves()
            curve_metadata: Metadata from generate_response_curves()
            figure_size: Size of the figure (width, height)
            n_columns: Number of columns in the subplot grid
            marker_size: Size of markers for key points
            legend_fontsize: Font size for legends
            save_plots: If True, save plots to files
            output_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import matplotlib.ticker as mticker
        except ImportError:
            print("❌ Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
            return

        # Helper function for number formatting
        def format_large_numbers(x, pos):
            if x >= 1e9:
                return f'{x/1e9:.1f}B'  # Billions
            elif x >= 1e6:
                return f'{x/1e6:.1f}M'  # Millions
            elif x >= 1e3:
                return f'{x/1e3:.1f}K'  # Thousands
            else:
                return f'{x:.0f}'  # Default

        def _calculate_number_rows_plot(n_media_channels: int, n_columns: int):
            """Calculate number of rows needed for n_channels + 1 combined plot."""
            return (n_media_channels + n_columns) // n_columns

        # Set up plotting style
        plt.style.use('default')
        _PALETTE = sns.color_palette(n_colors=100)
        
        media_names = list(response_curves.keys())
        n_media_channels = len(media_names)
        kpi_label = "KPI Contribution"
        
        # Calculate key spending points using actual historical data
        key_points = self._calculate_key_spending_points()
        historical_avg_spend = key_points['historical_avg_spend']
        half_saturation_spend = key_points['half_saturation_spend']
        
        # Create figure with subplots for each channel + combined plot
        fig = plt.figure(figsize=figure_size, tight_layout=True)
        n_rows = _calculate_number_rows_plot(n_media_channels, n_columns)
        
        # Combined plot at the bottom
        last_ax = fig.add_subplot(n_rows, 1, n_rows)
        
        # Individual plots for each channel
        for i, (channel_name, curve_data) in enumerate(response_curves.items()):
            ax = fig.add_subplot(n_rows, n_columns, i + 1)
            
            spend_data = curve_data['actual_spend']
            kpi_data = curve_data['actual_kpi_contributions']
            
            # Main response curve for individual channel
            sns.lineplot(
                x=spend_data,
                y=kpi_data,
                label=channel_name,
                color=_PALETTE[i],
                ax=ax
            )
            
            # Add same curve to combined plot
            sns.lineplot(
                x=spend_data,
                y=kpi_data,
                label=channel_name,
                color=_PALETTE[i],
                ax=last_ax
            )
            
            # Calculate and add key markers using actual historical data
            if len(spend_data) > 5:  # Only add markers if we have enough data points
                
                # Historical Average (actual historical average spend)
                hist_spend_target = historical_avg_spend[i]
                hist_idx = np.argmin(np.abs(spend_data - hist_spend_target))
                hist_spend = spend_data[hist_idx]
                hist_kpi = kpi_data[hist_idx]
                
                # Format label
                if hist_spend >= 1e6:
                    label_formatted = f"{hist_spend/1e6:.1f}M"
                elif hist_spend >= 1e3:
                    label_formatted = f"{hist_spend/1e3:.1f}K"
                else:
                    label_formatted = f"{hist_spend:.0f}"
                
                ax.plot(
                    hist_spend, hist_kpi,
                    marker="o", markersize=marker_size,
                    label=f"Historical Avg = {label_formatted}",
                    color="grey"
                )
                
                last_ax.plot(
                    hist_spend, hist_kpi,
                    marker="o", markersize=marker_size,
                    label="Historical Avg" if i + 1 == n_media_channels else None,
                    color="grey"
                )
                
                # Half Saturation (ec_50 * historical_median)
                half_spend_target = half_saturation_spend[i]
                half_sat_idx = np.argmin(np.abs(spend_data - half_spend_target))
                half_spend = spend_data[half_sat_idx]
                half_kpi = kpi_data[half_sat_idx]
                
                if half_spend >= 1e6:
                    label_formatted = f"{half_spend/1e6:.1f}M"
                elif half_spend >= 1e3:
                    label_formatted = f"{half_spend/1e3:.1f}K"
                else:
                    label_formatted = f"{half_spend:.0f}"
                
                ax.plot(
                    half_spend, half_kpi,
                    marker="o", markersize=marker_size,
                    label=f"Half Saturation = {label_formatted}",
                    color="green"
                )
                
                last_ax.plot(
                    half_spend, half_kpi,
                    marker="o", markersize=marker_size,
                    label="Half Saturation" if i + 1 == n_media_channels else None,
                    color="green"
                )
            
            # Format individual channel subplot
            ax.set_ylabel(kpi_label)
            ax.set_xlabel("Spend")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_large_numbers))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_large_numbers))
            ax.legend(fontsize=legend_fontsize, loc="lower right")
        
        # Format combined plot
        fig.suptitle("Response curves", fontsize=20)
        last_ax.set_ylabel(kpi_label)
        last_ax.set_xlabel("Spend per channel")
        last_ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_large_numbers))
        last_ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_large_numbers))
        last_ax.legend(fontsize=legend_fontsize, loc="lower right")
        
        if save_plots:
            filename = f"{output_dir}/response_curves_mpa_style.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   📊 Saved: {filename}")
        
        plt.show()
        plt.close()
        
        print(f"\n✅ MPA-style response curve visualizations created!")
        print(f"   📈 Individual channel plots with key saturation markers")
        print(f"   📈 Combined plot showing all channels together")
        print(f"   💡 Historical Avg (grey), Half Saturation (green)")
        print(f"   🔍 Using actual historical data for marker calculations")
        if save_plots:
            print(f"   💾 Plot saved to: {output_dir}")

    def generate_geo_media_scenarios(self,
                                   max_multiplier: float = 2.0,
                                   num_steps: int = 50,
                                   selected_geos: Optional[Sequence[str]] = None,
                                   selected_times: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, dict]:
        """Generate geo-level scaled media scenarios from 0 to max_multiplier * geo_max_scaled_media.

        For each geography, creates scaled media ranges from 0 to max_multiplier times the
        maximum historical scaled media for that geo and channel combination.

        Args:
            max_multiplier: Maximum multiplier for scaled media ranges (default 2.0 = 200% of max).
            num_steps: Number of scaled media steps to generate per geo.
            selected_geos: Optional subset of geographies to include.
            selected_times: Optional subset of time periods to consider.

        Returns:
            Tuple containing:
            - media_scenarios: numpy array of shape (n_geos, num_steps, n_channels)
            - metadata: dict with information about the simulation
        """
        print(f"🗺️ Generating geo-level scaled media scenarios...")
        print(f"   Max multiplier: {max_multiplier}x")
        print(f"   Steps per geo: {num_steps}")

        # Apply geo filtering if specified
        scaled_media_data = self.historical_scaled_media
        geo_names = self.model.input_data.geo.values.tolist()

        if selected_geos is not None:
            geo_indices = [i for i, geo in enumerate(geo_names) if geo in selected_geos]
            scaled_media_data = tf.gather(scaled_media_data, geo_indices, axis=0)
            geo_names = [geo_names[i] for i in geo_indices]
            print(f"   Filtered to {len(selected_geos)} geos: {selected_geos}")

        # Apply time filtering if specified
        if selected_times is not None:
            time_names = self.model.input_data.time.values.tolist()
            time_indices = [i for i, time in enumerate(time_names) if str(time) in selected_times]
            scaled_media_data = tf.gather(scaled_media_data, time_indices, axis=1)
            print(f"   Filtered to {len(selected_times)} time periods")

        # Calculate maximum scaled media per geo per channel
        # Shape: (n_geos, n_channels)
        geo_max_scaled_media = tf.reduce_max(scaled_media_data, axis=1)

        print(f"   Historical scaled media shape: {scaled_media_data.shape}")
        print(f"   Geo max scaled media shape: {geo_max_scaled_media.shape}")
        print(f"   Scaled media range: [{tf.reduce_min(geo_max_scaled_media).numpy():.3f}, {tf.reduce_max(geo_max_scaled_media).numpy():.3f}]")

        # Generate scaled media scenarios for each geo independently
        final_n_geos = geo_max_scaled_media.shape[0]
        media_scenarios = np.zeros((final_n_geos, num_steps, self.n_channels))

        # Create scaled media ranges for each geo
        for geo_idx in range(final_n_geos):
            geo_max_media = geo_max_scaled_media[geo_idx].numpy()  # Shape: (n_channels,)

            for channel_idx in range(self.n_channels):
                max_media_for_channel = geo_max_media[channel_idx]

                # Create linear range from 0 to max_multiplier * max_scaled_media
                media_range = np.linspace(
                    start=0.0,
                    stop=max_multiplier * max_media_for_channel,
                    num=num_steps
                )
                media_scenarios[geo_idx, :, channel_idx] = media_range

        # Create metadata
        metadata = {
            'max_multiplier': max_multiplier,
            'num_steps': num_steps,
            'n_geos': final_n_geos,
            'n_channels': self.n_channels,
            'channel_names': self.channel_names,
            'geo_names': geo_names,
            'selected_geos': selected_geos,
            'selected_times': selected_times,
            'original_shape': self.historical_scaled_media.shape,
            'final_shape': media_scenarios.shape,
            'data_type': 'scaled_media',
            'value_range': [float(tf.reduce_min(geo_max_scaled_media).numpy()),
                          float(tf.reduce_max(geo_max_scaled_media).numpy())],
            'simulation_range': [0.0, float(tf.reduce_max(geo_max_scaled_media).numpy()) * max_multiplier]
        }

        print(f"✅ Geo scaled media scenarios generated:")
        print(f"   Output shape: {media_scenarios.shape} (geos, steps, channels)")
        print(f"   Scaled media range per geo: 0 to {max_multiplier}x historical max")
        print(f"   Max simulation values: {[f'{x:.3f}' for x in media_scenarios.max(axis=(0,1))]}")

        return media_scenarios, metadata

    def get_geo_media_summary(self, media_scenarios: np.ndarray, metadata: dict,
                             include_actual_impressions: bool = True) -> pd.DataFrame:
        """Create summary DataFrame of scaled media scenarios for inspection.

        Args:
            media_scenarios: Output from generate_geo_media_scenarios.
            metadata: Metadata from generate_geo_media_scenarios.
            include_actual_impressions: Whether to include actual impression values.

        Returns:
            DataFrame with scaled media range summaries per geo per channel,
            optionally including actual impression values.
        """
        summary_rows = []

        # Get actual impressions if requested
        actual_impressions = None
        reverse_metadata = None
        if include_actual_impressions:
            try:
                actual_impressions, reverse_metadata = self.reverse_transform_media_scenarios(
                    media_scenarios, metadata
                )
                print(f"   ✅ Including actual impression values in summary")
            except Exception as e:
                print(f"   ⚠️  Could not compute actual impressions: {str(e)}")
                include_actual_impressions = False

        for geo_idx in range(metadata['n_geos']):
            geo_name = metadata['geo_names'][geo_idx]

            for channel_idx, channel_name in enumerate(metadata['channel_names']):
                channel_media_range = media_scenarios[geo_idx, :, channel_idx]

                # Base summary data
                row_data = {
                    'geo': geo_name,
                    'geo_index': geo_idx,
                    'channel': channel_name,
                    'channel_index': channel_idx,
                    'min_scaled_media': channel_media_range.min(),
                    'max_scaled_media': channel_media_range.max(),
                    'steps': len(channel_media_range),
                    'avg_step_size': (channel_media_range.max() - channel_media_range.min()) / (len(channel_media_range) - 1) if len(channel_media_range) > 1 else 0
                }

                # Add actual impression data if available
                if include_actual_impressions and actual_impressions is not None:
                    actual_channel_range = actual_impressions[geo_idx, :, channel_idx]
                    row_data.update({
                        'min_actual_impressions': actual_channel_range.min(),
                        'max_actual_impressions': actual_channel_range.max(),
                        'avg_actual_step_size': (actual_channel_range.max() - actual_channel_range.min()) / (len(actual_channel_range) - 1) if len(actual_channel_range) > 1 else 0,
                        'scale_factor': actual_channel_range.max() / channel_media_range.max() if channel_media_range.max() > 0 else 1.0
                    })

                summary_rows.append(row_data)

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def export_sample_data(self, media_scenarios: np.ndarray, metadata: dict,
                          output_file: str = "geo_media_scenarios_sample.csv",
                          include_actual_values: bool = True):
        """Export sample scaled media scenarios for inspection.

        Args:
            media_scenarios: Output from generate_geo_media_scenarios.
            metadata: Metadata from generate_geo_media_scenarios.
            output_file: CSV filename for export.
            include_actual_values: Whether to include actual impression values.
        """
        sample_rows = []

        # Get actual impressions if requested
        actual_impressions = None
        if include_actual_values:
            try:
                actual_impressions, _ = self.reverse_transform_media_scenarios(
                    media_scenarios, metadata
                )
                print(f"   ✅ Including actual impression values in export")
            except Exception as e:
                print(f"   ⚠️  Could not compute actual impressions for export: {str(e)}")
                include_actual_values = False

        # Sample every 5th step to keep file manageable
        step_sample = range(0, metadata['num_steps'], max(1, metadata['num_steps'] // 10))

        for geo_idx in range(min(5, metadata['n_geos'])):  # First 5 geos only
            geo_name = metadata['geo_names'][geo_idx]

            for step_idx in step_sample:
                row = {
                    'geo': geo_name,
                    'geo_index': geo_idx,
                    'step': step_idx,
                    'step_multiplier': step_idx / (metadata['num_steps'] - 1) * metadata['max_multiplier']
                }

                # Add scaled media and actual impressions for each channel
                for channel_idx, channel_name in enumerate(metadata['channel_names']):
                    scaled_value = media_scenarios[geo_idx, step_idx, channel_idx]
                    row[f'{channel_name}_scaled_media'] = scaled_value

                    if include_actual_values and actual_impressions is not None:
                        actual_value = actual_impressions[geo_idx, step_idx, channel_idx]
                        row[f'{channel_name}_actual_impressions'] = actual_value

                sample_rows.append(row)

        sample_df = pd.DataFrame(sample_rows)
        sample_df.to_csv(output_file, index=False)
        print(f"📄 Sample data exported to: {output_file}")
        print(f"   Rows: {len(sample_df)}")
        print(f"   Columns: {list(sample_df.columns)}")
        print(f"   Value range: [{sample_df.select_dtypes(include=[np.number]).min().min():.6f}, {sample_df.select_dtypes(include=[np.number]).max().max():.3f}]")

        return sample_df

    def export_transformation_results(self,
                                    media_scenarios: np.ndarray,
                                    media_effects: np.ndarray,
                                    metadata: dict,
                                    output_file: str = "complete_transformation_results.csv"):
        """Export complete transformation pipeline results including actual values.

        Args:
            media_scenarios: Scaled media scenarios from generate_geo_media_scenarios()
            media_effects: Media effects from apply_media_transformations()
            metadata: Metadata from generate_geo_media_scenarios()
            output_file: CSV filename for export
        """
        print(f"📄 Exporting complete transformation results...")

        # Get reverse transformations
        try:
            actual_impressions, _ = self.reverse_transform_media_scenarios(media_scenarios, metadata)
            actual_kpi_effects, _ = self.reverse_transform_media_effects(media_effects, metadata)
            reverse_success = True
        except Exception as e:
            print(f"   ⚠️  Could not compute reverse transformations: {str(e)}")
            actual_impressions = media_scenarios.copy()
            actual_kpi_effects = media_effects.copy()
            reverse_success = False

        export_rows = []

        # Sample strategically to keep file manageable
        step_sample = range(0, metadata['num_steps'], max(1, metadata['num_steps'] // 15))
        geo_sample = range(0, min(10, metadata['n_geos']))  # First 10 geos

        for geo_idx in geo_sample:
            geo_name = metadata['geo_names'][geo_idx]

            for step_idx in step_sample:
                multiplier = step_idx / (metadata['num_steps'] - 1) * metadata['max_multiplier']

                # Base row data
                row = {
                    'geo': geo_name,
                    'geo_index': geo_idx,
                    'step': step_idx,
                    'step_multiplier': multiplier
                }

                # Add data for each channel
                for channel_idx, channel_name in enumerate(metadata['channel_names']):
                    # Scaled values
                    scaled_media = media_scenarios[geo_idx, step_idx, channel_idx]
                    scaled_effects = media_effects[geo_idx, step_idx, channel_idx]

                    row.update({
                        f'{channel_name}_scaled_media': scaled_media,
                        f'{channel_name}_scaled_effects': scaled_effects
                    })

                    # Actual values if available
                    if reverse_success:
                        actual_impressions_val = actual_impressions[geo_idx, step_idx, channel_idx]
                        actual_kpi_val = actual_kpi_effects[geo_idx, step_idx, channel_idx]

                        row.update({
                            f'{channel_name}_actual_impressions': actual_impressions_val,
                            f'{channel_name}_actual_kpi_contribution': actual_kpi_val
                        })

                export_rows.append(row)

        # Create and export DataFrame
        results_df = pd.DataFrame(export_rows)
        results_df.to_csv(output_file, index=False)

        print(f"   File: {output_file}")
        print(f"   Rows: {len(results_df)}")
        print(f"   Columns: {len(results_df.columns)}")
        print(f"   Reverse transformations: {'✅ Included' if reverse_success else '❌ Failed'}")

        return results_df

    def apply_media_transformations(self,
                                  media_scenarios: np.ndarray,
                                  metadata: dict) -> Tuple[np.ndarray, dict]:
        """Apply adstock and hill transformations to scaled media scenarios.

        This method takes the output from generate_geo_media_scenarios() and applies
        the same adstock and hill transformations used in the Meridian model.

        Args:
            media_scenarios: Output from generate_geo_media_scenarios()
                           Shape: (n_geos, n_steps, n_channels)
            metadata: Metadata from generate_geo_media_scenarios()

        Returns:
            Tuple containing:
            - media_effects: Shape (n_geos, n_steps, n_channels) with transformed effects
            - transform_metadata: Dict with transformation information
        """
        print(f"🔄 Applying media transformations...")
        print(f"   Input shape: {media_scenarios.shape}")
        print(f"   Channels: {metadata['channel_names']}")

        # Validate inputs
        if not self.media_params and not self.rf_params:
            raise ValueError("No transformation parameters available. Check model fitting.")

        n_geos, n_steps, n_channels = media_scenarios.shape
        media_effects = np.zeros_like(media_scenarios)

        # Track transformation statistics
        transformation_stats = {}

        # Apply transformations for each geo independently
        for geo_idx in range(n_geos):
            geo_media = media_scenarios[geo_idx]  # Shape: (n_steps, n_channels)

            # Convert to tensor for transformations
            geo_media_tensor = tf.constant(geo_media, dtype=tf.float32)

            # Apply transformations to each channel
            geo_effects = []
            for channel_idx in range(n_channels):
                channel_name = metadata['channel_names'][channel_idx]

                # Determine if this is a media or RF channel
                if channel_idx < len(self.media_params.get('ec', [])):
                    # Standard media channel
                    params = self.media_params
                elif self.rf_params and (channel_idx - len(self.media_params.get('ec', []))) < len(self.rf_params.get('ec', [])):
                    # R&F channel
                    params = self.rf_params
                    channel_idx = channel_idx - len(self.media_params.get('ec', []))
                else:
                    # No parameters for this channel
                    geo_effects.append(tf.zeros(n_steps))
                    continue

                # Extract channel data: (n_steps,)
                channel_media = geo_media_tensor[:, channel_idx]

                # Extract parameters for this channel
                ec = params['ec'][channel_idx]
                slope = params['slope'][channel_idx]
                alpha = params['alpha'][channel_idx]
                beta = params['beta'][geo_idx, channel_idx] if len(params['beta'].shape) > 1 else params['beta'][channel_idx]

                # Apply transformations: media → adstock → hill → coefficients
                adstocked = self._apply_adstock_transformation(channel_media, alpha)
                saturated = self._apply_hill_transformation(adstocked, ec, slope)
                effects = saturated * beta

                geo_effects.append(effects)

                # Track stats for first geo
                if geo_idx == 0:
                    transformation_stats[channel_name] = {
                        'alpha': float(alpha),
                        'ec': float(ec),
                        'slope': float(slope),
                        'beta': float(beta),
                        'input_range': [float(tf.reduce_min(channel_media)), float(tf.reduce_max(channel_media))],
                        'adstock_range': [float(tf.reduce_min(adstocked)), float(tf.reduce_max(adstocked))],
                        'saturated_range': [float(tf.reduce_min(saturated)), float(tf.reduce_max(saturated))],
                        'effects_range': [float(tf.reduce_min(effects)), float(tf.reduce_max(effects))]
                    }

            # Stack channel effects for this geo
            if geo_effects:
                geo_effects_tensor = tf.stack(geo_effects, axis=1)  # Shape: (n_steps, n_channels)
                media_effects[geo_idx] = geo_effects_tensor.numpy()

        # Create transformation metadata
        transform_metadata = {
            'transformation_applied': True,
            'transformation_steps': ['adstock', 'hill', 'coefficients'],
            'input_shape': media_scenarios.shape,
            'output_shape': media_effects.shape,
            'channel_stats': transformation_stats,
            'value_range': [float(media_effects.min()), float(media_effects.max())],
            'n_media_channels': len(self.media_params.get('ec', [])),
            'n_rf_channels': len(self.rf_params.get('ec', [])),
        }

        print(f"✅ Media transformations applied:")
        print(f"   Output shape: {media_effects.shape}")
        print(f"   Value range: [{media_effects.min():.6f}, {media_effects.max():.3f}]")
        print(f"   Transformation steps: {transform_metadata['transformation_steps']}")

        return media_effects, transform_metadata

    def _apply_adstock_transformation(self, media_data: tf.Tensor, alpha: float) -> tf.Tensor:
        """Apply adstock transformation using Meridian's logic.

        Args:
            media_data: Media data tensor of shape (n_steps,)
            alpha: Adstock parameter (0 <= alpha < 1)

        Returns:
            Adstocked media tensor of shape (n_steps,)
        """
        # Use Meridian's official AdstockTransformer approach
        from meridian.model.adstock_hill import AdstockTransformer

        n_steps = media_data.shape[0]

        # Reshape for AdstockTransformer: (n_geos=1, n_times=n_steps, n_channels=1)
        media_reshaped = tf.reshape(media_data, (1, n_steps, 1))
        alpha_tensor = tf.constant([alpha], dtype=tf.float32)

        # Create transformer with appropriate parameters
        # Use model's actual max_lag setting
        model_max_lag = self.model.model_spec.max_lag if hasattr(self.model, 'model_spec') else 8
        max_lag = min(model_max_lag, n_steps - 1)  # Not more than available time steps
        transformer = AdstockTransformer(
            alpha=alpha_tensor,
            max_lag=max_lag,
            n_times_output=n_steps
        )

        # Apply transformation
        adstocked = transformer.forward(media_reshaped)

        # Reshape back to (n_steps,)
        return tf.squeeze(adstocked, axis=[0, 2])

    def _apply_hill_transformation(self, media_data: tf.Tensor, ec: float, slope: float) -> tf.Tensor:
        """Apply hill saturation transformation using Meridian's logic.

        Args:
            media_data: Media data tensor of shape (n_steps,)
            ec: Half-saturation parameter
            slope: Hill slope parameter

        Returns:
            Hill-transformed media tensor of shape (n_steps,)
        """
        # Use Meridian's official HillTransformer approach
        from meridian.model.adstock_hill import HillTransformer

        n_steps = media_data.shape[0]

        # Reshape for HillTransformer: (n_geos=1, n_times=n_steps, n_channels=1)
        media_reshaped = tf.reshape(media_data, (1, n_steps, 1))
        ec_tensor = tf.constant([ec], dtype=tf.float32)
        slope_tensor = tf.constant([slope], dtype=tf.float32)

        # Create transformer
        transformer = HillTransformer(ec=ec_tensor, slope=slope_tensor)

        # Apply transformation
        saturated = transformer.forward(media_reshaped)

        # Reshape back to (n_steps,)
        return tf.squeeze(saturated, axis=[0, 2])


def test_geo_media_simulator(model_path: str):
    """Simple test function for the ResponseCurveGenerator.

    Args:
        model_path: Path to the Meridian model pickle file.
    """
    print("🧪 Testing GeoMediaSimulator...")

    try:
        # Load model
        print(f"📁 Loading model: {model_path}")
        mmm = model.load_mmm(model_path)

        # Initialize response curve generator
        generator = ResponseCurveGenerator(mmm)

        # Generate scaled media scenarios
        media_scenarios, metadata = generator.generate_geo_media_scenarios(
            max_multiplier=2.0,
            num_steps=20  # Small number for testing
        )

        # Create summary
        summary = generator.get_geo_media_summary(media_scenarios, metadata)
        print(f"\n📊 Scaled Media Summary (first 10 rows):")
        print(summary.head(10).to_string(index=False))

        # Export sample with actual values
        sample_df = generator.export_sample_data(media_scenarios, metadata,
                                                output_file="geo_media_scenarios_sample.csv",
                                                include_actual_values=True)

        print(f"\n🔄 Testing media transformations...")

        # Apply transformations
        media_effects, transform_metadata = generator.apply_media_transformations(
            media_scenarios, metadata
        )

        print(f"\n🔄 Testing reverse transformations...")

        # Test reverse transformations
        actual_impressions, actual_spend, impression_metadata = generator.reverse_transform_media_scenarios(
            media_scenarios, metadata
        )

        actual_kpi_effects, kpi_metadata = generator.reverse_transform_media_effects(
            media_effects, metadata
        )

        # Test the new complete export functionality
        print(f"\n📄 Testing complete transformation export...")
        complete_results_df = generator.export_transformation_results(
            media_scenarios, media_effects, metadata,
            output_file="complete_transformation_results.csv"
        )

        # Test the new response curve generation
        print(f"\n🎯 Testing response curve generation...")
        response_curves, curve_metadata = generator.generate_response_curves(
            max_multiplier=2.0,
            num_steps=30,  # Smaller for testing
            aggregation_level="national"
        )

        # Export response curves
        curves_df = generator.export_response_curves(
            response_curves, curve_metadata,
            output_file="national_response_curves.csv"
        )

        print(f"\n✅ Test completed successfully!")
        print(f"📊 Key Validation:")
        print(f"   Scenarios shape: {media_scenarios.shape} (expected: ({generator.n_geos}, 20, {generator.n_channels}))")
        print(f"   Scenarios range (scaled): [{media_scenarios.min():.6f}, {media_scenarios.max():.3f}]")
        print(f"   Actual impressions range: [{actual_impressions.min():.0f}, {actual_impressions.max():.0f}]")
        print(f"   Effects shape: {media_effects.shape}")
        print(f"   Effects range (scaled): [{media_effects.min():.6f}, {media_effects.max():.6f}]")
        print(f"   Actual KPI effects range: [{actual_kpi_effects.min():.2f}, {actual_kpi_effects.max():.2f}]")
        print(f"   Transformations: {transform_metadata['transformation_steps']}")

        return generator, media_scenarios, metadata, summary, media_effects, transform_metadata, actual_impressions, actual_kpi_effects

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


if __name__ == "__main__":
    # Test with ALDI model
    MODEL_PATH = "/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/model_objects/0_test_working_spend_ALDI_US_Starcom.pkl"
    test_geo_media_simulator(MODEL_PATH)
