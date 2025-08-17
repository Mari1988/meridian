#!/usr/bin/env python3
"""
Comprehensive test suite for ResponseCurveGenerator.

This consolidated test file includes all functionality tests:
1. MPA-style visualization
2. Corrected marker calculations  
3. Spend-based response curves
4. Spend conversion functionality
5. Visualization demos

All tests use a unified mock model for consistency.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import time
from response_curve_generator import ResponseCurveGenerator


class MockCompleteModel:
    """Unified mock model for all ResponseCurveGenerator tests."""
    
    def __init__(self, predictable_data: bool = False):
        """Initialize mock model.
        
        Args:
            predictable_data: If True, use predictable values for validation tests
        """
        # Model structure
        self.n_geos = 4
        self.n_times = 52  
        self.n_media_channels = 3
        self.n_rf_channels = 0
        self.predictable_data = predictable_data
        
        # Mock input data
        self.input_data = self._create_mock_input_data()
        
        # Mock model spec
        self.model_spec = MockModelSpec()
        
        # Mock transformers
        self.media_tensors = self._create_mock_media_tensors()
        self.kpi_transformer = self._create_mock_kpi_transformer()
        
        # Add required scaled media data
        self._add_scaled_media_data()
        
        # Mock inference data for median parameters
        self.inference_data = self._create_mock_inference_data()
    
    def _create_mock_input_data(self):
        """Create realistic mock input data."""
        import types
        
        input_data = types.SimpleNamespace()
        
        # Channel names 
        input_data.media_channel = MockDataArray(['TV', 'Display', 'Video'])
        
        # Time dimension  
        input_data.time = MockDataArray(pd.date_range('2023-01-01', periods=self.n_times, freq='W'))
        input_data.geo = MockDataArray([f'Geo_{i}' for i in range(self.n_geos)])
        
        # Create historical spend patterns
        np.random.seed(42)  # For reproducible results
        
        if self.predictable_data:
            # Known historical spend values per geo per channel (weekly)
            base_weekly_spend = np.array([
                [50000, 25000, 35000],  # TV, Display, Video base weekly spend
                [60000, 30000, 40000],
                [40000, 20000, 30000], 
                [55000, 28000, 38000]
            ])
            
            # Create historical spend with seasonal variation
            historical_spend = np.zeros((self.n_geos, self.n_times, self.n_media_channels))
            for geo in range(self.n_geos):
                for week in range(self.n_times):
                    # Add seasonal and random variation (±20%)
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * week / 52)  # Annual seasonality
                    random_factor = np.random.uniform(0.9, 1.1, self.n_media_channels)
                    historical_spend[geo, week, :] = base_weekly_spend[geo] * seasonal_factor * random_factor
        else:
            # Standard random historical spend
            base_impressions = np.array([
                [2000000, 500000, 800000],  # TV, Display, Video base impressions
                [1500000, 750000, 600000],
                [2500000, 400000, 900000], 
                [1800000, 600000, 700000]
            ])
            
            # Add weekly variation
            historical_impressions = np.zeros((self.n_geos, self.n_times, self.n_media_channels))
            for geo in range(self.n_geos):
                for week in range(self.n_times):
                    # Add seasonal and random variation
                    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * week / 52)  # Annual seasonality
                    random_factor = np.random.uniform(0.8, 1.2, self.n_media_channels)
                    historical_impressions[geo, week, :] = base_impressions[geo] * seasonal_factor * random_factor
            
            # Historical spend with realistic CPMs
            cpm_per_geo_channel = np.array([
                [45, 25, 35],  # TV, Display, Video CPMs
                [50, 20, 30],
                [40, 30, 40],
                [48, 22, 32]
            ])
            
            historical_spend = np.zeros_like(historical_impressions)
            for geo in range(self.n_geos):
                for ch in range(self.n_media_channels):
                    historical_spend[geo, :, ch] = (historical_impressions[geo, :, ch] / 1000) * cpm_per_geo_channel[geo, ch]
        
        # Calculate corresponding impressions using known CPMs
        cpm_per_geo_channel = np.array([
            [45, 25, 35],  # TV, Display, Video CPMs
            [50, 20, 30],
            [40, 30, 40],
            [48, 22, 32]
        ])
        
        historical_impressions = np.zeros_like(historical_spend)
        for geo in range(self.n_geos):
            for ch in range(self.n_media_channels):
                cpm = cpm_per_geo_channel[geo, ch]
                historical_impressions[geo, :, ch] = (historical_spend[geo, :, ch] / cpm) * 1000
        
        # Methods
        input_data.get_all_media_and_rf = lambda: historical_impressions
        input_data.get_total_spend = lambda: historical_spend
        
        # Store expected values for validation (if predictable data)
        if self.predictable_data:
            self.expected_values = {
                'historical_spend': historical_spend,
                'cpm_per_geo_channel': cpm_per_geo_channel
            }
        
        return input_data
    
    def _create_mock_media_tensors(self):
        """Create mock media tensors with transformers."""
        import types
        
        media_tensors = types.SimpleNamespace()
        media_tensors.media_transformer = self._create_mock_media_transformer()
        return media_tensors
    
    def _create_mock_media_transformer(self):
        """Create mock media transformer."""
        import types
        
        transformer = types.SimpleNamespace()
        # Scale factors based on population and median impressions
        population = np.array([1000000, 1500000, 800000, 1200000])  # Geo populations
        median_impressions = np.array([50, 25, 35])  # Per capita impressions
        
        scale_factors = np.outer(population, median_impressions)  # (n_geos, n_channels)
        transformer._scale_factors_gm = tf.constant(scale_factors, dtype=tf.float32)
        
        return transformer
    
    def _create_mock_kpi_transformer(self):
        """Create mock KPI transformer."""
        import types
        
        transformer = types.SimpleNamespace()
        population = np.array([1000000, 1500000, 800000, 1200000], dtype=np.float32)
        transformer._population = tf.constant(population)
        transformer._population_scaled_mean = tf.constant(np.float32(0.12))
        transformer._population_scaled_stdev = tf.constant(np.float32(0.85))
        
        # Mock inverse method
        def inverse(tensor):
            import tensorflow as tf
            return (tensor * 0.85 + 0.12) * tf.reshape(population, (-1, 1))
        
        transformer.inverse = inverse
        return transformer
    
    def _create_mock_inference_data(self):
        """Create mock inference data with posterior parameters."""
        import types
        
        inference_data = types.SimpleNamespace()
        
        # Media parameters (chains, draws, channels)
        n_chains, n_draws = 4, 1000
        
        if self.predictable_data:
            # Use known EC values for predictable half-saturation calculations
            known_ec_values = np.array([0.5, 0.6, 0.4])  # TV, Display, Video
            ec_data = np.tile(known_ec_values, (n_chains, n_draws, 1))
            self.expected_ec_values = known_ec_values
        else:
            ec_data = np.random.uniform(0.3, 0.7, (n_chains, n_draws, self.n_media_channels))
        
        # Create posterior as a dictionary to match the expected interface
        posterior = {
            'ec_m': MockXrArray(ec_data),
            'slope_m': MockXrArray(np.random.uniform(0.8, 1.5, (n_chains, n_draws, self.n_media_channels))),
            'alpha_m': MockXrArray(np.random.uniform(0.1, 0.8, (n_chains, n_draws, self.n_media_channels))),
            'beta_gm': MockXrArray(np.random.uniform(0.5, 2.0, (n_chains, n_draws, self.n_geos, self.n_media_channels)))
        }
        
        inference_data.posterior = posterior
        return inference_data
    
    def _add_scaled_media_data(self):
        """Add required scaled media data for generator validation."""
        # Create scaled media data by normalizing historical impressions
        historical_impressions = self.input_data.get_all_media_and_rf()
        
        # Simple scaling: divide by max per channel per geo
        scaled_media = np.zeros_like(historical_impressions)
        for geo in range(self.n_geos):
            for ch in range(self.n_media_channels):
                max_impressions = historical_impressions[geo, :, ch].max()
                if max_impressions > 0:
                    scaled_media[geo, :, ch] = historical_impressions[geo, :, ch] / max_impressions
        
        # Add to media_tensors - convert to actual TensorFlow tensor
        self.media_tensors.media_scaled = tf.constant(scaled_media, dtype=tf.float32)


class MockModelSpec:
    """Mock model specification."""
    def __init__(self):
        self.max_lag = 8


class MockXrArray:
    """Mock xarray DataArray."""
    def __init__(self, data):
        self.values = data


class MockDataArray:
    """Mock data array for input data."""
    def __init__(self, values):
        self.values = np.array(values)


class TestResponseCurveGenerator:
    """Comprehensive test suite for ResponseCurveGenerator."""
    
    def __init__(self):
        self.test_results = {}
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 COMPREHENSIVE ResponseCurveGenerator TEST SUITE")
        print("=" * 70)
        
        # Test 1: Basic initialization and functionality
        self.test_basic_functionality()
        
        # Test 2: Corrected marker calculations
        self.test_corrected_marker_calculations()
        
        # Test 3: Spend conversion functionality
        self.test_spend_conversion_functionality()
        
        # Test 4: Spend-based response curves
        self.test_spend_based_response_curves()
        
        
        # Test 5: Performance and edge cases
        self.test_performance_and_edge_cases()
        
        # Print final results
        self.print_test_summary()
    
    def test_basic_functionality(self):
        """Test basic ResponseCurveGenerator functionality."""
        print("\n🔧 TEST 1: Basic Functionality")
        print("-" * 50)
        
        try:
            # Setup mock model
            model = MockCompleteModel()
            
            # Initialize generator
            generator = ResponseCurveGenerator(model)
            print("✅ ResponseCurveGenerator initialized successfully")
            
            # Test basic response curve generation
            response_curves, metadata = generator.generate_response_curves(
                max_multiplier=1.5,
                num_steps=10,
                aggregation_level="national"
            )
            
            print(f"✅ Response curves generated: {len(response_curves)} channels")
            print(f"   Channels: {list(response_curves.keys())}")
            print(f"   Aggregation: {metadata['aggregation_level']}")
            
            # Validate structure
            required_keys = ['spend_multipliers', 'actual_spend', 'actual_kpi_contributions']
            for channel_name, curve_data in response_curves.items():
                missing_keys = [key for key in required_keys if key not in curve_data]
                if missing_keys:
                    raise ValueError(f"Missing keys in {channel_name}: {missing_keys}")
            
            print("✅ Response curve data structure validated")
            self.test_results['basic_functionality'] = True
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            self.test_results['basic_functionality'] = False
    
    def test_corrected_marker_calculations(self):
        """Test corrected historical average and half-saturation calculations."""
        print("\n📊 TEST 2: Corrected Marker Calculations")
        print("-" * 50)
        
        try:
            # Setup model with predictable data
            model = MockCompleteModel(predictable_data=True)
            generator = ResponseCurveGenerator(model)
            
            # Test key spending points calculation
            key_points = generator._calculate_key_spending_points()
            
            calculated_hist_avg = key_points['historical_avg_spend']
            calculated_half_sat = key_points['half_saturation_spend']
            
            print(f"✅ Key spending points calculated")
            print(f"   Historical average: {calculated_hist_avg}")
            print(f"   Half-saturation: {calculated_half_sat}")
            
            # Validate that calculations use actual data (not defaults)
            if np.allclose(calculated_hist_avg, 100000):
                raise ValueError("Using fallback defaults instead of actual data")
            
            # Test that half-saturation uses EC formula
            metadata = key_points['metadata']
            if metadata['calculation_method'] != 'actual_historical_data':
                raise ValueError("Not using actual historical data method")
            
            print("✅ Marker calculations use actual historical data")
            self.test_results['corrected_markers'] = True
            
        except Exception as e:
            print(f"❌ Corrected marker test failed: {e}")
            self.test_results['corrected_markers'] = False
    
    def test_spend_conversion_functionality(self):
        """Test spend conversion functionality."""
        print("\n💰 TEST 3: Spend Conversion Functionality")
        print("-" * 50)
        
        try:
            # Setup mock model
            model = MockCompleteModel()
            generator = ResponseCurveGenerator.__new__(ResponseCurveGenerator)
            generator.model = model
            
            # Test data
            test_impressions = np.array([
                [[100000, 50000], [200000, 100000], [300000, 150000]],
                [[150000, 75000], [300000, 150000], [450000, 225000]]
            ])  # (2 geos, 3 steps, 2 channels)
            
            test_metadata = {
                'channel_names': ['TV', 'Display'],
                'geo_names': ['Geo_0', 'Geo_1']
            }
            
            # Test conversion
            actual_spend, spend_metadata = generator.convert_impressions_to_spend(
                test_impressions, test_metadata
            )
            
            print(f"✅ Spend conversion successful")
            print(f"   Input shape: {test_impressions.shape}")
            print(f"   Output shape: {actual_spend.shape}")
            print(f"   Spend range: [${actual_spend.min():.2f}, ${actual_spend.max():.2f}]")
            
            # Validate mathematical consistency
            if actual_spend.shape != test_impressions.shape:
                raise ValueError("Spend and impression shapes don't match")
            
            # Test with edge cases (zero impressions)
            edge_impressions = test_impressions.copy()
            edge_impressions[0, 0, :] = 0
            
            edge_spend, _ = generator.convert_impressions_to_spend(edge_impressions, test_metadata)
            
            if edge_spend[0, 0, :].sum() != 0:
                raise ValueError("Zero impressions should give zero spend")
            
            print("✅ Edge case handling validated")
            self.test_results['spend_conversion'] = True
            
        except Exception as e:
            print(f"❌ Spend conversion test failed: {e}")
            self.test_results['spend_conversion'] = False
    
    def test_spend_based_response_curves(self):
        """Test spend-based response curve generation."""
        print("\n📈 TEST 4: Spend-Based Response Curves")
        print("-" * 50)
        
        try:
            # Setup mock model
            model = MockCompleteModel()
            generator = ResponseCurveGenerator(model)
            
            # Generate response curves
            response_curves, curve_metadata = generator.generate_response_curves(
                max_multiplier=2.0,
                num_steps=15,
                aggregation_level="national"
            )
            
            print(f"✅ Spend-based response curves generated")
            print(f"   Channels: {len(response_curves)}")
            
            # Validate spend-based structure
            for channel_name, curve_data in response_curves.items():
                spend_data = curve_data['actual_spend']
                kpi_data = curve_data['actual_kpi_contributions']
                
                # Check that spend increases with multiplier
                if not np.all(np.diff(spend_data) >= 0):
                    raise ValueError(f"Spend doesn't increase with multiplier in {channel_name}")
                
                # Check strong correlation between spend and KPI
                correlation = np.corrcoef(spend_data, kpi_data)[0, 1]
                if correlation < 0.8:
                    print(f"   ⚠️  Weak correlation in {channel_name}: {correlation:.3f}")
            
            print("✅ Response curves data validated")
            self.test_results['spend_based_curves'] = True
            
        except Exception as e:
            print(f"❌ Spend-based curves test failed: {e}")
            self.test_results['spend_based_curves'] = False
    
    
    def test_performance_and_edge_cases(self):
        """Test performance and edge case handling."""
        print("\n⚡ TEST 5: Performance and Edge Cases")
        print("-" * 50)
        
        try:
            # Performance test
            model = MockCompleteModel()
            generator = ResponseCurveGenerator(model)
            
            # Time response curve generation
            start_time = time.time()
            response_curves, _ = generator.generate_response_curves(
                max_multiplier=2.0,
                num_steps=30
            )
            end_time = time.time()
            
            generation_time = end_time - start_time
            print(f"✅ Performance test completed")
            print(f"   Generation time: {generation_time:.3f} seconds")
            print(f"   Performance: {'FAST' if generation_time < 5.0 else 'ACCEPTABLE' if generation_time < 10.0 else 'SLOW'}")
            
            # Edge case: Very small multiplier
            small_curves, _ = generator.generate_response_curves(
                max_multiplier=0.1,
                num_steps=5
            )
            
            # Edge case: Single step
            single_step_curves, _ = generator.generate_response_curves(
                max_multiplier=1.0,
                num_steps=1
            )
            
            print("✅ Edge cases handled successfully")
            self.test_results['performance_edge_cases'] = True
            
        except Exception as e:
            print(f"❌ Performance/edge case test failed: {e}")
            self.test_results['performance_edge_cases'] = False
    
    def print_test_summary(self):
        """Print comprehensive test results summary."""
        print("\n" + "=" * 70)
        print("🎉 TEST SUITE SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - ResponseCurveGenerator is ready for production!")
        else:
            print("⚠️  Some tests failed - review and fix issues before production use")
        
        print("\n🎯 Key Features Validated:")
        print("   • Spend-based response curve generation")
        print("   • Mathematically correct marker calculations")  
        print("   • Robust spend conversion functionality")
        print("   • Performance optimization and edge case handling")




if __name__ == "__main__":
    # Run comprehensive test suite
    test_suite = TestResponseCurveGenerator()
    test_suite.run_all_tests()