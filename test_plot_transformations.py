#!/usr/bin/env python3
"""
Test script to verify the plot_transformations function works correctly.
"""

import numpy as np
import sys
import os

# Add the meridian package to the path
sys.path.insert(0, '/Users/mariappan.subramanian/Documents/repo/forked/meridian')

from meridian.analysis.response_curve_generator import ResponseCurveGenerator
from meridian.model import model

def test_plot_transformations():
    """Test the plot_transformations function with historical data."""
    
    # Test with Allergan model
    MODEL_PATH = "/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/model_objects/0_test_working_spend_Allergan.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    print("🧪 Testing plot_transformations function...")
    print(f"📁 Loading Allergan model: {MODEL_PATH}")
    
    try:
        # Load the model object
        mmm = model.load_mmm(MODEL_PATH)
        generator = ResponseCurveGenerator(mmm)
        
        print(f"✅ Model loaded - Channels: {generator.channel_names}")
        
        # Step 1: Generate historical scenarios (required for plot_transformations)
        print(f"\n🎯 Step 1: Generate historical media scenarios")
        media_scenarios, scenario_metadata = generator.generate_geo_media_scenarios(
            use_historical=True
        )
        
        print(f"   ✅ Historical scenarios generated: {media_scenarios.shape}")
        print(f"   - Use historical: {scenario_metadata['use_historical']}")
        print(f"   - Time periods: {scenario_metadata['num_steps']}")
        
        # Step 2: Apply transformations to get all intermediate steps
        print(f"\n🎯 Step 2: Apply transformations")
        adstocked_effects, saturated_effects, media_effects, transform_metadata = generator.apply_media_transformations(
            media_scenarios, scenario_metadata
        )
        
        print(f"   ✅ Transformations completed:")
        print(f"   - Adstocked effects: {adstocked_effects.shape}")
        print(f"   - Saturated effects: {saturated_effects.shape}")
        print(f"   - Media effects: {media_effects.shape}")
        
        # Step 3: Get actual KPI effects (reverse transformation)
        print(f"\n🎯 Step 3: Get actual KPI effects")
        actual_kpi_effects, kpi_metadata = generator.reverse_transform_media_effects(
            media_effects, scenario_metadata
        )
        
        print(f"   ✅ KPI effects generated: {actual_kpi_effects.shape}")
        print(f"   - Value range: [{actual_kpi_effects.min():.2f}, {actual_kpi_effects.max():.2f}]")
        
        # Step 4: Test the plot_transformations function
        print(f"\n🎯 Step 4: Create transformation plots")
        
        # Test input validation first (should fail with simulated data)
        print(f"   Testing input validation...")
        
        # Generate simulated data for validation test
        sim_scenarios, sim_metadata = generator.generate_geo_media_scenarios(use_historical=False, num_steps=10)
        sim_adstocked, sim_saturated, sim_effects, _ = generator.apply_media_transformations(sim_scenarios, sim_metadata)
        sim_kpi, _ = generator.reverse_transform_media_effects(sim_effects, sim_metadata)
        
        # This should raise an error
        try:
            generator.plot_transformations(sim_scenarios, sim_adstocked, sim_saturated, sim_kpi, sim_metadata)
            print(f"   ❌ Validation failed - should have raised error for use_historical=False")
            return False
        except ValueError as e:
            print(f"   ✅ Validation works - correctly rejected simulated data: {str(e)}")
        
        # Now test with historical data (should work)
        print(f"\n   Creating historical transformation plots...")
        generator.plot_transformations(
            media_scenarios, 
            adstocked_effects, 
            saturated_effects, 
            actual_kpi_effects, 
            scenario_metadata
        )
        
        print(f"\n✅ plot_transformations test completed successfully!")
        
        # Step 5: Test with subset of data (time filtering)
        print(f"\n🎯 Step 5: Test with time filtering")
        
        # Get a subset of time periods for testing
        time_values = mmm.input_data.time.values
        selected_times = [str(t) for t in time_values[:20]]  # First 20 time periods
        
        filtered_scenarios, filtered_metadata = generator.generate_geo_media_scenarios(
            use_historical=True,
            selected_times=selected_times
        )
        
        filtered_adstocked, filtered_saturated, filtered_effects, _ = generator.apply_media_transformations(
            filtered_scenarios, filtered_metadata
        )
        
        filtered_kpi, _ = generator.reverse_transform_media_effects(filtered_effects, filtered_metadata)
        
        print(f"   Creating filtered transformation plots (first 20 time periods)...")
        generator.plot_transformations(
            filtered_scenarios,
            filtered_adstocked, 
            filtered_saturated, 
            filtered_kpi, 
            filtered_metadata
        )
        
        print(f"\n✅ Time filtering test completed successfully!")
        
        # Usage example
        print(f"\n📝 Usage Example:")
        print(f"   # Generate historical scenarios")
        print(f"   scenarios, metadata = generator.generate_geo_media_scenarios(use_historical=True)")
        print(f"   ")
        print(f"   # Apply transformations")
        print(f"   adstocked, saturated, effects, _ = generator.apply_media_transformations(scenarios, metadata)")
        print(f"   ")  
        print(f"   # Get KPI effects")
        print(f"   kpi_effects, _ = generator.reverse_transform_media_effects(effects, metadata)")
        print(f"   ")
        print(f"   # Plot transformations over time")
        print(f"   generator.plot_transformations(scenarios, adstocked, saturated, kpi_effects, metadata)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plot_transformations()
    sys.exit(0 if success else 1)