#!/usr/bin/env python3
"""
Simple test for the updated plot_transformations function with per-channel plots.
"""

import numpy as np
import sys
import os

# Add the meridian package to the path
sys.path.insert(0, '/Users/mariappan.subramanian/Documents/repo/forked/meridian')

from meridian.analysis.response_curve_generator import ResponseCurveGenerator
from meridian.model import model

def test_simple_plot_transformations():
    """Simple test of the updated plot_transformations function."""
    
    # Test with Allergan model
    MODEL_PATH = "/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/model_objects/0_test_working_spend_Allergan.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    print("🧪 Testing updated plot_transformations function...")
    
    try:
        # Load model
        mmm = model.load_mmm(MODEL_PATH)
        generator = ResponseCurveGenerator(mmm)
        
        print(f"✅ Model loaded - {len(generator.channel_names)} channels: {generator.channel_names}")
        
        # Generate a small historical dataset for faster testing
        scenarios, metadata = generator.generate_geo_media_scenarios(
            use_historical=True,
            selected_times=None  # Use all time periods but could be filtered for speed
        )
        
        print(f"📊 Historical scenarios: {scenarios.shape}")
        
        # Apply transformations
        adstocked, saturated, effects, _ = generator.apply_media_transformations(scenarios, metadata)
        
        # Get KPI effects
        kpi_effects, _ = generator.reverse_transform_media_effects(effects, metadata)
        
        print(f"📈 All transformations ready")
        
        # Test the updated plotting (with separate plots per channel)
        print(f"\n🎨 Creating individual channel plots...")
        generator.plot_transformations(scenarios, adstocked, saturated, kpi_effects, metadata)
        
        print(f"\n✅ Test completed!")
        print(f"🎉 Improvements made:")
        print(f"   - Separate plot for each channel (reduces clutter)")
        print(f"   - Improved X-axis formatting (6-month intervals)")  
        print(f"   - Better color scheme and styling")
        print(f"   - Cleaner legends and titles")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_simple_plot_transformations()
    sys.exit(0 if success else 1)