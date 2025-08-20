#!/usr/bin/env python3
"""
Test script to demonstrate the difference in marker placement between 
use_historical=True and use_historical=False scenarios.
"""

import numpy as np
import sys
import os

# Add the meridian package to the path
sys.path.insert(0, '/Users/mariappan.subramanian/Documents/repo/forked/meridian')

from meridian.analysis.response_curve_generator import ResponseCurveGenerator
from meridian.model import model

def test_marker_placement_comparison():
    """Test how markers are placed differently in each scenario."""
    
    # Test with Allergan model
    MODEL_PATH = "/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/model_objects/0_test_working_spend_Allergan.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    print("🧪 Testing marker placement differences between scenarios...")
    print(f"📁 Loading Allergan model: {MODEL_PATH}")
    
    try:
        # Load the model object
        mmm = model.load_mmm(MODEL_PATH)
        generator = ResponseCurveGenerator(mmm)
        
        print(f"✅ Model loaded - Channels: {generator.channel_names}")
        
        # Get the calculated spending points (these should be identical in both cases)
        key_points = generator._calculate_key_spending_points()
        historical_avg_targets = key_points['historical_avg_spend']
        half_saturation_targets = key_points['half_saturation_spend']
        
        print(f"\n📊 Target Values (should be identical in both scenarios):")
        for i, channel in enumerate(generator.channel_names):
            print(f"   {channel}: Hist Avg = ${historical_avg_targets[i]:,.0f}, Half Sat = ${half_saturation_targets[i]:,.0f}")
        
        # Scenario 1: Simulation mode
        print(f"\n🎯 Scenario 1: Simulation mode (use_historical=False)")
        sim_curves, sim_metadata = generator.generate_response_curves(
            max_multiplier=2.0,
            num_steps=50,
            use_historical=False
        )
        
        # Scenario 2: Historical mode  
        print(f"🎯 Scenario 2: Historical mode (use_historical=True)")
        hist_curves, hist_metadata = generator.generate_response_curves(
            use_historical=True
        )
        
        print(f"\n🔍 Data Points Comparison:")
        print(f"   Simulation curves: {len(sim_curves[generator.channel_names[0]]['actual_spend'])} points")
        print(f"   Historical curves: {len(hist_curves[generator.channel_names[0]]['actual_spend'])} points")
        
        # Now simulate the marker placement logic for both scenarios
        print(f"\n📍 Marker Placement Analysis:")
        print(f"{'Channel':<10} {'Target Hist Avg':<15} {'Sim Closest':<15} {'Hist Closest':<15} {'Target Half Sat':<15} {'Sim Closest':<15} {'Hist Closest':<15}")
        print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        
        for i, channel in enumerate(generator.channel_names):
            # Get spend data for both scenarios
            sim_spend = sim_curves[channel]['actual_spend']
            hist_spend = hist_curves[channel]['actual_spend']
            
            # Target values
            hist_avg_target = historical_avg_targets[i]
            half_sat_target = half_saturation_targets[i]
            
            # Find closest points in simulation curve
            sim_hist_idx = np.argmin(np.abs(sim_spend - hist_avg_target))
            sim_half_idx = np.argmin(np.abs(sim_spend - half_sat_target))
            sim_hist_closest = sim_spend[sim_hist_idx]
            sim_half_closest = sim_spend[sim_half_idx]
            
            # Find closest points in historical curve
            hist_hist_idx = np.argmin(np.abs(hist_spend - hist_avg_target))
            hist_half_idx = np.argmin(np.abs(hist_spend - half_sat_target))
            hist_hist_closest = hist_spend[hist_hist_idx]
            hist_half_closest = hist_spend[hist_half_idx]
            
            print(f"{channel:<10} ${hist_avg_target:<14,.0f} ${sim_hist_closest:<14,.0f} ${hist_hist_closest:<14,.0f} ${half_sat_target:<14,.0f} ${sim_half_closest:<14,.0f} ${hist_half_closest:<14,.0f}")
        
        # Show the key insight
        print(f"\n💡 Key Insights:")
        print(f"   1. The TARGET values (Historical Avg & Half Saturation) are identical in both scenarios")
        print(f"   2. The CLOSEST MARKERS will be placed at different actual spend values")
        print(f"   3. This is because the spend arrays have different ranges and distributions")
        print(f"   4. Simulation: Theoretical 0-2x range with linear steps")
        print(f"   5. Historical: Actual historical spend patterns")
        
        # Show spend ranges
        print(f"\n📈 Spend Range Comparison:")
        for channel in generator.channel_names:
            sim_range = [sim_curves[channel]['actual_spend'].min(), sim_curves[channel]['actual_spend'].max()]
            hist_range = [hist_curves[channel]['actual_spend'].min(), hist_curves[channel]['actual_spend'].max()]
            
            print(f"   {channel}:")
            print(f"     Simulation range: ${sim_range[0]:,.0f} - ${sim_range[1]:,.0f}")
            print(f"     Historical range: ${hist_range[0]:,.0f} - ${hist_range[1]:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_marker_placement_comparison()
    sys.exit(0 if success else 1)