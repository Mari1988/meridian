#!/usr/bin/env python3
"""
Test script to compare Historical Avg and Half Saturation values between 
use_historical=True and use_historical=False scenarios.
"""

import numpy as np
import sys
import os

# Add the meridian package to the path
sys.path.insert(0, '/Users/mariappan.subramanian/Documents/repo/forked/meridian')

from meridian.analysis.response_curve_generator import ResponseCurveGenerator
from meridian.model import model

def test_historical_vs_simulated_comparison():
    """Test if Historical Avg and Half Saturation values match between modes."""
    
    # Test with Allergan model as requested
    MODEL_PATH = "/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/model_objects/0_test_working_spend_Allergan.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH to point to a valid Allergan model file.")
        return False
    
    print("🧪 Testing Historical Avg and Half Saturation consistency...")
    print(f"📁 Loading Allergan model: {MODEL_PATH}")
    
    try:
        # 1. Load the model object
        mmm = model.load_mmm(MODEL_PATH)
        print(f"✅ Allergan model loaded successfully")
        
        # Initialize response curve generator
        generator = ResponseCurveGenerator(mmm)
        print(f"✅ ResponseCurveGenerator initialized")
        print(f"   - Historical data shape: {generator.historical_scaled_media.shape}")
        print(f"   - Channels: {generator.channel_names}")
        
        # 2. Test scenario 1: use_historical=False (simulation mode)
        print(f"\n🎯 Scenario 1: Simulation mode (use_historical=False)")
        
        simulated_curves, simulated_metadata = generator.generate_response_curves(
            max_multiplier=2.0,
            num_steps=50,
            aggregation_level="national",
            use_historical=False
        )
        
        # Extract the calculated key spending points used in plotting
        sim_key_points = generator._calculate_key_spending_points()
        sim_historical_avg = sim_key_points['historical_avg_spend']
        sim_half_saturation = sim_key_points['half_saturation_spend']
        
        print(f"   ✅ Simulation response curves generated")
        print(f"   - Data points in simulation curves: {len(simulated_curves[generator.channel_names[0]]['actual_spend'])}")
        print(f"   - Calculated Historical Avg Spend: {sim_historical_avg}")
        print(f"   - Calculated Half Saturation Spend: {sim_half_saturation}")
        
        # 3. Test scenario 2: use_historical=True (historical mode)
        print(f"\n🎯 Scenario 2: Historical mode (use_historical=True)")
        
        historical_curves, historical_metadata = generator.generate_response_curves(
            max_multiplier=2.0,  # Should be ignored
            num_steps=50,        # Should be ignored
            aggregation_level="national",
            use_historical=True
        )
        
        # Extract the calculated key spending points used in plotting (should be same)
        hist_key_points = generator._calculate_key_spending_points()
        hist_historical_avg = hist_key_points['historical_avg_spend']
        hist_half_saturation = hist_key_points['half_saturation_spend']
        
        print(f"   ✅ Historical response curves generated")
        print(f"   - Data points in historical curves: {len(historical_curves[generator.channel_names[0]]['actual_spend'])}")
        print(f"   - Calculated Historical Avg Spend: {hist_historical_avg}")
        print(f"   - Calculated Half Saturation Spend: {hist_half_saturation}")
        
        print(f"\n🔍 Key Insight: The calculated values should be identical, but the PLOTTING will differ!")
        print(f"   - Simulation has {len(simulated_curves[generator.channel_names[0]]['actual_spend'])} points")
        print(f"   - Historical has {len(historical_curves[generator.channel_names[0]]['actual_spend'])} points")
        print(f"   - The markers will be placed at different positions on each curve!")
        
        # 4. Compare the values
        print(f"\n🔍 Comparison Analysis:")
        print(f"{'Channel':<15} {'Sim Hist Avg':<15} {'Hist Hist Avg':<15} {'Match':<8} {'Sim Half Sat':<15} {'Hist Half Sat':<15} {'Match':<8}")
        print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*8} {'-'*15} {'-'*15} {'-'*8}")
        
        all_historical_avg_match = True
        all_half_saturation_match = True
        
        for i, channel in enumerate(generator.channel_names):
            sim_hist_avg_val = sim_historical_avg[i]
            hist_hist_avg_val = hist_historical_avg[i]
            sim_half_sat_val = sim_half_saturation[i]
            hist_half_sat_val = hist_half_saturation[i]
            
            # Check if values match (using small tolerance for floating point comparison)
            hist_avg_match = np.isclose(sim_hist_avg_val, hist_hist_avg_val, rtol=1e-10)
            half_sat_match = np.isclose(sim_half_sat_val, hist_half_sat_val, rtol=1e-10)
            
            if not hist_avg_match:
                all_historical_avg_match = False
            if not half_sat_match:
                all_half_saturation_match = False
            
            print(f"{channel:<15} {sim_hist_avg_val:<15.2f} {hist_hist_avg_val:<15.2f} {'✅' if hist_avg_match else '❌':<8} {sim_half_sat_val:<15.2f} {hist_half_sat_val:<15.2f} {'✅' if half_sat_match else '❌':<8}")
        
        print(f"\n📊 Summary:")
        if all_historical_avg_match and all_half_saturation_match:
            print(f"✅ SUCCESS: All Historical Avg and Half Saturation values match perfectly between scenarios!")
            print(f"   This is expected behavior - these values should be consistent regardless of use_historical parameter.")
        else:
            print(f"❌ ISSUE DETECTED:")
            if not all_historical_avg_match:
                print(f"   - Historical Avg values differ between scenarios")
            if not all_half_saturation_match:
                print(f"   - Half Saturation values differ between scenarios")
            print(f"\n🤔 Analysis of why this might happen:")
            print(f"   1. The _calculate_key_spending_points() method should use the same underlying historical data")
            print(f"   2. It should NOT be affected by the use_historical parameter")
            print(f"   3. If values differ, there might be a bug in the implementation")
            
            # Let's investigate further
            print(f"\n🔍 Detailed Investigation:")
            
            # Check if the underlying model state changed between calls
            print(f"   - Generator object ID: {id(generator)}")
            print(f"   - Model object ID: {id(generator.model)}")
            print(f"   - Historical scaled media shape: {generator.historical_scaled_media.shape}")
            
            # Check the metadata for calculation method
            print(f"   - Sim key points metadata: {sim_key_points['metadata']}")
            print(f"   - Hist key points metadata: {hist_key_points['metadata']}")
        
        # 5. Additional verification: Check if the calculation is truly independent
        print(f"\n🔬 Independence Test:")
        
        # Call _calculate_key_spending_points multiple times to verify consistency
        test1_points = generator._calculate_key_spending_points()
        test2_points = generator._calculate_key_spending_points()
        
        test1_avg = test1_points['historical_avg_spend']
        test2_avg = test2_points['historical_avg_spend']
        
        consistency_check = np.allclose(test1_avg, test2_avg, rtol=1e-10)
        print(f"   Multiple calls to _calculate_key_spending_points() consistent: {'✅' if consistency_check else '❌'}")
        
        if not consistency_check:
            print(f"   ⚠️  This suggests the calculation method itself has inconsistency issues!")
        
        return all_historical_avg_match and all_half_saturation_match
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_historical_vs_simulated_comparison()
    sys.exit(0 if success else 1)