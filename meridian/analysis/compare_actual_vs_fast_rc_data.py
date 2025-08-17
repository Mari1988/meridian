import time
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from meridian.model import model
from meridian.analysis import fast_response_curves
from meridian.analysis import visualizer
from meridian import constants as c

print("✅ All imports successful!")

# Configuration
MODEL_OBJECTS_DIR = "/Users/mariappan.subramanian/Library/CloudStorage/OneDrive-TheTradeDesk/MMM/Media Parameter Analysis/Dev/MMMFeasibility/model_objects"
OUTPUT_CSV_PATH = os.path.join(MODEL_OBJECTS_DIR, "fastrc_accuracy_report.csv")

confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL
selected_times: frozenset[str] | None = None
by_reach: bool = True

def analyze_single_model(pkl_path):
    """
    Analyze a single model file and return channel-level accuracy metrics.
    
    Args:
        pkl_path: Path to the .pkl model file
        
    Returns:
        List of dictionaries with model analysis results
    """
    pkl_filename = os.path.basename(pkl_path)
    
    try:
        print(f"  Loading {pkl_filename}...")
        
        # Load the model
        mmm = model.load_mmm(pkl_path)
        
        # Get model info
        n_geos = mmm.n_geos
        n_times = mmm.n_times
        n_media_channels = mmm.n_media_channels
        channels = list(mmm.input_data.media_channel.values)
        
        print(f"    Model info: {n_geos} geos, {n_times} periods, {n_media_channels} channels: {channels}")
        
        # Get actual response curves
        media_effects = visualizer.MediaEffects(mmm)
        total_num_channels = len(media_effects._meridian.input_data.get_all_channels())
        actual_rc_data = media_effects._transform_response_curve_metrics(
            total_num_channels,
            confidence_level=confidence_level,
            selected_times=selected_times,
            by_reach=by_reach,
        )
        
        # Extract multipliers (use a subset for speed)
        all_multipliers = sorted(actual_rc_data['spend_multiplier'].unique())
        # Use every 5th multiplier to speed up analysis while maintaining accuracy
        sample_multipliers = all_multipliers[::5]  # e.g., 0.0, 0.05, 0.10, ...
        
        # Get fast response curves with sampled multipliers
        fast_rc = fast_response_curves.FastResponseCurves(mmm)
        fast_rc_data = fast_rc.response_curves_data(spend_multipliers=sample_multipliers).to_dataframe().reset_index()
        
        # Perform comparison for each channel
        results = []
        
        for channel in channels:
            # Get data for this channel
            actual_channel = actual_rc_data[actual_rc_data['channel'] == channel].copy()
            fast_channel = fast_rc_data[fast_rc_data['channel'] == channel].copy()
            
            # Find matching multipliers for comparison
            actual_sample = actual_channel[actual_channel['spend_multiplier'].isin(sample_multipliers)]
            
            if len(actual_sample) > 0 and len(fast_channel) > 0:
                # Merge on spend_multiplier for exact comparison
                merged = pd.merge(actual_sample, fast_channel, on='spend_multiplier', suffixes=('_actual', '_fast'))
                
                if len(merged) > 0:
                    # Calculate percentage differences (exclude zero cases)
                    non_zero = merged[merged['mean'] != 0]
                    if len(non_zero) > 0:
                        diff_pct = (non_zero['incremental_outcome'] - non_zero['mean']) / non_zero['mean'] * 100
                        avg_deviation_pct = diff_pct.mean()
                    else:
                        avg_deviation_pct = 0.0
                else:
                    avg_deviation_pct = np.nan
            else:
                avg_deviation_pct = np.nan
            
            results.append({
                'pkl_file_name': pkl_filename,
                'channel': channel,
                'average_deviation_percent_between_actual_fast': avg_deviation_pct,
                'n_geos': n_geos,
                'n_times': n_times,
                'n_comparisons': len(merged) if 'merged' in locals() else 0
            })
        
        print(f"    ✅ Successfully analyzed {len(results)} channels")
        return results
        
    except Exception as e:
        print(f"    ❌ Error analyzing {pkl_filename}: {str(e)}")
        return [{
            'pkl_file_name': pkl_filename,
            'channel': 'ERROR',
            'average_deviation_percent_between_actual_fast': np.nan,
            'n_geos': np.nan,
            'n_times': np.nan,
            'n_comparisons': 0
        }]

def main():
    """Main function to process all models and generate CSV report."""
    
    print("🚀 Starting Multi-Model FastRC vs Actual Analysis")
    print(f"📁 Model directory: {MODEL_OBJECTS_DIR}")
    print(f"📄 Output CSV: {OUTPUT_CSV_PATH}")
    
    # Find all .pkl files
    pkl_files = glob.glob(os.path.join(MODEL_OBJECTS_DIR, "*.pkl"))
    pkl_files.sort()  # Process in alphabetical order
    
    print(f"\n📊 Found {len(pkl_files)} model files to analyze")
    
    # Process each model
    all_results = []
    successful_models = 0
    failed_models = 0
    
    for i, pkl_path in enumerate(pkl_files, 1):
        print(f"\n[{i}/{len(pkl_files)}] Processing model...")
        
        results = analyze_single_model(pkl_path)
        all_results.extend(results)
        
        # Track success/failure
        if any(r['channel'] == 'ERROR' for r in results):
            failed_models += 1
        else:
            successful_models += 1
    
    # Create comprehensive DataFrame
    df = pd.DataFrame(all_results)
    
    # Remove error rows for clean CSV
    df_clean = df[df['channel'] != 'ERROR'].copy()
    
    # Sort by file name and channel for readability
    df_clean = df_clean.sort_values(['pkl_file_name', 'channel'])
    
    # Save main CSV report
    df_clean[['pkl_file_name', 'channel', 'average_deviation_percent_between_actual_fast']].to_csv(
        OUTPUT_CSV_PATH, index=False
    )
    
    # Save detailed CSV with extra info
    detailed_csv_path = OUTPUT_CSV_PATH.replace('.csv', '_detailed.csv')
    df_clean.to_csv(detailed_csv_path, index=False)
    
    print(f"\n📈 ANALYSIS COMPLETE!")
    print(f"✅ Successfully processed: {successful_models} models")
    print(f"❌ Failed to process: {failed_models} models")
    print(f"📊 Total channels analyzed: {len(df_clean)}")
    print(f"💾 Main report saved: {OUTPUT_CSV_PATH}")
    print(f"💾 Detailed report saved: {detailed_csv_path}")
    
    # Summary statistics
    if len(df_clean) > 0:
        print(f"\n📋 SUMMARY STATISTICS:")
        print(f"Average deviation across all channels: {df_clean['average_deviation_percent_between_actual_fast'].mean():.1f}%")
        print(f"Worst performing channel: {df_clean['average_deviation_percent_between_actual_fast'].min():.1f}%")
        print(f"Best performing channel: {df_clean['average_deviation_percent_between_actual_fast'].max():.1f}%")
        
        # Channel-level summary
        print(f"\n📊 BY CHANNEL SUMMARY:")
        channel_summary = df_clean.groupby('channel')['average_deviation_percent_between_actual_fast'].agg(['count', 'mean', 'std']).round(1)
        print(channel_summary)
    
    return df_clean

# Run the analysis
if __name__ == "__main__":
    results_df = main()
