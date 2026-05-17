"""
Simple script to run the MASC processing pipeline in TRIPLET mode on test data.

This script demonstrates how to use masc_process() with triplet matching.
It processes snowflakes using 3-camera triplets and matches ROIs across views.

Usage:
    python scripts/run_test_triplet.py
"""

from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.process import masc_process
from src.core.config import ProcessingConfig, LabelConfig


def main():
    """Run the MASC processing pipeline on sample data in triplet mode."""
    
    # Define paths
    project_root = Path(__file__).parent.parent
    sample_data = project_root / "tests" / "data"
    output_dir = project_root / "output" / "test_results_triplet"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MASC Processing Pipeline - Triplet Mode Test")
    print("=" * 60)
    print(f"Sample data: {sample_data}")
    print(f"Output dir:  {output_dir}")
    print()
    
    # Check if sample data exists
    if not sample_data.exists():
        print(f"ERROR: Sample data directory not found at {sample_data}")
        print("Please ensure tests/data_sample/ exists with MASC data.")
        return 1
    
    # Configure processing parameters for TRIPLET mode
    process_config = ProcessingConfig() 
    
    # Configure data paths and time window
    label_config = LabelConfig(
        campaigndir=sample_data,
        
        # Time window: June 20, 2015, 9:00-10:00
        starthr_vec=datetime(2015, 6, 20, 9, 0, 0),
        endhr_vec=datetime(2015, 6, 20, 9, 59, 59),
        outdir=output_dir,
    )
    
    print("Configuration:")
    print(f"  - Mode: Triplet matching")
    print(f"  - Parallel: {process_config.parallel}")
    print(f"  - Time range: {label_config.starthr_vec} to {label_config.endhr_vec}")
    print(f"  - Backthresh: {process_config.backthresh}")
    print(f"  - Min area: {process_config.min_area}")
    print(f"  - Matching tolerance (pixels): {process_config.matching_tol_pix}")
    print(f"  - Matching tolerance (percent): {process_config.matching_tol_percent}")
    print(f"  - Camera order: {process_config.camera_order}")
    print()
    
    print("Starting triplet processing...")
    print("-" * 60)
    
    try:
        # Run the processing pipeline
        masc_process(label_config, process_config)
        
        print("-" * 60)
        print()
        print("Triplet processing completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Statistics file: {output_dir / 'proc_stats.txt'}")
        print()
        print("Expected statistics:")
        print("  - N_triplet_full: Number of complete triplets (3+ images)")
        print("  - N_triplet_miss: Number of incomplete triplets (< 3 images)")
        print("  - N_matched: Number of successfully matched triplets")
        print("  - N_no_match: Number of triplets with no ROI match")
        print("  - N_processed_indep: Number of images processed independently")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        return 1
    
    except NotImplementedError as e:
        print(f"\nERROR: Feature not implemented - {e}")
        print("Triplet processing should now be fully implemented!")
        return 1
    
    except Exception as e:
        print(f"\nERROR: Processing failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
