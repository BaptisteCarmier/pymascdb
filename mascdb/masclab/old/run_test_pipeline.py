"""
Simple script to run the MASC processing pipeline on test data.

This script demonstrates how to use masc_process() with the sample data.
It can be used for manual testing and verification.

Usage:
    python scripts/run_test_pipeline.py
"""

from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.process import masc_process
from src.core.config import load_from_yaml, print_config_summary
import matplotlib.pyplot as plt


def main():
    """Run the MASC processing pipeline on sample data."""
    
    # Load configuration from YAML (config/config.yaml)
    label_config, process_config = load_from_yaml()
    
    print("=" * 60)
    print("MASC Processing Pipeline - Test Run")
    print("=" * 60)
    print(f"Sample data: {label_config.campaigndir}")
    print(f"Output dir:  {label_config.outdir}")
    print()
    
    # Check if sample data exists
    if not label_config.campaigndir.exists():
        print(f"ERROR: Sample data directory not found at {label_config.campaigndir}")
        print("Please ensure tests/data/ exists with MASC data.")
        return 1
    
    # Ensure output directory exists
    label_config.outdir.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    print_config_summary(label_config, process_config)
    print()
    
    print("Starting processing...")
    print("-" * 60)
    
    try:
        # Run the processing pipeline
        masc_process(label_config, process_config)
        
        print("-" * 60)
        print()
        print("Processing completed successfully!")
        print(f"Results saved to: {label_config.outdir}")
        print(f"Statistics file: {label_config.outdir / 'proc_stats.txt'}")
        
    except Exception as e:
        print(f"\nERROR: Processing failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if process_config.display_figs:
        print("\nFigures displayed. Close windows or press Ctrl+C to exit.")
        plt.show()  # Bloque jusqu'à fermeture des fenêtres
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
