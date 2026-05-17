"""
Main runner for MASC processing pipeline with parallel processing.

This is the primary entry point for running the MASC pipeline with Dask parallelism.
Configure parameters in src/core/config.py, then execute this script.

Usage:
    python scripts/runner_parallel.py
"""
import sys

from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.process_dask import masc_process
from src.core.config import load_from_yaml, print_config_summary
from src.dataio.clean_outdir import clean_output_dir


def main():
    """Run the MASC processing pipeline with configured parameters and Dask parallelism."""

    print("=" * 70)
    print("MASC Processing Pipeline - Parallel Runner (Dask)")
    print("=" * 70)
    print()
    
    # Load configuration from YAML (config/config.yaml)
    label_config, process_config = load_from_yaml()

    # Clean output directory once at runner start
    if process_config.saveresults:
        clean_output_dir(label_config.outdir)
    
    # Display configuration summary
    print_config_summary(label_config, process_config)
    
    print("-" * 70)
    print("Starting processing with Dask parallelism...")
    print("-" * 70)
    print()
    
    try:
        # Run the main processing pipeline with Dask
        masc_process(label_config, process_config)
        
        print()
        print("-" * 70)
        print("Processing completed successfully!")
        print(f"Results saved to: {label_config.outdir}")
        if process_config.saveresults:
            print(f"Statistics file: {label_config.outdir / 'proc_stats.txt'}")
        print("-" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        print()
        print(f"ERROR: File not found - {e}")
        print("Please check your campaign directory path in config.py")
        return 1
    
    except NotImplementedError as e:
        print()
        print(f"ERROR: Feature not implemented - {e}")
        return 1
    
    except Exception as e:
        print()
        print(f"ERROR: Processing failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
