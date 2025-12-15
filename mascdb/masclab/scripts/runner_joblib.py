"""
Main runner for MASC processing pipeline with joblib parallelism.

This is the primary entry point for running the MASC pipeline with joblib parallelism.
Configure parameters in src/core/config.py, then execute this script.

Usage:
    python scripts/runner_joblib.py
"""
import sys
import logging
from pathlib import Path

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(level=logging.ERROR, force=True)

# Add parent directory to path to import src moduless
sys.path.insert(0, str(Path(__file__).parent.parent))



from src.core.process_joblib import masc_process
from src.core.config import ProcessingConfig, LabelConfig
from src.utils.clean_outdir import clean_output_dir

def main():
    """Run the MASC processing pipeline with configured parameters and joblib parallelism."""
    
    print("=" * 70)
    print("MASC Processing Pipeline - Joblib Parallel Runner")
    print("=" * 70)
    print()
    
    # Use configurations as defined in config.py
    process_config = ProcessingConfig()
    label_config = LabelConfig()

    # Clean output directory once at runner start
    if process_config.saveresults:
        clean_output_dir(label_config.outdir)
    
    # Display configuration summary
    print("Processing Configuration:")
    print(f"  Mode: {'Triplet' if process_config.use_triplet_algo else 'Single-image'}")
    print(f"  Parallel: {process_config.parallel}")
    if process_config.parallel:
        print(f"  Max workers: {process_config.max_workers}")
    print(f"  Generate figures: {process_config.generate_figs}")
    if process_config.generate_figs:
        print(f"    - Display: {process_config.display_figs}")
        print(f"    - Save: {process_config.save_figs}")
    print(f"  Save results: {process_config.saveresults}")
    print()
    
    if process_config.use_triplet_algo:
        print("Triplet Parameters:")
        print(f"  Camera order: {process_config.camera_order}")
        print(f"  Matching tolerance (pixels): {process_config.matching_tol_pix}")
        print(f"  Matching tolerance (percent): {process_config.matching_tol_percent}")
        print()
    
    print("Data Configuration:")
    print(f"  Campaign directory: {label_config.campaigndir}")
    print(f"  Time window: {label_config.starthr_vec} to {label_config.endhr_vec}")
    print(f"  Output directory: {label_config.outdir}")
    print()
    
    print("-" * 70)
    print("Starting processing with joblib parallelism...")
    print("-" * 70)
    print()
    
    try:
        # Run the main processing pipeline with joblib
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
