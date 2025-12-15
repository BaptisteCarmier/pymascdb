"""
Configuration management for MASC processing.

This module handles all configuration parameters for the MASC processing pipeline,
including processing parameters, camera settings, and analysis thresholds.

Translated from create_proc_params_file.m (Christophe Praz 2015) and adapted for Python.
Last update: November 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

@dataclass
class ProcessingConfig:
    """Processing configuration parameters."""
    
    # Processing control
    parallel: bool = True 
    use_triplet_algo : bool = False # Enable TRIPLET processing mode
    max_workers: int = 1  # Maximum number of parallel workers (to avoid memory issues)

    generate_figs: bool = True  # Always set to True is you want to generate matplotlib figures for visualization
    display_figs: bool = False  # Display figures interactively (if generate_figs=True)
    save_figs: bool = True  # Save generated figures to disk (if generate_figs=True)

    saveresults: bool = True  # Save processing results and statistics
    save_format: str = "mat"  # ROI data format: 'joblib' (recommended), 'pkl', or 'mat'
    
    flakebrighten: bool = False  # Apply brightening to dim flakes

    # Thresholds
    backthresh: int =15 # Vient de process_param.m de 2015_APRES3/masclab
    sizemin: int = 30 # Vient de process_param.m de 2015_APRES3/masclab
    min_area: int = 100 #################### VALEUR A VERIFIER/TROUVER ####################
    minbright: float = 0.15 # Vient de process_param.m de 2015_APRES3/masclab
    max_intensthresh: float = 0.25 # Vient de process_param.m de 2015_APRES3/masclab
    min_hole_area: int = 10 # Vient de process_param.m de 2015_APRES3/masclab
    
    # Image margins to discard [top, bottom, left, right]
    discardmat: Tuple[int, int, int, int] = (400, 548, 320, 320) # Vient de process_param.m de 2015_APRES3/masclab
    
    # Triplet processing
    matching_tol_pix: int = 10
    matching_tol_percent: int = 20
    camera_order: List[int] = field(default_factory=lambda: [0, 1, 2])  # Expected camera IDs for triplets


@dataclass
class LabelConfig:
    """Labeling and data organization parameters."""
    # Data paths (required fields first)
    campaigndir: Path = Path(r"C:\Users\bapti\Desktop\Archive_Baptiste\LTE\masclab_codeReview\BC_py_masclab\tests\data")
    starthr_vec: datetime = datetime(2015, 6, 20, 9, 0, 0)  # Time window start
    endhr_vec: datetime = datetime(2015, 6, 20, 9, 0, 59)  # Time window end
    
    # Optional fields (with defaults must come last)
    outdir: Path = field(default_factory=lambda: Path("./PROCESSED"))  # Output directory for results



def create_proc_params_file(
    pathname: Path,
    label: LabelConfig,
    process: ProcessingConfig
) -> None: 
    """
    Create a processing parameters file.
    
    Args:
        pathname: Directory where to save the parameters file
        label: Labeling configuration
        process: Processing configuration

    Returns:
        None
        
    Equivalent to MATLAB create_proc_params_file.m
    """
    filename = Path(pathname) / 'proc_params.txt'
    filename.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists


    with open(filename, 'w') as f:
        f.write('Processing parameters associated with data in this folder\n\n\n')
        
        # Write timestamps and paths
        f.write(f'creation date          : {datetime.now().strftime("%Y-%m-%d")}\n')
        f.write(f'data processed         : {label.campaigndir}\n')
        f.write(f'starting time          : {label.starthr_vec.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'ending time            : {label.endhr_vec.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'processed in parallel  : {int(process.parallel)}\n\n')
        
        # Write processing parameters
        f.write(f'backtresh limit        : {process.backthresh}\n')
        f.write(f'size min               : {process.sizemin}\n')
        f.write(f'min area               : {process.min_area}\n')
        f.write(f'min brightness         : {process.minbright:.2f}\n')
        f.write(f'max intens thresh      : {process.max_intensthresh:.2f}\n')
        f.write(f'min hole area          : {process.min_hole_area}\n')
        
        # Write discard margins
        margins = process.discardmat
        f.write(f'discardmat [t,b,l,r]   : [{margins[0]} {margins[1]} {margins[2]} {margins[3]}]\n\n')
        
        # Write triplet parameters
        f.write(f'use triplet algo       : {int(process.use_triplet_algo)}\n')
        if process.use_triplet_algo:
            f.write(f'matching tol. pix.     : {process.matching_tol_pix}\n')
            f.write(f'matching tol. percent. : {process.matching_tol_percent}\n')
            f.write(f'camera order           : {process.camera_order}\n')


            

############# A SUPPRIMER !!! #############
def load_default_config() -> Tuple[LabelConfig, ProcessingConfig]:
    """
    Create default configuration objects.
    
    Returns:
        Tuple of (label_config, process_config) with default values
    """
    label = LabelConfig(
        campaigndir=Path("./data/campaign"),
        starthr_vec=datetime.now(),
        endhr_vec=datetime.now(),
        outdir=Path("./data/results")
    )
    
    process = ProcessingConfig()  # Uses default values from dataclass
    
    return label, process