"""
MASC Processing, Classification, and Quicklooks Pipeline

This module orchestrates the complete MASC workflow:
1. Image processing and cropping (identify and crop best snowflake triplet)
2. Hydrometeor classification (classify according to Praz et al., AMT 2017)
3. Generate daily quicklooks of classification output

Translated from MASC_process_classify_quicklooks.m (Christophe Praz 2018)
Author: Baptiste Carmier
Last update: December 2025
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import warnings

from src.core.config import ProcessingConfig, LabelConfig, create_proc_params_file
from src.core.process import masc_process


class QuicklooksOptions:
    """Configuration options for quicklooks generation."""
    
    def __init__(
        self,
        savefigs: bool = True,
        savepath: str = '',
        pixres: float = 33.5 / 1000,  # MASC pixel resolution in mm/pixel
        xi_thresh: float = 9.0,  # quality parameter threshold
        Nmin_interval: int = 30,  # time window in minutes
        Nmin_shift: int = 10,
        Nclasses_masc: int = 6,
        MASC_classes: Optional[List[str]] = None,
        MASC_classes_desired: Optional[List[int]] = None,
        N_MascSamples_min: int = 0,
        use_triplet: bool = False,  # whether to treat each image independently or by triplet
        use_quintuplet: bool = False,
        verbose: int = 0,
        OR_180: bool = True
    ):
        """
        Initialize quicklooks options.
        
        Parameters
        ----------
        savefigs : bool
            Whether to save figures
        savepath : str
            Path to save quicklooks
        pixres : float
            MASC pixel resolution in mm/pixel (default: 33.5/1000)
        xi_thresh : float
            Quality parameter threshold
        Nmin_interval : int
            Time window in minutes
        Nmin_shift : int
            Time shift in minutes
        Nclasses_masc : int
            Number of MASC classes
        MASC_classes : list of str
            MASC class names (default: ['SP','CC','PC','AG','GR','CPC'])
        MASC_classes_desired : list of int
            Desired MASC class indices (default: [1,2,3,4,5,6])
        N_MascSamples_min : int
            Minimum number of MASC samples
        use_triplet : bool
            Whether to treat images by triplet
        use_quintuplet : bool
            Whether to use quintuplets
        verbose : int
            Verbosity level
        OR_180 : bool
            180-degree orientation option
        """
        self.savefigs = savefigs
        self.savepath = Path(savepath) if savepath else Path('')
        self.pixres = pixres
        self.xi_thresh = xi_thresh
        self.Nmin_interval = Nmin_interval
        self.Nmin_shift = Nmin_shift
        self.Nclasses_masc = Nclasses_masc
        self.MASC_classes = MASC_classes if MASC_classes else ['SP', 'CC', 'PC', 'AG', 'GR', 'CPC']
        self.MASC_classes_desired = MASC_classes_desired if MASC_classes_desired else [1, 2, 3, 4, 5, 6]
        self.N_MascSamples_min = N_MascSamples_min
        self.use_triplet = use_triplet
        self.use_quintuplet = use_quintuplet
        self.verbose = verbose
        self.OR_180 = OR_180
        self.savedir = None


class DisplayOptions:
    """Display options for quicklooks visualization."""
    
    def __init__(
        self,
        mode: str = 'Off',
        now: bool = True,
        ht_props: bool = False,
        ht_piechart: bool = False,
        riming: bool = False,
        melting: bool = False,
        n_MASC: bool = False,
        overview_classif: bool = True,
        overview_microstruct: bool = True
    ):
        """
        Initialize display options.
        
        Parameters
        ----------
        mode : str
            Display mode ('Off', 'On')
        now : bool
            Display now
        ht_props : bool
            Display hydrometeor properties
        ht_piechart : bool
            Display hydrometeor pie chart
        riming : bool
            Display riming information
        melting : bool
            Display melting information
        n_MASC : bool
            Display number of MASC samples
        overview_classif : bool
            Display classification overview
        overview_microstruct : bool
            Display microstructure overview
        """
        self.mode = mode
        self.now = now
        self.ht_props = ht_props
        self.ht_piechart = ht_piechart
        self.riming = riming
        self.melting = melting
        self.n_MASC = n_MASC
        self.overview_classif = overview_classif
        self.overview_microstruct = overview_microstruct


class ClassifierPaths:
    """Paths to trained classification models."""
    
    def __init__(
        self,
        class_model: Optional[str] = None,
        riming_model: Optional[str] = None,
        melting_model: Optional[str] = None
    ):
        """
        Initialize classifier paths.
        
        Parameters
        ----------
        class_model : str, optional
            Path to main classification model
        riming_model : str, optional
            Path to riming classification model
        melting_model : str, optional
            Path to melting classification model
        """
        self.class_model = class_model if class_model else 'classification/logit_trained_models/logit_v.1.1_FINAL3_6classes_weighted_scheme.mat'
        self.riming = riming_model if riming_model else 'classification/logit_trained_models/logit_v.1.1_FINAL3_riming_5classes_weighted_scheme.mat'
        self.melting = melting_model if melting_model else 'classification/logit_trained_models/logit_v.1.1_FINAL3_melting_2classes_noweight.mat'


def merge_predictions_for_campaign(
    output_dir: Path,
    start_datestr: str,
    end_datestr: str
) -> Dict[str, Any]:
    """
    Merge classification predictions for a campaign time period.
    
    This function aggregates prediction results from processed MASC data
    within a specified time range.
    
    Parameters
    ----------
    output_dir : Path
        Path to output directory containing processed data
    start_datestr : str
        Start datetime string in format 'yyyymmddHHMMSS'
    end_datestr : str
        End datetime string in format 'yyyymmddHHMMSS'
    
    Returns
    -------
    data : dict
        Merged prediction data
    
    Notes
    -----
    This is a placeholder for the actual implementation.
    The MATLAB version calls merge_predictions_for_campaign.m
    """
    warnings.warn(
        "merge_predictions_for_campaign is not yet implemented. "
        "This requires classification model integration.",
        UserWarning
    )
    return {}


def make_predictions_for_campaign(
    classif_dir: Path,
    classif_starthr_vec: tuple,
    classif_endhr_vec: tuple,
    classifiers: ClassifierPaths,
    parallel_proc: bool = True
) -> None:
    """
    Apply classification models to processed MASC images.
    
    Parameters
    ----------
    classif_dir : Path
        Directory containing processed MASC data
    classif_starthr_vec : tuple
        Start time vector (year, month, day, hour, min, sec)
    classif_endhr_vec : tuple
        End time vector (year, month, day, hour, min, sec)
    classifiers : ClassifierPaths
        Paths to classification models
    parallel_proc : bool
        Whether to use parallel processing
    
    Notes
    -----
    This is a placeholder for the actual implementation.
    Requires trained classification models and feature extraction.
    """
    warnings.warn(
        "make_predictions_for_campaign is not yet implemented. "
        "This requires classification model integration.",
        UserWarning
    )
    pass


def make_masc_time_series(
    data: Dict[str, Any],
    start_vec: tuple,
    end_vec: tuple,
    options: QuicklooksOptions,
    disp: DisplayOptions
) -> None:
    """
    Generate MASC time series plots and quicklooks.
    
    Parameters
    ----------
    data : dict
        Merged prediction data from merge_predictions_for_campaign
    start_vec : tuple
        Start time vector (year, month, day, hour, min, sec)
    end_vec : tuple
        End time vector (year, month, day, hour, min, sec)
    options : QuicklooksOptions
        Quicklooks generation options
    disp : DisplayOptions
        Display options
    
    Notes
    -----
    This is a placeholder for the actual implementation.
    Generates daily quicklooks of classification output.
    """
    warnings.warn(
        "make_masc_time_series is not yet implemented. "
        "This requires visualization and plotting tools.",
        UserWarning
    )
    pass


def masc_process_classify_quicklooks(
    # Part 1: Image processing parameters
    masc_process_enable: bool = True,
    masc_regen_from_start: bool = True,
    process_params_file: Optional[str] = None,
    campaigndir: str = '',
    output_dir: str = '',
    starthr_vec: tuple = (2021, 1, 1, 0, 0, 0),
    endhr_vec: tuple = (2021, 1, 4, 0, 0, 0),
    # Part 2: Classification parameters
    classif_process: bool = True,
    classif_regen_from_start: bool = True,
    classif_parallel_proc: bool = True,
    classifiers: Optional[ClassifierPaths] = None,
    # Part 3: Quicklooks parameters
    gen_quicklooks: bool = True,
    regen_quicklooks: bool = True,
    quicklooks_options: Optional[QuicklooksOptions] = None,
    display_options: Optional[DisplayOptions] = None,
    # Processing configuration
    process_config: Optional[ProcessingConfig] = None
) -> None:
    """
    Main pipeline for MASC processing, classification, and quicklooks generation.
    
    This function orchestrates three main parts:
    1. Image processing and cropping (identify best snowflake triplet)
    2. Hydrometeor classification (classify according to Praz et al., AMT 2017)
    3. Generate daily quicklooks of classification output
    
    Parameters
    ----------
    masc_process_enable : bool
        Whether to run MASC image processing (part 1)
    masc_regen_from_start : bool
        Whether to regenerate all images from scratch or continue from last processed
    process_params_file : str, optional
        Path to file containing processing parameters
    campaigndir : str
        Path to directory where raw MASC data is located
    output_dir : str
        Path to directory where processed MASC data will be stored
    starthr_vec : tuple
        Start time (year, month, day, hour, min, sec)
    endhr_vec : tuple
        End time (year, month, day, hour, min, sec)
    classif_process : bool
        Whether to run classification (part 2)
    classif_regen_from_start : bool
        Whether to regenerate classifications from scratch
    classif_parallel_proc : bool
        Whether to use parallel processing for classification
    classifiers : ClassifierPaths, optional
        Paths to trained classification models
    gen_quicklooks : bool
        Whether to generate quicklooks (part 3)
    regen_quicklooks : bool
        Whether to regenerate all quicklooks
    quicklooks_options : QuicklooksOptions, optional
        Options for quicklooks generation
    display_options : DisplayOptions, optional
        Display options for visualization
    process_config : ProcessingConfig, optional
        Processing configuration (if None, uses defaults)
    
    Examples
    --------
    >>> from src.processing.quicklooks import masc_process_classify_quicklooks
    >>> masc_process_classify_quicklooks(
    ...     campaigndir='/path/to/raw/data',
    ...     output_dir='/path/to/processed/data',
    ...     starthr_vec=(2021, 1, 1, 0, 0, 0),
    ...     endhr_vec=(2021, 1, 4, 0, 0, 0)
    ... )
    
    Notes
    -----
    Translated from MASC_process_classify_quicklooks.m (Christophe Praz, 2018)
    """
    try:
        # Store initial start time
        starthr_vec_ini = starthr_vec
        
        # ====================================================================
        # PART 1: IMAGE PROCESSING AND CROPPING
        # ====================================================================
        if masc_process_enable:
            print("\n" + "="*70)
            print("PART I: MASC Image Processing")
            print("="*70)
            
            # Create label configuration
            label_config = LabelConfig(
                campaigndir=Path(campaigndir),
                outdir=Path(output_dir),
                starthr_vec=starthr_vec,
                endhr_vec=endhr_vec
            )
            
            # If not regenerating from start, find last processed image
            if not masc_regen_from_start:
                output_path = Path(output_dir)
                if output_path.exists():
                    # Find all PNG files recursively
                    proc_pic_list = list(output_path.rglob('*.png'))
                    if proc_pic_list:
                        # Sort by filename (which contains timestamp)
                        proc_pic_list.sort()
                        proc_last_pic = proc_pic_list[-1].name
                        
                        # Extract timestamp from filename (format: yyyy.mm.dd_HH.MM.SS)
                        try:
                            timestamp_str = proc_last_pic[:19]  # First 19 characters
                            # Parse timestamp
                            dt = datetime.strptime(timestamp_str, '%Y.%m.%d_%H.%M.%S')
                            # Update start time to continue from last image
                            label_config.starthr_vec = (
                                dt.year, dt.month, dt.day,
                                dt.hour, dt.minute, dt.second
                            )
                            print(f"Continuing from last processed image: {proc_last_pic}")
                        except ValueError:
                            print(f"Warning: Could not parse timestamp from {proc_last_pic}")
            
            # Display processing time range
            start_dt = datetime(*label_config.starthr_vec)
            end_dt = datetime(*label_config.endhr_vec)
            print(f"Processing MASC images from {start_dt.strftime('%Y.%m.%d %H:%M')} "
                  f"to {end_dt.strftime('%Y.%m.%d %H:%M')}")
            
            # Create processing configuration if not provided
            if process_config is None:
                process_config = ProcessingConfig()
            
            # Run MASC processing
            masc_process(label_config, process_config)
            
            print("\nPart I completed successfully.")
        
        # ====================================================================
        # PART 2: HYDROMETEOR CLASSIFICATION
        # ====================================================================
        if classif_process:
            print("\n" + "="*70)
            print("PART II: Hydrometeor Classification")
            print("="*70)
            
            classif_dir = Path(output_dir)
            
            # Set classification time range
            if classif_regen_from_start:
                classif_starthr_vec = (2010, 1, 1, 0, 0, 0)
            else:
                classif_starthr_vec = starthr_vec
            
            classif_endhr_vec = endhr_vec
            
            # Initialize classifiers if not provided
            if classifiers is None:
                classifiers = ClassifierPaths()
            
            # Display classification info
            start_dt = datetime(*classif_starthr_vec)
            end_dt = datetime(*classif_endhr_vec)
            print(f"Classifying images from {start_dt.strftime('%Y.%m.%d %H:%M')} "
                  f"to {end_dt.strftime('%Y.%m.%d %H:%M')}")
            print(f"Parallel processing: {classif_parallel_proc}")
            
            # Run classification
            make_predictions_for_campaign(
                classif_dir,
                classif_starthr_vec,
                classif_endhr_vec,
                classifiers,
                classif_parallel_proc
            )
            
            print("\nPart II completed successfully.")
        
        # ====================================================================
        # PART 3: QUICKLOOKS GENERATION
        # ====================================================================
        if gen_quicklooks:
            print("\n" + "="*70)
            print("PART III: Quicklooks Generation")
            print("="*70)
            
            # Initialize options if not provided
            if quicklooks_options is None:
                quicklooks_options = QuicklooksOptions(
                    savefigs=True,
                    savepath=output_dir + '/Quicklooks'
                )
            
            if display_options is None:
                display_options = DisplayOptions()
            
            # Determine day range for quicklooks
            day_stop = datetime(*endhr_vec)
            day_stop_midnight = datetime(day_stop.year, day_stop.month, day_stop.day)
            
            if regen_quicklooks:
                day_ini = datetime(starthr_vec_ini[0], starthr_vec_ini[1], starthr_vec_ini[2])
            else:
                # Start from previous day if not regenerating all
                day_ini = day_stop_midnight - timedelta(days=1)
            
            # Generate list of days to process
            current_day = day_ini
            days_to_process = []
            while current_day <= day_stop:
                days_to_process.append(current_day)
                current_day += timedelta(days=1)
            
            print(f"Generating quicklooks for {len(days_to_process)} day(s)")
            
            # Process each day
            for i, day_current in enumerate(days_to_process, 1):
                day_next = day_current + timedelta(days=1)
                
                print(f"\n[{i}/{len(days_to_process)}] Processing {day_current.strftime('%Y-%m-%d')}")
                
                # Merge predictions for this day
                start_datestr = day_current.strftime('%Y%m%d%H%M%S')
                end_datestr = day_next.strftime('%Y%m%d%H%M%S')
                
                data = merge_predictions_for_campaign(
                    Path(output_dir),
                    start_datestr,
                    end_datestr
                )
                
                # Set save directory if saving figures
                if quicklooks_options.savefigs:
                    month_str = day_current.strftime('%Y_%m')
                    quicklooks_options.savedir = quicklooks_options.savepath / month_str
                    quicklooks_options.savedir.mkdir(parents=True, exist_ok=True)
                
                # Generate time series for this day
                start_vec = (day_current.year, day_current.month, day_current.day, 0, 0, 0)
                end_vec = (day_next.year, day_next.month, day_next.day, 0, 0, 0)
                
                make_masc_time_series(
                    data,
                    start_vec,
                    end_vec,
                    quicklooks_options,
                    display_options
                )
            
            print("\nPart III completed successfully.")
        
        print("\n" + "="*70)
        print("Pipeline completed successfully!")
        print("="*70)
    
    except Exception as err:
        import traceback
        print(f"\nError occurred: {err}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    """Example usage of the quicklooks pipeline."""
    
    # Example configuration matching MATLAB script
    masc_process_classify_quicklooks(
        # Part 1: Processing
        masc_process_enable=True,
        masc_regen_from_start=True,
        campaigndir='/ltedata/Eriswil_2024/MASC/Raw_data',
        output_dir='/ltedata/Eriswil_2024/MASC/Proc_data',
        starthr_vec=(2021, 1, 1, 0, 0, 0),
        endhr_vec=(2021, 1, 4, 0, 0, 0),
        
        # Part 2: Classification
        classif_process=True,
        classif_regen_from_start=True,
        classif_parallel_proc=True,
        
        # Part 3: Quicklooks
        gen_quicklooks=True,
        regen_quicklooks=True,
        quicklooks_options=QuicklooksOptions(
            savefigs=True,
            savepath='/ltedata/Eriswil_2024/MASC/Quicklooks',
            pixres=33.5/1000,
            xi_thresh=9.0,
            Nmin_interval=30,
            Nmin_shift=10,
            Nclasses_masc=6,
            MASC_classes=['SP', 'CC', 'PC', 'AG', 'GR', 'CPC'],
            MASC_classes_desired=[1, 2, 3, 4, 5, 6],
            N_MascSamples_min=0,
            use_triplet=False,
            use_quintuplet=False,
            verbose=0,
            OR_180=True
        )
    )
