"""
Core batch processing function for MASC data.

Process and crop around snowflakes in pictures located in campaigndir.
Outputs are cropped images and structures containing process parameters
and snowflake features.

Translated from MASC_process.m (Christophe Praz, EPFL)
Author: Christophe Praz (christophe.praz@epfl.ch)
Last update: October 2017 (MATLAB), November 2025 (Python)
"""

################# ATTENTION PAS A JOUR NOTAMMENT PROCESSING SINGLE IMAGE #################
# Version la plus à jour dans process_joblib.py #

import time
import logging

from pathlib import Path

from src.core.config import ProcessingConfig, LabelConfig
from src.dataio.uploaddirs import uploaddirs
from src.dataio.upload import upload
from src.processing.single import process_single_image
from src.processing.triplet import process_triplet_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class ProcessingStats:
    """Statistics accumulated during batch processing."""
    def __init__(self, use_triplet_algo):
        self.N_tot: int = 0

        if use_triplet_algo:
            # Triplet-specific stats
            self.N_triplet_full: int = 0
            self.N_triplet_miss: int = 0
            self.N_matched: int = 0
            self.N_no_match: int = 0
            self.N_processed_indep: int = 0

        else:
            self.N_good: int = 0
            self.N_bad: int = 0
            self.N_blurry: int = 0


class Timings:
    """Timing statistics for various processing steps."""
    def __init__(self, use_triplet_algo):

        self.tt_uploading: float = 0.0
        self.tt_loading: float = 0.0
        self.tt_clutter: float = 0.0
        self.tt_edging: float = 0.0
        self.tt_roiying: float = 0.0
        self.tt_feature: float = 0.0
        self.tt_plotting: float = 0.0
        self.tt_saving: float = 0.0
        self.tt_program: float = 0.0

        if use_triplet_algo:
            self.tt_matching: float = 0.0


def masc_process(label: LabelConfig, process: ProcessingConfig) -> None:
    """
    Main batch processing function for MASC data.
    
    Process and crop around snowflakes in the pictures located in campaigndir.
    
    Args:
        label: Labeling configuration (campaign paths, date ranges, output dirs)
        process: Processing configuration (thresholds, flags, algorithms)
    
    Returns:
        None (writes results to disk and prints statistics)
    
    MATLAB equivalent: MASC_process(label, cam, process)
    
    Notes:
        - Camera parameters (cam) are integrated into ProcessingConfig in Python
        - Parallel processing controlled via process.parallel flag
        - Triplet matching controlled via process.use_triplet_algo flag
    """
    
    # Find all relevant snowflake directories
    try:
        dir_list = uploaddirs(label.campaigndir, label.starthr_vec, label.endhr_vec)
    except NotImplementedError:
        logger.error("You need to implement directory discovery based on campaign structure")
        return
    
    # Initialize statistics and timings
    stats = ProcessingStats(process.use_triplet_algo)
    timings = Timings(process.use_triplet_algo)
    
    # Start program timer
    t_startprogram = time.time()
    
    # Upload all snowflake pictures, dir-by-dir
    for i, current_dir in enumerate(dir_list):
        
        t_uploading = time.time()
        
        # Retrieve snowflakes info
        pic_list = upload(current_dir)
        # id_unique is now computed inside upload()
        
        ## Il manquerait pas outdirs ici ? ##

        # Number of new flakes added
        n_id = len(pic_list.id_unique)
        stats.N_tot += n_id
        
        timings.tt_uploading += time.time() - t_uploading
        
        # ===== Main processing loop =====
        
        if not process.use_triplet_algo:
            # ===== Single-image mode (no triplet matching) =====
            
            # Main loop on all the pictures located in current_dir
            for j in range(len(pic_list.id)):
                
                if pic_list.files:
                    logger.info("Processing %s...", pic_list.files[j])
                
                # Construct pic_info dict for this image
                pic_info = {
                    "filename": pic_list.files[j] if pic_list.files else f"image_{j}.png",
                    "cam": pic_list.cam[j],
                    "id": pic_list.id[j],
                    "fallspeed": pic_list.fallspeed[j] if j < len(pic_list.fallspeed) else None,
                    "time_num": pic_list.time_num[j] if j < len(pic_list.time_num) else None,
                }
                
                img_path = Path(current_dir) / pic_list.files[j]
                
                try:
                    roi, flag, timing = process_single_image(img_path, pic_info, label, process, verbose=False)
                except Exception as exc:
                    logger.error("Unknown error while processing image %s: %s", pic_list.files[j], exc)
                    flag = -1
                    timing = {}
                
                # Accumulate flags
                # flags: -1=error, 0=no ROI, 1=BAD, 2=GOOD
                if flag == 0:
                    stats.N_bad += 1
                elif flag == 1:
                    stats.N_blurry += 1
                elif flag == 2:
                    stats.N_good += 1
                # flag == -1 (load error) is not counted in statistics
                
                # Accumulate timings
                if timing:
                    timings.tt_loading += timing.get("loading", 0.0)
                    timings.tt_clutter += timing.get("clutter", 0.0)
                    timings.tt_edging += timing.get("edging", 0.0)
                    timings.tt_roiying += timing.get("roiying", 0.0)
                    timings.tt_feature += timing.get("feature", 0.0)
                    timings.tt_plotting += timing.get("plotting", 0.0)
                    timings.tt_saving += timing.get("saving", 0.0)
        
        else:
            # ===== Triplet matching mode =====
            
            logger.info("Processing %d unique snowflake IDs with triplet matching...", len(pic_list.id_unique))
            
            # Loop over unique picture IDs (triplets or incomplete triplets)
            for j, unique_id in enumerate(pic_list.id_unique):
                # Find all images with this ID
                idx_pics = [i for i, pic_id in enumerate(pic_list.id) if pic_id == unique_id]
                
                # Check if we have a full triplet (3+ images)
                if len(idx_pics) >= 3:
                    stats.N_triplet_full += 1
                    
                    # Process as triplet
                    timing, flag = process_triplet_image(
                        flake_id=unique_id,
                        idx_pics=idx_pics,
                        current_dir=current_dir,
                        pic_list=pic_list,
                        label=label,
                        process=process
                    )
                    
                    # Update statistics based on flag
                    if flag == 2:  # GOOD - matched triplet
                        stats.N_matched += 1
                    elif flag == 1:  # No match found
                        stats.N_no_match += 1
                        # When no match, images are processed independently
                        stats.N_processed_indep += len(idx_pics)
                    # flag == -1 (error) or 0 (no ROI) are not counted separately
                    
                    # Accumulate timings
                    if timing:
                        timings.tt_loading += timing.get("loading", 0.0)
                        timings.tt_clutter += timing.get("clutter", 0.0)
                        timings.tt_edging += timing.get("edging", 0.0)
                        timings.tt_roiying += timing.get("roiying", 0.0)
                        timings.tt_feature += timing.get("feature", 0.0)
                        timings.tt_matching += timing.get("matching", 0.0)
                        timings.tt_plotting += timing.get("plotting", 0.0)
                        timings.tt_saving += timing.get("saving", 0.0)
                
                else:
                    # Incomplete triplet (missing images) - process each image independently
                    stats.N_triplet_miss += 1
                    stats.N_processed_indep += len(idx_pics)
                    
                    for idx in idx_pics:
                        # Construct pic_info dict for this image
                        pic_info = {
                            "filename": pic_list.files[idx] if pic_list.files else f"image_{idx}.png",
                            "cam": pic_list.cam[idx],
                            "id": pic_list.id[idx],
                            "fallspeed": pic_list.fallspeed[idx] if idx < len(pic_list.fallspeed) else None,
                            "time_num": pic_list.time_num[idx] if idx < len(pic_list.time_num) else None,
                        }
                        
                        img_path = Path(current_dir) / pic_list.files[idx]
                        
                        try:
                            roi, flag, timing = process_single_image(img_path, pic_info, label, process, verbose=False)
                        except Exception as exc:
                            logger.error("Unknown error while processing image %s: %s", pic_list.files[idx], exc)
                            flag = -1
                            timing = {}
                        
                        # Accumulate timings (same as single-image mode)
                        if timing:
                            timings.tt_loading += timing.get("loading", 0.0)
                            timings.tt_clutter += timing.get("clutter", 0.0)
                            timings.tt_edging += timing.get("edging", 0.0)
                            timings.tt_roiying += timing.get("roiying", 0.0)
                            timings.tt_feature += timing.get("feature", 0.0)
                            timings.tt_plotting += timing.get("plotting", 0.0)
                            timings.tt_saving += timing.get("saving", 0.0)
            
            # Total is number of unique IDs processed
            stats.N_tot = len(pic_list.id_unique)
    
    # ===== End of processing loop =====
    
    timings.tt_program = time.time() - t_startprogram
    tt_all = (timings.tt_uploading + timings.tt_loading + timings.tt_clutter + 
              timings.tt_edging + timings.tt_feature + timings.tt_roiying + 
              timings.tt_plotting + timings.tt_saving)
    
    if process.use_triplet_algo:
        tt_all += timings.tt_matching
    
    # ===== Display statistics =====
    
    print("*********************************************************")
    print("*********************************************************")
    print("*********************************************************")
    print(f"***** Task finished ! Total Time spent    : {timings.tt_program:.2f} seconds")
    
    if process.use_triplet_algo:
        print(f"***** Number of triplets processed        : {stats.N_tot:d}")
        print(f"***** Number of full triplets             : {stats.N_triplet_full:d} {stats.N_triplet_full/stats.N_tot*100:.1f} %")
        print(f"***** Number of incomplete triplets       : {stats.N_triplet_miss:d} {stats.N_triplet_miss/stats.N_tot*100:.1f} %")
        print(f"***** Number of matched triplets          : {stats.N_matched:d} {stats.N_matched/stats.N_tot*100:.1f} %")
        print(f"***** Number of unmatched triplets        : {stats.N_no_match:d} {stats.N_no_match/stats.N_tot*100:.1f} %")
        print(f"***** Number of imgs processed indep.     : {stats.N_processed_indep:d} {stats.N_processed_indep/(stats.N_processed_indep+3*stats.N_triplet_full)*100:.1f} %")
    else:
        N_tot_im = stats.N_good + stats.N_bad + stats.N_blurry
        print(f"***** Number of flakes found              : {stats.N_tot:d}")
        print(f"***** Number of pictures processed        : {N_tot_im:d}")
        if N_tot_im > 0:
            print(f"***** Number of good snowflakes           : {stats.N_good:d} {stats.N_good/N_tot_im*100:.1f} %")
            print(f"***** Number of blurry snowflakes         : {stats.N_blurry:d} {stats.N_blurry/N_tot_im*100:.1f} %")
            print(f"***** Number of no/bad detections         : {stats.N_bad:d} {stats.N_bad/N_tot_im*100:.1f} %")
    
    if tt_all > 0:
        print(f"***** Total time uploading directories    : {timings.tt_uploading/tt_all*100:.1f} %")
        print(f"***** Total time loading pictures         : {timings.tt_loading/tt_all*100:.1f} %")
        print(f"***** Total time removing clutter         : {timings.tt_clutter/tt_all*100:.1f} %")
        print(f"***** Total time edging pictures          : {timings.tt_edging/tt_all*100:.1f} %")
        print(f"***** Total time computing feature        : {timings.tt_feature/tt_all*100:.1f} %")
        print(f"***** Total time selecting flake ROI      : {timings.tt_roiying/tt_all*100:.1f} %")
        if process.use_triplet_algo:
            print(f"***** Total time matching particules      : {timings.tt_matching/tt_all*100:.1f} %")

        print(f"***** Total time plotting                 : {timings.tt_plotting/tt_all*100:.1f} %")
        print(f"***** Total time saving processed data    : {timings.tt_saving/tt_all*100:.1f} %")
    
    if process.use_triplet_algo and stats.N_tot > 0:
        print(f"***** Net average time per triplet         : {timings.tt_program/stats.N_tot:.2f}")
    elif not process.use_triplet_algo and N_tot_im > 0:
        print(f"***** Net average time per image           : {timings.tt_program/N_tot_im:.2f}")
    
    print("*********************************************************")
    print("*********************************************************")
    print("*********************************************************")
    
    # ===== Save statistics to text file =====
    
    if process.saveresults:
        stats_file = Path(label.outdir) / "proc_stats.txt"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            f.write("*********************************************************\n")
            f.write("*********************************************************\n")
            f.write("*********************************************************\n")
            f.write(f"***** Task finished ! Total Time spent    : {timings.tt_program:.2f} seconds\n")
            
            if process.use_triplet_algo:
                f.write(f"***** Number of triplets processed        : {stats.N_tot:d}\n")
                f.write(f"***** Number of full triplets             : {stats.N_triplet_full:d} {stats.N_triplet_full/stats.N_tot*100:.1f} %\n")
                f.write(f"***** Number of incomplete triplets       : {stats.N_triplet_miss:d} {stats.N_triplet_miss/stats.N_tot*100:.1f} %\n")
                f.write(f"***** Number of matched triplets          : {stats.N_matched:d} {stats.N_matched/stats.N_tot*100:.1f} %\n")
                f.write(f"***** Number of unmatched triplets        : {stats.N_no_match:d} {stats.N_no_match/stats.N_tot*100:.1f} %\n")
                f.write(f"***** Number of imgs processed indep.     : {stats.N_processed_indep:d} {stats.N_processed_indep/(stats.N_processed_indep+3*stats.N_triplet_full)*100:.1f} %\n")
            else:
                N_tot_im = stats.N_good + stats.N_bad + stats.N_blurry
                f.write(f"***** Number of flakes found              : {stats.N_tot:d}\n")
                f.write(f"***** Number of pictures processed        : {N_tot_im:d}\n")
                if N_tot_im > 0:
                    f.write(f"***** Number of good snowflakes           : {stats.N_good:d} {stats.N_good/N_tot_im*100:.1f} %\n")
                    f.write(f"***** Number of blurry snowflakes         : {stats.N_blurry:d} {stats.N_blurry/N_tot_im*100:.1f} %\n")
                    f.write(f"***** Number of no/bad detections         : {stats.N_bad:d} {stats.N_bad/N_tot_im*100:.1f} %\n")
            
            if tt_all > 0:
                f.write(f"***** Total time uploading directories    : {timings.tt_uploading/tt_all*100:.1f} %\n")
                f.write(f"***** Total time loading pictures         : {timings.tt_loading/tt_all*100:.1f} %\n")
                f.write(f"***** Total time removing clutter         : {timings.tt_clutter/tt_all*100:.1f} %\n")
                f.write(f"***** Total time edging pictures          : {timings.tt_edging/tt_all*100:.1f} %\n")
                f.write(f"***** Total time computing feature        : {timings.tt_feature/tt_all*100:.1f} %\n")
                f.write(f"***** Total time selecting flake ROI      : {timings.tt_roiying/tt_all*100:.1f} %\n")
                if process.use_triplet_algo:
                    f.write(f"***** Total time matching particules      : {timings.tt_matching/tt_all*100:.1f} %\n")
                f.write(f"***** Total time plotting                 : {timings.tt_plotting/tt_all*100:.1f} %\n")
                f.write(f"***** Total time saving processed data    : {timings.tt_saving/tt_all*100:.1f} %\n")
            
            if process.use_triplet_algo and stats.N_tot > 0:
                f.write(f"***** Net average time per triplet         : {timings.tt_program/stats.N_tot:.2f}\n")
            elif not process.use_triplet_algo and N_tot_im > 0:
                f.write(f"***** Net average time per image           : {timings.tt_program/N_tot_im:.2f}\n")
            
            f.write("*********************************************************\n")
            f.write("*********************************************************\n")
            f.write("*********************************************************\n")
        
        print(f"Statistics saved to {stats_file}")
