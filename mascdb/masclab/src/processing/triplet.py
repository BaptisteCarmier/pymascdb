"""
Triplet image processing module for MASC analysis.

This module handles the processing of triplet MASC snowflake images
(3 cameras capturing the same snowflake simultaneously).

The processing pipeline includes:
1. Load all 3 (or more) camera images
2. Apply masking to remove clutter
3. Create binary masks
4. Detect ROIs on each image
5. Match ROIs across cameras based on vertical position and size
6. Select best matching triplet based on focus metric
7. Extract features for each camera view
8. Save results

Translated from MASC_triplet_process.m (Christophe Praz 2015)
Last update: November 2025
"""

########################### On abandonne ce fichier ###########################
# il ne sert à rien pour être correctement branché avec pymascdb #

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from src.core.config import LabelConfig, ProcessingConfig, create_proc_params_file
from src.features.basic import process_basic_descriptors
from src.preprocessing.masking import masking
from src.preprocessing.roi import ROIDetector
from src.utils.image import rangefilt
from src.utils.load_roi_data import save_roi_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def process_triplet_image(
    flake_id: int,
    idx_pics: List[int],
    current_dir: Path,
    pic_list,
    label: LabelConfig,
    process: ProcessingConfig,
) -> Tuple[Dict, int]:
    """
    Process a triplet of MASC images (3 cameras viewing same snowflake).

    This function implements the MASC_triplet_process.m workflow:
    1. Load images from all cameras
    2. Apply masking to remove clutter
    3. Create binary masks (edge detection)
    4. Detect ROIs on each image
    5. Match ROIs across cameras (vertical position + size)
    6. Select best matching triplet based on focus metric
    7. Extract descriptors for each view
    8. Save cropped images and metadata

    Args:
        flake_id: Unique snowflake ID
        idx_pics: List of indices in pic_list for this triplet
        current_dir: Directory containing the images
        pic_list: Picture list object with metadata
        label: LabelConfig with output paths
        process: ProcessingConfig with processing parameters

    Returns
    -------
        Tuple of (timing, flag):
            - timing: Dictionary with timing for each processing step
            - flag: Integer flag:
                -1 = Load error
                 0 = No ROI found on at least one image
                 1 = No matching triplet found
                 2 = GOOD (matching triplet found and saved)

    Notes
    -----
        This is the Python translation of MASC_triplet_process.m
        Supports 3-5 cameras (camera_order defines which cameras to use)
    """
    timing = {}

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    t_loading = time.time()

    try:
        # Initialize camera data dictionaries
        camera_data = {}

        # Get camera order from process config (default [0, 1, 2])
        camera_order = getattr(process, "camera_order", [0, 1, 2])

        # Load images for each camera
        for i in idx_pics:
            cam_id = pic_list.cam[i]

            if cam_id not in camera_order:
                continue

            # Read image
            img_path = Path(current_dir) / pic_list.files[i]

            # Check if file exists
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")

            img_data = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_data is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Ensure uint8
            if img_data.dtype != np.uint8:
                img_data = img_data.astype(np.uint8)

            # Store camera data
            camera_data[cam_id] = {
                "data": img_data,
                "filename": pic_list.files[i],
                "cam": cam_id,
                "time_num": pic_list.time_num[i] if hasattr(pic_list, "time_num") else None,
            }

        # Get fallspeed
        if hasattr(pic_list, "fallspeed") and hasattr(pic_list, "fallid"):
            fallspeed_idx = np.where(np.array(pic_list.fallid) == flake_id)[0]
            if len(fallspeed_idx) > 0:
                flake_fallspeed = pic_list.fallspeed[fallspeed_idx[0]]
            else:
                flake_fallspeed = np.nan
        else:
            flake_fallspeed = np.nan

    except Exception as e:
        logger.warning("Warning: Couldn't load flake %s. Triplet discarded: %s", flake_id, e)
        return {}, -1

    timing["loading"] = time.time() - t_loading

    # Check that we have at least 3 cameras
    if len(camera_data) < 3:
        logger.info("Warning: Not enough cameras for flake %s (found %d, need 3)", flake_id, len(camera_data))
        return timing, -1

    # ========================================================================
    # 2. REMOVE CLUTTER (MASKING)
    # ========================================================================
    t_clutter = time.time()

    for cam_id, cam_info in camera_data.items():
        cam_info["data"] = masking(cam_info["data"], cam_id, process)

    timing["clutter"] = time.time() - t_clutter

    # ========================================================================
    # 3. CREATE BINARY MASKS
    # ========================================================================
    t_edging = time.time()

    for cam_id, cam_info in camera_data.items():
        # Create binary mask
        eroded = np.zeros_like(cam_info["data"], dtype=np.uint8)
        eroded[cam_info["data"] > 0] = 1

        # Fill holes
        eroded = ndimage.binary_fill_holes(eroded).astype(np.uint8)

        cam_info["eroded"] = eroded

    timing["edging"] = time.time() - t_edging

    # ========================================================================
    # 4. DETECT ROIs ON EACH IMAGE
    # ========================================================================
    t_roiying = time.time()

    detector = ROIDetector(process)

    # Get main 3 cameras (first 3 in camera_order)
    main_cameras = camera_order[:3]
    left_cam, mid_cam, right_cam = main_cameras[0], main_cameras[1], main_cameras[2]

    # Detect ROIs for each camera
    for cam_id in main_cameras:
        if cam_id not in camera_data:
            logger.warning("Camera %s not found for flake ID %s", cam_id, flake_id)
            timing["roiying"] = time.time() - t_roiying
            timing["matching"] = 0
            timing["feature"] = 0
            timing["plotting"] = 0
            timing["saving"] = 0
            return timing, 0

        all_roi, idx_best, area_focus_ratio, flag_roi, status = detector.detect(
            camera_data[cam_id]["data"],
            camera_data[cam_id]["eroded"],
            cam_id,
            compute_best=False,  # We'll match triplets first
        )

        camera_data[cam_id]["all_roi"] = all_roi

        # Check if no ROI found
        if len(all_roi) == 0:
            timing["roiying"] = time.time() - t_roiying
            timing["matching"] = 0
            timing["feature"] = 0
            timing["plotting"] = 0
            timing["saving"] = 0
            return timing, 0

    # Process additional cameras (if any) - just find best ROI
    for cam_id in camera_order[3:]:
        if cam_id in camera_data:
            all_roi, idx_best, area_focus_ratio, flag_roi, status = detector.detect(
                camera_data[cam_id]["data"],
                camera_data[cam_id]["eroded"],
                cam_id,
                compute_best=True,
            )
            camera_data[cam_id]["all_roi"] = all_roi
            camera_data[cam_id]["idx_best"] = idx_best

    timing["roiying"] = time.time() - t_roiying

    # ========================================================================
    # 5. MATCH PARTICLES ACROSS CAMERAS (ref = mid camera)
    # ========================================================================
    t_matching = time.time()

    left_all_roi = camera_data[left_cam]["all_roi"]
    mid_all_roi = camera_data[mid_cam]["all_roi"]
    right_all_roi = camera_data[right_cam]["all_roi"]

    idx_left_match = []
    idx_right_match = []

    # For each ROI in mid camera, find matching ROIs in left and right
    for k, mid_roi in enumerate(mid_all_roi):
        mid_height = mid_roi.bbox[3]  # height
        mid_vert_pos = mid_roi.bbox[1]  # y position

        matching_tol_height = max(process.matching_tol_pix, process.matching_tol_percent * mid_height / 100.0)
        matching_tol_vert_pos = max(process.matching_tol_pix, process.matching_tol_percent * mid_height / 100.0)

        # Find match on left camera
        left_match = None
        best_left_var = float("inf")

        for left_idx, left_roi in enumerate(left_all_roi):
            left_height = left_roi.bbox[3]
            left_vert_pos = left_roi.bbox[1]

            # Check if within tolerance
            if (
                abs(left_height - mid_height) <= matching_tol_height
                and abs(left_vert_pos - mid_vert_pos) <= matching_tol_vert_pos
            ):

                # Compute variation metric
                height_var = abs(left_height - mid_height) + abs(left_vert_pos - mid_vert_pos)

                if height_var < best_left_var:
                    best_left_var = height_var
                    left_match = left_idx

        idx_left_match.append(left_match)

        # Find match on right camera
        right_match = None
        best_right_var = float("inf")

        for right_idx, right_roi in enumerate(right_all_roi):
            right_height = right_roi.bbox[3]
            right_vert_pos = right_roi.bbox[1]

            # Check if within tolerance
            if (
                abs(right_height - mid_height) <= matching_tol_height
                and abs(right_vert_pos - mid_vert_pos) <= matching_tol_vert_pos
            ):

                # Compute variation metric
                height_var = abs(right_height - mid_height) + abs(right_vert_pos - mid_vert_pos)

                if height_var < best_right_var:
                    best_right_var = height_var
                    right_match = right_idx

        idx_right_match.append(right_match)

    # ========================================================================
    # 6. SELECT BEST MATCHING TRIPLET BASED ON FOCUS
    # ========================================================================

    # Compute area*focus for each matching triplet
    area_focus_matrix = np.full((len(mid_all_roi), 3), np.nan)

    for k in range(len(mid_all_roi)):
        if idx_left_match[k] is not None and idx_right_match[k] is not None:

            # Compute mid camera area*focus
            mid_roi = mid_all_roi[k]
            mid_focus = _compute_roi_focus(camera_data[mid_cam]["data"], mid_roi)
            area_focus_matrix[k, 1] = mid_focus * mid_roi.area

            # Compute left camera area*focus
            left_roi = left_all_roi[idx_left_match[k]]
            left_focus = _compute_roi_focus(camera_data[left_cam]["data"], left_roi)
            area_focus_matrix[k, 0] = left_focus * left_roi.area

            # Compute right camera area*focus
            right_roi = right_all_roi[idx_right_match[k]]
            right_focus = _compute_roi_focus(camera_data[right_cam]["data"], right_roi)
            area_focus_matrix[k, 2] = right_focus * right_roi.area

    timing["matching"] = time.time() - t_matching

    # Check if we found any matching triplet
    if np.all(np.isnan(area_focus_matrix)):
        timing["feature"] = 0
        timing["plotting"] = 0
        timing["saving"] = 0
        return timing, 1

    # Select triplet with best average area*focus
    if area_focus_matrix.size == 0 or np.all(np.isnan(area_focus_matrix)):
        logger.warning("No valid triplets found for snowflake %s", flake_id)
        return timing, 1  # No match found

    mean_area_focus = np.nanmean(area_focus_matrix, axis=1)
    idx_mid = np.nanargmax(mean_area_focus)
    idx_left = idx_left_match[idx_mid]
    idx_right = idx_right_match[idx_mid]

    # ========================================================================
    # 7. COMPUTE FEATURES ON SELECTED ROIs
    # ========================================================================
    t_feature = time.time()

    # Process left camera
    roi_left_obj = left_all_roi[idx_left]
    roi_left_dict = {
        "BoundingBox": roi_left_obj.bbox,
        "image": roi_left_obj.mask,
        "coords": roi_left_obj.coords,
        "MajorAxisLength": roi_left_obj.major_axis_length,
        "MinorAxisLength": roi_left_obj.minor_axis_length,
        "Orientation": roi_left_obj.orientation,
        "Centroid": roi_left_obj.centroid,
    }
    roi_left = process_basic_descriptors(
        camera_data[left_cam]["data"],
        roi_left_dict,
        process,
    )
    roi_left["n_roi"] = len(left_all_roi)
    roi_left["name"] = camera_data[left_cam]["filename"]
    roi_left["id"] = flake_id
    roi_left["cam"] = left_cam
    roi_left["tnum"] = camera_data[left_cam]["time_num"]
    roi_left["fallspeed"] = flake_fallspeed
    roi_left["flag_roi"] = "GOOD"
    roi_left["flag"] = "GOOD"
    roi_left["status"] = "good detection."

    # Process mid camera
    roi_mid_obj = mid_all_roi[idx_mid]
    roi_mid_dict = {
        "BoundingBox": roi_mid_obj.bbox,
        "image": roi_mid_obj.mask,
        "coords": roi_mid_obj.coords,
        "MajorAxisLength": roi_mid_obj.major_axis_length,
        "MinorAxisLength": roi_mid_obj.minor_axis_length,
        "Orientation": roi_mid_obj.orientation,
        "Centroid": roi_mid_obj.centroid,
    }
    roi_mid = process_basic_descriptors(
        camera_data[mid_cam]["data"],
        roi_mid_dict,
        process,
    )
    roi_mid["n_roi"] = len(mid_all_roi)
    roi_mid["name"] = camera_data[mid_cam]["filename"]
    roi_mid["id"] = flake_id
    roi_mid["cam"] = mid_cam
    roi_mid["tnum"] = camera_data[mid_cam]["time_num"]
    roi_mid["fallspeed"] = flake_fallspeed
    roi_mid["flag_roi"] = "GOOD"
    roi_mid["flag"] = "GOOD"
    roi_mid["status"] = "good detection."

    # Process right camera
    roi_right_obj = right_all_roi[idx_right]
    roi_right_dict = {
        "BoundingBox": roi_right_obj.bbox,
        "image": roi_right_obj.mask,
        "coords": roi_right_obj.coords,
        "MajorAxisLength": roi_right_obj.major_axis_length,
        "MinorAxisLength": roi_right_obj.minor_axis_length,
        "Orientation": roi_right_obj.orientation,
        "Centroid": roi_right_obj.centroid,
    }
    roi_right = process_basic_descriptors(
        camera_data[right_cam]["data"],
        roi_right_dict,
        process,
    )
    roi_right["n_roi"] = len(right_all_roi)
    roi_right["name"] = camera_data[right_cam]["filename"]
    roi_right["id"] = flake_id
    roi_right["cam"] = right_cam
    roi_right["tnum"] = camera_data[right_cam]["time_num"]
    roi_right["fallspeed"] = flake_fallspeed
    roi_right["flag_roi"] = "GOOD"
    roi_right["flag"] = "GOOD"
    roi_right["status"] = "good detection."

    # Process additional cameras (if any)
    roi_additional = []
    for cam_id in camera_order[3:]:
        if cam_id in camera_data and len(camera_data[cam_id]["all_roi"]) > 0:
            idx_best = camera_data[cam_id]["idx_best"]
            roi_ad_obj = camera_data[cam_id]["all_roi"][idx_best]
            roi_ad_dict = {
                "BoundingBox": roi_ad_obj.bbox,
                "image": roi_ad_obj.mask,
                "coords": roi_ad_obj.coords,
                "MajorAxisLength": roi_ad_obj.major_axis_length,
                "MinorAxisLength": roi_ad_obj.minor_axis_length,
                "Orientation": roi_ad_obj.orientation,
                "Centroid": roi_ad_obj.centroid,
            }
            roi_ad = process_basic_descriptors(
                camera_data[cam_id]["data"],
                roi_ad_dict,
                process,
            )
            roi_ad["n_roi"] = len(camera_data[cam_id]["all_roi"])
            roi_ad["name"] = camera_data[cam_id]["filename"]
            roi_ad["id"] = flake_id
            roi_ad["cam"] = cam_id
            roi_ad["tnum"] = camera_data[cam_id]["time_num"]
            roi_ad["fallspeed"] = flake_fallspeed
            roi_ad["flag_roi"] = "GOOD"
            roi_ad["flag"] = "GOOD"
            roi_ad["status"] = "good detection."
            roi_additional.append(roi_ad)

    timing["feature"] = time.time() - t_feature

    # ========================================================================
    # 8. PLOTTING (if desired)
    # ========================================================================
    t_plotting = time.time()

    if process.generate_figs:
        try:
            # Set matplotlib backend based on display_figs setting
            if not process.display_figs:
                matplotlib.use("Agg")  # Non-interactive backend
            else:
                matplotlib.use("TkAgg")  # Interactive backend for display

            # Determine number of rows for subplot (1 or 2 depending on additional cameras)
            nline_fig = 1 if len(roi_additional) == 0 else 2

            # Create figure for triplet display (similar to MATLAB fig1)
            fig_triplet = plt.figure(figsize=(12, 4 * nline_fig))

            # Left camera (subplot 1)
            ax1 = plt.subplot(nline_fig, 3, 1)
            ax1.imshow(roi_left["data"], cmap="gray", vmin=0, vmax=255)
            ax1.set_xlabel(f"cam {roi_left['cam']}")
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Mid camera (subplot 2)
            ax2 = plt.subplot(nline_fig, 3, 2)
            ax2.imshow(roi_mid["data"], cmap="gray", vmin=0, vmax=255)
            ax2.set_xlabel(f"cam {roi_mid['cam']}")
            ax2.set_title(f"Triplet #{flake_id}")
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Right camera (subplot 3)
            ax3 = plt.subplot(nline_fig, 3, 3)
            ax3.imshow(roi_right["data"], cmap="gray", vmin=0, vmax=255)
            ax3.set_xlabel(f"cam {roi_right['cam']}")
            ax3.set_xticks([])
            ax3.set_yticks([])

            # Additional cameras (if any) in second row
            if len(roi_additional) > 0:
                # First additional camera (subplot 4)
                ax4 = plt.subplot(nline_fig, 3, 4)
                ax4.imshow(roi_additional[0]["data"], cmap="gray", vmin=0, vmax=255)
                ax4.set_xlabel(f"add. cam {roi_additional[0]['cam']}")
                ax4.set_xticks([])
                ax4.set_yticks([])

                # Second additional camera (subplot 6) if exists
                if len(roi_additional) > 1:
                    ax6 = plt.subplot(nline_fig, 3, 6)
                    ax6.imshow(roi_additional[1]["data"], cmap="gray", vmin=0, vmax=255)
                    ax6.set_xlabel(f"add. cam {roi_additional[1]['cam']}")
                    ax6.set_xticks([])
                    ax6.set_yticks([])

            plt.tight_layout()

            # Store figure for later saving
            roi_mid["_figure_triplet"] = fig_triplet

            if process.display_figs:
                plt.draw()  # Force draw
                plt.pause(0.1)  # Pause to let figure render

        except Exception as e:
            logger.warning("Could not generate triplet figure for flake %s: %s", flake_id, e)
            if not process.display_figs:
                plt.close("all")

    timing["plotting"] = time.time() - t_plotting

    # ========================================================================
    # 8b. CREATE INDIVIDUAL FIGURES FOR EACH CAMERA (if requested)
    # ========================================================================

    if process.generate_figs:
        try:
            # Create individual figures for each camera with ellipses/contours
            roi_left["_figure"] = _create_individual_figure(roi_left)
            roi_mid["_figure"] = _create_individual_figure(roi_mid)
            roi_right["_figure"] = _create_individual_figure(roi_right)

            # Create figures for additional cameras
            for roi_ad in roi_additional:
                roi_ad["_figure"] = _create_individual_figure(roi_ad)

            if process.display_figs:
                plt.draw()
                plt.pause(0.1)

        except Exception as e:
            logger.warning("Could not generate individual camera figures: %s", e)

    # ========================================================================
    # 9. SAVE RESULTS
    # ========================================================================
    t_saving = time.time()

    if process.saveresults:

        # Create metadata file if first time
        if not label.outdir.exists():
            label.outdir.mkdir(parents=True, exist_ok=True)
            create_proc_params_file(label.outdir, label, process)
            logger.info("Creation of metadata file proc_params.txt...")

        # Create date/hour folders
        tnum = roi_mid.get("tnum")
        if tnum is not None and not np.isnan(tnum):
            try:
                # Check if tnum is already a datetime object
                if isinstance(tnum, datetime):
                    dt = tnum
                elif isinstance(tnum, (int, float)):
                    # Try MATLAB datenum conversion
                    # MATLAB datenum: days since January 0, 0000
                    # Conversion: subtract 719529 days (difference between year 0 and Unix epoch)
                    # Then convert to Unix timestamp
                    # MATLAB epoch: day 1 = January 1, year 1 AD = 1.0
                    # Python/Unix epoch: January 1, 1970

                    # More robust: use the date from the filename instead
                    # Extract date from filename (format: YYYY.MM.DD_HH.MM.SS)
                    filename = roi_mid.get("name", "")
                    if filename:
                        # Parse date from filename
                        parts = filename.split("_")
                        if len(parts) >= 2:
                            date_str = parts[0]  # YYYY.MM.DD
                            time_str = parts[1]  # HH.MM.SS

                            date_parts = date_str.split(".")
                            time_parts = time_str.split(".")

                            if len(date_parts) >= 3 and len(time_parts) >= 3:
                                year = int(date_parts[0])
                                month = int(date_parts[1])
                                day = int(date_parts[2])
                                hour = int(time_parts[0])
                                dt = datetime(year, month, day, hour)
                            else:
                                raise ValueError("Cannot parse date from filename")
                        else:
                            raise ValueError("Cannot parse date from filename")
                    else:
                        raise ValueError("No filename available")
                else:
                    raise ValueError(f"Unexpected tnum type: {type(tnum)}")

                date_folder = dt.strftime("%Y.%m.%d")
                hour_folder = dt.strftime("%H")

            except Exception as e:
                logger.warning("Could not parse date from tnum or filename: %s", e)
                date_folder = "unknown_date"
                hour_folder = "00"
        else:
            # Fallback if no time info
            date_folder = "unknown_date"
            hour_folder = "00"

        path2save = label.outdir / date_folder / hour_folder

        # Compute triplet quality metrics
        triplet_mean_intens = np.mean(
            [
                roi_left.get("mean_intens", 0),
                roi_mid.get("mean_intens", 0),
                roi_right.get("mean_intens", 0),
            ]
        )

        triplet_max_intens = np.mean(
            [
                roi_left.get("max_intens", 0),
                roi_mid.get("max_intens", 0),
                roi_right.get("max_intens", 0),
            ]
        )

        triplet_max_dim = np.mean(
            [
                max(roi_left.get("width", 0), roi_left.get("height", 0)),
                max(roi_mid.get("width", 0), roi_mid.get("height", 0)),
                max(roi_right.get("width", 0), roi_right.get("height", 0)),
            ]
        )

        # Determine if GOOD or BAD based on triplet averages (same as MATLAB)
        # Calculate triplet mean intensity, max intensity, and max dimension
        triplet_mean_intens = np.mean(
            [
                roi_left.get("mean_intens", 0),
                roi_mid.get("mean_intens", 0),
                roi_right.get("mean_intens", 0),
            ]
        )
        triplet_max_intens = np.mean(
            [
                roi_left.get("max_intens", 0),
                roi_mid.get("max_intens", 0),
                roi_right.get("max_intens", 0),
            ]
        )
        triplet_max_dim = np.mean(
            [
                max(roi_left.get("width", 0), roi_left.get("height", 0)),
                max(roi_mid.get("width", 0), roi_mid.get("height", 0)),
                max(roi_right.get("width", 0), roi_right.get("height", 0)),
            ]
        )

        # Validate triplet quality (same thresholds as MATLAB)
        is_good = (
            triplet_mean_intens >= process.minbright
            and triplet_max_intens >= process.max_intensthresh
            and triplet_max_dim >= process.sizemin
        )

        if is_good:
            save_path = path2save / "IMAGES" / "GOOD"
        else:
            save_path = path2save / "IMAGES" / "BAD"

        save_path.mkdir(parents=True, exist_ok=True)

        # Save images
        _save_roi_image(roi_left, save_path)
        _save_roi_image(roi_mid, save_path)
        _save_roi_image(roi_right, save_path)

        # Save additional camera images
        for roi_ad in roi_additional:
            _save_roi_image(roi_ad, save_path)

        # Save ROI data (pickle format)
        if is_good:
            data_path = path2save / "DATA" / "GOOD"
        else:
            data_path = path2save / "DATA" / "BAD"

        data_path.mkdir(parents=True, exist_ok=True)

        # Determine file extension based on save format
        if process.save_format == "joblib":
            ext = "joblib"
        elif process.save_format == "pkl":
            ext = "pkl"
        elif process.save_format == "mat":
            ext = "mat"
        else:
            ext = "joblib"  # Default fallback

        # Save each camera's ROI data
        for roi_item in [roi_left, roi_mid, roi_right] + roi_additional:
            if "name" in roi_item:
                # Save ROI data using save_roi_data utility
                base_name = Path(roi_item["name"]).stem
                data_file_path = data_path / f"{base_name}.{ext}"

                save_roi_data(roi_item, data_file_path, format=process.save_format, compress=3)

        # Save individual camera figures if requested
        if process.save_figs:
            # Determine figure directory based on quality
            if is_good:
                fig_dir = path2save / "FIGURES" / "GOOD"
            else:
                fig_dir = path2save / "FIGURES" / "BAD"

            fig_dir.mkdir(parents=True, exist_ok=True)

            # Save figures for each camera
            for roi_item in [roi_left, roi_mid, roi_right] + roi_additional:
                if "_figure" in roi_item and "name" in roi_item:
                    base_name = Path(roi_item["name"]).stem
                    fig_path = fig_dir / f"{base_name}_fig.png"
                    roi_item["_figure"].savefig(fig_path, dpi=100, bbox_inches="tight")

        # # Save triplet figure if requested
        # if process.save_figs and '_figure_triplet' in roi_mid:
        #     triplet_fig_dir = label.outdir / date_folder / hour_folder / 'TRIPLETS'
        #     triplet_fig_dir.mkdir(parents=True, exist_ok=True)
        #     fig_path = triplet_fig_dir / f'triplet_{flake_id}.png'
        #     roi_mid['_figure_triplet'].savefig(fig_path, dpi=100, bbox_inches='tight')
        #     print(f"Saved triplet figure: {fig_path}")

    # Close figures if not displaying
    if process.generate_figs and not process.display_figs:
        plt.close("all")

    timing["saving"] = time.time() - t_saving

    # Return flag based on triplet quality
    # is_good was determined from roi_mid flag_roi
    if is_good:
        return timing, 2  # flag=2: GOOD matching triplet
    return timing, 1  # flag=1: Triplet matched but quality is BAD


def _create_individual_figure(roi: Dict) -> matplotlib.figure.Figure:
    """
    Create individual figure for a single camera ROI with ellipses and contours.

    Similar to single.py plotting section.

    Args:
        roi: ROI dictionary with data and geometric features

    Returns
    -------
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display ROI image in grayscale
    ax.imshow(roi["data"], cmap="gray", vmin=0, vmax=255)
    ax.set_aspect("equal")
    ax.set_facecolor("black")

    # Generate ellipse points (t from 0 to 2*pi)
    t = np.linspace(0, 2 * np.pi, 100)

    # Convert orientation from degrees to radians (negative as per MATLAB)
    if "E" in roi and "theta" in roi["E"]:
        theta = -roi["E"]["theta"] * np.pi / 180.0
    else:
        theta = 0

    # Inner ellipse (E)
    if "E" in roi and all(k in roi["E"] for k in ["X0", "Y0", "a", "b"]):
        X0, Y0, a, b = roi["E"]["X0"], roi["E"]["Y0"], roi["E"]["a"], roi["E"]["b"]
        xt1 = X0 + np.cos(theta) * a * np.cos(t) - np.sin(theta) * b * np.sin(t)
        yt1 = Y0 + np.sin(theta) * a * np.cos(t) + np.cos(theta) * b * np.sin(t)
        ax.plot(xt1, yt1, "r-", linewidth=2, label="Inner ellipse (E)")
        ax.plot(X0, Y0, "rx", markersize=10)

    # Outer ellipse (E_out)
    if "E_out" in roi and all(k in roi["E_out"] for k in ["X0", "Y0", "a", "b"]):
        X0, Y0, a, b = roi["E_out"]["X0"], roi["E_out"]["Y0"], roi["E_out"]["a"], roi["E_out"]["b"]
        xt2 = X0 + np.cos(theta) * a * np.cos(t) - np.sin(theta) * b * np.sin(t)
        yt2 = Y0 + np.sin(theta) * a * np.cos(t) + np.cos(theta) * b * np.sin(t)
        ax.plot(xt2, yt2, "c-", linewidth=2, label="Outer ellipse (E_out)")
        ax.plot(X0, Y0, "co", markersize=8)

    # Inscribed ellipse (E_in)
    if "E_in" in roi and all(k in roi["E_in"] for k in ["X0", "Y0", "a", "b"]):
        X0, Y0, a, b = roi["E_in"]["X0"], roi["E_in"]["Y0"], roi["E_in"]["a"], roi["E_in"]["b"]
        xt3 = X0 + np.cos(theta) * a * np.cos(t) - np.sin(theta) * b * np.sin(t)
        yt3 = Y0 + np.sin(theta) * a * np.cos(t) + np.cos(theta) * b * np.sin(t)
        ax.plot(xt3, yt3, "g-", linewidth=2, label="Inscribed ellipse (E_in)")
        ax.plot(X0, Y0, "gv", markersize=8)

    # Circumscribed circle (C_out)
    if "C_out" in roi and all(k in roi["C_out"] for k in ["X0", "Y0", "r"]):
        X0, Y0, r = roi["C_out"]["X0"], roi["C_out"]["Y0"], roi["C_out"]["r"]
        xt4 = X0 + r * np.cos(t)
        yt4 = Y0 + r * np.sin(t)
        ax.plot(xt4, yt4, "b-", linewidth=2, label="Circumscribed circle (C_out)")

    # Perimeter
    if "x_perim" in roi and "y_perim" in roi:
        ax.plot(roi["x_perim"], roi["y_perim"], "y-", linewidth=1, label="Perimeter")

    # Convex hull
    if "hull" in roi and "xh" in roi["hull"] and "yh" in roi["hull"]:
        ax.plot(roi["hull"]["xh"], roi["hull"]["yh"], "c--", linewidth=1, label="Convex hull")

    # Bounding rectangle
    if "Rect" in roi and "rectx" in roi["Rect"] and "recty" in roi["Rect"]:
        ax.plot(roi["Rect"]["rectx"], roi["Rect"]["recty"], "w-", linewidth=1, label="Bounding rect")

    # Labels and title
    ax.set_xlabel("x axis [pixels]")
    ax.set_ylabel("y axis [pixels]")
    if "E" in roi and "theta" in roi["E"]:
        ax.set_title(f"Camera {roi.get('cam', '?')} - Orientation: {roi['E']['theta']:.2f}°")
    else:
        ax.set_title(f"Camera {roi.get('cam', '?')}")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    return fig


def _compute_roi_focus(img_data: np.ndarray, roi) -> float:
    """
    Compute focus quality metric for an ROI.

    Focus = mean_intensity * range_intensity^2

    Args:
        img_data: Full image data
        roi: ROI object with bbox and mask

    Returns
    -------
        Focus quality metric
    """
    # Extract ROI region
    x, y, w, h = roi.bbox
    roi_data = img_data[y : y + h, x : x + w]

    # Get pixels within ROI mask
    roi_pixels = roi_data[roi.mask]

    # Compute mean intensity
    mean_intens = np.mean(roi_pixels) / 255.0

    # Compute range intensity (local variability)
    range_array = rangefilt(roi_data)
    range_intens = np.mean(range_array[roi.mask]) / 255.0

    # Focus metric
    focus = mean_intens * (range_intens**2)

    return focus


def _save_roi_image(roi: Dict, save_path: Path) -> None:
    """
    Save ROI image to disk.

    Args:
        roi: ROI dictionary with 'data' and 'name' keys
        save_path: Directory to save to
    """
    if "data" not in roi or "name" not in roi:
        return

    img_path = save_path / roi["name"]

    # Save image using cv2
    cv2.imwrite(str(img_path), roi["data"])
