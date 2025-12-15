"""
Basic feature extraction module for MASC analysis.

This module provides functions for extracting basic descriptors from
snowflake ROIs, including intensity, geometry, topology and texture features.

Translated from process_basic_descriptors.m

Features computed:
- Intensity features (mean, max, min, contrast, range)
- Geometric features (area, perimeter, dimensions, convex hull)
- Shape features (ellipses, circles, rectangularity, complexity)
- Topological features (skeleton, fractal dimension, holes)
- Texture features (using fmeasure, Haralick, local std)
- Symmetry features
"""

import logging
from typing import Dict

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from src.core.config import ProcessingConfig
from src.features.fmeasure import fmeasure
from src.preprocessing.brightening import brightening
from src.utils.descriptors_heplers import (
    compute_convex_hull,
    compute_D90,
    compute_Dmax,
    compute_rectangularity,
    compute_symmetry_features,
    fit_circle_around,
    fit_ellipse_around,
    fit_ellipse_inside,
    fractal_dim,
    haralick_props,
    skeleton_props,
)
from src.utils.image import rangefilt, stdfilt

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def process_basic_descriptors(
    data_in: np.ndarray,
    regionprops_roi: Dict,
    process: ProcessingConfig,
) -> Dict:
    """
    Process basic descriptors for a detected ROI.

    This is the main entry point, equivalent to MATLAB's process_basic_descriptors.m

    Args:
        data_in: Input image (grayscale uint8)
        regionprops_roi: Dictionary with regionprops output from scikit-image
                        Expected keys: BoundingBox, image, coords, Centroid,
                        MajorAxisLength, MinorAxisLength, Orientation
        process: ProcessingConfig object with parameters

    Returns
    -------
        Dictionary (roi) with all computed descriptors
    """
    roi = {}

    # Extract bounding box and ROI data from regionprops
    # scikit-image BoundingBox format: (min_row, min_col, max_row, max_col)
    bbox = regionprops_roi["BoundingBox"]
    min_row, min_col, max_row, max_col = bbox

    # Extract and crop ROI image
    roi_data = data_in[min_row:max_row, min_col:max_col]
    roi["data"] = roi_data.copy()

    # Get local binary mask from regionprops (already in local coordinates relative to bbox)
    if "image" in regionprops_roi and regionprops_roi["image"] is not None:
        # Use the binary mask from regionprops (already local to bbox)
        local_mask = regionprops_roi["image"].astype(bool)
    else:
        # Fallback: create mask from pixel coordinates
        local_mask = np.zeros_like(roi_data, dtype=bool)
        if "coords" in regionprops_roi and regionprops_roi["coords"] is not None:
            coords = regionprops_roi["coords"]
            # Shift coords to local coordinates (relative to bbox)
            local_coords = coords - np.array([[min_row, min_col]])
            # Filter coords that fall within local bbox
            valid = (
                (local_coords[:, 0] >= 0)
                & (local_coords[:, 0] < roi_data.shape[0])
                & (local_coords[:, 1] >= 0)
                & (local_coords[:, 1] < roi_data.shape[1])
            )
            local_coords_valid = local_coords[valid]
            if len(local_coords_valid) > 0:
                local_mask[tuple(local_coords_valid.T)] = True

        # Last resort fallback: use pixel values > 0
        if not np.any(local_mask):
            local_mask = (roi_data > 0).astype(bool)

    # Generate masks of the snowflake + holes
    roi["bw_mask"] = local_mask
    roi["bw_mask_filled"] = ndimage.binary_fill_holes(roi["bw_mask"])

    # Detect holes
    bw_holes_mask = roi["bw_mask_filled"] & ~roi["bw_mask"]

    if np.any(bw_holes_mask):
        # Use regionprops to detect holes
        holes_labeled = label(bw_holes_mask)
        holes_props = regionprops(holes_labeled)

        # Filter holes by minimum area

        valid_holes = [h for h in holes_props if h.area > process.min_hole_area]

        roi["nb_holes"] = len(valid_holes)
        roi["holes_mask"] = np.zeros_like(roi_data, dtype=bool)

        if roi["nb_holes"] > 0:
            all_coords = np.vstack([hole.coords for hole in valid_holes])
            roi["holes_mask"][tuple(all_coords.T)] = True

    else:
        roi["nb_holes"] = 0
        roi["holes_mask"] = np.zeros_like(roi_data, dtype=bool)

    # Final mask: filled - holes
    roi["bw_mask"] = roi["bw_mask_filled"] & ~roi["holes_mask"]

    # Get coordinates of filled mask
    roi["y"], roi["x"] = np.where(roi["bw_mask_filled"] > 0)
    roi["area"] = float(np.sum(roi["bw_mask_filled"]))
    roi["area2"] = roi["area"]  # MATLAB compatibility
    roi["area_porous"] = float(np.sum(roi["bw_mask"]))

    # Intensity features (normalized to [0, 1])
    pixels_filled = roi["data"][roi["bw_mask_filled"]]
    if len(pixels_filled) > 0:
        roi["mean_intens"] = float(np.mean(pixels_filled)) / 255.0
        roi["max_intens"] = float(np.max(pixels_filled)) / 255.0
        roi["min_intens"] = float(np.min(pixels_filled)) / 255.0  # Note: done later in matlab but coherent here

        # Range intensity
        range_array = rangefilt(roi["data"], size=3)
        roi["range_intens"] = float(np.mean(range_array[roi["bw_mask_filled"]])) / 255.0

        # Focus and derived metrics
        roi["focus"] = roi["mean_intens"] * roi["range_intens"]
        roi["area_focus"] = roi["area"] * roi["focus"]
        roi["area_range"] = roi["area"] * roi["range_intens"]

        # Contrast (note: MATLAB has inconsistent normalization here)
        roi["contrast"] = (
            float(roi["max_intens"] - roi["min_intens"]) / roi["mean_intens"]
        )  # Note : done later un matlab but coherent here

    else:  # No pixels in filled mask, set all intensity features to zero ### TO VERIFY IF CORRECT
        roi["mean_intens"] = 0.0
        roi["max_intens"] = 0.0
        roi["min_intens"] = 0.0
        roi["range_intens"] = 0.0
        roi["focus"] = 0.0
        roi["area_focus"] = 0.0
        roi["area_range"] = 0.0
        roi["contrast"] = 0.0

    # Location in original image (match MATLAB: x_loc stores row, y_loc stores column)
    roi["x_loc"] = min_row
    roi["y_loc"] = min_col

    # Fitted ellipse parameters (from regionprops, using scikit-image convention)
    # scikit-image orientation is in radians; MATLAB expects degrees
    maj_axis = regionprops_roi.get("MajorAxisLength", 0.0)
    min_axis = regionprops_roi.get("MinorAxisLength", 0.0)
    orientation_rad = regionprops_roi.get("Orientation", 0.0)
    orientation_deg = np.degrees(orientation_rad)  # Convert to degrees

    roi["E"] = {
        "a": maj_axis / 2.0,
        "b": min_axis / 2.0,
        "theta": orientation_deg,
    }
    roi["orientation"] = orientation_deg
    roi["major_axis_length"] = maj_axis
    roi["minor_axis_length"] = min_axis

    # Centroid (global and local versions)
    centroid_full = regionprops_roi.get("Centroid", (min_row, min_col))
    centroid_global = (float(centroid_full[1]), float(centroid_full[0]))  # (x, y) like MATLAB
    centroid_local = (float(centroid_full[1] - min_col), float(centroid_full[0] - min_row))
    roi["centroid_global"] = centroid_global
    roi["centroid"] = centroid_global  # Field 15: MATLAB-compatible naming
    roi["centroid_local_init"] = centroid_local
    roi["E"]["X0"] = centroid_local[0]
    roi["E"]["Y0"] = centroid_local[1]

    # Perimeter using boundary
    contours, _ = cv2.findContours(roi["bw_mask_filled"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        if len(contours) > 1:
            logger.warning("Warning: more than 1 particle detected on bw_mask_filled !!!")

        contour = contours[0].squeeze()
        if contour.ndim == 1:
            contour = contour.reshape(-1, 2)

        roi["x_perim"] = contour[:, 0]  # Field 24
        roi["y_perim"] = contour[:, 1]  # Field 23
        roi["perim"] = float(len(roi["x_perim"]))

        # Build a perimeter mask (bw_perim) similar to MATLAB output
        bw_perim = np.zeros_like(roi["bw_mask_filled"], dtype=bool)
        y_idx = np.clip(roi["y_perim"].astype(int), 0, bw_perim.shape[0] - 1)
        x_idx = np.clip(roi["x_perim"].astype(int), 0, bw_perim.shape[1] - 1)
        bw_perim[y_idx, x_idx] = True
        roi["bw_perim"] = bw_perim  # Field 20

    else:
        roi["x_perim"] = np.array([])
        roi["y_perim"] = np.array([])
        roi["perim"] = 0.0
        roi["bw_perim"] = np.zeros_like(roi["bw_mask_filled"], dtype=bool)

    # Convex hull
    roi["hull"] = compute_convex_hull(roi["x"], roi["y"])  ### a vérifier l'exactitude de convex_hull

    # Dimensions
    roi["width"] = float(roi["bw_mask"].shape[1])
    roi["height"] = float(roi["bw_mask"].shape[0])
    roi["Dmean"] = 0.5 * (roi["width"] + roi["height"])

    # Dmax and related
    roi["Dmax"], roi["Dmax_theta"], roi["DmaxA"], roi["DmaxB"] = compute_Dmax(roi["hull"]["xh"], roi["hull"]["yh"])

    roi["eq_radius"] = np.sqrt(roi["area"] / np.pi)
    roi["D90"] = compute_D90(roi["bw_mask_filled"], roi["Dmax"], roi["Dmax_theta"])

    # Complexity
    roi["complex"] = len(roi["y_perim"]) / (2.0 * np.pi * roi["eq_radius"])

    # Circumscribed circle
    roi["C_out"] = fit_circle_around(roi["x_perim"], roi["y_perim"], roi["hull"]["xh"], roi["hull"]["yh"])

    # Local centroid (in cropped coordinates)
    if len(roi["x"]) > 0:
        roi["centroid_local"] = (float(np.mean(roi["x"])), float(np.mean(roi["y"])))  ### ATTENTION sur cette ligne
    else:
        roi["centroid_local"] = (0.0, 0.0)

    roi["E"]["X0"] = roi["centroid_local"][0]
    roi["E"]["Y0"] = roi["centroid_local"][1]

    # Inscribed and circumscribed ellipses
    theta_rad = roi["E"]["theta"] * np.pi / 180.0
    roi["E_in"] = fit_ellipse_inside(roi["x"], roi["y"], roi["x_perim"], roi["y_perim"], theta_rad)
    roi["E_out"] = fit_ellipse_around(roi["x"], roi["y"], roi["hull"]["xh"], roi["hull"]["yh"], theta_rad)

    # Rectangularity
    roi["Rect"] = compute_rectangularity(roi["x"], roi["y"], roi["perim"])

    # Shape descriptors
    if roi["E_out"]["a"] > 0 and roi["E_out"]["b"] > 0:
        roi["compactness"] = roi["area"] / (np.pi * roi["E_out"]["a"] * roi["E_out"]["b"])
    else:
        roi["compactness"] = 0.0

    if roi["C_out"]["A"] > 0:
        roi["roundness"] = roi["area"] / roi["C_out"]["A"]
    else:
        roi["roundness"] = 0.0

    # Skeleton properties
    roi["skel"] = skeleton_props(roi["data"])

    # Fractal dimension
    roi["F"] = fractal_dim(roi["data"])
    roi["F_jac"] = 2.0 * np.log(roi["perim"] / 4.0) / np.log(roi["area"])

    # Symmetry features
    roi["Sym"] = compute_symmetry_features(roi["bw_mask_filled"], roi["Dmax"], roi["eq_radius"])

    # Haralick texture features
    roi["H"] = haralick_props(roi["data"])

    # More textural descriptors using fmeasure
    roi["lap"] = fmeasure(roi["data"], "LAPM", None)
    roi["area_lap"] = roi["lap"] * roi["area"]
    roi["hist_entropy"] = fmeasure(roi["data"], "HISE", None)
    roi["wavs"] = fmeasure(roi["data"], "WAVS", None)

    # Standard deviation features
    pixels_mask = roi["data"][roi["bw_mask_filled"]]
    if len(pixels_mask) > 0:
        roi["std"] = float(np.std(pixels_mask))
    else:
        roi["std"] = 0.0

    # Local standard deviation with different window sizes
    local_std = stdfilt(roi["data"], size=3)
    roi["local_std"] = float(np.mean(local_std[roi["bw_mask_filled"]])) if np.any(roi["bw_mask_filled"]) else 0.0

    local_std5 = stdfilt(roi["data"], size=5)
    roi["local_std5"] = float(np.mean(local_std5[roi["bw_mask_filled"]])) if np.any(roi["bw_mask_filled"]) else 0.0

    local_std7 = stdfilt(roi["data"], size=7)
    roi["local_std7"] = float(np.mean(local_std7[roi["bw_mask_filled"]])) if np.any(roi["bw_mask_filled"]) else 0.0

    # Combined metrics
    roi["range_complex"] = roi["complex"] * roi["range_intens"]
    # roi['min_intens'] done above
    # roi['contrast'] done above

    # Brightness adjustment if requested
    roi["new"] = {}
    if process.flakebrighten:
        roi["new"]["data"] = brightening(roi["data"])

        # Recompute textural descriptors on brightened image
        roi["new"]["lap"] = fmeasure(roi["new"]["data"], "LAPM", None)
        roi["new"]["area_lap"] = roi["new"]["lap"] * roi["area"]

        range_array_new = rangefilt(roi["new"]["data"], size=3)
        roi["new"]["range_intens"] = (
            float(np.mean(range_array_new[roi["bw_mask_filled"]])) / 255.0 if np.any(roi["bw_mask_filled"]) else 0.0
        )
        roi["new"]["range_complex"] = roi["complex"] * roi["new"]["range_intens"]

        pixels_new = roi["new"]["data"][roi["bw_mask_filled"]]
        if len(pixels_new) > 0:
            roi["new"]["std"] = float(np.std(pixels_new))

            local_std_new = stdfilt(roi["new"]["data"], size=3)
            roi["new"]["local_std"] = (
                float(np.mean(local_std_new[roi["bw_mask_filled"]])) if np.any(roi["bw_mask_filled"]) else 0.0
            )

            local_std5_new = stdfilt(roi["new"]["data"], size=5)
            roi["new"]["local_std5"] = (
                float(np.mean(local_std5_new[roi["bw_mask_filled"]])) if np.any(roi["bw_mask_filled"]) else 0.0
            )

            local_std7_new = stdfilt(roi["new"]["data"], size=7)
            roi["new"]["local_std7"] = (
                float(np.mean(local_std7_new[roi["bw_mask_filled"]])) if np.any(roi["bw_mask_filled"]) else 0.0
            )

            max_new = float(np.max(pixels_new))
            min_new = float(np.min(pixels_new))
            mean_new = float(np.mean(pixels_new))
            roi["new"]["contrast"] = (max_new - min_new) / mean_new if mean_new > 0 else 0.0

        else:
            roi["new"]["std"] = 0.0
            roi["new"]["local_std"] = 0.0
            roi["new"]["local_std5"] = 0.0
            roi["new"]["local_std7"] = 0.0
            roi["new"]["contrast"] = 0.0

    else:  ## ATTENTION Pas forcément nécessaire, à vérifier le comportement
        # Brightening disabled: copy original values
        roi["new"]["data"] = roi["data"].copy()
        roi["new"]["lap"] = roi["lap"]
        roi["new"]["area_lap"] = roi["area_lap"]
        roi["new"]["range_intens"] = roi["range_intens"]
        roi["new"]["range_complex"] = roi["range_complex"]
        roi["new"]["std"] = roi["std"]
        roi["new"]["local_std"] = roi["local_std"]
        roi["new"]["local_std5"] = roi["local_std5"]
        roi["new"]["local_std7"] = roi["local_std7"]
        roi["new"]["contrast"] = roi["contrast"]

    # Compute magic quality parameter (xhi)
    avg_lap = (roi["lap"] + roi["new"]["lap"]) / 2.0
    avg_local_std = (roi["local_std"] + roi["new"]["local_std"]) / 2.0

    # Ensure all components are positive before taking log (matches MATLAB behavior)
    if avg_lap > 0 and roi["complex"] > 0 and avg_local_std > 0 and roi["Dmean"] > 0:
        roi["xhi"] = np.log(avg_lap * roi["complex"] * avg_local_std * roi["Dmean"])
    else:
        roi["xhi"] = 0.0  # Default value when log cannot be computed

    return roi
