"""
Advanced feature extraction module for MASC analysis.

This module provides functions for extracting advanced features from
snowflake ROIs, including additional texture features and blur indices.

Translated from process_new_descriptors.m

Features computed:
- Blur index (based on Gaussian filtering)
- Additional texture features if missing
- Conditional feature updates (hull, symmetry, D90, etc.)
"""

from typing import Dict

from src.features.fmeasure import fmeasure
from src.utils.descriptors_heplers import (
    compute_blur_index,
    compute_convex_hull,
    compute_D90,
    compute_Dmax,
    compute_symmetry_features,
)


def process_new_descriptors(roi: Dict) -> Dict:
    """
    Add missing advanced descriptors to an existing ROI dictionary.

    This function checks for missing fields in the ROI dictionary and computes
    them if necessary. This is useful for updating older ROI files that were
    processed with earlier versions of the code.

    Translated from process_new_descriptors.m logic.

    Args:
        roi: ROI dictionary (typically loaded from .mat file or from basic processing)

    Returns
    -------
        Updated ROI dictionary with all advanced descriptors
    """
    is_modified = False

    # Convex hull (if missing)
    if "hull" not in roi:
        if "x" in roi and "y" in roi and "perim" in roi:
            roi["hull"] = compute_convex_hull(roi["x"], roi["y"])
            is_modified = True

    # Symmetry features (if missing)
    if "Sym" not in roi:
        if "bw_mask_filled" in roi and "Dmax" in roi and "eq_radius" in roi:
            roi["Sym"] = compute_symmetry_features(
                roi["bw_mask_filled"],
                roi["Dmax"],
                roi["eq_radius"],
            )
            is_modified = True

    # Wavelet sharpness (if missing)
    if "wavs" not in roi:
        if "data" in roi:
            roi["wavs"] = fmeasure(roi["data"], "WAVS", None)
            is_modified = True

    # Histogram entropy (if missing)
    if "hist_entropy" not in roi:
        if "data" in roi:
            roi["hist_entropy"] = fmeasure(roi["data"], "HISE", None)
            is_modified = True

    # Area × range (if missing)
    if "area_range" not in roi:
        if "area" in roi and "range_intens" in roi:
            roi["area_range"] = roi["area"] * roi["range_intens"]
            is_modified = True

    # Blur index (if missing)
    if "blur_idx" not in roi:
        if "data" in roi:
            roi["blur_idx"] = compute_blur_index(roi["data"])
            is_modified = True

    # D90 and updated Dmax (if missing)
    if "D90" not in roi:
        if "hull" in roi and "bw_mask_filled" in roi:
            # Recompute Dmax from hull
            Dmax, Dmax_theta, DmaxA, DmaxB = compute_Dmax(
                roi["hull"]["xh"],
                roi["hull"]["yh"],
            )
            roi["Dmax"] = Dmax
            roi["Dmax_theta"] = Dmax_theta
            roi["DmaxA"] = DmaxA
            roi["DmaxB"] = DmaxB

            # Compute D90
            roi["D90"] = compute_D90(roi["bw_mask_filled"], Dmax, Dmax_theta)
            is_modified = True

    return roi
