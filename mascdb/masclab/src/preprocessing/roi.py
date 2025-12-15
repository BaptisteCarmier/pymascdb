"""
ROI (Region of Interest) detection module.

This module handles the detection and validation of regions of interest
in MASC snowflake images.

Translated from roi_detection.m and adapted for Python.
Last update: November 2025
"""

import numpy as np
import cv2

from dataclasses import dataclass
from typing import List, Tuple, Optional
from skimage.measure import label, regionprops

from src.core.config import ProcessingConfig
from src.utils.image import rangefilt

"""
Classes:
    ROI: Class representing a detected region of interest
    ROIDetector: Class for detecting and managing ROIs
"""
@dataclass
class ROI:
    """
    Class representing a detected region of interest.
    
    Attributes:
        bbox: Bounding box as (min_row, min_col, max_row, max_col) from scikit-image
        mask: Binary mask of the ROI (local to bbox region)
        area: Area in pixels
        perimeter: Perimeter in pixels
        centroid: Centroid as (row, col) in full image coordinates
        major_axis_length: Length of major axis of fitted ellipse
        minor_axis_length: Length of minor axis of fitted ellipse
        orientation: Orientation angle in radians (scikit-image convention)
        mean_intensity: Mean intensity [0, 1]
        max_intensity: Maximum intensity [0, 1]
        range_intensity: Average local variability [0, 1]
        focus: Focus quality metric
        area_focus: Area × focus product (used for selection)
        coords: Array of pixel coordinates (n_pixels, 2) where each row is (row, col)
    """
    bbox: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    mask: np.ndarray  # Binary mask local to bbox
    area: float
    perimeter: float
    centroid: Tuple[float, float]  # (row, col)
    major_axis_length: float
    minor_axis_length: float
    orientation: float  # in radians
    mean_intensity: float = 0.0
    max_intensity: float = 0.0
    range_intensity: float = 0.0
    focus: float = 0.0
    area_focus: float = 0.0
    coords: Optional[np.ndarray] = None  # (n_pixels, 2) array of (row, col)


class ROIDetector:
    """
    Class for detecting regions of interest in MASC images.
    
    This class implements the ROI detection algorithm from roi_detection.m,
    which detects all connected components in a binary mask, filters them
    based on size and position, computes quality metrics, and selects the
    best ROI based on area × focus criterion.
    """
    
    def __init__(self, process_config: ProcessingConfig):
        """
        Initialize ROI detector with processing configuration.
        
        Args:
            process_config: Processing configuration with thresholds
        """
        self.config = process_config
        
    def detect(
        self,
        image: np.ndarray,
        edge_mask: np.ndarray,
        cam_id: int,
        compute_best: bool = True
    ) -> Tuple[List[ROI], Optional[int], Optional[float], Optional[str], Optional[str]]:
        """
        Detect all ROIs in image using binary edge mask.
        
        Implements the complete roi_detection.m algorithm using scikit-image regionprops:
        1. Label connected components using skimage.measure.label
        2. Extract regionprops for each component
        3. Filter by minimum area
        4. Filter ROIs touching discarded borders
        5. Compute quality metrics (intensity, focus)
        6. Select best ROI based on area × focus
        7. Validate selected ROI against quality thresholds
        
        Args:
            image: Original grayscale image (uint8)
            edge_mask: Binary mask from edge detection (uint8, 0 or 1)
            cam_id: Camera ID (1=left LED, 2=center, 3=right LED in MATLAB convention)
            compute_best: Whether to compute best ROI index (default: True)
            
        Returns:
            Tuple of:
                - all_roi: List of detected ROIs
                - idx_best: Index of best ROI (None if compute_best=False or no ROIs)
                - area_focus_ratio: Ratio of best to second-best area_focus (None if <2 ROIs)
                - flag: 'GOOD' or 'BAD' validation flag (None if no best ROI)
                - status: Status message describing validation issues (None if no best ROI)
        """
        # Label connected components using scikit-image
        labeled_img = label(edge_mask, connectivity=2)
        
        all_roi = []
        
        # Extract properties for each labeled region (skip label 0 which is background)
        props_list = regionprops(labeled_img)
        
        for prop in props_list:
            # Filter by minimum area early
            if prop.area <= self.config.min_area:
                continue
            
            # Create ROI object with scikit-image properties
            # bbox format: (min_row, min_col, max_row, max_col)
            roi = ROI(
                bbox=prop.bbox,
                mask=prop.image,  # Binary mask local to bbox
                area=float(prop.area),
                perimeter=float(prop.perimeter),
                centroid=prop.centroid,  # (row, col) in full image
                major_axis_length=float(prop.major_axis_length),
                minor_axis_length=float(prop.minor_axis_length),
                orientation=float(prop.orientation),  # in radians
                coords=prop.coords  # (n_pixels, 2) array of (row, col)
            )
            
            all_roi.append(roi)
        
        # Filter ROIs touching the discarded borders
        all_roi = self._filter_border_rois(all_roi, edge_mask.shape, cam_id)
        
        # If not computing best or no ROIs, return early
        if not compute_best or len(all_roi) == 0:
            return all_roi, None, None, None, None
        
        # Compute quality metrics for all ROIs
        for roi in all_roi:
            self._compute_roi_metrics(roi, image)
        
        # Select best ROI based on area × focus
        all_area_focus = [roi.area_focus for roi in all_roi]
        idx_best = int(np.argmax(all_area_focus))
        
        # Compute area_focus ratio (best / second best)
        if len(all_roi) > 1:
            sorted_area_focus = sorted(all_area_focus, reverse=True)
            area_focus_ratio = sorted_area_focus[0] / sorted_area_focus[1]
        else:
            area_focus_ratio = float('inf')
        
        # Validate best ROI
        flag_roi, status = self._validate_roi(all_roi[idx_best])
        
        return all_roi, idx_best, area_focus_ratio, flag_roi, status
    
    def _filter_border_rois(
        self,
        rois: List[ROI],
        image_shape: Tuple[int, int],
        cam_id: int
    ) -> List[ROI]:
        """
        Filter ROIs touching the discarded borders.
        
        Args:
            rois: List of ROIs to filter
            image_shape: Shape of the image (height, width)
            cam_id: Camera ID (1=left LED, 2=center, 3=right LED)
            
        Returns:
            Filtered list of ROIs
            
        Notes:
            MATLAB logic from roi_detection.m:
            - cam=1 (left LED): filters left, top, bottom
            - cam=2 (center): filters only top, bottom
            - cam=3 (right LED): filters right, top, bottom
        """
        if len(rois) == 0:
            return rois
        
        height, width = image_shape
        filtered_rois = []
        
        for roi in rois:
            # bbox is (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = roi.bbox
            
            # Compute distances from borders
            dist_from_top = min_row
            dist_from_bot = height - max_row
            dist_from_left = min_col
            dist_from_right = width - max_col
            
            # Apply filtering based on camera position (MATLAB convention)
            # MATLAB: cam == process.camera_order(1) means cam == 1 (left LED)
            if cam_id == 1:  # Left LED camera
                valid = (dist_from_top > self.config.discardmat[0] + 1 and
                        dist_from_bot > self.config.discardmat[1] + 1 and
                        dist_from_left > self.config.discardmat[2] + 1 and
                        dist_from_right > 1)
            # MATLAB: cam == process.camera_order(3) means cam == 3 (right LED)
            elif cam_id == 3:  # Right LED camera
                valid = (dist_from_top > self.config.discardmat[0] + 1 and
                        dist_from_bot > self.config.discardmat[1] + 1 and
                        dist_from_right > self.config.discardmat[3] + 1 and
                        dist_from_left > 1)
            # MATLAB: cam == process.camera_order(2) means cam == 2 (center)
            elif cam_id == 2:  # Center camera
                valid = (dist_from_top > self.config.discardmat[0] + 1 and
                        dist_from_bot > self.config.discardmat[1] + 1 and
                        dist_from_left > 1 and
                        dist_from_right > 1)
            else:
                # Default: check all borders
                valid = (dist_from_top > self.config.discardmat[0] + 1 and
                        dist_from_bot > self.config.discardmat[1] + 1 and
                        dist_from_left > 1 and
                        dist_from_right > 1)
            
            if valid:
                filtered_rois.append(roi)
        
        return filtered_rois
    
    def _compute_roi_metrics(self, roi: ROI, image: np.ndarray) -> None:
        """
        Compute intensity and focus metrics for an ROI.
        
        Modifies the ROI object in-place with computed metrics.
        
        Args:
            roi: ROI object to compute metrics for
            image: Original grayscale image (uint8)
        """
        # bbox is (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = roi.bbox
        
        # Crop image around ROI bounding box
        cropped_image = image[min_row:max_row, min_col:max_col]
        
        # Get pixels within ROI mask (roi.mask is already local to the bbox)
        roi_pixels = cropped_image[roi.mask > 0]
        
        if len(roi_pixels) == 0:
            return
        
        # Maximal brightness [0, 1]
        roi.max_intensity = float(np.max(roi_pixels)) / 255.0
        
        # Average brightness [0, 1]
        roi.mean_intensity = float(np.mean(roi_pixels)) / 255.0
        
        # Local variability using rangefilt (3x3 box)
        range_array = rangefilt(cropped_image, size=3)
        roi.range_intensity = float(np.mean(range_array[roi.mask > 0])) / 255.0
        
        # Focus parameter: mean_intensity × range_intensity
        roi.focus = roi.mean_intensity * roi.range_intensity
        
        # Area × focus (used for ROI selection)
        roi.area_focus = roi.focus * roi.area
    
    def _validate_roi(self, roi: ROI) -> Tuple[str, str]:
        """
        Validate ROI against quality thresholds.
        
        Args:
            roi: ROI to validate
            
        Returns:
            Tuple of (flag, status):
                - flag: 'GOOD' if passes all checks, 'BAD' otherwise
                - status: String describing any validation failures
        """
        flag = 'GOOD'
        status = ''
        
        # Check maximum size of ROI bounding box
        # bbox is (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = roi.bbox
        height = max_row - min_row
        width = max_col - min_col
        max_size = max(width, height)
        
        if max_size < self.config.sizemin:
            flag = 'BAD'
            status += ' too small.'
        
        # Check mean brightness
        if roi.mean_intensity < self.config.minbright:
            flag = 'BAD'
            status += ' too dark (mean).'
        
        # Check maximum brightness
        if roi.max_intensity < self.config.max_intensthresh:
            flag = 'BAD'
            status += ' too dark (max).'
        
        return flag, status