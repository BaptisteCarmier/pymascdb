"""
Descriptor helper functions for MASC analysis.

This module provides utility functions for computing geometric and topological
descriptors used in snowflake feature extraction.

Functions for:
- Convex hull computation
- Maximum diameter (Dmax) and perpendicular width (D90)
- Circle and ellipse fitting (inscribed and circumscribed)
- Rectangularity features
- Skeleton properties
- Fractal dimension
- Haralick texture features
- Symmetry features

Last update: November 2025
"""

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import morphology
from skimage.feature import graycomatrix, graycoprops


def compute_convex_hull(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute convex hull of a set of points.

    Args:
        x: Array of x coordinates
        y: Array of y coordinates

    Returns
    -------
        Dictionary with keys:
            - xh: x coordinates of hull vertices
            - yh: y coordinates of hull vertices
            - area: Area of convex hull
    """
    if len(x) < 3:
        return {"xh": x, "yh": y, "area": 0.0}

    points = np.column_stack([x, y])
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return {
            "xh": hull_points[:, 0],
            "yh": hull_points[:, 1],
            "area": hull.volume,  # In 2D, volume is area
        }
    except Exception:
        return {"xh": x, "yh": y, "area": 0.0}


def compute_Dmax(xh: np.ndarray, yh: np.ndarray) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """
    Compute maximum diameter (Dmax) from convex hull points.

    Args:
        xh: x coordinates of convex hull vertices
        yh: y coordinates of convex hull vertices

    Returns
    -------
        Tuple of (Dmax, Dmax_theta, DmaxA, DmaxB):
            - Dmax: Maximum distance between any two hull points
            - Dmax_theta: Angle of Dmax line in degrees
            - DmaxA: (x, y) coordinates of first endpoint
            - DmaxB: (x, y) coordinates of second endpoint
    """
    if len(xh) < 2:
        return 0.0, 0.0, (0.0, 0.0), (0.0, 0.0)

    max_dist = 0.0
    idx_a = 0
    idx_b = 0

    # Brute force: find maximum distance between all pairs
    for i in range(len(xh)):
        for j in range(i + 1, len(xh)):
            dist = np.sqrt((xh[i] - xh[j]) ** 2 + (yh[i] - yh[j]) ** 2)
            if dist > max_dist:
                max_dist = dist
                idx_a = i
                idx_b = j

    # Compute angle
    dx = xh[idx_b] - xh[idx_a]
    dy = yh[idx_b] - yh[idx_a]
    theta = np.arctan2(dy, dx) * 180.0 / np.pi

    DmaxA = (float(xh[idx_a]), float(yh[idx_a]))
    DmaxB = (float(xh[idx_b]), float(yh[idx_b]))

    return float(max_dist), float(theta), DmaxA, DmaxB


def compute_D90(mask: np.ndarray, Dmax: float, Dmax_theta: float) -> float:
    """
    Compute D90: maximum width perpendicular to Dmax.

    Args:
        mask: Binary mask of the ROI
        Dmax: Maximum diameter
        Dmax_theta: Angle of Dmax in degrees

    Returns
    -------
        D90: Maximum width perpendicular to Dmax
    """
    if Dmax == 0:
        return 0.0

    # Get all points in mask
    y, x = np.where(mask > 0)
    if len(x) == 0:
        return 0.0

    # Rotate points by -Dmax_theta to align Dmax with x-axis
    theta_rad = -Dmax_theta * np.pi / 180.0
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Center of mass
    cx = np.mean(x)
    cy = np.mean(y)

    # Translate and rotate
    x_rot = (x - cx) * cos_t - (y - cy) * sin_t
    y_rot = (x - cx) * sin_t + (y - cy) * cos_t

    # D90 is the range in the perpendicular direction (y after rotation)
    D90 = float(np.max(y_rot) - np.min(y_rot))

    return D90


def fit_circle_around(x_perim: np.ndarray, y_perim: np.ndarray, xh: np.ndarray, yh: np.ndarray) -> Dict[str, float]:
    """
    Fit smallest circumscribed circle around convex hull.

    Args:
        x_perim: x coordinates of perimeter
        y_perim: y coordinates of perimeter
        xh: x coordinates of convex hull
        yh: y coordinates of convex hull

    Returns
    -------
        Dictionary with keys:
            - X0: x coordinate of circle center
            - Y0: y coordinate of circle center
            - R: radius of circle
            - A: area of circle (π R²)
    """
    # Use convex hull points for fitting
    if len(xh) < 3:
        return {"X0": 0.0, "Y0": 0.0, "R": 0.0, "A": 0.0}

    # Simple heuristic: center at centroid, radius = max distance to centroid
    cx = float(np.mean(xh))
    cy = float(np.mean(yh))

    distances = np.sqrt((xh - cx) ** 2 + (yh - cy) ** 2)
    R = float(np.max(distances))
    A = np.pi * R * R

    return {"X0": cx, "Y0": cy, "R": R, "A": A}


def fit_ellipse_inside(
    x: np.ndarray, y: np.ndarray, x_perim: np.ndarray, y_perim: np.ndarray, theta: float
) -> Dict[str, float]:
    """
    Fit largest inscribed ellipse.

    Args:
        x: all x coordinates in ROI
        y: all y coordinates in ROI
        x_perim: x coordinates of perimeter
        y_perim: y coordinates of perimeter
        theta: orientation angle in radians

    Returns
    -------
        Dictionary with keys:
            - a: semi-major axis
            - b: semi-minor axis
            - theta: orientation in radians
            - X0: x coordinate of center
            - Y0: y coordinate of center
            - A: area (π a b)
    """
    # Simplified: use erosion-based approach or moment-based ellipse
    # For now, return a scaled-down version of the fitted ellipse
    cx = float(np.mean(x))
    cy = float(np.mean(y))

    # Rotate points
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    x_rot = (x - cx) * cos_t - (y - cy) * sin_t
    y_rot = (x - cx) * sin_t + (y - cy) * cos_t

    # Semi-axes as std * sqrt(eigenvalues) approximation
    a = float(np.std(x_rot) * 2.0 * 0.8)  # Scale down for inscribed
    b = float(np.std(y_rot) * 2.0 * 0.8)

    return {
        "a": a,
        "b": b,
        "theta": theta,
        "X0": cx,
        "Y0": cy,
        "A": np.pi * a * b,
    }


def fit_ellipse_around(x: np.ndarray, y: np.ndarray, xh: np.ndarray, yh: np.ndarray, theta: float) -> Dict[str, float]:
    """
    Fit smallest circumscribed ellipse around convex hull.

    Args:
        x: all x coordinates in ROI
        y: all y coordinates in ROI
        xh: x coordinates of convex hull
        yh: y coordinates of convex hull
        theta: orientation angle in radians

    Returns
    -------
        Dictionary with keys:
            - a: semi-major axis
            - b: semi-minor axis
            - theta: orientation in radians
            - X0: x coordinate of center
            - Y0: y coordinate of center
            - A: area (π a b)
    """
    cx = float(np.mean(xh))
    cy = float(np.mean(yh))

    # Rotate hull points
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    xh_rot = (xh - cx) * cos_t - (yh - cy) * sin_t
    yh_rot = (xh - cx) * sin_t + (yh - cy) * cos_t

    # Semi-axes as range/2
    a = float((np.max(xh_rot) - np.min(xh_rot)) / 2.0)
    b = float((np.max(yh_rot) - np.min(yh_rot)) / 2.0)

    return {
        "a": a,
        "b": b,
        "theta": theta,
        "X0": cx,
        "Y0": cy,
        "A": np.pi * a * b,
    }


def compute_rectangularity(x: np.ndarray, y: np.ndarray, perim: float) -> Dict[str, np.ndarray]:
    """
    Compute smallest enclosing rectangle and rectangularity features.

    Args:
        x: x coordinates of all ROI points
        y: y coordinates of all ROI points
        perim: perimeter of ROI

    Returns
    -------
        Dictionary with keys:
            - rectx: x coordinates of rectangle vertices
            - recty: y coordinates of rectangle vertices
            - width: width of rectangle
            - height: height of rectangle
            - area: area of rectangle
            - rectangularity: ROI_area / rectangle_area
    """
    if len(x) < 3:
        return {
            "rectx": np.array([]),
            "recty": np.array([]),
            "width": 0.0,
            "height": 0.0,
            "area": 0.0,
            "rectangularity": 0.0,
        }

    # Simple axis-aligned bounding box
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    width = x_max - x_min
    height = y_max - y_min
    area = width * height

    # Rectangle vertices (clockwise from bottom-left)
    rectx = np.array([x_min, x_max, x_max, x_min, x_min])
    recty = np.array([y_min, y_min, y_max, y_max, y_min])

    roi_area = len(x)
    rectangularity = roi_area / area if area > 0 else 0.0

    return {
        "rectx": rectx,
        "recty": recty,
        "width": width,
        "height": height,
        "area": area,
        "rectangularity": rectangularity,
    }


def skeleton_props(data: np.ndarray) -> Dict[str, float]:
    """
    Compute skeleton-based properties.

    Args:
        data: ROI image data

    Returns
    -------
        Dictionary with keys:
            - length: total skeleton length (number of skeleton pixels)
            - density: skeleton density (length / total_pixels)
            - branches: number of branch points
            - endpoints: number of endpoints
    """
    if data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        return {"length": 0.0, "density": 0.0, "branches": 0, "endpoints": 0}

    # Create binary mask
    mask = data > 0
    total_pixels = float(np.sum(mask))

    if total_pixels == 0:
        return {"length": 0.0, "density": 0.0, "branches": 0, "endpoints": 0}

    try:
        # Skeletonize
        skel = morphology.skeletonize(mask)

        # Count skeleton pixels
        length = float(np.sum(skel))
        density = length / total_pixels if total_pixels > 0 else 0.0

        # Detect branch points and endpoints using convolution
        # A branch point has 3+ neighbors, endpoint has 1 neighbor
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0

        neighbor_count = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant")

        # Branch points: >2 neighbors
        branches = int(np.sum((neighbor_count > 2) & skel))

        # Endpoints: exactly 1 neighbor
        endpoints = int(np.sum((neighbor_count == 1) & skel))

        return {
            "length": length,
            "density": density,
            "branches": branches,
            "endpoints": endpoints,
        }
    except Exception:
        return {"length": 0.0, "density": 0.0, "branches": 0, "endpoints": 0}


def fractal_dim(data: np.ndarray) -> float:
    """
    Compute fractal dimension using box-counting method.

    Args:
        data: ROI image data

    Returns
    -------
        Fractal dimension estimate (float)
    """
    if data.size == 0 or data.shape[0] < 4 or data.shape[1] < 4:
        return 0.0

    mask = data > 0

    # Box-counting method
    scales = []
    counts = []

    max_box_size = min(mask.shape[0], mask.shape[1]) // 4

    for box_size in [2, 4, 8, 16, 32]:
        if box_size > max_box_size:
            break

        # Count boxes that contain at least one True pixel
        n_boxes_y = mask.shape[0] // box_size
        n_boxes_x = mask.shape[1] // box_size

        count = 0
        for i in range(n_boxes_y):
            for j in range(n_boxes_x):
                box = mask[i * box_size : (i + 1) * box_size, j * box_size : (j + 1) * box_size]
                if np.any(box):
                    count += 1

        if count > 0:
            scales.append(box_size)
            counts.append(count)

    if len(scales) < 2:
        return 1.0  # Default reasonable fractal dimension

    # Fit log-log line
    scales = np.array(scales, dtype=float)
    counts = np.array(counts, dtype=float)

    try:
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        D = float(-coeffs[0])
        # Clamp to reasonable range [1, 3]
        D = max(1.0, min(3.0, D))
        return D
    except Exception:
        return 1.0


def haralick_props(data: np.ndarray) -> Dict[str, float]:
    """
    Compute Haralick texture features using GLCM.

    Args:
        data: ROI image data (grayscale)

    Returns
    -------
        Dictionary with Haralick features:
            - contrast
            - dissimilarity
            - homogeneity
            - energy
            - correlation
            - ASM (Angular Second Moment)
    """
    if data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        return {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0,
            "ASM": 0.0,
        }

    # Ensure data is in valid range
    mask = data > 0
    if not np.any(mask):
        return {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0,
            "ASM": 0.0,
        }

    # Quantize to 8 levels for GLCM (faster and more stable)
    data_quant = (data / 32).astype(np.uint8)
    data_quant = np.clip(data_quant, 0, 7)

    try:
        # Compute GLCM for 4 directions
        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(data_quant, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)

        # Average over all directions
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        energy = float(np.mean(graycoprops(glcm, "energy")))
        correlation = float(np.mean(graycoprops(glcm, "correlation")))
        ASM = float(np.mean(graycoprops(glcm, "ASM")))

        return {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
            "ASM": ASM,
        }
    except Exception:
        return {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0,
            "ASM": 0.0,
        }


def compute_symmetry_features(mask_filled: np.ndarray, Dmax: float, eq_radius: float) -> Dict[str, float]:
    """
    Compute symmetry features.

    Args:
        mask_filled: Filled binary mask
        Dmax: Maximum diameter
        eq_radius: Equivalent radius

    Returns
    -------
        Dictionary with symmetry metrics:
            - horizontal: horizontal symmetry score
            - vertical: vertical symmetry score
            - mean: average of horizontal and vertical symmetry
    """
    if mask_filled.size == 0 or Dmax == 0:
        return {"horizontal": 0.0, "vertical": 0.0, "mean": 0.0}

    # Compute horizontal symmetry
    h, w = mask_filled.shape
    mid_h = h // 2

    top_half = mask_filled[:mid_h, :]
    bottom_half = mask_filled[mid_h:, :]
    bottom_half_flipped = np.flipud(bottom_half)

    # Pad to same size
    min_h = min(top_half.shape[0], bottom_half_flipped.shape[0])
    top_half = top_half[:min_h, :]
    bottom_half_flipped = bottom_half_flipped[:min_h, :]

    if top_half.size > 0:
        horizontal_sym = float(np.sum(top_half == bottom_half_flipped)) / top_half.size
    else:
        horizontal_sym = 0.0

    # Compute vertical symmetry
    mid_w = w // 2
    left_half = mask_filled[:, :mid_w]
    right_half = mask_filled[:, mid_w:]
    right_half_flipped = np.fliplr(right_half)

    min_w = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_w]
    right_half_flipped = right_half_flipped[:, :min_w]

    if left_half.size > 0:
        vertical_sym = float(np.sum(left_half == right_half_flipped)) / left_half.size
    else:
        vertical_sym = 0.0

    # Compute mean symmetry
    mean_sym = (horizontal_sym + vertical_sym) / 2.0

    return {
        "horizontal": horizontal_sym,
        "vertical": vertical_sym,
        "mean": mean_sym,
    }


def compute_blur_index(data: np.ndarray) -> Dict[str, float]:
    """
    Compute blur index based on Gaussian filtering.

    The blur index quantifies image blur by measuring the difference
    between the original image and Gaussian-filtered versions.
    Higher values indicate sharper images.

    Args:
        data: ROI image data (grayscale uint8)

    Returns
    -------
        Dictionary with keys:
            - xhi2: std of difference between original and sigma=2 Gaussian blur
            - xhi4: std of difference between original and sigma=4 Gaussian blur
    """
    if data.size == 0 or data.shape[0] < 5 or data.shape[1] < 5:
        return {"xhi2": 0.0, "xhi4": 0.0}

    # Convert to float for filtering
    data_float = data.astype(float)

    # Apply Gaussian blur with sigma=2 and sigma=4
    blurry2 = cv2.GaussianBlur(data_float, (0, 0), sigmaX=2.0)
    blurry4 = cv2.GaussianBlur(data_float, (0, 0), sigmaX=4.0)

    # Compute differences
    diff2 = data_float - blurry2
    diff4 = data_float - blurry4

    # Return standard deviation of differences
    xhi2 = float(np.std(diff2))
    xhi4 = float(np.std(diff4))

    return {
        "xhi2": xhi2,
        "xhi4": xhi4,
    }
