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

import numpy as np
import cv2

from typing import Dict, Tuple
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import morphology
from skimage.feature import graycomatrix, graycoprops


def compute_convex_hull(x: np.ndarray, y: np.ndarray, perim: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Compute convex hull of a set of points.
    Translated from compute_convex_hull.m

    Returns:
        Dictionary with keys: xh, yh, solidity, perim, convexity
    """
    x = x.ravel()
    y = y.ravel()

    if len(x) < 3:
        return {'xh': x, 'yh': y, 'solidity': 0.0, 'perim': 0.0, 'convexity': 0.0}

    points = np.column_stack([x, y])
    try:
        hull = ConvexHull(points)
        # MATLAB convhull returns indices that close the polygon (first == last)
        k = np.append(hull.vertices, hull.vertices[0])
        xh = x[k]
        yh = y[k]
    except Exception:
        return {'xh': x, 'yh': y, 'solidity': 0.0, 'perim': 0.0, 'convexity': 0.0}

    # solidity = n_pixels / polyarea(hull)
    hull_area = float(np.abs(np.dot(xh[:-1], yh[1:]) - np.dot(xh[1:], yh[:-1])) / 2.0)
    solidity = len(x) / hull_area if hull_area > 0 else 0.0

    # hull perimeter
    hull_perim = 0.0
    for i in range(len(xh) - 1):
        hull_perim += np.sqrt((xh[i+1] - xh[i])**2 + (yh[i+1] - yh[i])**2)

    return {
        'xh': xh,
        'yh': yh,
        'solidity': float(solidity),
        'perim': float(hull_perim),
        'convexity': float(hull_perim / perim) if perim > 0 else 0.0
    }


def compute_Dmax(xh: np.ndarray, yh: np.ndarray) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """
    Compute maximum diameter (Dmax) from convex hull points.
    
    Args:
        xh: x coordinates of convex hull vertices
        yh: y coordinates of convex hull vertices
        
    Returns:
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
            dist = np.sqrt((xh[i] - xh[j])**2 + (yh[i] - yh[j])**2)
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
        
    Returns:
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
    Translated from fit_circle_around.m (heuristic iterative algorithm).
    """
    xh = xh.ravel()
    yh = yh.ravel()

    if len(xh) < 2:
        return {'X0': 0.0, 'Y0': 0.0, 'r': 0.0, 'A': 0.0}

    stepsize = 0.1
    tol = 1.0

    x0_new = float(np.mean(xh))
    y0_new = float(np.mean(yh))
    r_old = np.inf
    x0_old, y0_old = x0_new, y0_new

    for _ in range(10000):  # safety cap
        xd = xh - x0_new
        yd = yh - y0_new
        r_vec = np.sqrt(xd**2 + yd**2)
        r_new = float(np.max(r_vec))

        if r_new < r_old:
            idx_support = np.where(r_vec >= r_new - tol)[0]
            mx = float(np.mean(xd[idx_support]))
            my = float(np.mean(yd[idx_support]))
            n = np.sqrt(mx**2 + my**2)
            if n == 0:
                break
            normal = stepsize * np.array([mx, my]) / n
            x0_old, y0_old = x0_new, y0_new
            x0_new += normal[0]
            y0_new += normal[1]
            r_old = r_new
        else:
            break

    r = r_old
    return {'X0': x0_old, 'Y0': y0_old, 'r': r, 'A': float(np.pi * r**2)}


def fit_ellipse_inside(x: np.ndarray, y: np.ndarray, x_perim: np.ndarray, y_perim: np.ndarray, theta: float) -> Dict[str, float]:
    """
    Fit largest inscribed ellipse.
    Translated from fit_ellipse_inside.m (focal-distance optimization).

    Searches over focal distances d to maximize ellipse area while keeping
    all perimeter points outside or on the ellipse.
    """
    x = x.ravel().astype(float)
    y = y.ravel().astype(float)
    xP = x_perim.ravel().astype(float)
    yP = y_perim.ravel().astype(float)

    centroid = np.array([np.mean(x), np.mean(y)])
    xPc = xP - centroid[0]
    yPc = yP - centroid[1]

    theta_eff = -theta  # MATLAB applies theta = -theta

    # Untilt the perimeter points
    cos_t = np.cos(-theta_eff)
    sin_t = np.sin(-theta_eff)
    xPc0 = cos_t * xPc - sin_t * yPc
    yPc0 = sin_t * xPc + cos_t * yPc

    maxd = float(np.max(xPc0) - np.min(xPc0))
    if maxd <= 0:
        return {'a': 0.0, 'b': 0.0, 'theta': -theta_eff, 'X0': centroid[0], 'Y0': centroid[1], 'A': 0.0}

    d_arr = np.arange(0, maxd + 1, 1.0)
    s_arr = np.zeros(len(d_arr))
    A_arr = np.zeros(len(d_arr))

    for i, d in enumerate(d_arr):
        # sum of distances from each perimeter point to the two foci
        s_vals = (np.sqrt((xPc0 - 0.5*d)**2 + yPc0**2) +
                  np.sqrt((xPc0 + 0.5*d)**2 + yPc0**2))
        s = float(np.min(s_vals))  # smallest sum = tightest constraint
        s_arr[i] = s
        s2d2 = s**2 - d**2
        A_arr[i] = np.pi * s / 4.0 * np.sqrt(s2d2) if s2d2 > 0 else 0.0

    idx = int(np.argmax(A_arr))
    s = s_arr[idx]
    d = d_arr[idx]
    a = s / 2.0
    b = float(np.sqrt(max((s/2.0)**2 - (d/2.0)**2, 0.0)))

    return {
        'a': a,
        'b': b,
        'theta': -theta_eff,
        'X0': float(centroid[0]),
        'Y0': float(centroid[1]),
        'A': float(np.pi * a * b)
    }


def fit_ellipse_around(x: np.ndarray, y: np.ndarray, xh: np.ndarray, yh: np.ndarray, theta: float) -> Dict[str, float]:
    """
    Fit smallest circumscribed ellipse around convex hull.
    Translated from fit_ellipse_around.m (focal-distance optimization).

    Searches over focal distances d to minimize ellipse area while keeping
    all hull points inside or on the ellipse.
    """
    x = x.ravel().astype(float)
    y = y.ravel().astype(float)
    xh = xh.ravel().astype(float)
    yh = yh.ravel().astype(float)

    centroid = np.array([np.mean(x), np.mean(y)])
    xc = x - centroid[0]
    yc = y - centroid[1]

    theta_eff = -theta  # MATLAB applies theta = -theta

    # Untilt data and hull
    cos_t = np.cos(-theta_eff)
    sin_t = np.sin(-theta_eff)
    xc0 = cos_t * xc - sin_t * yc

    xhc = xh - centroid[0]
    yhc = yh - centroid[1]
    xhc0 = cos_t * xhc - sin_t * yhc
    yhc0 = sin_t * xhc + cos_t * yhc

    maxd = float(np.max(xc0) - np.min(xc0))
    if maxd <= 0:
        return {'a': 0.0, 'b': 0.0, 'theta': -theta_eff, 'X0': centroid[0], 'Y0': centroid[1], 'A': 0.0}

    d_arr = np.arange(0, maxd + 1, 1.0)
    s_arr = np.zeros(len(d_arr))
    A_arr = np.zeros(len(d_arr))

    for i, d in enumerate(d_arr):
        # sum of distances from each hull point to the two foci
        s_vals = (np.sqrt((xhc0 - 0.5*d)**2 + yhc0**2) +
                  np.sqrt((xhc0 + 0.5*d)**2 + yhc0**2))
        s = float(np.max(s_vals))  # largest sum = tightest constraint
        s_arr[i] = s
        s2d2 = s**2 - d**2
        A_arr[i] = np.pi * s / 4.0 * np.sqrt(s2d2) if s2d2 > 0 else np.inf

    idx = int(np.argmin(A_arr))
    s = s_arr[idx]
    d = d_arr[idx]
    a = s / 2.0
    b = float(np.sqrt(max((s/2.0)**2 - (d/2.0)**2, 0.0)))

    return {
        'a': a,
        'b': b,
        'theta': -theta_eff,
        'X0': float(centroid[0]),
        'Y0': float(centroid[1]),
        'A': float(np.pi * a * b)
    }


def compute_rectangularity(x: np.ndarray, y: np.ndarray, perim: float) -> Dict[str, np.ndarray]:
    """
    Compute smallest enclosing rectangle and rectangularity features.
    
    Args:
        x: x coordinates of all ROI points
        y: y coordinates of all ROI points
        perim: perimeter of ROI
        
    Returns:
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
            'rectx': np.array([]),
            'recty': np.array([]),
            'width': 0.0, 
            'height': 0.0, 
            'area': 0.0, 
            'rectangularity': 0.0
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
        'rectx': rectx,
        'recty': recty,
        'width': width,
        'height': height,
        'area': area,
        'rectangularity': rectangularity
    }


def skeleton_props(data: np.ndarray) -> Dict[str, float]:
    """
    Compute skeleton-based properties.
    
    Args:
        data: ROI image data
        
    Returns:
        Dictionary with keys:
            - length: total skeleton length (number of skeleton pixels)
            - density: skeleton density (length / total_pixels)
            - branches: number of branch points
            - endpoints: number of endpoints
    """
    if data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        return {'length': 0.0, 'density': 0.0, 'branches': 0, 'endpoints': 0}
    
    # Create binary mask
    mask = data > 0
    total_pixels = float(np.sum(mask))
    
    if total_pixels == 0:
        return {'length': 0.0, 'density': 0.0, 'branches': 0, 'endpoints': 0}
    
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
        
        neighbor_count = ndimage.convolve(skel.astype(np.uint8), kernel, mode='constant')
        
        # Branch points: >2 neighbors
        branches = int(np.sum((neighbor_count > 2) & skel))
        
        # Endpoints: exactly 1 neighbor
        endpoints = int(np.sum((neighbor_count == 1) & skel))
        
        return {
            'length': length,
            'density': density,
            'branches': branches,
            'endpoints': endpoints
        }
    except Exception:
        return {'length': 0.0, 'density': 0.0, 'branches': 0, 'endpoints': 0}


def fractal_dim(data: np.ndarray) -> float:
    """
    Compute fractal dimension using box-counting method.
    
    Args:
        data: ROI image data
        
    Returns:
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
                box = mask[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
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

    Matches masclab/texture/haralick_props.m exactly:
      - 256 gray levels, no quantization
      - 4 directions (0°, 45°, 90°, 135°), distance = 1, symmetric
      - Background co-occurrences (pixel value 0) excluded by slicing
        glcm[1:, 1:] — equivalent to MATLAB's glcm(pix_value_thresh:end, ...)
        with pix_value_thresh = 2
      - Mean over the 4 directions

    Returns:
        Dictionary with keys 'Contrast', 'Correlation', 'Energy', 'Homogeneity'
    """
    _zero = {'Contrast': 0.0, 'Correlation': 0.0, 'Energy': 0.0, 'Homogeneity': 0.0}

    if data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        return _zero

    if not np.any(data > 0):
        return _zero

    # Ensure uint8 in [0, 255] — no quantization (matches MATLAB NumLevels=256)
    data_u8 = np.clip(data, 0, 255).astype(np.uint8)

    try:
        # Compute un-normalised GLCM; normalisation happens inside graycoprops
        # after we remove the background slice (same as MATLAB graycoprops on
        # the sliced matrix)
        glcm = graycomatrix(data_u8, distances=[1],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=False)

        # Remove background (pixel value 0) co-occurrences — matches
        # MATLAB: glcm(pix_value_thresh:end, pix_value_thresh:end), thresh=2
        glcm = glcm[1:, 1:, :, :]  # (255, 255, 1, 4)

        return {
            'Contrast':    float(np.mean(graycoprops(glcm, 'contrast'))),
            'Correlation': float(np.mean(graycoprops(glcm, 'correlation'))),
            'Energy':      float(np.mean(graycoprops(glcm, 'energy'))),
            'Homogeneity': float(np.mean(graycoprops(glcm, 'homogeneity'))),
        }
    except Exception:
        return _zero


def compute_symmetry_features(mask_filled: np.ndarray, Dmax: float, 
                              eq_radius: float) -> Dict[str, float]:
    """
    Compute symmetry features.
    
    Args:
        mask_filled: Filled binary mask
        Dmax: Maximum diameter
        eq_radius: Equivalent radius
        
    Returns:
        Dictionary with symmetry metrics:
            - horizontal: horizontal symmetry score
            - vertical: vertical symmetry score
            - mean: average of horizontal and vertical symmetry
    """
    if mask_filled.size == 0 or Dmax == 0:
        return {'horizontal': 0.0, 'vertical': 0.0, 'mean': 0.0}
    
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
        'horizontal': horizontal_sym,
        'vertical': vertical_sym,
        'mean': mean_sym
    }


def compute_blur_index(data: np.ndarray) -> Dict[str, float]:
    """
    Compute blur index based on Gaussian filtering.
    
    The blur index quantifies image blur by measuring the difference
    between the original image and Gaussian-filtered versions.
    Higher values indicate sharper images.
    
    Args:
        data: ROI image data (grayscale uint8)
        
    Returns:
        Dictionary with keys:
            - xhi2: std of difference between original and sigma=2 Gaussian blur
            - xhi4: std of difference between original and sigma=4 Gaussian blur
    """
    if data.size == 0 or data.shape[0] < 5 or data.shape[1] < 5:
        return {'xhi2': 0.0, 'xhi4': 0.0}
    
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
        'xhi2': xhi2,
        'xhi4': xhi4
    }
