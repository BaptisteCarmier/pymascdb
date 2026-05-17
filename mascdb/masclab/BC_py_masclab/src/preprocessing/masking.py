"""
Image masking module for MASC analysis.

This module creates a dark background for MASC images by removing clutter
and applying thresholds based on camera position.

Translated from masking.m (Christophe Praz 2015) and adapted for Python.
Last update: November 2025
"""

import numpy as np

from src.core.config import ProcessingConfig

def masking(flake_data: np.ndarray, flake_cam: int, process: ProcessingConfig) -> np.ndarray:
    """
    Create a dark background for MASC images.
    
    Args:
        flake_data: MASC picture (2D matrix)
        flake_cam: ID of the camera (1=left LED, 2=center, 3=right LED)
        process: Processing parameters
        
    Returns:
        MASC picture with dark background mask applied
        
    Notes:
        - For left LED camera (cam=0): masks left side
        - For right LED camera (cam=2): masks right side
        - For cameras 0, 1, 2: masks top and bottom
        - Applies threshold to remove low luminosity pixels
    """
    # Make a copy to avoid modifying the input
    data = flake_data.copy()
    
    # Remove clutter according to the discard matrix based on camera position.
    # Guard every strip: NumPy treats slices like data[-0:, :] / data[:, -0:] as
    # "from index 0 to end" i.e. the whole axis — so discardmat[k]==0 must mean
    # "do not discard", not "blank the entire image".

    t, b, l, r = process.discardmat[0], process.discardmat[1], process.discardmat[2], process.discardmat[3]

    if flake_cam == 0 and l > 0:  # LED on left
        data[:, :l] = 0

    elif flake_cam == 2 and r > 0:  # LED on right
        data[:, -r:] = 0

    if t > 0:
        data[:t, :] = 0  # Top
    if b > 0:
        data[-b:, :] = 0  # Bottom

    # Remove clutter below luminosity threshold
    data[data <= process.backthresh] = 0
    
    return data