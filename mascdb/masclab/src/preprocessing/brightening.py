"""
Image brightening module for MASC analysis.

This module provides functions for enhancing the brightness and contrast
of MASC snowflake images using CLAHE histogram equalization algorithm.

Translated from brightening.m (Christophe Praz 2015) and adapted for Python.
Last update: Novemer 2025
"""

import numpy as np
import cv2


def brightening(image_in: np.ndarray) -> np.ndarray:
    """
    Brighten a MASC snowflake picture using CLAHE histogram equalization algorithm.
    
    Args:
        image_in: Input grayscale image as numpy array (uint8)
        
    Returns:
        Brightened image using CLAHE
    
    Notes:
        The function handles small images by padding them to a minimum size
        before applying CLAHE, then cropping back to original size.
    """

    # Fixed tile dimension
    tile_dim = 8
    n_lines, n_cols = image_in.shape
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_dim, tile_dim))
    # Note on createCLAHE parameters : 
    # clipLimit=40.0 and tileGridSize=(8,8) are parameters chosen to match MATLAB's default CLAHE settings (adapthisteq())
    # cliplimit : Threshold for contrast limiting. By default value is 40
    # tileGridSize : Size of the grid for the histogram equalization. Must be a multiple of 8. By default value is (8,8) 

    if n_lines < tile_dim and n_cols < tile_dim:
        # Padding both lines and columns
        image_mod = np.zeros((tile_dim, tile_dim), dtype=np.uint8) # Note : dtype must be uint8, because cv2 functions only work with uint8 images
        image_mod[:n_lines, :n_cols] = image_in
        image_out = clahe.apply(image_mod) # Note : apply would raise error if inptut image is not an uint8 array
        return image_out[:n_lines, :n_cols]
        
    elif n_lines < tile_dim:
        # Padding only lines
        image_mod = np.zeros((tile_dim, n_cols), dtype=np.uint8) 
        image_mod[:n_lines, :n_cols] = image_in
        image_out = clahe.apply(image_mod)
        return image_out[:n_lines, :n_cols]
        
    elif n_cols < tile_dim:
        # Padding only columns
        image_mod = np.zeros((n_lines, tile_dim), dtype=np.uint8)
        image_mod[:n_lines, :n_cols] = image_in
        image_out = clahe.apply(image_mod)
        return image_out[:n_lines, :n_cols]
        
    else:
        # No padding 
        return clahe.apply(image_in)
    
    

# Peut être plus rapide en fonction de comment cv2 gère le padding en interne ??
# A vérifier si gain de temps significatif 

# def brightening(image_in: np.ndarray) -> np.ndarray:
#     """Brighten image using CLAHE algorithm.
    
#     Args:
#         image_in: Input image as numpy array (uint8)
        
#     Returns:
#         Brightened image as numpy array (uint8)
#     """
#     # Create CLAHE object with proper parameters
#     clahe = cv2.createCLAHE(
#         clipLimit=2.0,  # Limit contrast enhancement
#         tileGridSize=(8, 8)  # 8x8 tiles as in MATLAB
#     )
    
#     # Apply CLAHE
#     return clahe.apply(image_in)