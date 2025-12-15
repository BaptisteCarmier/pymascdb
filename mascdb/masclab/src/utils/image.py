"""
Image utility functions for MASC analysis.

This module provides utility functions for handling MASC images,
including file operations and image transformations.
"""

import numpy as np
from scipy import ndimage


def rangefilt(image: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Compute local range filter (max - min in neighborhood).

    Equivalent to MATLAB rangefilt(image) with default 3x3 neighborhood.

    Args:
        image: Input grayscale image
        size: Size of the square neighborhood (default: 3)

    Returns
    -------
        Range filtered image
    """

    def range_func(values):
        return values.max() - values.min()

    rangefilt_im = ndimage.generic_filter(image, range_func, size=size, mode="nearest")

    return rangefilt_im


def stdfilt(image: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Compute local standard deviation filter.

    Equivalent to MATLAB stdfilt(image) or stdfilt(image, ones(size)).

    Args:
        image: Input grayscale image
        size: Size of the square neighborhood (default: 3)

    Returns
    -------
        Standard deviation filtered image
    """

    def std_func(values):
        return np.std(values)

    stdfilt_im = ndimage.generic_filter(image.astype(float), std_func, size=size, mode="nearest")

    return stdfilt_im
