"""
Unit tests for image brightening module.
"""

import pytest
import numpy as np
from src.preprocessing.brightening import brightening

def test_brightening_normal_size():
    """Test brightening with image larger than tile size."""
    # Create a test image larger than tile_dim (8)
    test_image = np.zeros((16, 16), dtype=np.uint8)
    test_image[4:12, 4:12] = 255  # Create a white square in the middle
    
    # Apply brightening
    result = brightening(test_image)
    
    # Verify results
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8
    # Result should not be identical to input due to CLAHE
    assert not np.array_equal(result, test_image)

def test_brightening_small_both():
    """Test brightening with image smaller than tile size in both dimensions."""
    # Create a small test image (smaller than tile_dim in both dimensions)
    test_image = np.zeros((6, 6), dtype=np.uint8)
    test_image[2:4, 2:4] = 255  # Create a white square in the middle
    
    # Apply brightening
    result = brightening(test_image)
    
    # Verify results
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8

def test_brightening_small_height():
    """Test brightening with image smaller than tile size in height."""
    # Create a test image with small height
    test_image = np.zeros((6, 16), dtype=np.uint8)
    test_image[2:4, 4:12] = 255
    
    # Apply brightening
    result = brightening(test_image)
    
    # Verify results
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8

def test_brightening_small_width():
    """Test brightening with image smaller than tile size in width."""
    # Create a test image with small width
    test_image = np.zeros((16, 6), dtype=np.uint8)
    test_image[4:12, 2:4] = 255
    
    # Apply brightening
    result = brightening(test_image)
    
    # Verify results
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8

def test_brightening_uniform():
    """Test brightening with uniform image."""
    # Create a uniform test image
    test_image = np.full((16, 16), 128, dtype=np.uint8)
    
    # Apply brightening
    result = brightening(test_image)
    
    # Verify results
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8
    # For uniform image, result should be similar to input
    assert abs(np.mean(result) - np.mean(test_image)) < 20  # Allow small variation

def test_brightening_empty():
    """Test brightening with black image."""
    # Create a black test image
    test_image = np.zeros((16, 16), dtype=np.uint8)
    
    # Apply brightening
    result = brightening(test_image)
    
    # Verify results
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8