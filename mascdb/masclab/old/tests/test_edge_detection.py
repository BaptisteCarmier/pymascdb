"""
Unit tests for edge detection module.

Tests the edge detection functionality translated from MATLAB edge_detection.m
"""

import pytest
import numpy as np
from src.preprocessing.edge_detection import detect_edges


def test_detect_edges_basic():
    """Test basic edge detection on simple synthetic image."""
    # Create test image with a white square on black background
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[30:70, 30:70] = 255
    
    result = detect_edges(test_image)
    
    # Verify output properties
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8
    # Result should be binary (0 or 1)
    assert set(np.unique(result)).issubset({0, 1})
    # Should detect some edges
    assert np.sum(result) > 0


def test_detect_edges_with_strel_size():
    """Test edge detection with different structuring element sizes."""
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[30:70, 30:70] = 255
    
    # Test with different strel sizes
    result_small = detect_edges(test_image, strel_size=3)
    result_large = detect_edges(test_image, strel_size=10)
    
    # Both should produce valid outputs
    assert result_small.shape == test_image.shape
    assert result_large.shape == test_image.shape
    
    # Larger strel_size typically produces larger filled regions
    assert np.sum(result_large) >= np.sum(result_small)


def test_detect_edges_threshold_sensitivity():
    """Test edge detection with different Sobel thresholds."""
    test_image = np.zeros((100, 100), dtype=np.uint8)
    # Create gradual gradient
    for i in range(50):
        test_image[25:75, 25+i] = i * 5
    
    # Lower threshold should detect more edges
    result_low = detect_edges(test_image, sobel_threshold=0.001)
    result_high = detect_edges(test_image, sobel_threshold=0.05)
    
    assert result_low.shape == test_image.shape
    assert result_high.shape == test_image.shape
    # Lower threshold detects more
    assert np.sum(result_low) >= np.sum(result_high)


def test_detect_edges_empty_image():
    """Test edge detection on empty (black) image."""
    test_image = np.zeros((100, 100), dtype=np.uint8)
    
    result = detect_edges(test_image)
    
    # Empty image should produce no or minimal edges
    assert result.shape == test_image.shape
    # Should be mostly zeros
    assert np.sum(result) < 10  # Allow for minimal noise


def test_detect_edges_uniform_image():
    """Test edge detection on uniform gray image."""
    test_image = np.full((100, 100), 128, dtype=np.uint8)
    
    result = detect_edges(test_image)
    
    # Uniform image should produce no edges
    assert result.shape == test_image.shape
    assert np.sum(result) == 0


def test_detect_edges_circle():
    """Test edge detection on circular shape."""
    test_image = np.zeros((100, 100), dtype=np.uint8)
    # Create circle
    y, x = np.ogrid[:100, :100]
    mask = (x - 50)**2 + (y - 50)**2 <= 20**2
    test_image[mask] = 255
    
    result = detect_edges(test_image)
    
    # Should detect the circular boundary
    assert result.shape == test_image.shape
    assert np.sum(result) > 0
    # The filled region should be approximately circular
    assert np.sum(result) > np.pi * 20**2 * 0.5  # At least half the circle area


def test_detect_edges_preserves_input():
    """Test that input image is not modified."""
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[30:70, 30:70] = 255
    original = test_image.copy()
    
    _ = detect_edges(test_image)
    
    # Check original image was not modified
    assert np.array_equal(test_image, original)


def test_detect_edges_small_image():
    """Test edge detection on small image."""
    test_image = np.zeros((50, 50), dtype=np.uint8)
    test_image[15:35, 15:35] = 255
    
    result = detect_edges(test_image, strel_size=3)
    
    assert result.shape == test_image.shape
    assert result.dtype == np.uint8


def test_detect_edges_rectangle():
    """Test edge detection on rectangular shape."""
    test_image = np.zeros((100, 120), dtype=np.uint8)
    test_image[20:80, 30:90] = 255
    
    result = detect_edges(test_image)
    
    # Should detect and fill the rectangle
    assert result.shape == test_image.shape
    assert np.sum(result) > 0
    # Most of the rectangle should be filled
    assert np.sum(result[20:80, 30:90]) > 0.5 * (60 * 60)


def test_detect_edges_multiple_objects():
    """Test edge detection with multiple separate objects."""
    test_image = np.zeros((100, 100), dtype=np.uint8)
    # Two separate squares
    test_image[20:40, 20:40] = 255
    test_image[60:80, 60:80] = 255
    
    result = detect_edges(test_image)
    
    # Should detect both objects
    assert result.shape == test_image.shape
    assert np.sum(result) > 0
    # Should have filled regions for both squares
    assert np.sum(result[20:40, 20:40]) > 0
    assert np.sum(result[60:80, 60:80]) > 0
