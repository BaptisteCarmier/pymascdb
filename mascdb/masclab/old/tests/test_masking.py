"""
Unit tests for image masking module.
"""

import pytest
import numpy as np
from src.preprocessing.masking import masking
from src.core.config import ProcessingConfig

@pytest.fixture
def process_config():
    """Fixture for processing configuration."""
    return ProcessingConfig(
        discardmat=(10, 10, 20, 20),  # top, bottom, left, right
        backthresh=40,
        # Autres paramètres gardent leurs valeurs par défaut
        sizemin=32,
        min_area=100,
        minbright=0.1,
        max_intensthresh=0.9,
        min_hole_area=10,
        parallel=False,
        use_triplet_algo=True,
        matching_tol_pix=10,
        matching_tol_percent=20
    )

def test_masking_left_camera(process_config):
    """Test masking with left LED camera."""
    # Create test image
    test_image = np.full((100, 100), 100, dtype=np.uint8)
    
    result = masking(test_image, flake_cam=1, process=process_config)
    
    # Check left side is masked
    assert np.all(result[:, :process_config.discardmat[2]] == 0)
    # Check top and bottom are masked
    assert np.all(result[:process_config.discardmat[0], :] == 0)
    assert np.all(result[-process_config.discardmat[1]:, :] == 0)
    # Check middle area is preserved
    assert np.any(result[process_config.discardmat[0]:-process_config.discardmat[1], 
                       process_config.discardmat[2]:] != 0)

def test_masking_right_camera(process_config):
    """Test masking with right LED camera."""
    test_image = np.full((100, 100), 100, dtype=np.uint8)
    
    result = masking(test_image, flake_cam=3, process=process_config)
    
    # Check right side is masked
    assert np.all(result[:, -process_config.discardmat[3]:] == 0)
    # Check top and bottom are masked
    assert np.all(result[:process_config.discardmat[0], :] == 0)
    assert np.all(result[-process_config.discardmat[1]:, :] == 0)
    # Check middle area is preserved
    assert np.any(result[process_config.discardmat[0]:-process_config.discardmat[1], 
                       :-process_config.discardmat[3]] != 0)

def test_masking_center_camera(process_config):
    """Test masking with center camera."""
    test_image = np.full((100, 100), 100, dtype=np.uint8)
    
    result = masking(test_image, flake_cam=2, process=process_config)
    
    # Check only top and bottom are masked
    assert np.all(result[:process_config.discardmat[0], :] == 0)
    assert np.all(result[-process_config.discardmat[1]:, :] == 0)
    # Check sides are preserved
    assert np.any(result[process_config.discardmat[0]:-process_config.discardmat[1], 
                       :process_config.discardmat[2]] != 0)
    assert np.any(result[process_config.discardmat[0]:-process_config.discardmat[1], 
                       -process_config.discardmat[3]:] != 0)

def test_masking_threshold(process_config):
    """Test luminosity threshold masking."""
    # Create image with varying intensities
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[25:75, 25:75] = 30  # Below threshold
    test_image[40:60, 40:60] = 50  # Above threshold
    
    result = masking(test_image, flake_cam=2, process=process_config)
    
    # Check values below threshold are masked
    assert np.all(result[result <= process_config.backthresh] == 0)
    # Check values above threshold are preserved (except in masked regions)
    assert np.any(result[40:60, 40:60] > 0)

def test_masking_preserves_input(process_config):
    """Test that input image is not modified."""
    test_image = np.full((100, 100), 100, dtype=np.uint8)
    original = test_image.copy()
    
    _ = masking(test_image, flake_cam=1, process=process_config)
    
    # Check original image was not modified
    assert np.array_equal(test_image, original)