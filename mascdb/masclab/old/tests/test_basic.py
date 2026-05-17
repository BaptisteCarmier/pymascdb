"""
Unit tests for basic feature extraction module.

Tests the process_basic_descriptors function and validates behavior
against MATLAB's process_basic_descriptors.m
"""

import pytest
import numpy as np
from skimage import measure
from scipy import ndimage

from src.features.basic import process_basic_descriptors
from src.core.config import ProcessingConfig


@pytest.fixture
def process_config():
    config = ProcessingConfig()
    config.min_hole_area = 5
    config.flakebrighten = False
    return config


def create_regionprops_dict(image):
    """Helper to create regionprops dictionary matching MATLAB format."""
    labeled = measure.label(image > 0, connectivity=2)
    props = measure.regionprops(labeled)[0]
    
    min_row, min_col, max_row, max_col = props.bbox
    regionprops_roi = {
        'BoundingBox': np.array([min_col, min_row, max_col - min_col, max_row - min_row], dtype=float),
        'PixelIdxList': props.coords[:, 0] * image.shape[1] + props.coords[:, 1],
        'MajorAxisLength': props.major_axis_length,
        'MinorAxisLength': props.minor_axis_length,
        'Orientation': props.orientation * 180 / np.pi,
        'Centroid': props.centroid
    }
    
    return regionprops_roi


def test_simple_square_roi(process_config):
    """Test processing a simple square ROI."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:70, 30:70] = 200
    
    regionprops_roi = create_regionprops_dict(image)
    roi = process_basic_descriptors(image, regionprops_roi, process_config)
    
    # Basic validations
    assert isinstance(roi, dict)
    assert roi['area'] == 1600
    assert roi['width'] == 40
    assert roi['height'] == 40
    assert 0 < roi['mean_intens'] <= 1
    assert roi['nb_holes'] == 0


def test_roi_with_texture(process_config):
    """Test ROI with texture features."""
    np.random.seed(42)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:70, 30:70] = 150
    noise = np.random.randint(-30, 30, (40, 40)).astype(np.int16)
    image[30:70, 30:70] = np.clip(image[30:70, 30:70].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    regionprops_roi = create_regionprops_dict(image)
    roi = process_basic_descriptors(image, regionprops_roi, process_config)
    
    # Texture features should be computed
    assert roi['range_intens'] > 0
    assert roi['lap'] > 0
    assert roi['hist_entropy'] > 0
    assert roi['std'] > 0
    assert roi['local_std'] > 0


def test_roi_with_holes(process_config):
    """Test ROI with holes detection."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 200
    image[40:50, 40:50] = 0  # Create hole
    
    regionprops_roi = create_regionprops_dict(image)
    roi = process_basic_descriptors(image, regionprops_roi, process_config)
    
    # Should detect hole
    assert roi['nb_holes'] >= 1
    assert roi['area'] > roi['area_porous']


def test_roi_geometric_features(process_config):
    """Test geometric features computation."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:70, 30:70] = 200
    
    regionprops_roi = create_regionprops_dict(image)
    roi = process_basic_descriptors(image, regionprops_roi, process_config)
    
    # Geometric features
    assert 'hull' in roi
    assert roi['Dmax'] > 0
    assert roi['D90'] > 0
    assert 'E' in roi
    assert 'C_out' in roi
    assert 0 < roi['compactness'] <= 1


def test_roi_shape_features(process_config):
    """Test shape features computation."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:70, 30:70] = 200
    
    regionprops_roi = create_regionprops_dict(image)
    roi = process_basic_descriptors(image, regionprops_roi, process_config)
    
    # Shape features
    assert 'skel' in roi
    assert 'F' in roi
    assert 'Sym' in roi
    assert roi['skel']['length'] > 0


def test_all_required_fields(process_config):
    """Test that all MATLAB required fields are present."""
    np.random.seed(42)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:70, 30:70] = 150
    noise = np.random.randint(-20, 20, (40, 40)).astype(np.int16)
    image[30:70, 30:70] = np.clip(image[30:70, 30:70].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    regionprops_roi = create_regionprops_dict(image)
    roi = process_basic_descriptors(image, regionprops_roi, process_config)
    
    # Check key MATLAB fields
    required_fields = [
        'data', 'bw_mask', 'area', 'mean_intens', 'perim',
        'hull', 'width', 'height', 'Dmax', 'eq_radius',
        'E', 'C_out', 'skel', 'F', 'Sym', 'H',
        'lap', 'hist_entropy', 'wavs', 'std', 'xhi'
    ]
    
    for field in required_fields:
        assert field in roi, f"Missing field: {field}"


def test_tiny_roi(process_config):
    """Test with very small ROI."""
    image = np.zeros((20, 20), dtype=np.uint8)
    image[10:13, 10:13] = 200
    
    regionprops_roi = create_regionprops_dict(image)
    roi = process_basic_descriptors(image, regionprops_roi, process_config)
    
    assert roi['area'] > 0
    assert roi['area'] == 9
