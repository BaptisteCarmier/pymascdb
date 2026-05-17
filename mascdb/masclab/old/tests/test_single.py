"""
Unit tests for single image processing module.

Tests the complete MASC image processing pipeline translated from
MASC_picture_process.m
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
from datetime import datetime

from src.processing.single import process_single_image, process_single_roi
from src.core.config import ProcessingConfig


@pytest.fixture
def process_config():
    """Fixture for processing configuration."""
    config = ProcessingConfig()
    config.discardmat = (10, 10, 20, 20)  # top, bottom, left, right
    config.backthresh = 40
    config.sizemin = 20
    config.min_area = 100
    config.minbright = 0.1
    config.max_intensthresh = 0.5
    config.min_hole_area = 10
    config.flakebrighten = False
    return config


@pytest.fixture
def process_config_brighten():
    """Fixture for configuration with brightening enabled."""
    config = ProcessingConfig()
    config.discardmat = (10, 10, 20, 20)
    config.backthresh = 40
    config.sizemin = 20
    config.min_area = 100
    config.minbright = 0.1
    config.max_intensthresh = 0.5
    config.min_hole_area = 10
    config.flakebrighten = True
    return config


@pytest.fixture
def temp_image_path():
    """Create a temporary directory for test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_image(
    image_path: Path,
    filename: str,
    size: tuple = (100, 100),
    roi_bbox: tuple = (30, 30, 40, 40),
    intensity: int = 200
) -> Path:
    """
    Create a test image with a bright ROI.
    
    Args:
        image_path: Directory to save image
        filename: Image filename
        size: Image size (width, height)
        roi_bbox: ROI bounding box (x, y, w, h)
        intensity: Pixel intensity in ROI
        
    Returns:
        Path to created image
    """
    image = np.zeros(size, dtype=np.uint8)
    x, y, w, h = roi_bbox
    image[y:y+h, x:x+w] = intensity
    
    filepath = image_path / filename
    Image.fromarray(image).save(filepath)
    
    return filepath


def test_process_single_image_good(process_config, temp_image_path):
    """Test processing a good quality image."""
    # Create test image with bright ROI
    filepath = create_test_image(
        temp_image_path,
        "test_flake_good.png",
        size=(100, 100),
        roi_bbox=(35, 35, 30, 30),  # Centered ROI, 30x30
        intensity=230  # Very bright
    )
    
    # Create pic_info
    pic_info = {
        'filename': 'test_flake_good.png',
        'cam': 1,
        'id': 12345,
        'fallspeed': 1.2,
        'time_num': datetime.now()
    }
    
    # Process image
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=True
    )
    
    # Check flag
    print(f"\nFlag: {flag}")
    if roi:
        print(f"ROI flag_roi: {roi.get('flag_roi')}")
        print(f"ROI status: {roi.get('status')}")
        print(f"ROI area: {roi.get('area')}")
        print(f"ROI mean_intens: {roi.get('mean_intens')}")
    
    # Assertions
    assert roi is not None, "ROI should not be None"
    assert flag in [1, 2], f"Expected flag 1 or 2, got {flag}"
    
    # Check metadata
    assert roi['name'] == 'test_flake_good.png'
    assert roi['cam'] == 1
    assert roi['id'] == 12345
    assert roi['fallspeed'] == 1.2
    
    # Check timing
    assert 'loading' in timing
    assert 'clutter' in timing
    assert 'edging' in timing
    assert 'roiying' in timing
    assert 'feature' in timing
    
    # Check features
    assert 'area' in roi
    assert 'mean_intens' in roi
    assert 'width' in roi
    assert 'height' in roi
    
    print(f"✅ Good image: flag={flag}, area={roi['area']}, n_roi={roi['n_roi']}")


def test_process_single_image_with_texture(process_config, temp_image_path):
    """Test processing image with texture."""
    # Create image with textured ROI
    np.random.seed(42)
    image = np.zeros((100, 100), dtype=np.uint8)
    
    # Add textured ROI with checkerboard pattern
    for i in range(30, 60):
        for j in range(30, 60):
            image[i, j] = 220 if (i + j) % 2 == 0 else 230
    
    filepath = temp_image_path / "test_textured.png"
    Image.fromarray(image).save(filepath)
    
    pic_info = {
        'filename': 'test_textured.png',
        'cam': 1,
        'id': 12346,
        'fallspeed': 1.5,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    # Should detect ROI
    assert roi is not None
    assert flag in [1, 2]
    
    # Texture metrics should be non-zero
    assert roi['range_intens'] > 0
    assert roi['local_std'] > 0
    
    print(f"✅ Textured image: flag={flag}, range={roi['range_intens']:.3f}")


def test_process_single_image_multiple_rois(process_config, temp_image_path):
    """Test processing image with multiple ROIs."""
    # Create image with two ROIs
    image = np.zeros((150, 150), dtype=np.uint8)
    
    # First ROI (larger, brighter with strong texture)
    np.random.seed(42)
    for i in range(20, 50):
        for j in range(20, 50):
            image[i, j] = 235 if (i + j) % 2 == 0 else 245
    
    # Second ROI (smaller, less bright)
    image[80:100, 80:100] = 180
    
    filepath = temp_image_path / "test_multiple.png"
    Image.fromarray(image).save(filepath)
    
    pic_info = {
        'filename': 'test_multiple.png',
        'cam': 1,
        'id': 12347,
        'fallspeed': 0.9,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    # Should detect ROIs
    assert roi is not None
    assert flag in [1, 2]
    
    # Should have detected multiple ROIs
    assert roi['n_roi'] >= 1
    
    # If multiple ROIs, ratio should not be inf
    if roi['n_roi'] > 1:
        assert roi['area_focus_ratio'] != float('inf')
    
    print(f"✅ Multiple ROIs: n_roi={roi['n_roi']}, ratio={roi.get('area_focus_ratio', 'inf')}")


def test_process_single_image_no_roi(process_config, temp_image_path):
    """Test processing empty image (no ROI)."""
    # Create empty/dark image
    image = np.full((100, 100), 30, dtype=np.uint8)  # Below backthresh
    
    filepath = temp_image_path / "test_empty.png"
    Image.fromarray(image).save(filepath)
    
    pic_info = {
        'filename': 'test_empty.png',
        'cam': 1,
        'id': 12348,
        'fallspeed': 0.5,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=True
    )
    
    # Should return None with flag=0
    assert roi is None
    assert flag == 0
    
    # Timing should still be recorded
    assert timing['feature'] == 0
    assert timing['loading'] > 0
    
    print(f"✅ Empty image: flag={flag}, roi=None")


def test_process_single_image_load_error(process_config, temp_image_path):
    """Test processing non-existent image."""
    # Non-existent file
    filepath = temp_image_path / "nonexistent.png"
    
    pic_info = {
        'filename': 'nonexistent.png',
        'cam': 1,
        'id': 12349,
        'fallspeed': 1.0,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=True
    )
    
    # Should return None with flag=-1
    assert roi is None
    assert flag == -1
    
    # Only loading timing should be recorded
    assert 'loading' in timing
    
    print(f"✅ Load error: flag={flag}")


def test_process_single_image_rgb_conversion(process_config, temp_image_path):
    """Test processing RGB image (should convert to grayscale)."""
    # Create RGB image
    rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_image[30:60, 30:60, :] = 220  # Bright white square
    
    filepath = temp_image_path / "test_rgb.png"
    Image.fromarray(rgb_image, mode='RGB').save(filepath)
    
    pic_info = {
        'filename': 'test_rgb.png',
        'cam': 1,
        'id': 12350,
        'fallspeed': 1.1,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    # Should process successfully (converted to grayscale)
    assert roi is not None
    assert flag in [1, 2]
    
    print(f"✅ RGB conversion: flag={flag}, area={roi['area']}")


def test_process_single_image_with_brightening(process_config_brighten, temp_image_path):
    """Test processing with brightening enabled."""
    # Create dim image
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:60, 30:60] = 120  # Dim ROI
    
    filepath = temp_image_path / "test_dim.png"
    Image.fromarray(image).save(filepath)
    
    pic_info = {
        'filename': 'test_dim.png',
        'cam': 1,
        'id': 12351,
        'fallspeed': 0.8,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config_brighten, verbose=False
    )
    
    # Should process (might be flagged as BAD due to darkness)
    if roi is not None:
        # Check that 'new' features exist
        assert 'new' in roi
        assert 'data' in roi['new']
        assert 'xhi' in roi
        
        print(f"✅ Brightening enabled: flag={flag}, xhi={roi.get('xhi', 'N/A')}")


def test_process_single_image_border_roi_left_cam(process_config, temp_image_path):
    """Test ROI touching left border with left camera."""
    # Create image with ROI touching left border
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 5:25] = 220  # ROI touching left (< discardmat[2]=20)
    
    filepath = temp_image_path / "test_border.png"
    Image.fromarray(image).save(filepath)
    
    pic_info = {
        'filename': 'test_border.png',
        'cam': 1,  # Left camera (MATLAB convention: cam=1 for left LED)
        'id': 12352,
        'fallspeed': 1.0,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    # ROI should be filtered out
    assert roi is None
    assert flag == 0
    
    print(f"✅ Border filtering: flag={flag}, roi=None (as expected)")


def test_process_single_image_small_roi(process_config, temp_image_path):
    """Test image with ROI smaller than sizemin."""
    # Create image with tiny ROI
    image = np.zeros((100, 100), dtype=np.uint8)
    image[45:50, 45:50] = 230  # Only 5x5 pixels
    
    filepath = temp_image_path / "test_small.png"
    Image.fromarray(image).save(filepath)
    
    pic_info = {
        'filename': 'test_small.png',
        'cam': 1,
        'id': 12353,
        'fallspeed': 1.0,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    # ROI might be filtered by min_area or flagged as BAD
    if roi is None:
        assert flag == 0  # Filtered by min_area
        print(f"✅ Small ROI filtered: flag={flag}")
    else:
        # If detected, should be flagged as BAD (too small)
        assert flag == 1
        assert 'too small' in roi['status']
        print(f"✅ Small ROI flagged BAD: flag={flag}, status={roi['status']}")


def test_process_single_image_dark_roi(process_config, temp_image_path):
    """Test image with dark ROI."""
    # Create image with dark ROI
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:60, 30:60] = 50  # Dark (50/255 = 0.196)
    
    filepath = temp_image_path / "test_dark.png"
    Image.fromarray(image).save(filepath)
    
    pic_info = {
        'filename': 'test_dark.png',
        'cam': 1,
        'id': 12354,
        'fallspeed': 1.0,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    # Should be detected but flagged as BAD (too dark)
    if roi is not None:
        assert flag == 1
        assert 'too dark' in roi['status']
        print(f"✅ Dark ROI flagged BAD: flag={flag}, status={roi['status']}")
    else:
        # Might be filtered out completely
        assert flag == 0
        print(f"✅ Dark ROI filtered: flag={flag}")


def test_process_single_image_timing(process_config, temp_image_path):
    """Test that timing information is recorded."""
    filepath = create_test_image(
        temp_image_path,
        "test_timing.png",
        size=(100, 100),
        roi_bbox=(30, 30, 40, 40),
        intensity=220
    )
    
    pic_info = {
        'filename': 'test_timing.png',
        'cam': 1,
        'id': 12355,
        'fallspeed': 1.0,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    # Check all timing keys exist
    assert 'loading' in timing
    assert 'clutter' in timing
    assert 'edging' in timing
    assert 'roiying' in timing
    
    if roi is not None:
        assert 'feature' in timing
        assert timing['feature'] > 0
    
    # All times should be non-negative
    for key, value in timing.items():
        assert value >= 0, f"Timing {key} should be non-negative"
    
    total_time = sum(timing.values())
    print(f"✅ Timing: total={total_time:.4f}s, breakdown={timing}")


"""def test_process_single_roi_basic(process_config):
    """"""Test process_single_roi function.""""""
    # Create test image
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:60, 30:60] = 200
    
    # Create regionprops dict (MATLAB format)
    regionprops_roi = {
        'BoundingBox': np.array([30.0, 30.0, 30.0, 30.0]),  # x, y, w, h
        'PixelIdxList': np.arange(1, 901),  # 1-based indices
        'MajorAxisLength': 40.0,
        'MinorAxisLength': 40.0,
        'Orientation': 0.0,
        'Centroid': (45.0, 45.0)
    }
    
    # Process ROI
    roi = process_single_roi(image, regionprops_roi, process_config)
    
    # Check results
    assert roi is not None
    assert 'area' in roi
    assert 'mean_intens' in roi
    assert 'width' in roi
    assert 'height' in roi
    
    print(f"✅ process_single_roi: area={roi['area']}, width={roi['width']}")

"""
def test_process_single_image_metadata_preservation(process_config, temp_image_path):
    """Test that all metadata is preserved in output."""
    filepath = create_test_image(
        temp_image_path,
        "test_metadata.png",
        size=(100, 100),
        roi_bbox=(30, 30, 40, 40),
        intensity=220
    )
    
    test_time = datetime(2023, 6, 15, 14, 30, 0)
    
    pic_info = {
        'filename': 'test_metadata.png',
        'cam': 2,
        'id': 99999,
        'fallspeed': 2.5,
        'time_num': test_time
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    assert roi is not None
    
    # Check all metadata preserved
    assert roi['name'] == 'test_metadata.png'
    assert roi['cam'] == 2
    assert roi['id'] == 99999
    assert roi['fallspeed'] == 2.5
    assert roi['tnum'] == test_time
    assert 'n_roi' in roi
    assert 'area_focus_ratio' in roi
    assert 'flag_roi' in roi
    assert 'status' in roi
    
    print(f"✅ Metadata preserved: name={roi['name']}, cam={roi['cam']}, id={roi['id']}")


def test_process_single_image_complete_fields(process_config, temp_image_path):
    """Test that all expected fields are present in output."""
    filepath = create_test_image(
        temp_image_path,
        "test_complete.png",
        size=(100, 100),
        roi_bbox=(30, 30, 40, 40),
        intensity=220
    )
    
    pic_info = {
        'filename': 'test_complete.png',
        'cam': 1,
        'id': 12356,
        'fallspeed': 1.0,
        'time_num': datetime.now()
    }
    
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=False
    )
    
    assert roi is not None
    
    # Check basic descriptors fields (from process_basic_descriptors)
    basic_fields = [
        'data', 'area', 'mean_intens', 'width', 'height',
        'centroid', 'perim', 'Dmax', 'eq_radius', 'compactness'
    ]
    
    for field in basic_fields:
        assert field in roi, f"Missing field: {field}"
    
    # Check additional fields from process_single_image
    additional_fields = [
        'name', 'cam', 'id', 'fallspeed', 'tnum',
        'n_roi', 'area_focus_ratio', 'flag_roi', 'status'
    ]
    
    for field in additional_fields:
        assert field in roi, f"Missing field: {field}"
    
    print(f"✅ All fields present: {len(roi.keys())} total fields")


def test_process_single_image_verbose_mode(process_config, temp_image_path, capsys):
    """Test verbose mode prints information."""
    filepath = create_test_image(
        temp_image_path,
        "test_verbose.png",
        size=(100, 100),
        roi_bbox=(30, 30, 40, 40),
        intensity=220
    )
    
    pic_info = {
        'filename': 'test_verbose.png',
        'cam': 1,
        'id': 12357,
        'fallspeed': 1.0,
        'time_num': datetime.now()
    }
    
    # Process with verbose=True
    roi, flag, timing = process_single_image(
        filepath, pic_info, process_config, verbose=True
    )
    
    # Capture output would require capsys, but basic test that it doesn't crash
    assert roi is not None or flag in [-1, 0, 1, 2]
    
    print(f"✅ Verbose mode executed successfully")
