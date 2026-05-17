"""
Unit tests for identifier extraction module.
"""

import pytest
from src.utils.identifiers import (
    get_cam_id,
    get_snowflake_id,
    extract_identifiers,
    validate_camera_id,
    validate_snowflake_id
)

def test_get_cam_id():
    """Test camera ID extraction."""
    # Test valid cases
    assert get_cam_id("image_cam_2_timestamp.jpg") == 2
    assert get_cam_id("prefix_cam_1_suffix.jpg") == 1
    assert get_cam_id("test_cam_3_other.jpg") == 3
    assert get_cam_id("sample_cam_1.jpg") == 1
    
    # Test invalid cases
    assert get_cam_id("no_camera_id.jpg") is None
    assert get_cam_id("") is None
    assert get_cam_id("cam_invalid.jpg") is None
    assert get_cam_id("wrong_format_cam1.jpg") is None

def test_get_snowflake_id():
    """Test snowflake ID extraction."""
    # Test valid cases with exact format XXX_flake_id_YYY.ZZZ
    assert get_snowflake_id("prefix_flake_20201217_suffix.jpg") == 20201217
    assert get_snowflake_id("test_flake_12345_other.jpg") == 12345
    assert get_snowflake_id("img_flake_98765_end.jpg") == 98765
    
    # Test invalid cases
    assert get_snowflake_id("no_flake_id.jpg") is None
    assert get_snowflake_id("") is None
    assert get_snowflake_id("flake_invalid.jpg") is None
    assert get_snowflake_id("wrong_format_flake.jpg") is None
    assert get_snowflake_id("flake_123.jpg") is None  # Missing required format

def test_extract_identifiers():
    """Test combined ID extraction."""
    # Test valid case with both IDs
    cam_id, flake_id = extract_identifiers("prefix_flake_20201217_cam_1_suffix.jpg")
    assert cam_id == 1
    assert flake_id == 20201217
    
    # Test partial matches
    cam_id, flake_id = extract_identifiers("test_cam_2_other.jpg")
    assert cam_id == 2
    assert flake_id is None
    
    cam_id, flake_id = extract_identifiers("test_flake_12345_suffix.jpg")
    assert cam_id is None
    assert flake_id == 12345
    
    # Test no match
    cam_id, flake_id = extract_identifiers("invalid_filename.jpg")
    assert cam_id is None
    assert flake_id is None

def test_validate_camera_id():
    """Test camera ID validation."""
    assert validate_camera_id(1) is True
    assert validate_camera_id(2) is True
    assert validate_camera_id(3) is True
    assert validate_camera_id(0) is False
    assert validate_camera_id(4) is False
    assert validate_camera_id(None) is False

def test_validate_snowflake_id():
    """Test snowflake ID validation."""
    assert validate_snowflake_id(20201217235959) is True
    assert validate_snowflake_id(1) is True
    assert validate_snowflake_id(0) is False
    assert validate_snowflake_id(-1) is False
    assert validate_snowflake_id(None) is False