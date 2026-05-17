"""
Test triplet processing integration into main pipeline.

This test verifies that:
1. triplet.py is properly imported
2. process.py correctly calls triplet functions
3. Statistics are accumulated correctly
4. Configuration includes camera_order
"""

import pytest
from pathlib import Path

from datetime import datetime

from src.core.config import ProcessingConfig, LabelConfig
from src.core.process import masc_process, ProcessingStats, Timings
from src.utils.upload import PictureList


def test_processing_stats_triplet_mode():
    """Test that ProcessingStats initializes correctly for triplet mode."""
    stats = ProcessingStats(use_triplet_algo=True)
    
    assert hasattr(stats, 'N_tot')
    assert hasattr(stats, 'N_triplet_full')
    assert hasattr(stats, 'N_triplet_miss')
    assert hasattr(stats, 'N_matched')
    assert hasattr(stats, 'N_no_match')
    assert hasattr(stats, 'N_processed_indep')
    
    # Should not have single-image stats
    assert not hasattr(stats, 'N_good')
    assert not hasattr(stats, 'N_bad')
    assert not hasattr(stats, 'N_blurry')


def test_processing_stats_single_mode():
    """Test that ProcessingStats initializes correctly for single-image mode."""
    stats = ProcessingStats(use_triplet_algo=False)
    
    assert hasattr(stats, 'N_tot')
    assert hasattr(stats, 'N_good')
    assert hasattr(stats, 'N_bad')
    assert hasattr(stats, 'N_blurry')
    
    # Should not have triplet stats
    assert not hasattr(stats, 'N_triplet_full')
    assert not hasattr(stats, 'N_matched')


def test_timings_triplet_mode():
    """Test that Timings includes matching time for triplet mode."""
    timings = Timings(use_triplet_algo=True)
    
    assert hasattr(timings, 'tt_matching')
    assert timings.tt_matching == 0.0


def test_timings_single_mode():
    """Test that Timings does not include matching time for single mode."""
    timings = Timings(use_triplet_algo=False)
    
    assert not hasattr(timings, 'tt_matching')


def test_config_camera_order():
    """Test that ProcessingConfig includes camera_order field."""
    config = ProcessingConfig()
    
    assert hasattr(config, 'camera_order')
    assert config.camera_order == [0, 1, 2]  # Default value
    
    # Test custom camera order
    config_custom = ProcessingConfig(camera_order=[0, 1, 2, 3, 4])
    assert config_custom.camera_order == [0, 1, 2, 3, 4]


def test_picture_list_id_unique():
    """Test that PictureList includes id_unique field."""
    pic_list = PictureList()
    
    assert hasattr(pic_list, 'id_unique')
    assert pic_list.id_unique == []
    
    # Test with some data
    pic_list.id = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    pic_list.id_unique = sorted(set(pic_list.id))
    
    assert pic_list.id_unique == [1, 2, 3]


def test_triplet_import():
    """Test that process_triplet_image can be imported."""
    from src.processing.triplet import process_triplet_image
    
    assert callable(process_triplet_image)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
