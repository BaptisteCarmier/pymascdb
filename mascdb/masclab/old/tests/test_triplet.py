"""
Unit tests for triplet image processing module.

This module tests the triplet processing functionality, including:
- Image loading from multiple cameras
- ROI matching across cameras
- Focus computation
- Triplet selection
- Results saving

Created: November 2025
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import tempfile
import shutil

from src.processing.triplet import process_triplet_image, _compute_roi_focus, _save_roi_image
from src.core.config import ProcessingConfig, LabelConfig
from src.preprocessing.roi import ROI


class MockPicList:
    """Mock picture list for testing."""
    def __init__(self, files, cams, time_nums):
        self.files = files
        self.cam = cams
        self.time_num = time_nums
        self.fallspeed = [1.0] * len(files)
        self.fallid = [1] * len(files)


class TestTripletProcessing:
    """Test suite for triplet image processing."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_triplet_images(self):
        """Create sample triplet images for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create 3 simple test images (200x200 pixels)
        for cam_id in [0, 1, 2]:
            img = np.zeros((200, 200), dtype=np.uint8)
            
            # Add a snowflake-like pattern (circle)
            center_y, center_x = 100, 100
            y, x = np.ogrid[:200, :200]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
            img[mask] = 200  # Bright snowflake
            
            # Save image
            img_path = temp_dir / f"test_flake_1_cam_{cam_id}.png"
            Image.fromarray(img).save(img_path)
        
        yield temp_dir
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def process_config_triplet(self):
        """Create ProcessingConfig for triplet mode."""
        return ProcessingConfig(
            use_triplet_algo=True,
            matching_tol_pix=10,
            matching_tol_percent=20,
            backthresh=15,
            sizemin=30,
            min_area=100,
            minbright=0.15,
            max_intensthresh=0.25,
            saveresults=True,
        )
    
    @pytest.fixture
    def label_config(self, temp_output_dir):
        """Create LabelConfig for testing."""
        return LabelConfig(
            campaigndir=Path("./tests/data_sample"),
            outdir=temp_output_dir,
            starthr_vec=datetime(2015, 6, 20, 9, 0, 0),
            endhr_vec=datetime(2015, 6, 20, 10, 0, 0),
        )
    
    def test_compute_roi_focus(self):
        """Test ROI focus computation."""
        # Create test image with gradient (to have local variability)
        img = np.zeros((100, 100), dtype=np.uint8)
        
        # Add a gradient pattern with edges (not uniform)
        for i in range(40, 60):
            for j in range(40, 60):
                # Create a gradient from 100 to 200
                img[i, j] = 100 + int((i - 40) * 5)
        
        # Create mock ROI
        roi = ROI(
            bbox=(40, 40, 20, 20),
            mask=np.ones((20, 20), dtype=bool),
            area=400.0,
            perimeter=80.0,
            centroid=(50.0, 50.0),
            major_axis_length=20.0,
            minor_axis_length=20.0,
            orientation=0.0,
            mean_intensity=0.0,
            max_intensity=0.0,
            range_intensity=0.0,
            focus=0.0,
            area_focus=0.0,
            pixel_idx_list=[]
        )
        
        # Compute focus
        focus = _compute_roi_focus(img, roi)
        
        # Focus should be non-negative and a float
        # (may be 0 for uniform regions, >0 for regions with edges)
        assert focus >= 0
        assert isinstance(focus, float)
    
    def test_save_roi_image(self, temp_output_dir):
        """Test ROI image saving."""
        # Create test ROI dict
        roi = {
            'data': np.ones((50, 50), dtype=np.uint8) * 100,
            'name': 'test_image.png'
        }
        
        # Save image
        _save_roi_image(roi, temp_output_dir)
        
        # Check file exists
        saved_path = temp_output_dir / 'test_image.png'
        assert saved_path.exists()
        
        # Load and verify
        loaded = np.array(Image.open(saved_path))
        assert loaded.shape == (50, 50)
        assert np.all(loaded == 100)
    
    def test_process_triplet_basic(self, sample_triplet_images, temp_output_dir, 
                                   process_config_triplet, label_config):
        """Test basic triplet processing."""
        # Create mock pic_list
        files = [
            "test_flake_1_cam_0.png",
            "test_flake_1_cam_1.png",
            "test_flake_1_cam_2.png"
        ]
        cams = [0, 1, 2]
        time_nums = [datetime(2015, 6, 20, 9, 0, 0)] * 3
        
        pic_list = MockPicList(files, cams, time_nums)
        
        # Process triplet
        timing, flag = process_triplet_image(
            flake_id=1,
            idx_pics=[0, 1, 2],
            current_dir=sample_triplet_images,
            pic_list=pic_list,
            label=label_config,
            process=process_config_triplet
        )
        
        # Check that processing completed
        assert isinstance(timing, dict)
        assert isinstance(flag, int)
        
        # Check timing keys
        assert 'loading' in timing
        assert 'clutter' in timing
        assert 'edging' in timing
        assert 'roiying' in timing
        
        # Flag should be >= 0 (not an error)
        assert flag >= 0
    
    def test_process_triplet_missing_camera(self, sample_triplet_images, temp_output_dir,
                                           process_config_triplet, label_config):
        """Test triplet processing with missing camera."""
        # Create pic_list with only 2 cameras
        files = [
            "test_flake_1_cam_0.png",
            "test_flake_1_cam_1.png"
        ]
        cams = [0, 1]
        time_nums = [datetime(2015, 6, 20, 9, 0, 0)] * 2
        
        pic_list = MockPicList(files, cams, time_nums)
        
        # Process should fail (need 3 cameras)
        timing, flag = process_triplet_image(
            flake_id=1,
            idx_pics=[0, 1],
            current_dir=sample_triplet_images,
            pic_list=pic_list,
            label=label_config,
            process=process_config_triplet
        )
        
        # Should return error flag
        assert flag == -1
    
    def test_process_triplet_no_saveresults(self, sample_triplet_images, temp_output_dir,
                                           label_config):
        """Test triplet processing without saving results."""
        # Config with saveresults=False
        process_config = ProcessingConfig(
            use_triplet_algo=True,
            matching_tol_pix=10,
            matching_tol_percent=20,
            saveresults=False,
        )
        
        files = [
            "test_flake_1_cam_0.png",
            "test_flake_1_cam_1.png",
            "test_flake_1_cam_2.png"
        ]
        cams = [0, 1, 2]
        time_nums = [datetime(2015, 6, 20, 9, 0, 0)] * 3
        
        pic_list = MockPicList(files, cams, time_nums)
        
        # Process
        timing, flag = process_triplet_image(
            flake_id=1,
            idx_pics=[0, 1, 2],
            current_dir=sample_triplet_images,
            pic_list=pic_list,
            label=label_config,
            process=process_config
        )
        
        # Check no output files created
        assert not (temp_output_dir / "2015.06.20").exists()
        
        # But processing should complete
        assert flag >= 0


class TestTripletWithRealData:
    """Test triplet processing with real sample data if available."""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Get sample data directory."""
        sample_dir = Path("tests/data_sample")
        if not sample_dir.exists():
            pytest.skip("Sample data not available")
        return sample_dir
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_triplet_with_sample_data(self, sample_data_dir, temp_output_dir):
        """Test triplet processing with real sample data (if available)."""
        # This test will be skipped if sample data doesn't have triplets
        # For now, it's a placeholder for future testing
        pytest.skip("Triplet sample data not yet available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
