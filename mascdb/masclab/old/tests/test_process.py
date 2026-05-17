"""
Test suite for the complete MASC processing pipeline.

This module tests the masc_process function with real sample data,
validating the end-to-end workflow including:
- Directory discovery (uploaddirs)
- Metadata loading (upload)
- Single-image processing
- Statistics accumulation
- Timing measurements

Author: Test suite for MASC Python processing
Last update: November 2025
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from src.core.process import masc_process, ProcessingStats, Timings
from src.core.config import ProcessingConfig, LabelConfig
from src.utils.uploaddirs import uploaddirs
from src.utils.upload import upload


class TestProcessingStats:
    """Test ProcessingStats class initialization."""
    
    def test_stats_single_mode(self):
        """Test stats initialization for single-image mode."""
        stats = ProcessingStats(use_triplet_algo=False)
        
        assert stats.N_tot == 0
        assert stats.N_good == 0
        assert stats.N_bad == 0
        assert stats.N_blurry == 0
        
        # Triplet fields should not exist
        assert not hasattr(stats, 'N_triplet_full')
        assert not hasattr(stats, 'N_matched')
    
    def test_stats_triplet_mode(self):
        """Test stats initialization for triplet mode."""
        stats = ProcessingStats(use_triplet_algo=True)
        
        assert stats.N_tot == 0
        assert stats.N_triplet_full == 0
        assert stats.N_triplet_miss == 0
        assert stats.N_matched == 0
        assert stats.N_no_match == 0
        assert stats.N_processed_indep == 0
        
        # Single-image fields should not exist
        assert not hasattr(stats, 'N_good')
        assert not hasattr(stats, 'N_bad')


class TestTimings:
    """Test Timings class initialization."""
    
    def test_timings_single_mode(self):
        """Test timings initialization for single-image mode."""
        timings = Timings(use_triplet_algo=False)
        
        assert timings.tt_uploading == 0.0
        assert timings.tt_loading == 0.0
        assert timings.tt_clutter == 0.0
        assert timings.tt_edging == 0.0
        assert timings.tt_roiying == 0.0
        assert timings.tt_feature == 0.0
        assert timings.tt_plotting == 0.0
        assert timings.tt_saving == 0.0
        assert timings.tt_program == 0.0
        
        # Triplet timing should not exist
        assert not hasattr(timings, 'tt_matching')
    
    def test_timings_triplet_mode(self):
        """Test timings initialization for triplet mode."""
        timings = Timings(use_triplet_algo=True)
        
        assert timings.tt_uploading == 0.0
        assert timings.tt_matching == 0.0


class TestMASCProcessPipeline:
    """Test the complete MASC processing pipeline with sample data."""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Path to sample data directory."""
        return Path(__file__).parent / "data_sample"
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def label_config(self, sample_data_dir, temp_output_dir):
        """Create LabelConfig for sample data."""
        return LabelConfig(
            campaigndir=sample_data_dir,
            outdir=temp_output_dir,
            starthr_vec=datetime(2015, 6, 20, 9, 0, 0),
            endhr_vec=datetime(2015, 6, 20, 9, 59, 59)
        )
    
    @pytest.fixture
    def process_config_single(self):
        """Create ProcessingConfig for single-image mode."""
        return ProcessingConfig(
            parallel=False,
            use_triplet_algo=False,
            backthresh=15,
            sizemin=30,
            min_area=100,
            minbright=0.15,
            max_intensthresh=0.25,
            min_hole_area=10,
            discardmat=(400, 400, 320, 320),
            flakebrighten=False
        )
    
    def test_uploaddirs_with_sample_data(self, sample_data_dir):
        """Test directory discovery with sample data."""
        starthr_vec = datetime(2015, 6, 20, 9, 0, 0)
        endhr_vec = datetime(2015, 6, 20, 9, 59, 59)
        
        dir_list = uploaddirs(sample_data_dir, starthr_vec, endhr_vec)
        
        # Should find the 2015.06.20/09 directory
        assert len(dir_list) >= 1
        assert any("09" in str(d) for d in dir_list)
        assert any("2015.06.20" in str(d) for d in dir_list)
    
    def test_upload_with_sample_data(self, sample_data_dir):
        """Test metadata loading with sample data."""
        sample_dir = sample_data_dir / "2015.06.20" / "09"
        
        if not sample_dir.exists():
            pytest.skip(f"Sample directory not found: {sample_dir}")
        
        pic_list = upload(sample_dir)
        
        # Verify data was loaded
        assert len(pic_list.files) > 0
        assert len(pic_list.id) > 0
        assert len(pic_list.cam) > 0
        assert len(pic_list.time_vec) > 0
        
        # Verify data consistency
        assert len(pic_list.files) == len(pic_list.id)
        assert len(pic_list.files) == len(pic_list.cam)
        
        # Check camera IDs are valid (0, 1, 2)
        assert all(cam in [0, 1, 2] for cam in pic_list.cam)
        
        # Check fallspeeds were loaded (should not all be NaN)
        if pic_list.fallspeed:
            assert any(not np.isnan(fs) for fs in pic_list.fallspeed)
    
    def test_masc_process_single_mode_limited(
        self, 
        label_config, 
        process_config_single,
        temp_output_dir,
        capsys
    ):
        """Test masc_process in single-image mode with limited data."""
        
        # Check if sample data exists
        sample_dir = label_config.campaigndir / "2015.06.20" / "09"
        if not sample_dir.exists():
            pytest.skip(f"Sample directory not found: {sample_dir}")
        
        # Update label config with output directory
        label_config.outdir = temp_output_dir
        
        # Modify config to avoid errors on missing advanced features
        process_config_single.saveresults = False  # Don't save results for now
        
        # Run the processing pipeline
        # Note: This will process ALL images in the sample directory
        # which might take some time. In production, you'd want to limit this.
        try:
            masc_process(label_config, process_config_single)
        except NotImplementedError as e:
            # Expected if triplet mode is accidentally enabled
            pytest.fail(f"Unexpected NotImplementedError: {e}")
        except Exception as e:
            # Capture any other errors for debugging
            pytest.fail(f"Processing failed with error: {e}")
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Verify output contains expected statistics
        assert "Task finished" in captured.out
        assert "Number of flakes found" in captured.out
        assert "Number of pictures processed" in captured.out
        
        # Verify statistics contain reasonable values
        assert "good snowflakes" in captured.out or "blurry snowflakes" in captured.out
    
    def test_masc_process_triplet_mode_raises_error(
        self,
        label_config,
        temp_output_dir
    ):
        """Test that triplet mode raises NotImplementedError."""
        
        # Check if sample data exists
        sample_dir = label_config.campaigndir / "2015.06.20" / "09"
        if not sample_dir.exists():
            pytest.skip(f"Sample directory not found: {sample_dir}")
        
        # Create config with triplet mode enabled
        process_config_triplet = ProcessingConfig(
            use_triplet_algo=True,
            parallel=False,
            saveresults=False
        )
        
        label_config.outdir = temp_output_dir
        
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Triplet processing not yet implemented"):
            masc_process(label_config, process_config_triplet)
    
    def test_masc_process_with_stats_output(
        self,
        label_config,
        process_config_single,
        temp_output_dir,
        capsys
    ):
        """Test that processing outputs statistics correctly."""
        
        # Check if sample data exists
        sample_dir = label_config.campaigndir / "2015.06.20" / "09"
        if not sample_dir.exists():
            pytest.skip(f"Sample directory not found: {sample_dir}")
        
        # Enable result saving
        label_config.outdir = temp_output_dir
        process_config_single.saveresults = True
        
        # Run processing
        try:
            masc_process(label_config, process_config_single)
        except Exception as e:
            pytest.fail(f"Processing failed: {e}")
        
        # Verify stats file was created
        stats_file = temp_output_dir / "proc_stats.txt"
        assert stats_file.exists(), f"Stats file not created at {stats_file}"
        
        # Read and verify stats file content
        with open(stats_file, 'r') as f:
            content = f.read()
        
        assert "Task finished" in content
        assert "Number of flakes found" in content
        assert "Total Time spent" in content
    
    def test_masc_process_timing_measurements(
        self,
        label_config,
        process_config_single,
        temp_output_dir,
        capsys
    ):
        """Test that timing measurements are collected."""
        
        # Check if sample data exists
        sample_dir = label_config.campaigndir / "2015.06.20" / "09"
        if not sample_dir.exists():
            pytest.skip(f"Sample directory not found: {sample_dir}")
        
        label_config.outdir = temp_output_dir
        process_config_single.saveresults = False
        
        # Run processing
        try:
            masc_process(label_config, process_config_single)
        except Exception as e:
            pytest.fail(f"Processing failed: {e}")
        
        # Capture output
        captured = capsys.readouterr()
        
        # Verify timing percentages are displayed
        assert "Total time uploading directories" in captured.out
        assert "Total time loading pictures" in captured.out
        assert "Total time removing clutter" in captured.out
        assert "Total time edging pictures" in captured.out
        assert "Total time computing feature" in captured.out
        assert "Total time selecting flake ROI" in captured.out
        assert "Net average time per image" in captured.out


class TestMASCProcessEdgeCases:
    """Test edge cases and error handling in MASC processing."""
    
    def test_masc_process_nonexistent_directory(self, temp_output_dir):
        """Test that processing handles nonexistent directories gracefully."""
        
        label = LabelConfig(
            campaigndir=Path("/nonexistent/path"),
            outdir=temp_output_dir,
            starthr_vec=datetime(2015, 6, 20, 9, 0, 0),
            endhr_vec=datetime(2015, 6, 20, 9, 59, 59)
        )
        
        process = ProcessingConfig(
            use_triplet_algo=False,
            saveresults=False
        )
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            masc_process(label, process)
    
    def test_masc_process_empty_time_range(self, sample_data_dir, temp_output_dir, capsys):
        """Test processing with time range that yields no directories."""
        
        label = LabelConfig(
            campaigndir=sample_data_dir,
            outdir=temp_output_dir,
            starthr_vec=datetime(2020, 1, 1, 0, 0, 0),  # Future date
            endhr_vec=datetime(2020, 1, 1, 1, 0, 0)
        )
        
        process = ProcessingConfig(
            use_triplet_algo=False,
            saveresults=False
        )
        
        # Run processing - should complete but process 0 images
        masc_process(label, process)
        
        captured = capsys.readouterr()
        
        # Should report 0 flakes found
        assert "Number of flakes found              : 0" in captured.out


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
