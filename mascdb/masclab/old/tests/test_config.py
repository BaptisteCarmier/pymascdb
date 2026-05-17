"""
Unit tests for MASC configuration module.
"""

from pathlib import Path
import sys
import pytest
from datetime import datetime, timedelta

# Add src to PYTHONPATH for local import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.core.config import load_default_config, create_proc_params_file

def test_load_default_config():
    """Test that default configuration loads with expected values."""
    label, process = load_default_config()
    
    # Check ProcessingConfig defaults
    assert process.parallel is False
    assert process.backthresh == 50
    assert process.sizemin == 32
    assert process.discardmat == (20, 20, 20, 20)
    
    # Check LabelConfig defaults
    assert isinstance(label.campaigndir, Path)
    assert isinstance(label.starthr_vec, datetime)

def test_create_proc_params_file(tmp_path):
    """Test parameter file creation and content."""
    # Setup
    label, process = load_default_config()
    outdir = tmp_path / "test_output"
    outdir.mkdir()
    
    # Test file creation
    create_proc_params_file(outdir, label, process)
    param_file = outdir / "proc_params.txt"
    
    # Verify file exists
    assert param_file.exists()
    
    # Check file content
    content = param_file.read_text()
    assert "Processing parameters associated with data in this folder" in content
    assert "processed in parallel  : 0" in content
    assert "use triplet algo" in content

def test_modified_config(tmp_path):
    """Test that modified configuration values are correctly written."""
    label, process = load_default_config()
    
    # Modify some values
    process.parallel = True
    process.matching_tol_pix = 25
    process.backthresh = 75
    
    # Create parameter file
    outdir = tmp_path / "modified_test"
    outdir.mkdir()
    create_proc_params_file(outdir, label, process)
    
    # Verify modified values in file
    content = (outdir / "proc_params.txt").read_text()
    assert "processed in parallel  : 1" in content
    assert "matching tol. pix.     : 25" in content
    assert "backtresh limit        : 75" in content