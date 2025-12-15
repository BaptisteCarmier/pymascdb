"""
Utility functions for saving and loading ROI data.

Supports multiple formats:
- joblib: Optimized for numpy arrays (recommended, default)
- pkl: Standard Python pickle
- mat: MATLAB compatible (requires scipy)
"""

import pickle
import numpy as np
import logging

from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

try:
    import joblib # for .joblib format
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import scipy.io # for .mat format
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def save_roi_data(
    roi: Dict,
    filepath: Path,
    format: str = "joblib",
    compress: Union[bool, int] = 3
) -> None:
    """
    Save ROI data to disk in specified format.
    
    Args:
        roi: Dictionary containing ROI features and metadata
        filepath: Path where to save the file (without extension)
        format: Save format - "joblib", "pkl", or "mat"
        compress: Compression level (0-9 for joblib, True/False for others)
    
    Examples:
        >>> save_roi_data(roi, Path("output/data"), format="joblib", compress=3)
        >>> save_roi_data(roi, Path("output/data"), format="pkl")
        >>> save_roi_data(roi, Path("output/data"), format="mat")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove non-serializable items (matplotlib figures)
    roi_clean = {k: v for k, v in roi.items() if not k.startswith('_figure')}
    
    if format == "joblib":
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib not installed. Install with: pip install joblib")
        
        # joblib is optimized for numpy arrays with compression
        joblib.dump(roi_clean, filepath, compress=compress)
        
    elif format == "pkl":
        # Standard pickle
        with open(filepath, 'wb') as f:
            pickle.dump(roi_clean, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif format == "mat":
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not installed. Install with: pip install scipy")
        
        # Convert to MATLAB-compatible format
        mat_data = _convert_roi_to_matlab(roi_clean)
        scipy.io.savemat(filepath, mat_data)
    
    else:
        raise ValueError(f"Unknown format: {format}. Use 'joblib', 'pkl', or 'mat' in config.py")


def load_roi_data(filepath: Path, format: Optional[str] = None) -> Dict:
    """
    Load ROI data from disk.
    
    Args:
        filepath: Path to the file
        format: File format (auto-detected from extension if None)
    
    Returns:
        Dictionary containing ROI data
    
    Examples:
        >>> roi = load_roi_data(Path("output.joblib"))
        >>> roi = load_roi_data(Path("output.pkl"))
        >>> roi = load_roi_data(Path("output.mat"))
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format from extension
    if format is None:
        ext = filepath.suffix.lower()
        if ext in ['.joblib', '.jl']:
            format = 'joblib'
        elif ext == '.pkl':
            format = 'pkl'
        elif ext == '.mat':
            format = 'mat'
        else:
            # Try joblib first (most common), fallback to pickle
            format = 'joblib'
    
    if format == "joblib":
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib not installed. Install with: pip install joblib")
        return joblib.load(filepath)
    
    elif format == "pkl":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    elif format == "mat":
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not installed. Install with: pip install scipy")
        mat_data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        # Remove MATLAB metadata keys
        return {k: v for k, v in mat_data.items() if not k.startswith('__')}
    
    else:
        raise ValueError(f"Unknown format: {format}")


def _convert_roi_to_matlab(roi: Dict) -> Dict:
    """Convert ROI dict to MATLAB-compatible format."""
    mat_data = {}
    
    for key, value in roi.items():
        if isinstance(value, dict):
            # Nested dict -> struct in MATLAB
            mat_data[key] = _convert_roi_to_matlab(value)
        elif isinstance(value, np.ndarray):
            mat_data[key] = value
        elif isinstance(value, (list, tuple)):
            mat_data[key] = np.array(value)
        elif value is None:
            mat_data[key] = np.nan
        else:
            mat_data[key] = value
    
    return mat_data


def load_all_roi_in_folder(
    folder_path: Path,
    quality: Optional[str] = None,
    format: str = "joblib"
) -> List[Dict]:
    """
    Load all ROI files from a folder.
    
    Args:
        folder_path: Path to DATA/ folder (or DATA/GOOD, DATA/BAD)
        quality: Filter by quality ("GOOD" or "BAD"), or None for all
        format: File format to load
    
    Returns:
        List of ROI dictionaries
    
    Example:
        >>> rois = load_all_roi_in_folder(Path("output/2015.06.20/09/DATA"), quality="GOOD")
    """
    folder_path = Path(folder_path)
    
    if quality:
        folder_path = folder_path / quality
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all files with correct extension
    if format == "joblib":
        pattern = "*.joblib"
    elif format == "pkl":
        pattern = "*.pkl"
    elif format == "mat":
        pattern = "*.mat"
    else:
        raise ValueError(f"Unknown format: {format}")
    
    roi_files = sorted(folder_path.glob(pattern))
    
    rois = []
    for file in roi_files:
        try:
            roi = load_roi_data(file, format=format)
            rois.append(roi)
        except Exception as e:
            logger.warning("Could not load %s: %s", file, e)
    
    return rois


def export_roi_to_dict(roi: Dict, fields: Optional[List[str]] = None) -> Dict:
    """
    Export selected fields from ROI to a clean dictionary.
    
    Useful for creating dataframes or CSV exports.
    
    Args:
        roi: ROI dictionary
        fields: List of field names to export. If None, exports common scalar fields.
        
    Returns:
        Dictionary with selected fields
        
    Example:
        >>> roi = load_roi_data(Path("data.joblib"))
        >>> clean = export_roi_to_dict(roi, fields=['Area', 'Perimeter', 'mean_intens'])
    """
    if fields is None:
        # Auto-select scalar fields (exclude arrays and dicts)
        fields = [k for k, v in roi.items() 
                  if not isinstance(v, (dict, np.ndarray, list)) or k == 'name']
    
    result = {}
    for field in fields:
        if field in roi:
            value = roi[field]
            # Convert numpy scalars to Python types
            if isinstance(value, np.generic):
                value = value.item()
            result[field] = value
    
    return result


def print_roi_summary(roi: Dict) -> None:
    """
    Print a summary of ROI features.
    
    Args:
        roi: ROI dictionary loaded from file
    """
    print("=" * 60)
    print("ROI Summary")
    print("=" * 60)
    
    # Basic info
    print(f"Filename: {roi.get('name', 'N/A')}")
    print(f"Camera: {roi.get('cam', 'N/A')}")
    print(f"Flake ID: {roi.get('id', 'N/A')}")
    print(f"Timestamp: {roi.get('tnum', 'N/A')}")
    print()
    
    # Geometric features
    print("Geometric Features:")
    for key in ['Area', 'Perimeter', 'width', 'height', 'Dmax', 'AspectRatio', 'Rectangularity', 'Nb_holes']:
        if key in roi:
            val = roi[key]
            if isinstance(val, (int, float, np.number)):
                print(f"  {key:20s}: {val:.2f}")
    print()
    
    # Intensity features
    print("Intensity Features:")
    for key in ['mean_intens', 'max_intens']:
        if key in roi:
            print(f"  {key:20s}: {roi[key]:.4f}")
    print()
    
    # Ellipse fit
    if 'E' in roi and isinstance(roi['E'], dict):
        print("Ellipse Fit:")
        print(f"  Semi-major axis (a): {roi['E'].get('a', 'N/A')}")
        print(f"  Semi-minor axis (b): {roi['E'].get('b', 'N/A')}")
        print(f"  Orientation (theta): {roi['E'].get('theta', 'N/A')}°")
        if 'EllipseAreaRatio' in roi:
            print(f"  Ellipse Area Ratio: {roi['EllipseAreaRatio']:.4f}")
        print()
    
    # Quality metrics
    print("Quality Metrics:")
    for key in ['Complexity', 'Compactness', 'Convexity', 'Solidity']:
        if key in roi:
            print(f"  {key:20s}: {roi[key]:.4f}")
    print()
    
    print("=" * 60)


if __name__ == "__main__":
    """Example usage from command line"""
    import sys
    
    if len(sys.argv) > 1:
        pkl_file = Path(sys.argv[1])
        if pkl_file.exists():
            roi = load_roi_data(pkl_file)
            print_roi_summary(roi)
        else:
            logger.error("File not found: %s", pkl_file)    
    else:
        print("Usage: python load_roi_data.py <path_to_file>")
        print("Supported formats: .joblib, .pkl, .mat")
        print("Example: python load_roi_data.py output/2015.06.20/09/DATA/GOOD/2015.06.20_09.00.01_cam0.joblib")
