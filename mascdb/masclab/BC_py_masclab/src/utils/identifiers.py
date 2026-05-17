"""

"""

import logging

from typing import Tuple, Optional


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def get_cam_id(filename: str) -> Optional[int]:
    """
    Extract camera ID from filename.
    
    Args:
        filename: String containing camera ID (format: XXX_cam_id_YYY.ZZZ)
        
    Returns:
        Camera ID as integer, or None if not found
    """
    try:
        # Find start position 
        idx_start = filename.find('cam_') + 4 # +4 as cam_ is of length 4
        filename = filename[idx_start:]
        
        # Find stop positions 
        idx_stop_1 = filename.find('_')
        idx_stop_2 = filename.find('.')
            
        # Choose the appropriate stop position
        if idx_stop_1 == -1:  # -1 as .find() returns -1 if not found
            idx_stop = idx_stop_2
        elif idx_stop_2 == -1:
            idx_stop = idx_stop_1
        else:
            idx_stop = min(x for x in [idx_stop_1, idx_stop_2] if x >= 0)
            
        # Extract and convert ID
        filename = filename[:idx_stop] 
        return int(filename)
        
    except Exception as e:
        logger.error("Could not retrieve camera ID from string: %s", filename)

        return None

def get_snowflake_id(filename: str) -> Optional[int]:
    """
    Extract snowflake ID from filename.
    
    Args:
        filename: String containing snowflake ID (format: XXX_flake_id_YYY.ZZZ)
        
    Returns:
        Snowflake ID as integer, or None if not found
    """
    try:
        # Find start position 
        idx_start = filename.find('flake_') + 6 # +6 as flake_ is of length 6
        filename = filename[idx_start:]
        
        # Find stop position 
        idx_stop = filename.find('_')

        if idx_stop == -1:  # -1 because .find() returns -1 if not found
            return None
            
        # Extract and convert ID
        filename = filename[:idx_stop]  
        return int(filename)
    
    
    except Exception as e:
        logger.error("Could not retrieve snowflake ID from string: %s", filename)

        return None

def extract_identifiers(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract both camera and snowflake IDs from filename.
    
    Args:
        filename: MASC image filename
        
    Returns:
        Tuple of (camera_id, snowflake_id), None for any ID not found
    """
    cam_id = get_cam_id(filename)
    flake_id = get_snowflake_id(filename)

    return cam_id, flake_id