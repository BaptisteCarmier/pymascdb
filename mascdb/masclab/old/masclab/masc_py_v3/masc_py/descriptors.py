
import numpy as np
from skimage.measure import regionprops
from typing import Dict, Any

def process_basic_descriptors(mask: np.ndarray) -> Dict[str, Any]:
    props = regionprops(mask.astype(int))
    if not props:
        return {}
    p = max(props, key=lambda x: x.area)
    return {
        "area": float(p.area),
        "perimeter": float(p.perimeter),
        "bbox": p.bbox,
        "eccentricity": float(getattr(p, "eccentricity", 0.0)),
        "solidity": float(getattr(p, "solidity", 0.0)),
    }

def process_new_descriptors(mask: np.ndarray) -> Dict[str, Any]:
    props = regionprops(mask.astype(int))
    if not props:
        return {}
    p = max(props, key=lambda x: x.area)
    return {
        "convex_area": float(getattr(p, "convex_area", p.area)),
        "extent": float(getattr(p, "extent", 0.0)),
    }
