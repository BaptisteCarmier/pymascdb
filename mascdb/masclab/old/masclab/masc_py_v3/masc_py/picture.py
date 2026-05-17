
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
from .preprocess import brightening, edge_detection, masking
from .roi import roi_detection
from .descriptors import process_basic_descriptors, process_new_descriptors

def MASC_picture_process(image_path: Path, process) -> Dict[str, Any]:
    img = np.array(Image.open(image_path))
    img_b = brightening(img, process)
    edges, filled = edge_detection(img_b, process)
    rois = roi_detection(img_b, process)

    feats: List[Dict[str, Any]] = []
    for (r0,c0,r1,c1) in rois:
        roi_mask = np.zeros_like(edges, dtype=bool)
        roi_mask[r0:r1, c0:c1] = filled[r0:r1, c0:c1]
        basic = process_basic_descriptors(roi_mask)
        newd  = process_new_descriptors(roi_mask)
        feats.append({"basic": basic, "extra": newd})
    return {
        "path": str(image_path),
        "num_rois": len(rois),
        "features": feats
    }
