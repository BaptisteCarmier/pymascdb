
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_opening, square

def roi_detection(img: np.ndarray, process):
    if img.ndim == 3:
        img = (0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]).astype(img.dtype)
    thr = img > (img.mean() + img.std())
    thr = binary_opening(thr, square(3))
    lbl = label(thr)
    lbl = remove_small_objects(lbl, 50)
    rois = []
    for r in regionprops(lbl):
        rois.append(r.bbox)  # (min_row, min_col, max_row, max_col)
    return rois
