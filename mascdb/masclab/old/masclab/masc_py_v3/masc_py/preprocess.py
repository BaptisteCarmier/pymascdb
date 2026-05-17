
import numpy as np
from skimage import exposure, filters, morphology, util

def brightening(img: np.ndarray, process) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        img = util.img_as_ubyte(0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2])
    eq = exposure.equalize_adapthist(img, clip_limit=0.01)
    return util.img_as_ubyte(eq)

def edge_detection(img: np.ndarray, process):
    if img.ndim == 3:
        img = util.img_as_ubyte(0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2])
    edges = filters.sobel(img)
    binary = edges > (edges.mean() + edges.std()*0.5)
    binary = morphology.binary_opening(binary, morphology.square(3))
    binary = morphology.remove_small_objects(binary, 64)
    filled = morphology.remove_small_holes(binary, area_threshold=64)
    return binary, filled

def masking(img: np.ndarray, process):
    if img.ndim == 3:
        img = util.img_as_ubyte(0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2])
    t = filters.threshold_otsu(img)
    mask = img > t
    out = img.copy()
    out[~mask] = 0
    return out
