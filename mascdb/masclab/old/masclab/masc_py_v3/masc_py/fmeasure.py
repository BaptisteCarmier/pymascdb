
import numpy as np
from scipy.ndimage import sobel, gaussian_laplace
from pywt import wavedec2

def fmeasure(image: np.ndarray, method: str = 'TENG'):
    if image.ndim == 3:
        image = np.round(0.2989*image[...,0] + 0.5870*image[...,1] + 0.1140*image[...,2]).astype(image.dtype)
    I = image.astype(np.float64)
    m = method.upper()
    if m == 'TENG':
        gx = sobel(I, axis=1); gy = sobel(I, axis=0)
        return float(np.mean(gx*gx + gy*gy))
    elif m == 'GRAE':
        gx = sobel(I, axis=1); gy = sobel(I, axis=0)
        return float(np.sum(gx*gx + gy*gy))
    elif m == 'LAPV':
        L = gaussian_laplace(I, sigma=1.0)
        return float(np.var(L))
    elif m == 'WAVV':
        coeffs = wavedec2(I, 'db2', level=2)
        cH1, cV1, cD1 = coeffs[1]
        cH2, cV2, cD2 = coeffs[2]
        vals = [np.var(c) for c in (cH1,cV1,cD1,cH2,cV2,cD2)]
        return float(np.mean(vals))
    else:
        raise ValueError(f"Unsupported focus measure method: {method}")
