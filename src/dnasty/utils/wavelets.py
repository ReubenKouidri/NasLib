from __future__ import annotations

import numpy as np
import pywt
from skimage.transform import resize

__all__ = ["cmor", "mexh"]

IMAGE_SIZE = (128, 128)


def mexh(signal: np.ndarray, max_width: int) -> np.ndarray:
    """Mexican-hat continuous wavelet transform resized to a 128x128 image."""
    widths = range(1, max_width)
    img, _ = pywt.cwt(signal, widths, "mexh")
    return resize(img, IMAGE_SIZE)


def cmor(signal: np.ndarray, max_width: int) -> np.ndarray:
    """Complex-Morlet CWT magnitude resized to a 128x128 image."""
    widths = range(1, max_width)
    img, _ = pywt.cwt(signal, widths, "cmor1.5-1")
    return resize(np.abs(img), IMAGE_SIZE)
