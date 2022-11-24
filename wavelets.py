import pywt
from skimage.transform import resize
import numpy as np


def mexh(signal: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    :param signal: 1D ndarray of the timeseries signal to be processed
    :param widths: 1D ndarray of the widths that the CWT will be computed for
    :return img: 2D ndarray representing the image of the transformed signal
    """
    img, _ = pywt.cwt(signal, widths, "mexh")
    img = resize(img, (128, 128))
    return img


def cmor(signal: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    uses the complex-Morlet CWT
    :param signal: 1D ndarray of the timeseries signal to be processed
    :param widths: 1D ndarray of the widths that the CWT will be computed for
    :return img: 2D ndarray representing the image of the transformed signal
    """
    img, _ = pywt.cwt(signal, widths, 'cmor1.5-1')
    img = abs(img)
    img = resize(img, (128, 128))
    return img
