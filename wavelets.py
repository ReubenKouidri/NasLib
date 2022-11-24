import pywt
from skimage.transform import resize


def mexh(signal, widths):
    img, _ = pywt.cwt(signal, widths, "mexh")
    img.resize((128, 128))
    return img


def cmor(signal, widths):
    img, _ = pywt.cwt(signal, widths, 'cmor1.5-1')
    img = abs(img)
    img = resize(img, (128, 128))
    return img
