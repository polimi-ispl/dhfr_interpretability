"""
Common utils functions
Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
"""

# --- Libraries import
import numpy as np


# --- Utils
def uint82float32(img: np.array) -> np.array:
    """
    Normalize a uint8 image as float with values between 0-1
    :param img: np.array, uint8 image to normalize
    :return: np.array, float32 array normalized
    """
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img
