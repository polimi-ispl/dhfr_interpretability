"""
Data utilities functions.
"""

# --- Libraries import
import rasterio
import numpy as np
import cv2
import ntpath
from sklearn.preprocessing import QuantileTransformer

# --- Helpers functions
def load_and_normalize(path: str, use_he: bool = False) -> np.array:
    """
    Load a generic image and normalize it as a float32 between -1 and 1
    following a normal distribution
    :param path: str, the path of the image
    :param use_he: bool, whether to perform histogram equalization following a normal distribution
    :return: np.array float32 of the normalized image
    """
    # Find extension first
    extension = ntpath.split(path)[-1].split('.')[-1]

    # Load the image according to the file type
    if extension == 'tiff':
        with rasterio.open(path) as src:
            img = src.read()
    elif extension == 'png':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif extension == 'npy':
        img = np.load(path)

    if use_he:
        # Histogram equalization
        img = QuantileTransformer(output_distribution='normal',
                                  random_state=42).fit_transform(img.reshape(-1, 1)).reshape(img.shape).astype(np.float32)
    else:
        # Perform a simple multiplicative scaling (used for the "standard" detectors)
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img /= 255
        elif img.dtype == np.uint16:
            img = img.astype(np.float32)
            img /= (2 ** 16 - 1)
            img *= 100
            img = np.clip(img, 0, 1).astype(np.float32)
        else:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())

    return np.squeeze(img)