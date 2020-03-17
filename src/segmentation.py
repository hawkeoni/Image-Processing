import numpy as np
from skimage import segmentation
from PIL import Image


def felzenszwalb(image: Image) -> np.ndarray:
    return segmentation.felzenszwalb(
        image, scale=1000, min_size=image.height * image.width // 200
    )
