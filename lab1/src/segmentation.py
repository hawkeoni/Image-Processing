import numpy as np
from skimage import segmentation
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes
from skimage.measure import label
from PIL import Image


def felzenszwalb(image: Image) -> np.ndarray:
    return segmentation.felzenszwalb(
        image, scale=1000, min_size=image.height * image.width // 200
    )


def motley_segmentation(image: Image) -> np.ndarray:
    image = np.array(image)
    grayscale = rgb2gray(image)
    grayscale_bin = binary_erosion(grayscale > 0.28, selem=np.ones((10, 10)))
    grayscale_bin = remove_small_objects(
        grayscale_bin, min_size=image.shape[0] * image.shape[1] // 200
    )
    grayscale_bin = remove_small_holes(
        grayscale_bin, image.shape[0] * image.shape[1] // 900
    )
    return label(grayscale_bin, background=0)


SEGMENTATION_TECHNIQUES = {"monochrome": felzenszwalb, "motley": motley_segmentation}
