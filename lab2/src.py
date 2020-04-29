from typing import Tuple, List
from operator import itemgetter

from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.morphology import binary_closing, remove_small_objects, remove_small_holes
import numpy as np
import cv2


def load_image(filepath: str) -> np.ndarray:
    """Takes image path and returns it as grayscale matrix."""
    return np.array(Image.open(filepath))


def show_image(image: np.ndarray):
    """Display image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")


def binarize(image: np.ndarray) -> np.ndarray:
    """Binarizes image and applies morphology to it."""
    space = image.shape[0] * image.shape[1]
    image = rgb2gray(image)
    image = image > image.mean()
    image = remove_small_objects(image, min_size=space // 5)
    image = remove_small_holes(image, space // 3)
    image = binary_closing(image)
    image = image.astype(np.uint8)
    image[image > 0] = 255
    return image


def get_contours_and_hull(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns binarized image biggest contour and its convex hull point indices.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=False)
    return contour, hull


def get_tips_and_valleys(
    contour: np.ndarray, hull: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns all possible tips and valleys.
    """
    defects = cv2.convexityDefects(contour, hull).reshape(-1, 4)
    contour = contour.reshape(-1, 2)
    tips, valleys = [], []
    mindistance = 3000
    while len(tips) < 5 and mindistance > 0:
        tips, valleys = [], []
        for start, end, farpoint, distance in defects:
            if distance > 3000:
                tips.append(contour[start])
                valleys.append(contour[farpoint])
        mindistance -= 300
    tips, valleys = np.array(tips), np.array(valleys)

    if tips.shape[0] < 5:
        raise Exception("Failed to find 5 tip candidates.")
    # find the best tips
    tips_center = np.mean(tips, axis=0)
    tips_to_center = tips - tips_center
    norms = np.linalg.norm(tips_to_center, axis=1)
    tips = tips[np.argsort(norms)[:5]]

    valleys_to_center = valleys - tips_center
    norms = np.linalg.norm(valleys_to_center, axis=1)
    valleys = valleys[np.argsort(norms)[:4]]
    return tips, valleys


def get_tip_valley_sequence(
    tips: np.ndarray, valleys: np.ndarray
) -> List[Tuple[int, int]]:
    tips = tips.tolist()
    valleys = valleys.tolist()
    tips = sorted(tips, key=itemgetter(0))
    valleys = sorted(valleys, key=itemgetter(0))
    dots = []
    for i in range(9):
        if i % 2 == 0:
            dots.append(tips[i // 2])
        else:
            dots.append(valleys[i // 2])
    return dots


def save_image(source: np.ndarray, dots: List[Tuple[int, int]], filename: str):
    fig, ax = plt.subplots(figsize=(10, 10))
    dots = list(zip(*dots))
    ax.imshow(source)
    ax.plot(*dots, "go")
    ax.plot(*dots, "green")
    plt.savefig(filename)
    print(f"Successfully saved into {filename}")
