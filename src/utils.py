from typing import Tuple
from math import factorial

import numpy as np
from PIL import Image
from scipy import ndimage


def load_image(filepath: str):
    """Takes image path and returns it as PIL Image."""
    return Image.open(filepath)


def center_of_mass(matr: np.ndarray) -> Tuple[int, int]:
    """Takes image and returns its center of mass."""
    coords = ndimage.measurements.center_of_mass(matr)
    return tuple(map(int, coords))


def get_cutout(matr: np.ndarray) -> np.array:
    """
    Takes a puzzle and returns its cutout.
    Input matr should be matrix of 1 and 0.
    Pads image by 2 from all sides.
    """
    coords = np.argwhere(matr == 1)
    ys, xs = zip(*coords)
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    xlen = xmax - xmin
    dx = int(xlen * 0.2) + 4
    ylen = ymax - ymin
    dy = int(ylen * 0.2) + 4
    new_matr = np.zeros((ylen + dy, xlen + dx))
    new_matr[dy // 2: dy // 2 + ymax - ymin, dx // 2: dx // 2 + xmax - xmin] = matr[
        ymin:ymax, xmin:xmax
    ]
    return new_matr


def get_angle_cos(points) -> float:
    """Returns the cosine of angle a, b, c where a, b, c are the points in list."""
    assert len(points) == 3
    a, b, c = points
    ba = a - b
    bc = c - b
    banorm = np.linalg.norm(ba)
    bcnorm = np.linalg.norm(bc)
    cos = np.dot(ba, bc) / (banorm * bcnorm)
    return abs(cos)


def num_combinations(n: int, k: int):
    """Number of combinations of k elements from n elements."""
    return factorial(n) // factorial(n - k) // factorial(k)


def point_to_line_distance(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate distance from point p3 to line between p1 and p2."""
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
