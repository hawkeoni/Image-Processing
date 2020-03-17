from typing import Tuple, List
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
    new_matr[dy // 2 : dy // 2 + ymax - ymin, dx // 2 : dx // 2 + xmax - xmin] = matr[
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


def get_sides_from_rectangle(rectangle: np.ndarray) -> List[Tuple[np.ndarray]]:
    a, b, c, d = rectangle
    sides = [(a, b), (b, c), (c, d), (d, a)]
    return sides


def _area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a: int, b: int, c: int, d: int) -> bool:
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    return max(a, c) <= min(b, d)


def segment_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    """Return True if 2 segments intersect."""
    return (
        _intersect(a[0], b[0], c[0], d[0])
        and _intersect(a[1], b[1], c[1], d[1])
        and _area(a, b, c) * _area(a, b, d) <= 0
        and _area(c, d, a) * _area(c, d, b) <= 0
    )
