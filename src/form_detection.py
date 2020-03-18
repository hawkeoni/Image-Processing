from typing import Tuple, List
from itertools import combinations, permutations
import random

import numpy as np
from tqdm import tqdm
from skimage.morphology import convex_hull_image
from sklearn.metrics import pairwise_distances
from skimage import segmentation
from skimage.feature import corner_harris, corner_peaks

from src.utils import (
    get_angle_cos,
    num_combinations,
    point_to_segment_distance,
    segment_intersect,
    get_sides_from_rectangle,
)
from src.heap import Heap


def get_boundaries_and_corners(cutout: np.ndarray, num_candidates: int = 22) -> Tuple[np.ndarray, np.ndarray]:
    """Gets dots creating boundaries and possible corners from Harris algorithm."""
    boundaries = segmentation.boundaries.find_boundaries(cutout)
    corners = corner_peaks(
        corner_harris(boundaries, k=0.2), min_distance=10, num_peaks=200
    ).tolist()
    random.shuffle(corners)
    return boundaries, np.array(corners[:num_candidates])


def find_best_angle_permutation(points: Tuple[np.array]) -> Tuple[float, np.array]:
    """Returns best permutation of 4 points to create a rectangle based on angles."""
    assert len(points) == 4
    best_cos = 6
    best_perm = None
    for perm in permutations(points):
        cur_cos = (
            get_angle_cos(np.array([perm[0], perm[1], perm[2]]))
            + get_angle_cos(np.array([perm[1], perm[2], perm[3]]))
            + get_angle_cos(np.array([perm[2], perm[3], perm[0]]))
            + get_angle_cos(np.array([perm[3], perm[0], perm[1]]))
        )
        if cur_cos < best_cos:
            best_cos = cur_cos
            best_perm = perm
    return best_cos, best_perm


def find_best_rectangle(points: List[Tuple[np.array]]) -> Tuple[float, np.array]:
    """Finds the best rectangle by angles from rectangle list."""
    points = list(map(find_best_angle_permutation, points))
    return sorted(points, key=lambda x: x[0])[0][1]


def find_rectangle_candidates(
    cutout: np.array, points: np.array, candidate_limit: int = 20
) -> List[Tuple[np.array]]:
    """Returns a list of rectangles with maximum space overlap with original figure."""
    heap = Heap()
    for quad in tqdm(combinations(points, 4), total=num_combinations(len(points), 4)):
        zeros = np.zeros_like(cutout)
        for p in quad:
            zeros[p[0], p[1]] = 1
        chull = convex_hull_image(zeros)
        space_part = np.sum(chull * cutout)
        heap.insert((space_part, quad))
    best_space, _ = heap.peek()
    retlist = []
    total = 0
    while total < candidate_limit and heap.peek()[0] >= 0.9 * best_space:
        space, cur_points = heap.pop()
        retlist.append(cur_points)
    return retlist


def classify_points(
    rectangle: np.ndarray, boundary_points: np.ndarray, thr: float = 0.75
):
    """Classifies boundary points into 4 sides of a rectangle."""
    sides = get_sides_from_rectangle(rectangle)
    unclassified_points = []
    return_points = []
    classes = []
    # classifying points based on closest side
    for point_idx, point in enumerate(boundary_points):
        distances = np.array(
            [point_to_segment_distance(*side, point) for side in sides]
        )
        top1, top2 = distances.argsort()[:2]
        dist1, dist2 = distances[top1], distances[top2]
        if dist1 >= thr * dist2 and len(classes) > 5:
            unclassified_points.append(point_idx)
        else:
            return_points.append(point)
            classes.append(top1)
    point_distances = pairwise_distances(
        boundary_points[unclassified_points], np.array(return_points)
    )
    # point_distances - unclassified_points, classified_points
    # classifying points via nearest neighbours
    for i, point_idx in enumerate(unclassified_points):
        closest = point_distances[i].argsort()
        closest = closest[:25].tolist()
        closest_classes = [classes[idx] for idx in closest]
        classes.append(np.bincount(closest_classes).argmax())
        return_points.append(boundary_points[point_idx])
    return np.array(return_points), np.array(classes)


def get_object_description(
    rectangle: np.ndarray,
    boundary_points: np.ndarray,
    classes: np.ndarray,
    mass_center: np.ndarray,
    cutout_shape: Tuple[int, int] = None,
    center_thr: int = None,
    max_thr: int = None,
) -> str:
    if center_thr is None and max_thr is None:
        assert cutout_shape is not None
        side_len = (cutout_shape[0] + cutout_shape[1]) / 2
        center_thr = side_len / 15
        max_thr = side_len / 7
    assert len(classes) == len(boundary_points)
    peninsula, bay = 0, 0
    sides = get_sides_from_rectangle(rectangle)
    for side_cls, (p1, p2) in enumerate(sides):
        side_points = boundary_points[classes == side_cls]
        side_mass_center = np.mean(boundary_points[classes == side_cls], axis=0)
        intersection = segment_intersect(side_mass_center, mass_center, p1, p2)
        center_distance = point_to_segment_distance(p1, p2, side_mass_center)
        side_points_distance = []
        for p in side_points:
            side_points_distance.append(point_to_segment_distance(p1, p2, p))
        most_distance = np.array(side_points_distance)
        most_distance.sort()

        most_distance = most_distance[-int(len(side_points_distance) * 0.1)]
        print(most_distance, center_distance)
        if most_distance >= max_thr and center_distance >= center_thr:
            if intersection:
                peninsula += 1
            else:
                bay += 1
    return f"P{peninsula}B{bay}"
