from typing import Tuple, List
from itertools import combinations, permutations


import numpy as np
from tqdm import tqdm
from skimage.morphology import convex_hull_image

from src.utils import get_angle_cos, num_combinations
from src.heap import Heap


def find_best_rectangle(points: Tuple[np.array]) -> Tuple[float, np.array]:
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
    points = list(map(find_best_rectangle, points))
    return sorted(points, key=lambda x: x[0])[0][1]


def find_rectangle_candidate(
    cutout: np.array, points: np.array, candidate_limit: int = 100
) -> np.array:
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
    while total < candidate_limit and heap.peek()[0] >= 0.8 * best_space:
        space, cur_points = heap.pop()
        retlist.append(cur_points)
    return find_best_rectangle(retlist)
