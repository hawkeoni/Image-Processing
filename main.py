import argparse
import logging

import numpy as np
from skimage.morphology import convex_hull_image

from src.utils import load_image, center_of_mass, get_cutout, draw_type
from src.segmentation import SEGMENTATION_TECHNIQUES
from src.form_detection import (
    find_rectangle_candidates,
    find_best_rectangle,
    classify_points,
    get_object_description,
    get_boundaries_and_corners,
)


# uncomment to disable loggint
# logging.basicConfig()
logging.basicConfig(level=logging.INFO)


def main(args):
    logging.info("Started loading image.")
    image = load_image(args.input_file)
    logging.info("Finished loading image.")
    segmentation_function = SEGMENTATION_TECHNIQUES[args.segmentation]
    logging.info("Started image segmentation.")
    segmented_image = segmentation_function(image)
    logging.info("Finished image segmentation.")
    num_segments = np.max(segmented_image)
    print(f"Number of detected jigsaw pieces: {num_segments}.")
    for i in range(1, num_segments + 1):
        logging.info(f"Start processing segment {i}.")
        subimage = segmented_image == i

        cutout = get_cutout(subimage)
        total_space = np.sum(cutout)
        best_space = 0
        space_target = 0.8
        while best_space <= space_target * total_space:
            boundaries, corners = get_boundaries_and_corners(cutout)
            space_target -= 0.05
            possible_rectangles = find_rectangle_candidates(
                cutout, corners, candidate_limit=10
            )
            best_rectangle = find_best_rectangle(possible_rectangles)
            zeros = np.zeros_like(cutout)
            for p in best_rectangle:
                zeros[p[0], p[1]] = 1
            best_space = np.sum(convex_hull_image(zeros))
            logging.info(
                f"Current space is {best_space} which is {best_space / total_space} of total space. "
                f"Required space is {space_target}."
            )
        boundary_points = np.argwhere(boundaries == 1)
        new_boundary_points, classes = classify_points(best_rectangle, boundary_points)
        description = get_object_description(
            best_rectangle,
            new_boundary_points,
            classes,
            center_of_mass(cutout),
            None,
            center_thr=12,
            max_thr=75,
        )
        print(f"Figure_{i}'s code is {description}.")
        center = center_of_mass(subimage)
        logging.info(f"Saving image to output.png")
        draw_type(image, center, description)
        image.save("output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Input image.")
    parser.add_argument(
        "--segmentation", type=str, required=True, help="Segmentation type."
    )
    parser.add_argument("--corner-candidates", type=int, help="Number of candidate corners for best rectangle "
                                                              "approximation.")
    args = parser.parse_args()
    main(args)
