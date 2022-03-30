import argparse
import pathlib
import os
import scipy.optimize
import numpy as np
import os.path
import pathlib
import argparse

import numpy as np
from PIL import Image
from time import perf_counter

from utils import plot_images, load_images
from image_manipulation import transform_image
from cost_functions import cost_function
from similarity_metrics import compute_similarity_metric
from lc2_similarity_2 import LC2_similarity_patch, gradient_magnitude


def main(params):
    # load images
    fixed_image, moving_image = load_images(params)

    # Give some initial values to the transformation parameters

    perspective_transform = [[0.258, 0.966, 0.001],
                             [0.966, 0.258, 0.001],
                             [0.001, 0.001, 1.001]]
    affine_transform = [70, -5, -15, 1.01, 1.01]
    rigid_transform = [75, -10, 10]
    start_time = perf_counter()
    end_time = perf_counter()

    initial_transform = np.asarray(rigid_transform)

    # Choose with similarity metric to use
    similarity_metric = "mi"
    optimiser = "scipy"

    start_time = perf_counter()
    result_params = optimise(optimiser, initial_transform, fixed_image, moving_image, similarity_metric,
                             params.patch_size)
    end_time = perf_counter()

    print(f"\nTime: {end_time - start_time}\n")

    # Transform the moving images with the found parameters
    result_image = transform_image(moving_image, result_params)

    initial_metric = compute_similarity_metric(fixed_image, moving_image, similarity_metric, params.patch_size)
    final_metric = compute_similarity_metric(fixed_image, result_image, similarity_metric, params.patch_size)

    moving_image = transform_image(moving_image, initial_transform)

    plot_images(fixed_image, moving_image, result_image)

    print(f"\n\n{initial_metric=}\n"
          f"{final_metric=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-fp", "--fixed_path", default=os.path.join(current_directory, "misc/ct_fixed.png"),
                        help="Path to the fixed image")
    parser.add_argument("-mp", "--moving_path", default=os.path.join(current_directory, "misc/ct_moving.png"),
                        help="Path to the moving image")
    parser.add_argument("-ps", "--patch_size", default=19, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
