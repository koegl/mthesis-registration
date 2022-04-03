import os.path
import pathlib
import argparse
import numpy as np
from time import perf_counter

from utils import plot_images, load_images
from image_manipulation import transform_image
from similarity_metrics import compute_similarity_metric
from optimisers import optimise


def main(params):
    # load images
    fixed_image, moving_image = load_images(params)

    # Give some initial values to the transformation parameters

    perspective_transform = [[0.258, 0.966, 0.001],
                             [0.966, 0.258, 0.001],
                             [0.001, 0.001, 1.001]]
    affine_transform = [70, -5, -15, 0.8, 0.8]
    rigid_transform = [75, -15, -15]

    initial_transform = np.asarray(perspective_transform)

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

    # calculate the initial and final metrics
    initial_metric = compute_similarity_metric(fixed_image, moving_image, similarity_metric, params.patch_size)
    final_metric = compute_similarity_metric(fixed_image, result_image, similarity_metric, params.patch_size)

    # transform the original moving image with the initial transformation
    moving_image = transform_image(moving_image, initial_transform)

    plot_images(fixed_image, moving_image, result_image,
                main_title=f"{similarity_metric} --- {optimiser}")

    print(f"Initial metric({similarity_metric}) value= {initial_metric:0.3f}\n"
          f"Final metric({similarity_metric}) value= {final_metric:0.3f}")
    print(result_params)


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
