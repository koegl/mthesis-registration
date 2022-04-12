import os.path
import pathlib
import argparse
import numpy as np
from time import perf_counter
from skimage import data, color, io, img_as_float
import matplotlib.pyplot as plt
from PIL import Image

from utils import plot_images, load_images, plot_one_image, create_two_image_overlay
from image_manipulation import transform_image
from similarity_metrics import compute_similarity_metric
from optimisers import optimise


def main(params):
    # load images
    fixed_image, moving_image = load_images(params)

    # Choose with similarity metric to use
    similarity_metric = "ssd"
    optimiser = "bobyqa"

    # Give some initial values to the transformation parameters
    perspective_transform = [[0.258, 0.966, 0.001],
                             [0.966, 0.258, 0.001],
                             [0.001, 0.001, 1.001]]
    affine_transform = [70, -5, -15, 0.8, 0.8]
    rigid_transform = [1, 5, -6]

    initial_transform = np.asarray(rigid_transform)

    rigid_transform_list = [[1.1, 5, -5.1],
                            [1, 5, 5],
                            [1, -5, 5],
                            [1, -5, -5]]

    result_image_list = []
    moving_image_display_list = []
    combined = np.zeros((fixed_image.shape[0], fixed_image.shape[1], len(rigid_transform_list)))

    for i in range(len(rigid_transform_list)):

        initial_transform = np.asarray(rigid_transform_list[i])

        start_time = perf_counter()
        result_params = optimise(optimiser, initial_transform, fixed_image, moving_image, similarity_metric,
                                 params.patch_size)
        end_time = perf_counter()

        print(f"\nTime: {end_time - start_time}\n")

        # Transform the moving images with the found parameters
        result_image = transform_image(moving_image, result_params)
        result_image_list.append(result_image)
        combined[:, :, i] = result_image

        # calculate the initial and final metrics
        initial_metric = compute_similarity_metric(fixed_image, moving_image, similarity_metric, params.patch_size)
        final_metric = compute_similarity_metric(fixed_image, result_image, similarity_metric, params.patch_size)

        # transform the original moving image with the initial transformation
        moving_image_display = transform_image(moving_image, initial_transform)
        moving_image_display_list.append(moving_image_display)

        # print the initial and final metrics with 2 siffificant digits
        print(f"Initial metric({similarity_metric}) value = {initial_metric:0.3f}\n"
              f"Final metric({similarity_metric}) value =   {final_metric:0.3f}")

        # print numpy array with three significant digits
        # print(f"Initial transformation: {initial_transform:0.3f}")
        print(f"\nInitial transform: {np.around(initial_transform, 3)}")
        print(f"Final transform:   {np.around(result_params, 3)}")

        # plot_images(fixed_image, moving_image_display_list[i], result_image_list[i],
        #             main_title=f"{similarity_metric} --- {optimiser}")

    # calculate variance of the combined results
    combined_variance = np.var(combined, axis=2)
    combined_variance = combined_variance / np.max(combined_variance)

    moving_overlayed_with_var = create_two_image_overlay(moving_image, combined_variance, alpha=0.6, cmap="plasma")

    # Display the output
    plt.figure(figsize=(12, 6))
    ax0 = plt.subplot(2, 4, 1)
    ax1 = plt.subplot(2, 4, 2)
    ax2 = plt.subplot(2, 4, 5)
    ax3 = plt.subplot(2, 4, 6)
    ax4 = plt.subplot(1, 2, 2)

    ax0.imshow(fixed_image + moving_image_display_list[0], cmap=plt.cm.gray)
    ax0.set_title("Initial state 1")
    ax1.imshow(fixed_image + moving_image_display_list[1], cmap=plt.cm.gray)
    ax1.set_title("Initial state 2")
    ax2.imshow(fixed_image + moving_image_display_list[2], cmap=plt.cm.gray)
    ax2.set_title("Initial state 3")
    ax3.imshow(fixed_image + moving_image_display_list[3], cmap=plt.cm.gray)
    ax3.set_title("Initial state 4")
    ax4.imshow(moving_overlayed_with_var)
    ax4.set_title("Variance of the results of the 4 registrations")
    plt.show()

    print(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-fp", "--fixed_path", default=os.path.join(current_directory, "misc/ct_fixed.png"),
                        help="Path to the fixed image")
    parser.add_argument("-mp", "--moving_path", default=os.path.join(current_directory, "misc/ct_fixed.png"),
                        help="Path to the moving image")
    parser.add_argument("-ps", "--patch_size", default=19, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
