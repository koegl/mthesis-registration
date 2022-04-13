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
    similarity_metric = "ncc"
    optimiser = "scipy"

    # Give some initial values to the transformation parameters
    perspective_transform = np.eye(3, dtype=np.float64) - 0.001 * np.random.randn(3, 3)
    affine_transform = [70, -5, -15, 0.8, 0.8]
    rigid_transform = [1, 5, -6]

    rigid_transform_list = [[1.1,   5,  -5.1],
                            [1,     5,   5],
                            [1,    -5,   5],
                            [1,    -5,  -5]]

    affine_transform_list = [[5,   5,  -5,   0.90,   1.10],
                             [5, -5, -5, 0.99, 0.88],
                             [5, -5, 5, 0.9, 0.90],
                             [5, 10, 10, 0.95, 0.95]]
    perspective_transform_list = [
                                  [[1.00508982e+00,  1.87032770e-02, -1.84965684e-02],
                                   [1.03065424e-02,  1.00111112e+00, -8.10246664e-03],
                                   [5.19517157e-03, -4.79838201e-05,  1.00094246e+00]],

                                  [[1.00527385e+00, - 1.83330496e-03,  2.60316135e-02],
                                   [-7.79680251e-04,  9.95856185e-01, - 8.55454218e-03],
                                   [-2.09975064e-03,  6.75339944e-03,  9.97413635e-01]],

                                  [[1.00014738e+00, - 4.26601060e-04, - 9.73921893e-04],
                                   [-1.78478695e-03,  9.98879786e-01,  3.71800820e-04],
                                   [-1.40905630e-03,  1.95891125e-03,  1.00154902e+00]],

                                  [[1.00052328e+00, - 1.87763715e-03,  5.09990405e-04],
                                   [2.11848499e-03, 9.98868515e-01, - 4.70762342e-04],
                                   [-9.03543672e-04, - 1.15540717e-04,  9.98057178e-01]]
                                  ]

    transform_list = perspective_transform_list
    result_params_list = []
    moving_image_display_list = []
    combined = np.zeros((fixed_image.shape[0], fixed_image.shape[1], len(transform_list)))

    for i in range(len(transform_list)):

        initial_transform = np.asarray(transform_list[i])

        result_params_list.append(optimise(optimiser, initial_transform, fixed_image, moving_image, similarity_metric,
                                  params.patch_size))

        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
        # ax0.imshow(transform_image(moving_image, initial_transform) - fixed_image, cmap='gray')
        # ax0.set_title("before reg")
        # ax1.imshow(transform_image(moving_image, result_params_list[i]) - fixed_image, cmap='gray')
        # ax1.set_title("after reg")
        # plt.show()
        # return
        # Transform the moving images with the found parameters
        combined[:, :, i] = transform_image(moving_image, result_params_list[i])

        # transform the original moving image with the initial transformation
        moving_image_display_list.append(transform_image(moving_image, initial_transform))

    # calculate variance of the combined results
    combined_variance = np.var(combined, axis=2)
    combined_variance = combined_variance / np.max(combined_variance)

    moving_overlayed_with_var = create_two_image_overlay(moving_image, combined_variance, alpha=0.6, cmap="plasma")

    # Display the output
    fig = plt.figure(figsize=(12, 6))
    ax0 = plt.subplot(2, 4, 1)
    ax1 = plt.subplot(2, 4, 2)
    ax2 = plt.subplot(2, 4, 5)
    ax3 = plt.subplot(2, 4, 6)
    ax4 = plt.subplot(1, 2, 2)

    ax0.imshow(fixed_image + moving_image_display_list[0], cmap=plt.cm.gray)
    ax0.set_title("Perspective state 1")
    ax0.axis('off')
    ax1.imshow(fixed_image + moving_image_display_list[1], cmap=plt.cm.gray)
    ax1.set_title("Perspective state 2")
    ax1.axis('off')
    ax2.imshow(fixed_image + moving_image_display_list[2], cmap=plt.cm.gray)
    ax2.set_title("Perspective state 3")
    ax2.axis('off')
    ax3.imshow(fixed_image + moving_image_display_list[3], cmap=plt.cm.gray)
    ax3.set_title("Perspective state 4")
    ax3.axis('off')
    ax4.imshow(moving_overlayed_with_var)
    ax4.set_title("Variance of the results of the 4 perspective registrations")
    ax4.axis('off')

    fig.tight_layout()

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
