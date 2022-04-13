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
    similarity_metric = "mi"
    optimiser = "scipy"

    # Give some initial values to the transformation parameters
    perspective_transform = np.eye(3, dtype=np.float64) - 0.002 * np.random.randn(3, 3)
    # perspective_transform[0, 2] = 5
    # perspective_transform[1, 2] = 15

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
                                  [[9.97962760e-01,  1.87656307e-03, -5.00000000e+00],
                                   [-1.77597560e-04,  1.00062371e+00, -5.00000000e+00],
                                   [-4.64119328e-04, -8.44921859e-05,  1.00048039e+00]],

                                  [[9.99622225e-01, - 1.45607355e-03, - 5.00000000e+00],
                                   [8.72527131e-04,  9.99464513e-01,  5.00000000e+00],
                                   [-1.43820521e-03,  1.38813994e-03,  1.00001295e+00]],

                                  # [[1.00022373e+00,  1.79532565e-03,  5.00000000e+00],
                                  #  [1.49052863e-03,  1.00074024e+00, -5.00000000e+00],
                                  #  [2.18009167e-03, - 1.62779992e-03,  1.00009343e+00]],

                                  [[1.00047720e+00, - 1.92540739e-04,  5.00000000e+00],
                                   [2.17379389e-03,  1.00074264e+00,  5.00000000e+00],
                                   [2.75700615e-04,  1.97223155e-04,  1.00077639e+00]],

                                  [[9.99974479e-01, - 8.74546788e-04,  5.00000000e+00],
                                   [-1.11169385e-03,  9.99651386e-01,  5.00000000e+00],
                                   [2.95366985e-04,  4.35728917e-04,  9.99527605e-01]]
    ]

    # perspective_transform_list = [perspective_transform]

    initial_perspective_transform = [[1.00000100e+00, 4.48563412e-07, 1.72053412e-07],
                                     [1.21248831e-06, 9.99999259e-01, 2.72091069e-06],
                                     [8.67668241e-07, 2.35931466e-06, 9.99999636e-01]]

    initial_transform = np.asarray(initial_perspective_transform)

    transform_list = perspective_transform_list
    result_params_list = []

    moving_image_list = []
    combined = np.zeros((fixed_image.shape[0], fixed_image.shape[1], len(transform_list)))

    for i in range(len(transform_list)):

        # apply transformation to the moving image (because moving and fixed are the same image)
        moving_image_list.append(transform_image(moving_image, np.asarray(perspective_transform_list[i])))

        result_params_list.append(optimise(optimiser, initial_transform, fixed_image, moving_image_list[i],
                                           similarity_metric, params.patch_size))

        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
        # ax0.imshow(moving_image_list[i] - fixed_image, cmap='gray')
        # ax0.set_title("before reg")
        # ax1.imshow(transform_image(moving_image, result_params_list[i]) - fixed_image, cmap='gray')
        # ax1.set_title("after reg")
        # plt.show()
        # return

        # Transform the moving images with the found parameters
        combined[:, :, i] = transform_image(moving_image, result_params_list[i])


    # calculate variance of the combined results
    combined_variance = np.var(combined, axis=2)
    combined_variance = combined_variance / np.max(combined_variance)

    # create an overlay of one of the registered images (in this case it's the first one) with the variance map
    moving_overlayed_with_var = create_two_image_overlay(combined[:, :, 0], combined_variance, alpha=0.6, cmap="plasma")

    # Display the output
    fig = plt.figure(figsize=(12, 6))
    ax0 = plt.subplot(2, 4, 1)
    ax1 = plt.subplot(2, 4, 2)
    ax2 = plt.subplot(2, 4, 5)
    ax3 = plt.subplot(2, 4, 6)
    ax4 = plt.subplot(1, 2, 2)

    ax0.imshow(fixed_image + moving_image_list[0], cmap=plt.cm.gray)
    ax0.set_title("Perspective state 1")
    ax0.axis('off')
    ax1.imshow(fixed_image + moving_image_list[1], cmap=plt.cm.gray)
    ax1.set_title("Perspective state 2")
    ax1.axis('off')
    ax2.imshow(fixed_image + moving_image_list[2], cmap=plt.cm.gray)
    ax2.set_title("Perspective state 3")
    ax2.axis('off')
    ax3.imshow(fixed_image + moving_image_list[3], cmap=plt.cm.gray)
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
