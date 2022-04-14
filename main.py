import os.path
import pathlib
import argparse
import numpy as np
from time import perf_counter
from skimage import data, color, io, img_as_float
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from utils import plot_images, load_images, plot_one_image, mark_points_in_image, create_gird, calculate_distance_between_points
from image_manipulation import transform_image
from similarity_metrics import compute_similarity_metric
from optimisers import optimise


def main(params):
    # load images
    fixed_image, moving_image = load_images(params)

    # Choose with similarity metric to use
    similarity_metric = "mi"
    optimiser = "scipy"

    # load png image with cv2
    grid_image = cv2.imread("/Users/fryderykkogl/Downloads/grid.png")
    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)

    # Give some initial values to the transformation parameters
    perspective_transform = np.eye(3, dtype=np.float64) - 0.001 * np.random.randn(3, 3)
    perspective_transform[0, 2] = 5
    perspective_transform[1, 2] = 5

    perspective_transform_list = [
        [[9.98973341e-01, - 7.31692941e-05,  9.37318374e-04],
         [-2.23558816e-03,  1.00117543e+00, - 3.75353094e-04],
         [8.88458505e-04,  3.31434113e-04,  1.00444452e+00]],

        [[9.99975115e-01, - 3.13277331e-03,  2.01608106e-03],
         [-1.78004689e-03,  9.99731592e-01,  1.75828657e-03],
         [-1.67729496e-04,  5.81974154e-04,  1.00318653e+00]],

        [[1.00027271e+00, - 1.49952337e-04,  1.58446339e-03],
         [9.65119604e-04,  9.99892158e-01,  5.00000000e+00],
         [-4.65758781e-04, - 2.84519347e-04,  1.00073029e+00]],

        [[9.99458100e-01,  2.03181632e-03,  5.00000000e+00],
         [-1.64182089e-04,  9.99429672e-01,  5.00000000e+00],
         [-4.30677510e-04, - 2.53577463e-04,  9.99401458e-01]]
    ]

    # perspective_transform_list = [perspective_transform]

    initial_perspective_transform = [[1.00000100e+00, 4.48563412e-07, 1.72053412e-07],
                                     [1.21248831e-06, 9.99999259e-01, 2.72091069e-06],
                                     [8.67668241e-07, 2.35931466e-06, 9.99999636e-01]]

    initial_transform = np.asarray(initial_perspective_transform)

    transform_list = perspective_transform_list
    result_params_list = []
    true_transform_params_list = []

    moving_image_list = []
    registered_images_list = np.zeros((fixed_image.shape[0], fixed_image.shape[1], len(transform_list)))

    # extract grid from the moving image
    grid, points = create_gird(moving_image.shape, 8)

    # mark points in moving image
    moving_image = mark_points_in_image(moving_image, points)

    for i in range(len(transform_list)):

        # apply transformation to the moving image (because moving and fixed are the same image)
        moving_image_list.append(transform_image(moving_image, np.asarray(perspective_transform_list[i])))
        points_moving = cv2.perspectiveTransform(points, np.asarray(perspective_transform_list[i]))

        result_params_list.append(optimise(optimiser, initial_transform, fixed_image, moving_image_list[i],
                                           similarity_metric, params.patch_size))
        true_transform_params_list.append(np.linalg.inv(np.reshape(transform_list[i], (3, 3))))

        # Transform the moving images with the found parameters
        registered_images_list[:, :, i] = transform_image(moving_image_list[i], result_params_list[i])
        points_registered = cv2.perspectiveTransform(points_moving, np.reshape(result_params_list[i], (3, 3)))

        # calculate the error between the registered and true points (error being ecuclidean distance)
        dist = calculate_distance_between_points(points, points_registered)

        # transform points
        plt.figure()
        plt.imshow(registered_images_list[:, :, i], cmap='gray')
        plt.plot(points[:, :, 0], points[:, :, 1], 'ro', markersize=2)
        plt.plot(points_registered[:, :, 0], points_registered[:, :, 1], 'bo', markersize=2)

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


