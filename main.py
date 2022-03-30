import argparse
import pathlib
import os
import scipy.optimize

from utils import plot_images, load_images
from image_manipulation import transform_image
from cost_functions import cost_function


def main(params):
    # load images
    fixed_image, moving_image = load_images(params)

    # Give some initial values to the transformation parameters
    x0 = [75, -15, -15]

    # Choose with similarity metric to use

    similarity_metric = "MI"

    result_params = scipy.optimize.fmin(cost_function, x0, args=(fixed_image, moving_image, similarity_metric))

    # Transform the moving images with the found parameters
    result_image = transform_image(moving_image, result_params)

    plot_images(result_image, fixed_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-fp", "--fixed_path", default=os.path.join(current_directory, "misc/ct_fixed.png"),
                        help="Path to the fixed image")
    parser.add_argument("-mp", "--moving_path", default=os.path.join(current_directory, "misc/ct_moving.png"),
                        help="Path to the moving image")

    args = parser.parse_args()

    main(args)
