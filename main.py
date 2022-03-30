import numpy as np
from PIL import Image
import argparse
import pathlib
import os
import scipy.optimize

from utils import plot_images
from image_manipulation import transform_image
from cost_functions import cost_function


def main(params):
    # load images
    fixed_image = Image.open(params.fixed_path)
    fixed_image = np.asarray(fixed_image).astype('float64') / 255
    moving_image = Image.open(params.moving_path)
    moving_image = np.asarray(moving_image).astype('float64') / 255

    # Give some initial values to the transformation parameters
    x0 = [75, -15, -15]

    # Choose with similarity metric to use
    # similarity_metric = "SSD"
    # similarity_metric = "SAD"
    # similarity_metric = "NCC"
    similarity_metric = "MI"

    # Call fmin : https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.fmin.html
    # In short fmin needs a function (cost_function) and initial values for the parameters to minimize for (x0),
    # what is in args are additional parameters that are needed for our specific function.
    # fmin will return the values that after running the algorithm are minimazing the function
    result_params = scipy.optimize.fmin(cost_function, x0, args=(fixed_image, moving_image, similarity_metric))

    # Transform the moving images with the found parameters
    result_image = transform_image(moving_image, result_params)
    # Let's have a look at the result!
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
