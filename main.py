from cost_functions import rigid_transformation_cost_function, affine_transformation_cost_function
from image_manipulation import rigid_transform, affine_transform
from utils import plot_images, pad_smaller_image, plot

import os.path
import numpy as np
from PIL import Image
import argparse
import pathlib
import time
import scipy.optimize
import pybobyqa
import cv2


def main(params):
    # load images
    fixed = Image.open(params.mr_path)
    fixed = np.asarray(fixed).astype('float64') / 255
    moving = Image.open(params.ultrasound_path)
    moving = np.asarray(moving).astype('float64') / 255

    fixed = fixed[:, :, 0] + fixed[:, :, 1] + fixed[:, :, 2] + fixed[:, :, 3]

    fixed, moving = pad_smaller_image(fixed, moving)

    # resize
    quotient = 500/fixed.shape[0]
    width = int(fixed.shape[0] * quotient)
    height = int(fixed.shape[1] * quotient)
    fixed = cv2.resize(fixed, (height, width), interpolation=cv2.INTER_AREA)
    moving = cv2.resize(moving, (height, width), interpolation=cv2.INTER_AREA)

    fixed = np.pad(fixed, 100)
    moving = np.pad(moving, 100)

    # this initial transformation is important - changing it too much will lead the optimiser into a local minimum
    initial_transform = [1, 1, 1, 1, 1]

    start_time = time.time()
    # you can use two metrics: lc2 and mi (mutual information)
    optimisation_result = scipy.optimize.fmin(affine_transformation_cost_function, initial_transform,
                                              args=(fixed, moving, "ssd"))
    result_parameters_pybobyqa = optimisation_result# .x
    compute_time_pybobyqa = time.time() - start_time

    result_image = affine_transform(moving,
                                    result_parameters_pybobyqa[0],
                                    result_parameters_pybobyqa[1],
                                    result_parameters_pybobyqa[2],
                                    result_parameters_pybobyqa[3],
                                    result_parameters_pybobyqa[4]
                                   )
    plot_images(fixed, moving, result_image)

    print("\033[0;0m\n")
    print("Compute time scipy = {}".format(compute_time_pybobyqa))
    print("Resulting parameters (scipy):\t{}".format(result_parameters_pybobyqa))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-up", "--ultrasound_path", default=os.path.join(current_directory, "misc/us_cone_template.png"),
                        help="Path to the Ultrasound image")
    parser.add_argument("-mp", "--mr_path", default=os.path.join(current_directory, "misc/us_cone.png"),
                        help="Path to the MR image")
    parser.add_argument("-ps", "--patch_size", default=9, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
