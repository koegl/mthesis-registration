from cost_functions import rigid_transformation_cost_function, affine_transformation_cost_function
from image_manipulation import rigid_transform, affine_transform
from utils import plot_images, pad_images_to_same_size, plot, resize_image

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
    template = cv2.imread(params.template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    us = cv2.imread(params.us_path)
    us = cv2.cvtColor(us, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    us, template = pad_images_to_same_size(us, template)
    us = resize_image(us, 500)
    template = resize_image(template, 500)
    us = np.pad(us, 100)
    template = np.pad(template, 100)

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

    parser.add_argument("-tp", "--template_path", default=os.path.join(current_directory, "misc/us_cone_template.png"),
                        help="Path to the cone template")
    parser.add_argument("-up", "--us_path", default=os.path.join(current_directory, "misc/us_cone.png"),
                        help="Path to the Ultrasound image")

    args = parser.parse_args()

    main(args)
