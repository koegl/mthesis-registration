from cost_functions import rigid_transformation_cost_function
from image_manipulation import rigid_transform
from utils import plot_images

import os.path
import numpy as np
from PIL import Image
import argparse
import pathlib
import time
import scipy.optimize


def main(params):
    # load images
    mr = Image.open(params.mr_path)
    mr = np.asarray(mr).astype('float64') / 255
    us = Image.open(params.ultrasound_path)
    us = np.asarray(us).astype('float64') / 255

    # this initial transformation is important - changing it too much will lead the optimiser into a local minimum
    initial_transform = [75, -15, -15]

    start_time_parallel = time.time()
    result_parameters = scipy.optimize.fmin(rigid_transformation_cost_function, initial_transform, args=(mr, us, "mi"))
    compute_time = time.time() - start_time_parallel

    result_image = rigid_transform(us, result_parameters[0], result_parameters[1], result_parameters[2])

    plot_images(result_image, mr)

    print("\033[0;0m\n")
    print("Compute time = {}".format(compute_time))
    print("Resulting parameters:\n{}".format(result_parameters))

    print(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-up", "--ultrasound_path", default=os.path.join(current_directory, "misc/ct_moving.png"),
                        help="Path to the Ultrasound image")
    parser.add_argument("-mp", "--mr_path", default=os.path.join(current_directory, "misc/ct_fixed.png"),
                        help="Path to the MR image")
    parser.add_argument("-ps", "--patch_size", default=9, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
