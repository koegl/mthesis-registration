from cost_functions import cost_function, extract_mask, symmetry_regulariser
from image_manipulation import transform, scale_image, translate_image, affine_transform
from utils import plot_images, pad_images_to_same_size, plot, resize_image

import os.path
import numpy as np
import argparse
import pathlib
import scipy.optimize
import cv2
import pybobyqa


def preprocess_images(params):
    # load images
    template = cv2.imread(params.template_path)
    try:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    except:
        template = template.astype('float64') / 255

    us = cv2.imread(params.us_path)
    try:
        us = cv2.cvtColor(us, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    except:
        us = us.astype('float64') / 255

    us, template = pad_images_to_same_size(us, template)
    us = resize_image(us, 500)
    template = resize_image(template, 500)
    us = np.pad(us, 100)
    template = np.pad(template, 100)

    return us, template


def main(params):

    us, template = preprocess_images(params)

    us_ori = us.copy()
    template_ori = template.copy()

    # us = np.gradient(us)[0]
    # template = np.gradient(template)[0]

    # us[us > 0] = 1
    # template[template > 0] = 1

    # initial transformation
    transform_perspective = np.identity(3) + (np.random.rand(3, 3) - 0.5) * 0.0001
    transform_affine = np.asarray([0.0, -20.05, -50, 1.0, 1.0])
    transform_rigid = np.asarray([0.1, 0.1, 0.1])
    transform_init = transform_perspective

    optimisation_result = scipy.optimize.fmin(cost_function, transform_init,
                                               args=(us, template, "ssd", False))
    # transform_init = transform_init.flatten()
    # optimisation_result = pybobyqa.solve(cost_function, transform_init,
    #                                      args=(us, template, "mi", False))
    # optimisation_result = optimisation_result.x

    result_image = transform(template_ori, optimisation_result)

    template_ori = transform(template_ori, transform_init)

    plot_images(us_ori, template_ori, result_image)

    print("\033[0;0m\n")
    print("Resulting parameters:\n{}".format(optimisation_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-tp", "--template_path",
                        default=os.path.join(current_directory, "misc/ct_fixed.png"),
                        help="Path to the cone template")
    parser.add_argument("-up", "--us_path", default=os.path.join(current_directory, "misc/ct_moving.png"),
                        help="Path to the Ultrasound image")

    args = parser.parse_args()

    main(args)
