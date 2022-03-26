from cost_functions import cost_function, extract_mask, symmetry_regulariser
from image_manipulation import transform, scale_image, translate_image
from utils import plot_images, pad_images_to_same_size, plot, resize_image

import os.path
import numpy as np
import argparse
import pathlib
import scipy.optimize
import cv2


def preprocess_images(params):
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

    return us, template


def main(params):

    us, template = preprocess_images(params)

    us_ori = us.copy()
    template_ori = template.copy()

    us[us > 0] = 1
    template[template > 0] = 1

    # us = np.gradient(us)[0]
    # template = np.gradient(template)[0]

    # initial transformation
    transform_perspective = np.identity(3) # + (np.random.rand(3, 3) - 0.5) * 0.000001
    transform_affine = np.asarray([10, 70, 60, 0.9, 0.9])
    transform_rigid = np.asarray([30, 30, 60])
    transform_init = transform_affine

    optimisation_result = scipy.optimize.fmin(cost_function, transform_init,
                                              args=(us, template, "ncc", True))

    result_image = transform(template_ori, optimisation_result)

    template_ori = transform(template_ori, transform_init)

    plot_images(us_ori, template_ori, result_image)

    print("\033[0;0m\n")
    print("Resulting parameters:\n{}".format(optimisation_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-tp", "--template_path",
                        default=os.path.join(current_directory, "misc/us_cone_template.png"),
                        help="Path to the cone template")
    parser.add_argument("-up", "--us_path", default=os.path.join(current_directory, "misc/us_cone.png"),
                        help="Path to the Ultrasound image")

    args = parser.parse_args()

    main(args)
