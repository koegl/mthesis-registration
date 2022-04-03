import os.path
import pathlib
import argparse
import numpy as np
from utils import plot_images, load_images
from image_manipulation import transform_image
from skimage.transform import ProjectiveTransform


def main(params):
    # load images
    fixed_image, moving_image = load_images(params)
    plot_images(moving_image, main_title="Initial transform")
    # Give some initial values to the transformation parameters

    us_transform = [[0.66993, 0.321128, 0.669381],
                    [-0.367, 0.926994, -0.077414],
                    [-0.645372, - 0.193801, 0.738875]]
    initial_transform = np.asarray(us_transform)

    # transform the original moving image with the initial transformation
    moving_image = transform_image(moving_image, initial_transform)

    # plot_images(moving_image, main_title="After initial transformation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-fp", "--fixed_path", default=os.path.join(current_directory, "/Users/fryderykkogl/Data/datasets/resect/NIFTI/Case2/MRI/Case2-T1.nii"),
                        help="Path to the fixed image")
    parser.add_argument("-mp", "--moving_path", default=os.path.join(current_directory, "/Users/fryderykkogl/Data/datasets/resect/NIFTI/Case2/US/Case2-US-before.nii"),
                        help="Path to the moving image")
    parser.add_argument("-ps", "--patch_size", default=19, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
