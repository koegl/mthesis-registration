import os.path
import pathlib
import argparse
import numpy as np
from time import perf_counter
from scipy.ndimage import affine_transform
import nibabel as nib
import matplotlib.pyplot as plt

from utils import plot_images, load_nifti_image
from similarity_metrics import compute_similarity_metric
from optimisers import optimise
from utils_vis.visualization_utils import image_viewer
from LC2_similarity_metric_3D import lc2_3d, gradient_magnitude


def main(params):
    # load images
    fixed_image = load_nifti_image(params.fixed_path)
    moving_image = load_nifti_image(params.moving_path)

    # create initial affine transform (it's not homogeneous)
    initial_transform = np.zeros((3, 4))
    initial_transform[:3, :3] = np.eye(3) + np.random.randn(3, 3) * 0.0001
    initial_transform[:, 3] = np.random.randn(3) * 3

    # get transform from moving to fixed
    moving_to_fixed_transform = np.linalg.inv(moving_image.affine).dot(fixed_image.affine)

    # get image data
    fixed_data = fixed_image.get_fdata()
    moving_data = moving_image.get_fdata()
    moving_data_transformed = np.zeros_like(fixed_data)

    # move moving image to fixed image space
    moving_data = affine_transform(moving_data, moving_to_fixed_transform, output=moving_data_transformed)

    # combine images for display
    # combination = np.where(moving_data >= 1, moving_data * 0.8 + fixed_data * 0.2, fixed_data)
    # fixed_image_voxel_size = fixed_image.header['pixdim'][1:4]
    # image_viewer(combination, voxel_size=fixed_image_voxel_size)

    # Choose which similarity metric to use
    similarity_metric = "lc2"
    optimiser = "bobyqa"
    deformation_type = "affine"

    test_metric = lc2_3d(fixed_data, moving_data, gradient_magnitude(moving_data), 19)

    return

    start_time = perf_counter()
    result_params = optimise(optimiser, deformation_type, initial_transform, fixed_data, moving_data, similarity_metric,
                             params.patch_size)
    end_time = perf_counter()
    print(f"\nTime: {end_time - start_time}\n")

    # reshape transform to be homogeneous
    result_params = np.reshape(result_params, (3, 4))
    homogeneous_vec = np.zeros((1, 4))
    homogeneous_vec[0, 3] = 1
    result_params = np.concatenate((result_params, homogeneous_vec), axis=0)

    # result_params = np.eye(4)
    # translation = np.zeros((1, 3))
    # translation[0][0] = 50
    # result_params[:3, 3] = translation

    # Transform the moving images with the found parameters
    result_data = affine_transform(moving_data, result_params)

    # calculate the initial and final metrics
    initial_metric = compute_similarity_metric(fixed_data, moving_data, similarity_metric, params.patch_size)
    final_metric = compute_similarity_metric(fixed_data, result_data, similarity_metric, params.patch_size)

    # print the initial and final metrics and transform parameters with 2 siffificant digits
    print(f"Initial metric({similarity_metric}) value = {initial_metric:0.3f}\n"
          f"Final metric({similarity_metric}) value =   {final_metric:0.3f}")
    print(f"\nInitial transform:\n{np.around(initial_transform, 3)}\n")
    print(f"Final transform:\n{np.around(result_params, 3)}\n")

    # combination = np.where(result_data >= 1, result_data * 0.8 + fixed_data * 0.2, fixed_data)
    # image_viewer(combination, voxel_size=fixed_image_voxel_size)

    # create a nifti image from the result data
    result_image = nib.nifti1.Nifti1Image(result_data, fixed_image.affine, header=moving_image.header)
    nib.save(result_image, params.moving_path.replace(".nii", "_result.nii"))
    print(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-fp", "--fixed_path", default=os.path.join(current_directory, "/Users/fryderykkogl/Data/nifti_test/exported/20210713_craini_bi/Intra-op MR/3D AX T1 Post-contrast Intra-op Thin-cut.nii"),
                        help="Path to the fixed image")
    parser.add_argument("-mp", "--moving_path", default=os.path.join(current_directory, "/Users/fryderykkogl/Data/nifti_test/exported/20210713_craini_bi/Intra-op US/US1.nii"),
                        help="Path to the moving image")
    parser.add_argument("-ps", "--patch_size", default=19, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
