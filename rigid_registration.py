import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.ndimage import affine_transform

from logic.patcher import Patcher
import helpers.visualisations as visualisations
import helpers.utils as utils
from helpers.volumes import mark_patch_borders


def calculate_resulting_vector(variance_list, ed_list, mode="inverse"):
    assert mode in ["inverse", "oneminus"], "mode must be either 'inverse' or 'oneminus'"
    assert len(variance_list) == len(ed_list), "variance and ed lists must have the same length"
    assert isinstance(variance_list, list), "variance list must be a list"
    assert isinstance(ed_list, list), "ed list must be a list"

    variance_list = np.array(variance_list)
    ed_list = np.array(ed_list)

    if mode == "inverse":
        variance = 1 / variance_list
        resulting_vector = np.dot(variance, ed_list) / np.sum(variance)
    else:
        variance_list = variance_list / np.max(variance_list)
        variance_list = 1 - variance_list
        resulting_vector = np.dot(variance_list, ed_list) / np.sum(variance_list)

    return resulting_vector# * 1.5


def get_uniform_patch_centres(volume_size, cpd=10, patch_size=32):

    patch_centres = []

    # get maximum dimension
    max_dim = np.max(volume_size)
    dimension_factor = volume_size / max_dim  # so we have a uniform grid in all dimensions

    offset_x = volume_size[0] / int(cpd * dimension_factor[0])
    offset_y = volume_size[1] / int(cpd * dimension_factor[1])
    offset_z = volume_size[2] / int(cpd * dimension_factor[2])

    for i in range(int(cpd * dimension_factor[0])):
        for j in range(int(cpd * dimension_factor[1])):
            for k in range(int(cpd * dimension_factor[2])):

                # multiply the offsets with the indices to get the i,j,k centre (starting at 16)
                centre_x = int(i * offset_x + 16)
                centre_y = int(j * offset_y + 16)
                centre_z = int(k * offset_z + 16)

                # check for out of bounds
                if centre_x < patch_size or centre_x > volume_size[0] - patch_size:
                    continue
                if centre_y < patch_size or centre_y > volume_size[1] - patch_size:
                    continue
                if centre_z < patch_size or centre_z > volume_size[2] - patch_size:
                    continue

                centre = [centre_x, centre_y, centre_z]

                patch_centres.append(centre)

    return patch_centres


def main(params):
    fig = None
    np.set_printoptions(suppress=True)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    # generate patches
    patcher = Patcher(load_directory="", save_directory="", file_type="nii.gz",
                      centres_per_dimension=6, perfect_truth=False, rescale=True, save_type="float16")

    volume_fixed = np.load(params.fixed_volume_path).astype(np.float32)
    volume_offset = np.load(params.offset_volume_path).astype(np.float32)

    model = utils.load_model_for_inference(params.model_path)

    original_offsets = patcher.offsets
    original_offsets[0, :] = np.array([0., 0., 0.])

    patch_centres = get_uniform_patch_centres(volume_fixed.shape)

    offset = np.array([12., 16., 16.])
    initial_offset = offset.copy()
    resulting_vector_list = []

    start = perf_counter()

    counter = 0

    transform = np.eye(4)
    transform[0:3, 3] = offset
    volume_offset_t = affine_transform(volume_offset, transform)
    volume_offset_t_original = np.copy(volume_offset_t)

    while True:
        _ = visualisations.display_two_volume_slices(np.stack((volume_offset_t, volume_offset_t_original), 0))

        variance_list = []
        ed_list = []

        for centre in patch_centres:
            patch_fixed, patch_offset = patcher.extract_overlapping_patches(volume_fixed, volume_offset_t, centre, None)
            # patch_fixed, patch_offset = patcher.extract_overlapping_patches(volume_fixed, volume_offset, centre, offset)
            patch = np.stack((patch_fixed, patch_offset), 0)

            e_d, model_output, predicted_probabilities = utils.patch_inference(model, patch, original_offsets)

            variance_list.append(utils.calculate_variance(predicted_probabilities, original_offsets))

            ed_list.append(e_d)
            ss = 7

        resulting_vector = calculate_resulting_vector(variance_list, ed_list, mode="oneminus")

        resulting_vector_list.append(resulting_vector)

        offset = offset - resulting_vector

        transform[0:3, 3] = - np.sum(np.array(resulting_vector_list), 0)
        volume_offset_t = affine_transform(volume_offset_t_original, transform)

        print(f"It: {counter+1}, predicted vector {resulting_vector}, current acc vector {np.sum(np.array(resulting_vector_list), 0)}, new offset {offset}")
        counter += 1

        if all(np.abs(offset[i]) <= 1 for i in range(3)) or counter >= 40:
            break
        if counter >= 20 or all(np.abs(resulting_vector[i]) <= 0.2 for i in range(3)):
            break

    print(f"\n\n\nTrue vector: {initial_offset}")
    print(f"Resulting vector: {np.sum(np.array(resulting_vector_list), 0)}")
    print(f"Time: {perf_counter() - start} for {counter} iterations")
    print(f"{len(patch_centres)} patches used")

    plt.close(1)
    visualisations.plot_offset_convergence(initial_offset, resulting_vector_list)

    print("\n\nDONE\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-mp", "--model_path", default="/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Models/"
                                                       "39-dense-515k-mr-vocal-sweep3-10/model_epoch7_valacc0.910.pt",
                        help="Path to the trained .pt model file")
    parser.add_argument("-fvp", "--fixed_volume_path", default="/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/"
                                                               "Experiments/mr patch convergence/data/49.npy",
                        help="Path to the fixed .npy volume")
    parser.add_argument("-ovp", "--offset_volume_path", default="/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/"
                                                                "Experiments/mr patch convergence/data/50_49.npy",
                        help="Path to the offset .npy volume")

    args = parser.parse_args()

    main(args)
