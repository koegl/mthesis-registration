import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import scipy.ndimage as ndimage

import logic.patcher
from logic.patcher import Patcher
import helpers.visualisations as visualisations
import helpers.utils as utils
import helpers.volumes as volumes


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

    return resulting_vector


def main(params):
    print("oneminus")

    np.set_printoptions(suppress=True)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    np.set_printoptions(precision=2)

    # create Patcher
    patcher = Patcher()

    volume_fixed = np.load(params.fixed_volume_path).astype(np.float32)
    volume_offset = np.load(params.offset_volume_path).astype(np.float32)

    model = utils.load_model_for_inference(params.model_path)

    original_offsets = patcher.offsets
    original_offsets[0, :] = np.array([0., 0., 0.])

    patch_centres = logic.patcher.Patcher.get_uniform_patch_centres(volume_fixed.shape)

    start = perf_counter()

    counter = 0

    transform = volumes.create_transform_matrix(0, 0, 0, 16, 16, 16)
    volume_offset_t = ndimage.affine_transform(volume_offset, transform)
    volume_offset_t_original = np.copy(volume_offset_t)

    compounded_transform = np.eye(4)

    while True:
        # _ = visualisations.display_two_volume_slices(np.stack((volume_offset_t, volume_offset_t_original), 0))

        variance_list = []
        ed_list = []

        for centre in patch_centres:
            patch_fixed, patch_offset = patcher.extract_overlapping_patches(volume_fixed, volume_offset_t, centre, None)

            patch = np.stack((patch_fixed, patch_offset), 0)

            predicted_probabilities = utils.patch_inference_simple(model, patch)

            e_d, variance = utils.calculate_ed_and_variance(predicted_probabilities, original_offsets, remove_unrelated=True)
            ed_list.append(e_d)
            variance_list.append(variance)

        plt.show()

        points_in = np.array(patch_centres)
        temp_points_in = np.array(points_in)
        temp_ed_list = np.array(ed_list)
        points_out = temp_points_in + temp_ed_list

        # weights = 1 / np.array(variance_list)
        weights = 1 - (np.array(variance_list) / np.max(np.array(variance_list)))
        affine_transformation = utils.calculate_affine_transform(points_in, points_out, weights)

        for ix, iy in np.ndindex(affine_transformation.shape):
            if np.abs(affine_transformation[ix, iy]) < 0.001:
                affine_transformation[ix, iy] = 0.0

        compounded_transform = np.matmul(compounded_transform, np.linalg.inv(affine_transformation))
        compounded_transform[3, 0] = 0
        compounded_transform[3, 1] = 0
        compounded_transform[3, 2] = 0
        compounded_transform[3, 3] = 1

        volume_offset_t = ndimage.affine_transform(volume_offset_t_original, compounded_transform)

        print(f"It: {counter+1}\n"
              f"predicted transform:\n{affine_transformation}\n"
              f"accumulated transform:\n{compounded_transform}\n\n\n")
        counter += 1

        # if counter >= 20 or all(np.abs(resulting_vector[i]) <= 0.2 for i in range(3)):
        temp = affine_transformation.copy()
        temp[0, 0] = 0
        temp[1, 1] = 0
        temp[2, 2] = 0
        temp[3, 3] = 0

        if counter >= 20 or all(np.abs(temp[iy, ix]) <= 0.5 for iy, ix in np.ndindex(temp.shape)):
            break

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    print(f"\n\n\nTrue transform:\n{transform}\n")
    print(f"Resulting transform:\n{np.linalg.inv(compounded_transform)}\n")
    print(f"Time: {perf_counter() - start} for {counter} iterations")
    print(f"{len(patch_centres)} patches used")

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
