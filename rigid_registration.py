# script to test inference one patch at a time
import argparse
import numpy as np
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
import glob
import os
import nibabel as nib
from time import perf_counter

from logic.patcher import Patcher
import helpers.visualisations as visualisations
import helpers.utils as utils
from helpers.volumes import mark_patch_borders


def main(params):
    fig = None
    np.set_printoptions(suppress=True)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    # generate patches
    patcher = Patcher(load_directory="", save_directory="", file_type="nii.gz",
                      centres_per_dimension=6, perfect_truth=False, rescale=True, save_type="float16")

    volume_fixed = np.load(params.fixed_volume_path)
    volume_offset = np.load(params.offset_volume_path)

    model = utils.load_model_for_inference(params.model_path)

    original_offsets = patcher.offsets
    original_offsets[0, :] = np.array([0., 0., 0.])

    patch_centres = []

    cpd = 10
    patch_size = 32

    volume_size = volume_fixed.shape

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

    offset = np.array([2., -2., 0.])
    initial_offset = offset.copy()
    resulting_vector_list = []

    start = perf_counter()

    counter = 0

    while True:

        if all(np.abs(offset[i]) <= 1 for i in range(3)) or counter >= 40:
            break

        variance = []
        ed_list = []

        for centre in patch_centres:
            patch_fixed, patch_offset = patcher.extract_overlapping_patches(volume_fixed, volume_offset, centre, offset)
            patch = np.stack((patch_fixed, patch_offset), 0)

            e_d, model_output, predicted_probabilities = utils.patch_inference(model, patch, original_offsets)

            variance.append(utils.calculate_variance(predicted_probabilities, original_offsets))

            ed_list.append(e_d)
            ss = 7

        variance = np.array(variance)
        variance = 1 - (variance / np.max(variance))
        ed_list = np.array(ed_list)

        resulting_vector = np.dot(variance, ed_list) / len(variance)
        resulting_vector_list.append(resulting_vector)

        new_offset = offset - resulting_vector
        offset = new_offset

        print(f"It: {counter}, predicted vector {resulting_vector}, current acc vector {np.sum(np.array(resulting_vector_list), 0)}, new offset {offset}")
        counter += 1

    print(f"\n\n\nTrue vector: {initial_offset}")
    print(f"Resulting vector: {np.sum(np.array(resulting_vector_list), 0)}")
    print(f"Time: {perf_counter() - start} for {counter} iterations")

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
