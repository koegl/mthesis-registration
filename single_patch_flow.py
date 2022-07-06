# script to test inference one patch at a time
import argparse
import numpy as np
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
import glob
import os
import nibabel as nib

from logic.patcher import Patcher
import helpers.visualisations as visualisations
import helpers.utils as utils
from helpers.volumes import mark_patch_borders


def convergence_check(offset: 'np.ndarray', patch_centres: list, success_rate: float, idx, max_iter: int = 10)\
                      -> (bool, float, str):

    if all(np.abs(d) < 2 for d in offset):
        break_reason = "\nAll offsets are below 1.0. Done iterating"
        success_rate += 1 / len(patch_centres)
        return True, success_rate, break_reason, idx
    elif idx == max_iter:
        break_reason = f"\nMax iteration of {max_iter} reached. Done iterating"
        return True, success_rate, break_reason, idx
    elif any(np.abs(d) > 20 for d in offset):
        break_reason = "\nAn offset exceeded 20. Done iterating"
        return True, success_rate, break_reason, idx
    else:
        return False, success_rate, "", idx


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

    # patch_centres = [[59, 154, 54], [59, 154, 108], [59, 154, 162], [59, 231, 54], [59, 231, 108], [59, 231, 162],
    # [59, 308, 54], [59, 308, 108], [59, 308, 162], [59, 385, 54], [59, 385, 108], [59, 385, 162], [118, 77, 54],
    # [118, 77, 108], [118, 77, 162], [118, 154, 54], [118, 154, 108], [118, 154, 162], [118, 231, 54], [118, 231,
    # 108], [118, 231, 162], [118, 308, 54], [118, 308, 108], [118, 308, 162], [118, 385, 54], [118, 385, 108], [118,
    # 385, 162], [177, 77, 54], [177, 77, 108], [177, 77, 162], [177, 154, 54], [177, 154, 108], [177, 154, 162],
    # [177, 231, 54], [177, 231, 108], [177, 231, 162], [177, 308, 54], [177, 308, 108], [177, 308, 162], [177, 385,
    # 54], [177, 385, 108], [177, 385, 162]] subset which at offset 0 are in bounds etc
    patch_centres = [[118, 154, 54],
                     [118, 154, 108],
                     [118, 154, 162],
                     [118, 231, 54], [118, 231, 108], [118, 231, 162],
                     [118, 308, 54], [118, 308, 108], [118, 308, 162], [118, 385, 54], [118, 385, 108], [118, 385, 162],
                     [177, 77, 54], [177, 77, 162], [177, 154, 54], [177, 154, 108], [177, 154, 162], [177, 231, 54],
                     [177, 231, 108], [177, 231, 162], [177, 308, 54], [177, 308, 108], [177, 308, 162], [177, 385, 54],
                     [177, 385, 108], [177, 385, 162]]

    success_rate = 0
    idx = 0

    for centre in patch_centres:
        offset = np.array([-16., 0., -16.])
        initial_offset = offset.copy()
        offset_list = []
        file_string = ""
        file_string += f"centre: {centre}\n\n"

        # iterate until convergence
        while True:

            break_val, success_rate, break_reason, idx = convergence_check(offset, patch_centres, success_rate, idx)
            idx += 1
            if break_val:
                break

            patch_fixed, patch_offset = patcher.extract_overlapping_patches(volume_fixed, volume_offset, centre, offset)
            patch = np.stack((patch_fixed, patch_offset), 0)

            e_d, model_output, predicted_probabilities = utils.patch_inference(model, patch, original_offsets)

            offset_list.append(e_d)

            offset = offset - e_d
            offset = np.round(offset).astype(int)

        total_offset = np.array(offset_list)

        file_string += f"Initial offset:\t\t[{initial_offset[0]:.2f} {initial_offset[1]:.2f} {initial_offset[2]:.2f}]\n"\
                       f"Total offset:\t\t{np.sum(total_offset, 0)}\n"\
                       f"Remaining offset:\t{initial_offset - np.sum(total_offset, 0)}"\
                       f"\n"

        file_string += "\nAll offsets:\n"
        for idx, offset in enumerate(offset_list):
            file_string += f"{idx}: {offset}\n"

        file_string += break_reason

        plt.close(1)
        plt.close(2)
        visualisations.plot_offset_convergence(initial_offset, offset_list)

        # with open(
        #         f"/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Experiments/mr patch convergence/"
        #         f"centre{centre}_{initial_offset}.txt",
        #         "w+") as f:
        #     f.write(file_string)

        print("\n\n\n" + file_string)

        # create borders in volumes showing the patches
        centre_with_offset = list(np.array(centre).astype(int) + np.array(initial_offset).astype(int))
        volume_fixed_border = mark_patch_borders(volume_fixed, centre, 1.0, 16)
        volume_offset_border = mark_patch_borders(volume_offset, centre_with_offset, 1.0, 16)

        volumes = np.stack((volume_fixed_border, volume_offset_border), 0)

        _ = visualisations.display_two_volume_slices(volumes)
        ss = 7

    print("\n\nDONE\n\n")
    print(f"Success rate: {success_rate}")


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


"""
# check bounds of patches
bounds = patcher.get_bounds(centre, offset)
if patcher.in_bounds(volume_fixed.shape, bounds) is False:
    continue
if any(i < 0 for i in bounds):  # check if any bounds are negative
    continue

        # check black pixels
patch_thresh = patch.copy()
patch_thresh[patch_thresh < 0.05] = 0
if np.count_nonzero(patch_thresh[0, ...]) / (32 ** 3) < 0.25 or \
    np.count_nonzero(patch_thresh[1, ...]) / (32 ** 3) < 0.25:
    continue
"""


