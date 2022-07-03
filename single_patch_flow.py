# script to test inference one patch at a time
import numpy as np
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
import glob
import os
import nibabel as nib

from logic.patcher import Patcher
from helpers.visualisations import display_two_volume_slices
from helpers.utils import patch_inference
from architectures.densenet3d import DenseNet


def main():
    np.set_printoptions(suppress=True)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    # generate patches
    patcher = Patcher(
        load_directory=f"/Users/fryderykkogl/Data/patches/val_nii_small",
        save_directory=f"/Users/fryderykkogl/Data/patches/val_npy",
        file_type="nii.gz",
        centres_per_dimension=6,
        perfect_truth=False,
        patch_size=32,
        scale_dist=1.5,
        rescale=True,
        save_type="float16",
    )

    volume_fixed = np.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Experiments/mr patch convergence/data/49.npy")
    volume_offset = np.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Experiments/mr patch convergence/data/50_49.npy")

    model = DenseNet(num_init_features=64)
    model_params = torch.load(
        "/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Models/39-dense-515k-mr-vocal-sweep3-10/model_epoch7_valacc0.910.pt",
        map_location=torch.device('cpu'))
    model.load_state_dict(model_params['model_state_dict'])
    model.eval()

    original_offsets = patcher.offsets
    original_offsets[0, :] = np.array([0., 0., 0.])

    # patch_centres = [[59, 154, 54], [59, 154, 108], [59, 154, 162], [59, 231, 54], [59, 231, 108], [59, 231, 162],
    # [59, 308, 54], [59, 308, 108], [59, 308, 162], [59, 385, 54], [59, 385, 108], [59, 385, 162], [118, 77, 54],
    # [118, 77, 108], [118, 77, 162], [118, 154, 54], [118, 154, 108], [118, 154, 162], [118, 231, 54], [118, 231,
    # 108], [118, 231, 162], [118, 308, 54], [118, 308, 108], [118, 308, 162], [118, 385, 54], [118, 385, 108], [118,
    # 385, 162], [177, 77, 54], [177, 77, 108], [177, 77, 162], [177, 154, 54], [177, 154, 108], [177, 154, 162],
    # [177, 231, 54], [177, 231, 108], [177, 231, 162], [177, 308, 54], [177, 308, 108], [177, 308, 162], [177, 385,
    # 54], [177, 385, 108], [177, 385, 162]] subset which at offset 0 are in bounds etc
    patch_centres = [[118, 154, 54], [118, 154, 108], [118, 154, 162], [118, 231, 54], [118, 231, 108], [118, 231, 162],
                     [118, 308, 54], [118, 308, 108], [118, 308, 162], [118, 385, 54], [118, 385, 108], [118, 385, 162],
                     [177, 77, 54], [177, 77, 162], [177, 154, 54], [177, 154, 108], [177, 154, 162], [177, 231, 54],
                     [177, 231, 108], [177, 231, 162], [177, 308, 54], [177, 308, 108], [177, 308, 162], [177, 385, 54],
                     [177, 385, 108], [177, 385, 162]]

    success_rate = 0

    for centre in patch_centres:
        # print("\n\n\n\n\n\n\n\n======================================")
        # print("======================================")
        # print(f"centre: {centre}\n")
        # centre = [118, 308, 54]
        offset = np.array([3., 3., 3.])
        initial_offset = offset.copy()
        idx = 0
        max_iter = 10
        offset_list = []
        file_string = ""
        file_string += f"centre: {centre}\n\n"

        while True:

            if all(np.abs(d) < 2 for d in offset):
                break_reason = "\nAll offsets are below 1.0. Done iterating"
                success_rate += 1/len(patch_centres)
                break
            if idx == max_iter:
                break_reason = f"\nMax iteration of {max_iter} reached. Done iterating"
                break
            if any(np.abs(d) > 20 for d in offset):
                break_reason = "\nAn offset exceeded 20. Done iterating"
                break

            # plt.close("all")

            patch_fixed, patch_offset = patcher.extract_overlapping_patches(volume_fixed, volume_offset, centre, offset)
            patch = np.stack((patch_fixed, patch_offset), 0)

            e_d, model_output, predicted_probabilities = patch_inference(model, patch, original_offsets)

            offset_list.append(e_d)

            offset = offset - e_d
            offset = np.round(offset).astype(int)

            # print(f"[{e_d[0]:.2f}, {e_d[1]:.2f}, {e_d[2]:.2f}];\tcentre={centre}")
            idx += 1

            # _ = display_two_volume_slices(patch)

        total_offset = np.array(offset_list)
        # print(f"Initial offset:\t\t[{initial_offset[0]:.2f} {initial_offset[1]:.2f} {initial_offset[2]:.2f}]\n"
        #       f"Total offset:\t\t{np.sum(total_offset, 0)}\n"
        #       f"Remaining offset:\t{initial_offset - np.sum(total_offset, 0)}"
        #       f"\n")

        file_string += f"Initial offset:\t\t[{initial_offset[0]:.2f} {initial_offset[1]:.2f} {initial_offset[2]:.2f}]\n"\
                       f"Total offset:\t\t{np.sum(total_offset, 0)}\n"\
                       f"Remaining offset:\t{initial_offset - np.sum(total_offset, 0)}"\
                       f"\n"

        # print("All offsets:")
        file_string += "\nAll offsets:\n"
        for idx, offset in enumerate(offset_list):
            # print(f"{idx}: {np.round(offset)}")
            file_string += f"{idx}: {np.round(offset)}\n"

        # print(break_reason)
        file_string += break_reason

        # with open(
        #         f"/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Experiments/mr patch convergence/"
        #         f"centre{centre}_{initial_offset}.txt",
        #         "w+") as f:
        #     f.write(file_string)

        ss = 7

    print("\n\nDONE\n\n")
    print(f"Success rate: {success_rate}")


if __name__ == "__main__":
    main()


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


