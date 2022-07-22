import scipy.ndimage as ndimage


import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import scipy.ndimage as ndimage
import nibabel as nib

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
    volume = np.load(params.offset_volume_path).astype(np.float32)
    transform = volumes.create_transform_matrix(0, 0, 0, 7, 7, 7)

    t = perf_counter()
    volume_normal_affine = ndimage.affine_transform(volume, transform)
    t_normal_affine = perf_counter() - t

    t = perf_counter()
    i, j, k = volume.shape
    i_vals, j_vals, k_vals = np.meshgrid(range(j), range(j), range(k), indexing='ij')
    coords = np.array([i_vals, j_vals, k_vals]).transpose((1, 2, 3, 0))

    new_coords = nib.affines.apply_affine(transform, coords)
    new_coords = new_coords.transpose(3, 0, 1, 2)
    volume_map_coordinates = ndimage.map_coordinates(volume, new_coords)
    t_map_coordinates = perf_counter() - t

    print(5)

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

