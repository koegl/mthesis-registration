# script to test inference one patch at a time
import numpy as np
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
import glob
import os
import nibabel as nib

from logic.patcher import Patcher
from helpers.visualisations import display_volume_slice, visualise_per_class_accuracies, plot_offsets
from helpers.utils import patch_inference
from architectures.densenet3d import DenseNet


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

patch_centres = [[59, 154, 54], [59, 154, 108], [59, 154, 162], [59, 231, 54], [59, 231, 108], [59, 231, 162], [59, 308, 54], [59, 308, 108], [59, 308, 162], [59, 385, 54], [59, 385, 108], [59, 385, 162], [118, 77, 54], [118, 77, 108], [118, 77, 162], [118, 154, 54], [118, 154, 108], [118, 154, 162], [118, 231, 54], [118, 231, 108], [118, 231, 162], [118, 308, 54], [118, 308, 108], [118, 308, 162], [118, 385, 54], [118, 385, 108], [118, 385, 162], [177, 77, 54], [177, 77, 108], [177, 77, 162], [177, 154, 54], [177, 154, 108], [177, 154, 162], [177, 231, 54], [177, 231, 108], [177, 231, 162], [177, 308, 54], [177, 308, 108], [177, 308, 162], [177, 385, 54], [177, 385, 108], [177, 385, 162]]


volume_fixed = nib.load("/Users/fryderykkogl/Data/patches/test_nii_small/1.nii.gz").get_fdata()
volume_fixed /= np.max(volume_fixed)
volume_fixed = volume_fixed.astype(np.float16)
volume_offset = nib.load("/Users/fryderykkogl/Data/patches/test_nii_small/2_1.nii.gz").get_fdata()
volume_offset /= np.max(volume_offset)
volume_offset = volume_offset.astype(np.float16)

model = DenseNet(num_init_features=64)
model_params = torch.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Models/39-dense-515k-mr-vocal-sweep3-10/model_epoch7_valacc0.910.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()

original_offsets = patcher.offsets
original_offsets[0, :] = np.array([0, 0, 0])

offset = np.array([6, 0, 0])

for centre in patch_centres:

    # check bounds of patches
    bounds = patcher.get_bounds(centre, offset)
    if patcher.in_bounds(volume_fixed.shape, bounds) is False:
        continue
    if any(i < 0 for i in bounds):  # check if any bounds are negative
        continue

    patch_fixed, patch_offset = patcher.extract_overlapping_patches(volume_fixed, volume_offset, centre, offset)
    patch = np.stack((patch_fixed, patch_offset), 0)

    # check black pixels
    patch_thresh = patch.copy()
    patch_thresh[patch_thresh < 0.05] = 0
    if np.count_nonzero(patch[0, ...]) / (32 ** 3) < 0.25 or \
        np.count_nonzero(patch[1, ...]) / (32 ** 3) < 0.25:
        continue

    e_d, model_output, predicted_probabilities = patch_inference(model, patch, original_offsets)

    print(f"[{e_d[0]:.2f}, {e_d[1]:.2f}, {e_d[2]:.2f} ]")

print(5)
