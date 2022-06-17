import numpy as np
import torch
import glob
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from offsets import offsets_single as offsets
from joblib import Parallel, delayed
import multiprocessing
from ast import literal_eval

from logic.patcher import Patcher
from helpers.visualisations import display_volume_slice, visualise_per_class_accuracies, plot_offsets
from helpers.utils import patch_inference, get_label_from_label_id, softmax_sq
from architectures.densenet3d import DenseNet
from architectures.vit_standard_3d import ViTStandard3D
from time import perf_counter, sleep


def create_irs(new_offset, i, model):
    # delete directory if it exists
    if os.path.exists(f"/Users/fryderykkogl/Data/patches/val_npy_{i}"):
        shutil.rmtree(f"/Users/fryderykkogl/Data/patches/val_npy_{i}")
    os.mkdir(f"/Users/fryderykkogl/Data/patches/val_npy_{i}")

    # generate patches
    patcher = Patcher(
        load_directory=f"/Users/fryderykkogl/Data/patches/val_nii_small",
        save_directory=f"/Users/fryderykkogl/Data/patches/val_npy_{i}",
        file_type="nii.gz",
        centres_per_dimension=3,
        perfect_truth=False,
        patch_size=32,
        scale_dist=1.5,
        offset_multiplier=1,
        rescale=True,
        save_type="float16",
        offsets=[new_offset]
    )
    original_patcher = Patcher("", "", "", 10, "")
    original_offsets = original_patcher.offsets.copy()
    original_offsets[0] = np.asarray([0, 0, 0])
    # create new patches
    patcher.create_and_save_all_patches_and_labels()

    # get a list of patches
    patches = glob.glob(f"/Users/fryderykkogl/Data/patches/val_npy_{i}/*.npy")
    patches.sort()
    counter = 0

    for patch_path in tqdm(patches):
        patch = np.load(patch_path)
        true_offset = new_offset

        # predict displacement
        e_d, model_output, predicted_probabilities = patch_inference(model, patch, original_offsets)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

        # display current offsets
        model_output[model_output < 0.005] = 0
        model_output /= np.max(model_output)
        packets = []
        for k in range(len(model_output)):
            prob = model_output[k]
            off = original_offsets[k]
            if prob > 0:
                packets.append((off, prob))

        path = "/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Experiments/" + f"/{true_offset}_{counter}.png"
        plot_offsets(true_offset, e_d, packets, path)

        # plt.close()
        counter += 1

    if len(patches) == 0:
        return

    if os.path.exists(f"/Users/fryderykkogl/Data/patches/val_npy_{i}"):
        shutil.rmtree(f"/Users/fryderykkogl/Data/patches/val_npy_{i}")


# get the model
model_inference = DenseNet(num_init_features=64)
model_dict = torch.load(
    "/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Models/39-dense-515k-mr-vocal-sweep3-10/model_epoch7_valacc0.910.pt",
    map_location=torch.device('cpu'))
model_inference.load_state_dict(model_dict['model_state_dict'])
model_inference.eval()

# create_irs(offsets[0], 0, model_inference)

# for i in range(len(offsets)):
#     create_irs(offsets[i], i, model_inference)

num_cores = multiprocessing.cpu_count()
return_list = Parallel(n_jobs=num_cores)(delayed(create_irs)(offsets[i], i, model_inference)
                                         for i in range(len(offsets)))

print("="*30)
print("Done")
print("="*30)
