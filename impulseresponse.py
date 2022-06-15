import numpy as np
import torch
import glob
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from offsets import offsets_xy as offsets
from joblib import Parallel, delayed
import multiprocessing

from logic.patcher import Patcher
from helpers.visualisations import display_volume_slice, visualise_per_class_accuracies
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
        load_directory=f"/Users/fryderykkogl/Data/patches/val_nii",
        save_directory=f"/Users/fryderykkogl/Data/patches/val_npy_{i}",
        file_type="nii.gz",
        centres_per_dimension=6,
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

    ir = np.zeros(20)
    ed = np.zeros(3)

    for patch_path in tqdm(patches):
        patch = np.load(patch_path)

        # predict displacement
        _, output, _ = patch_inference(model, patch, original_offsets)

        ir += output
        np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
        output[output < 0] = 0
        output[0] = 0
        output[1] = 0
        output = np.exp(output)
        sum_output = np.sum(output)
        if sum_output == 0:
            pass
        else:
            output /= sum_output
            ed_temp = np.matmul(output, original_offsets)
            print(f"True offset {new_offset}: {ed_temp}")
            ed += ed_temp

    if len(patches) == 0:
        return

    ir /= len(patches)
    ed /= len(patches)

    visualise_per_class_accuracies(ir, [np.array2string(x, separator=',') for x in original_patcher.offsets],
                                   f"Average IR for {len(patches)} patches with offset f{new_offset}\n" +
                                   f"Average E(d) f{ed}",
                                   lim=(-15, 15))
    plt.savefig(
        "/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Experiments/" +
        f"/ir_{new_offset}.png",
        dpi=250)
    plt.close()

    if os.path.exists(f"/Users/fryderykkogl/Data/patches/val_npy_{i}"):
        shutil.rmtree(f"/Users/fryderykkogl/Data/patches/val_npy_{i}")


# get the model
model_inference = DenseNet(num_init_features=64)
model_dict = torch.load(
    "/Users/fryderykkogl/Desktop/model_epoch3_valacc0.870.pt",
    map_location=torch.device('cpu'))
model_inference.load_state_dict(model_dict['model_state_dict'])
model_inference.eval()

# create_irs(offsets[0], 0, model_inference)

for i in range(len(offsets)):
    create_irs(offsets[i], i, model_inference)

num_cores = multiprocessing.cpu_count()
# return_list = Parallel(n_jobs=num_cores)(delayed(create_irs)(offsets[i], i, model_inference)
#                                          for i in range(len(offsets)))

print("="*30)
print("Done")
print("="*30)