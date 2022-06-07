import nibabel as nib
import numpy as np
import torch
from ast import literal_eval
from itertools import permutations

from logic.patcher import Patcher
from visualisations import display_volume_slice
from utils import patch_inference
from architectures.densenet3d import DenseNet
from architectures.vit_standard_3d import ViTStandard3D


def ssd(a, b):
    dif = a.ravel() - b.ravel()
    return np.dot(dif, dif)


# Load the volume
# volume = nib.load("/Users/fryderykkogl/Data/patches/data_nii/0.nii.gz").get_fdata()
# np.save("/Users/fryderykkogl/Data/patches/numpy_volume", volume)
volume = np.load("/Users/fryderykkogl/Data/patches/numpy_volume.npy")

# get a patch pair
patcher = Patcher("", "", "", 10, False)

results = []

true_offsets = [[]]

single_offsets = [-16, -16, -16,
                  -8, -8, -8,
                  -4, -4, -4,
                  0,
                  4, 4, 4,
                  8, 8, 8,
                  16, 16, 16]

# get all possible permutations of length 3 of the displacements
all_permutations = permutations(single_offsets, 3)

# many duplicates, so remove them with set() and then change to list()
all_permutations = list(set(all_permutations))

all_permutations = [[0, 0, 0], [0, 0, 16], [4, 0, 0], [-8, 0, 0], [0, 16, 0]]

center = [350, 100, 300]

# get the model
model = ViTStandard3D(dim=128, volume_size=32, patch_size=4, num_classes=20, channels=2, depth=6, heads=8, mlp_dim=2048,
                      dropout=0.1, emb_dropout=0.1, device="cpu")
model = DenseNet()
model_params = torch.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments/5-dense-400k-tripleL-rosy-moon/model_epoch36_valacc0.994.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()
acc = 0

for i in range(len(all_permutations)):
    true_offset = all_permutations[i]
    packet = patcher.extract_overlapping_patches(volume, volume, center, offset=true_offset)

    patches = np.zeros((2, 32, 32, 32))
    patches[0, ...] = packet[0]
    patches[1, ...] = packet[1]

    # get offsets
    # offsets = np.asarray([np.asarray(literal_eval(temp)) for temp in patcher.offsets]).astype(np.float32)
    # offsets[0] = np.asarray([0.0, 0.0, 0.0])

    # predict displacement
    print(true_offset)
    dim_vals = np.asarray([
        [-16, -8,  -4, 4, 8, 16, 0, 0],
        [-16, -8, -4, 4, 8, 16, 0, 0],
        [-16, -8, -4, 4, 8, 16, 0, 0]
    ])
    e_d_x, e_d_y, e_d_z = patch_inference(model, patches, dim_vals)

    acc_x = (e_d_x - true_offset[0])**2
    acc_y = (e_d_y - true_offset[1])**2
    acc_z = (e_d_z - true_offset[2])**2

    acc += (acc_x + acc_y + acc_z) / 3

    # print results
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print(f"True displacement:\t\t{true_offset}\n"
          f"Predicted displacement:\t[{np.round(e_d_x[0])}, {np.round(e_d_y[0])}, {np.round(e_d_z[0])}]\n")
    print(f"SSD:\t\t\t{acc[0]:.2f}")

    # display_volume_slice(patches, f"True displacement:         {true_offset}\n"
    #                               f"Predicted displacement: {e_d}")

