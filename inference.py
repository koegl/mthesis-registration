import nibabel as nib
import numpy as np
import torch
from ast import literal_eval

from logic.patcher import Patcher
from helpers.visualisations import display_volume_slice, visualise_per_class_accuracies
from helpers.utils import patch_inference
from architectures.densenet3d import DenseNet
from architectures.vit_standard_3d import ViTStandard3D


def ssd(a, b):
    dif = a.ravel() - b.ravel()
    return np.dot(dif, dif)


# Load the volume
volume_fixed = nib.load("/Users/fryderykkogl/Data/patches/val_nii/49.nii.gz").get_fdata()
volume_offset = nib.load("/Users/fryderykkogl/Data/patches/val_nii/50_49.nii.gz").get_fdata()

# get a patch pair
patcher = Patcher("", "", "", 10, False)

results = []

true_offsets = [[0, 0, 0], [0, 0, 8], [4, 0, -4], [-8, 0, -8], [-16, -16, 0]]
center = [350, 200, 300]

# get the model
# model = ViTStandard3D(dim=128, volume_size=32, patch_size=4, num_classes=20, channels=2, depth=6, heads=8, mlp_dim=2048,
#                       dropout=0.1, emb_dropout=0.1, device="cpu")
model = DenseNet(num_init_features=64)
model_params = torch.load("/Users/fryderykkogl/Desktop/model_epoch3_valacc0.870.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()
acc = 0

x = patcher.offsets
x = [np.array2string(offset) for offset in x]

for i in range(len(true_offsets)):
    true_offset = true_offsets[i]
    packet = patcher.extract_overlapping_patches(volume_fixed, volume_offset, center, offset=true_offset)

    patches = np.zeros((2, 32, 32, 32))
    patches[0, ...] = packet[0]
    patches[1, ...] = packet[1]

    # get offsets
    offsets = patcher.offsets
    offsets[0] = np.asarray([0.0, 0.0, 0.0])

    # predict displacement
    print(true_offset)
    e_d = patch_inference(model, patches, offsets)

    acc += ssd(e_d, np.asarray(true_offset)) / len(true_offsets)

    # print results
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print(f"True displacement:\t\t{true_offset}\n"
          f"Predicted displacement:\t{e_d}\n")

    temp = 5
    # display_volume_slice(patches, f"True displacement:         {true_offset}\n"
    #                               f"Predicted displacement: {e_d}")


