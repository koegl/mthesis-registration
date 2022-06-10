import nibabel as nib
import numpy as np
import torch
from ast import literal_eval

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

true_offsets = [[0, 0, 0], [16, 4, -8], [4, 4, 4], [4, 0, 4], [0, 4, 4], [-4, -4, 0], [-4, 0, -4], [0, -4, -4]]
center = [350, 100, 300]

# get the model
# model = ViTStandard3D(dim=128, volume_size=32, patch_size=4, num_classes=20, channels=2, depth=6, heads=8, mlp_dim=2048,
#                       dropout=0.1, emb_dropout=0.1, device="cpu")
model = DenseNet(num_init_features=10)
model_dict = torch.load("/Users/fryderykkogl/Desktop/model_epoch14_valacc0.678.pt",
                        map_location=torch.device('cpu'))
model.load_state_dict(model_dict['model_state_dict'])
model.eval()
acc = 0

for i in range(len(true_offsets)):
    true_offset = true_offsets[i]
    packet = patcher.extract_overlapping_patches(volume, volume, center, offset=true_offset)

    patches = np.stack((packet[0], packet[1]), 0)

    # get offsets
    offsets = patcher.offsets.astype(np.float32)
    offsets[0] = np.asarray([0.0, 0.0, 0.0])

    # predict displacement
    print(true_offset)
    e_d = patch_inference(model, patches, offsets)

    acc += ssd(e_d, np.asarray(true_offset)) / len(true_offsets)

    # print results
    np.set_printoptions(precision=2)
    print(f"True displacement:\t\t{true_offset}\n"
          f"Predicted displacement:\t{e_d}\n")
    texx = 2
    # _ = display_volume_slice(patches, f"True displacement:         {true_offset}\n"
    #                                  f"Predicted displacement: {e_d}")

