import nibabel as nib
import numpy as np
import torch
from ast import literal_eval
import matplotlib.pyplot as plt
import glob
import os

from logic.patcher import Patcher
from helpers.visualisations import display_volume_slice, visualise_per_class_accuracies, plot_offsets
from helpers.utils import patch_inference
from architectures.densenet3d import DenseNet
from architectures.vit_standard_3d import ViTStandard3D


# get a patch pair
patcher = Patcher("", "", "", 10, False)

model = DenseNet(num_init_features=64)
model_params = torch.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Models/39-dense-515k-mr-vocal-sweep3-10/model_epoch7_valacc0.910.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()
acc = 0

x = patcher.offsets
x = [np.array2string(offset) for offset in x]
labels_to_offsets = patcher.label_to_offset_dict

patch_paths = glob.glob(os.path.join("/Users/fryderykkogl/Data/patches/val_npy", "*.npy"))
patch_paths.sort()

for path in patch_paths:

    patches = np.load(path)
    label = path.split('/')[-1].split("_")[1]
    true_offset = labels_to_offsets[label]

    # get offsets
    offsets = patcher.offsets
    offsets[0] = np.asarray([0.0, 0.0, 0.0])

    # predict displacement
    print(true_offset)

    e_d, model_output, predicted_probabilities = patch_inference(model, patches, offsets)

    # print results
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print(f"True displacement:\t\t{true_offset}\n"
          f"Predicted displacement:\t{e_d}\n")

    # display current offsets
    predicted_probabilities[predicted_probabilities < 0.005] = 0
    packets = []
    for k in range(len(predicted_probabilities)):
        prob = predicted_probabilities[k]
        off = offsets[k]
        if prob > 0:
            packets.append((off, prob))

    if true_offset == '[7,7,7]':
        true_offset = [0, 0, 0]
    else:
        true_offset = literal_eval(true_offset)

    plot_offsets(true_offset, e_d, packets)
    plt.savefig(
        "/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Experiments/" +
        f"/{true_offset}.png")
    plt.close()


