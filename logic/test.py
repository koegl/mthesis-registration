import matplotlib.pyplot as plt
import numpy as np
import tqdm
from tqdm import trange

import torch
import torch.nn as nn

from utils import calculate_accuracy, get_label_id_from_label
from visualisations import display_tensor_and_label, display_volume_slice
from dataloader import get_test_loader
from architectures.densenet3d import DenseNet
from logic.patcher import Patcher
from visualisations import visualise_per_class_accuracies
from logic.patcher import Patcher


def test(model, test_loader):
    patcher = Patcher("", "", "", 10, False)
    x = patcher.label_to_offset_dict

    length_of_test_loader = len(test_loader)

    test_accuracy = 0.0

    acc_list = np.zeros(24)
    class_amount = np.zeros(24)

    counter = 0
    labels_per_disp_dict = {"-16": 0, "-8": 1, "-4": 2, "4": 3, "8": 4, "16": 5, "0": 6, "7": 7}
    disp_per_label_dict = {str(v): int(k) for k, v in labels_per_disp_dict.items()}

    with torch.no_grad():

        t = trange(len(test_loader), desc=f"Testing patches ({test_accuracy:.2f})", leave=True)
        test_loader = iter(test_loader)
        for _ in t:
            data, label = next(test_loader)

            data = data.to(torch.float32)

            label_np = label.to(torch.float32).squeeze().detach().cpu().numpy()

            idx_label_x = label_np[0:8].argmax()
            idx_label_y = label_np[8:16].argmax()
            idx_label_z = label_np[16:24].argmax()

            class_amount[idx_label_x] += 1
            class_amount[idx_label_y + 8] += 1
            class_amount[idx_label_z + 16] += 1

            test_output = model(data)

            # idx_output_x = test_output[:, 0:8].squeeze().detach().cpu().numpy().argmax()
            # idx_output_y = test_output[:, 8:16].squeeze().detach().cpu().numpy().argmax()
            # idx_output_z = test_output[:, 16:24].squeeze().detach().cpu().numpy().argmax()

            acc_x = calculate_accuracy(test_output[:, 0:8], label[:, 0:8])
            acc_y = calculate_accuracy(test_output[:, 8:16], label[:, 8:16])
            acc_z = calculate_accuracy(test_output[:, 16:24], label[:, 16:24])
            acc = (acc_x + acc_y + acc_z) / 3

            test_accuracy += acc / length_of_test_loader

            if acc_x >= 1.0:
                acc_list[idx_label_x] += 1
            if acc_y >= 1.0:
                acc_list[idx_label_y + 8] += 1
            if acc_z >= 1.0:
                acc_list[idx_label_z + 16] += 1

            # if the label is 'registered'
            # if idx_label == 1:
            #
            #     if idx_label == idx_output:
            #         title = "Comparison of two patches that have [0, 0, 0] displacement.\n\nPrediction: CORRECT"
            #     else:
            #         title = "Comparison of two patches that have [0, 0, 0] displacement.\n\nPrediction: WRONG"

                # _ = display_volume_slice(data.squeeze().detach().cpu().numpy(), title)
                # print(5)

            t.set_description(f"Testing patches ({test_accuracy:.2f})")
            t.refresh()

            counter += 1
            if counter == 200:
                break

    return test_accuracy, acc_list, class_amount


# load model
model = DenseNet()
model_params = torch.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments/5-dense-400k-tripleL-rosy-moon/model_epoch36_valacc0.994.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()

# load test data
loader = get_test_loader("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments/test data - 20211129_craini_golby/resliced_perfect_false")
# loader = get_test_loader("/Users/fryderykkogl/Data/patches/data_npy")

test_accuracy, acc_list, class_amount = test(model, loader)

# visualise
patcher = Patcher("", "", "", 10, False)
x = patcher.offsets
visualise_per_class_accuracies(acc_list/class_amount, x, f"Test per-class accuracies for {len(loader)} 'perfect' patches")
plt.savefig("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments" + "/test_per_class_accuracies_true.png", dpi=250)
