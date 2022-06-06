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

    acc_list = np.zeros(20)
    class_amount = np.zeros(20)

    counter = 0

    # todo also log another array only when prediction is not 'unrelated'

    with torch.no_grad():

        t = trange(len(test_loader), desc=f"Testing patches ({test_accuracy:.2f})", leave=True)
        test_loader = iter(test_loader)
        for _ in t:
            data, label = next(test_loader)
            data = data.to(torch.float32)
            label = label.to(torch.float32)

            label_np = label.squeeze().detach().cpu().numpy()
            idx_label = label_np.argmax()

            class_amount[idx_label] += 1

            label_id = get_label_id_from_label(label_np)
            offset = x[label_id]
            # display_volume_slice(data.squeeze().detach().cpu().numpy(), offset)

            test_output = model(data)

            idx_output = test_output.squeeze().detach().cpu().numpy().argmax()

            acc = calculate_accuracy(test_output, label)
            test_accuracy += acc / length_of_test_loader

            if acc >= 1.0:
                acc_list[idx_label] += 1

            # if the label is 'registered'
            if idx_label == 1:

                if idx_label == idx_output:
                    title = "Comparison of two patches that have [0, 0, 0] displacement.\n\nPrediction: CORRECT"
                else:
                    title = "Comparison of two patches that have [0, 0, 0] displacement.\n\nPrediction: WRONG"

                # _ = display_volume_slice(data.squeeze().detach().cpu().numpy(), title)
                # print(5)

            t.set_description(f"Testing patches ({test_accuracy:.2f})")
            t.refresh()

            # counter += 1
            # if counter == 200:
            #     break

    return test_accuracy, acc_list, class_amount


# load model
model = DenseNet()
model_params = torch.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments/dense net 100k classification robust-oath-133/model_epoch23_valacc0.954.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()

# load test data
loader = get_test_loader("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments/dense net 100k classification robust-oath-133/data - 20211129_craini_golby/resliced_perfect_false")

test_accuracy, acc_list, class_amount = test(model, loader)

# visualise
patcher = Patcher("", "", "", 10, False)
x = patcher.offsets
visualise_per_class_accuracies(acc_list/class_amount, x, f"Test per-class accuracies for {len(loader)} 'perfect' patches")
plt.savefig("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments" + "/test_per_class_accuracies_true.png", dpi=250)

# 165 classes
# test accuracy = 0.9679