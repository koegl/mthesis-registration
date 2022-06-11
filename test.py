import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import torch

from helpers.utils import calculate_accuracy, get_label_id_from_label
from logic.dataloader import get_loader
from architectures.densenet3d import DenseNet
from helpers.visualisations import visualise_per_class_accuracies
from logic.patcher import Patcher


def test(model, test_loader):
    patcher = Patcher("", "", "", 10, False)
    x = patcher.label_to_offset_dict

    length_of_test_loader = len(test_loader)

    test_accuracy = 0.0

    acc_list = np.zeros(20)
    class_amount = np.zeros(20)
    counter = 0

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
            # _ = display_volume_slice(data.squeeze().detach().cpu().numpy(), offset)

            test_output = model(data)

            acc = calculate_accuracy(test_output, label)
            test_accuracy += acc / length_of_test_loader

            if acc >= 1.0:
                acc_list[idx_label] += 1

            # _ = display_volume_slice(data.squeeze().detach().cpu().numpy(), title)
            # print(5)

            t.set_description(f"Testing patches ({test_accuracy:.2f})")
            t.refresh()

            # counter += 1
            # if counter == 200:
            #     break

    return test_accuracy, acc_list, class_amount


# load model
model = DenseNet(num_init_features=64)
model_params = torch.load("/Users/fryderykkogl/Desktop/model_epoch3_valacc0.870.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()

# load test data
loader = get_loader("/Users/fryderykkogl/Data/patches/val_npy", 1, 10000000, loader_type="test")

test_accuracy, acc_list, class_amount = test(model, loader)

# visualise
patcher = Patcher("", "", "", 10, False)
x = patcher.offsets
x = [np.array2string(offset) for offset in x]
visualise_per_class_accuracies(acc_list/class_amount, x, f"Test per-class accuracies for {len(loader)} 'real' patches\n"
                                                         f"Test accuracy: {test_accuracy:.2f}")
plt.savefig("/Users/fryderykkogl/Dropbox (Partners HealthCare)/Experiments" + "/test_per_class_accuracies_true.png", dpi=250)
