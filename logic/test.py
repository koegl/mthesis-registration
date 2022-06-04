import numpy as np
import tqdm
from tqdm import trange

import torch
import torch.nn as nn

from utils import calculate_accuracy, display_volume_slice, get_label_id_from_label
from dataloader import get_test_loader
from architectures.densenet3d import DenseNet
from logic.patcher import Patcher


def test(model, test_loader):
    patcher = Patcher("", "", "", 10, False)
    x = patcher.label_to_offset_dict

    test_accuracy = 0.0

    acc_list = np.zeros(20)

    with torch.no_grad():

        t = trange(len(test_loader), desc=f"Testing patches ({test_accuracy:.2f})", leave=True)
        test_loader = iter(test_loader)
        for i in t:
            data, label = next(test_loader)
            data = data.to(torch.float32)
            label = label.to(torch.float32)

            label_np = label.squeeze().detach().cpu().numpy()
            idx = label_np.argmax()
            label_id = get_label_id_from_label(label_np)
            offset = x[label_id]
            # display_volume_slice(data.squeeze().detach().cpu().numpy(), offset)

            test_output = model(data)
            acc = calculate_accuracy(test_output, label)
            test_accuracy += acc / len(test_loader)

            if acc >= 1.0:
                acc_list[idx] += 1

            t.set_description(f"Testing patches ({test_accuracy:.2f})")
            t.refresh()

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Accuracy per label: {acc_list}")


# load model
model = DenseNet()
model_params = torch.load("/Users/fryderykkogl/Data/models/robust-oath-133/model_epoch23_valacc0.954.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()

# load test data
loader = get_test_loader("/Users/fryderykkogl/Data/patches/test_data/20211129_craini_golby/reslcied_perfect_true")

test(model, loader)
