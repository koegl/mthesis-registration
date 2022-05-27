import torch
from utils import get_data_loaders, display_tensor_and_label, calculate_accuracy
import matplotlib.pyplot as plt
import numpy as np

# todo - name attentions in the ViT so they can be recognised by the unroller


model = torch.load("/Users/fryderykkogl/Documents/university/master/thesis/code.nosync/mthesis-registration/models/model.pt")

_, _, test_loader = get_data_loaders(
    {"train_and_val_dir": "/Users/fryderykkogl/Data/ViT_training_data/data_overfit/train",
     "test_dir": "/Users/fryderykkogl/Data/ViT_training_data/data_overfit/train",
     "batch_size": 1,
     "dataset_size": 4,
     "validate": False})

model_accuracy = 0


def display_tensor_and_label2(tensor, label):
    """
    Display a tensor and its label
    :param tensor: the tensor
    :param label: the label
    :return:
    """
    tensor = tensor.squeeze()
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    label = label.numpy().squeeze()

    label = "dog" if all(label == [1.0, 0.0]) else "cat"

    plt.imshow(tensor)
    plt.title(label)
    plt.show()


predicted_label = [0, 0]

with torch.no_grad():
    for data, label in test_loader:

        display_tensor_and_label2(data, label)

        model_output = model(data)
        one = model_output.argmax(dim=1)
        if one == 1:
            predicted_label = [0, 1]
        else:
            predicted_label = [1, 0]

        accuracy = calculate_accuracy(model_output, label)

        model_accuracy += accuracy / len(test_loader)