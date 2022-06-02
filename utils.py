import numpy as np
import nibabel as nib
import random
import os
import glob
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from logic.dataloader import PatchDataset

# todo add way of encoding of patches with offset bigger than patch (unrelated)
# todo would just reversing the order of the patches be enough to be treated as 'augmented data'?


def save_np_array_as_nifti(array, path, affine, header=None):
    """
    Save an nd array as a nifti file.
    :param array: the nd array to save
    :param path: the path to save the nifti file
    :param header: the header of the nifti file
    :param affine: the affine of the nifti file
    """

    img = nib.Nifti1Image(array, affine=affine, header=header)

    nib.save(img, path)


def create_radial_gradient(width, height, depth):
    """
    Create a radial gradient.
    :param width: width of the volume
    :param height: height of the volume
    :param depth: depth of the volume
    :return: the gradient volume as a nd array
    """

    x, y, z = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height), np.linspace(-1, 1, depth))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    r = r / np.max(r)

    return r


def crop_volume_borders(volume):
    """
    Removes as much surrounding black pixels from a volume as possible
    :param volume: The entire volume
    :return: volume_cropped
    """

    shape = volume.shape

    assert len(shape) == 3, "Volume must be 3D"

    # set maximum and minimum boundaries in case the volume touches the sides of the image
    min_x = 0
    min_y = 0
    min_z = 0
    max_x = shape[0] - 1
    max_y = shape[1] - 1
    max_z = shape[2] - 1

    # find first plane in the x-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[0]):
        if np.count_nonzero(volume[i, :, :]) > 0:
            min_x = i
            break
    # find first plane in the x-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[0] - 1, -1, -1):
        if np.count_nonzero(volume[i, :, :]) > 0:
            max_x = i
            break

    # find first plane in the y-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[1]):
        if np.count_nonzero(volume[:, i, :]) > 0:
            min_y = i
            break
    # find first plane in the y-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[1] - 1, -1, -1):
        if np.count_nonzero(volume[:, i, :]) > 0:
            max_y = i
            break

    # find first plane in the z-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[2]):
        if np.count_nonzero(volume[:, :, i]) > 0:
            min_z = i
            break
    # find first plane in the z-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[2] - 1, -1, -1):
        if np.count_nonzero(volume[:, :, i]) > 0:
            max_z = i
            break

    # crop the volume
    volume_cropped = volume[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

    return volume_cropped


def display_volume_slice(volume):
    """
    Displays a slice of a 3D volume in a matplotlib figure
    :param volume: the volume
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    ax.imshow(volume[volume.shape[0] // 2, :, :], cmap='gray')

    ax_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
    slice = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=volume.shape[0] // 2)

    def update(val):
        ax.clear()
        ax.imshow(volume[int(slice.val), :, :], cmap='gray')
        fig.canvas.draw_idle()

    slice.on_changed(update)

    plt.show()


def get_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    return train_transforms, val_transforms, test_transforms


def get_labels(params):

    train_and_val_list = glob.glob(os.path.join(params["train_and_val_dir"], "*.jpg"))

    # get the labels which are the first part of each file name
    labels = [path.split('/')[-1].split('.')[0] for path in train_and_val_list]

    return labels


def get_data_loaders(params):

    # get a list of all files (.jpgs)
    train_and_val_list = glob.glob(os.path.join(params["train_and_val_dir"], "*.jpg"))
    train_and_val_list.sort()

    if params["dataset_size"] >= len(train_and_val_list):
        params["dataset_size"] = len(train_and_val_list)

    train_and_val_list = train_and_val_list[0:params["dataset_size"]]
    test_list = glob.glob(os.path.join(params["test_dir"], "*.jpg"))

    # get train and val split -> here "test" refers to validation
    if params["validate"] is True:
        train_list, val_list = train_test_split(train_and_val_list, test_size=0.2, random_state=params["seed"])
    else:
        train_list = train_and_val_list
        val_list = train_and_val_list

    # get transforms
    train_transforms, val_transforms, test_transforms = get_transforms()

    train_data = PatchDataset(train_list, transform=train_transforms)
    val_data = PatchDataset(val_list, transform=val_transforms)
    test_data = PatchDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=params["batch_size"], shuffle=True)

    return train_loader, val_loader, test_loader


def display_tensor_and_label(tensor, label):
    """
    Display a tensor and its label
    :param tensor: the tensor
    :param label: the label
    :return:
    """
    tensor = tensor.squeeze()
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    label = label.numpy()

    label = "dog" if all(label == [1.0, 0.0]) else "cat"

    plt.imshow(tensor)
    plt.title(label)
    plt.show()

