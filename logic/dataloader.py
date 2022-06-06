import glob
import os
import numpy as np
from ast import literal_eval

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import get_label_from_label_id
from logic.patcher import Patcher


class PatchDataset(Dataset):
    def __init__(self, patch_file_path_list, transform=None):
        self.patch_file_path_list = patch_file_path_list
        self.transform = transform

        self.patcher = Patcher("", "", "", 32, False)
        self.labels_per_d_dict = {"-16": 0, "-8": 1, "-4": 2, "4": 3, "8": 4, "16": 5, "0": 6, "7": 7}

    def __len__(self):
        return len(self.patch_file_path_list)

    def get_triple_label(self, label_id):
        offset = literal_eval(self.patcher.label_to_offset_dict[label_id])

        if offset == [7, 7, 7] or offset == [0, 0, 0]:
            temp = 0
            pass

        # get displacement in each dimension
        x_disp = offset[0]
        y_disp = offset[1]
        z_disp = offset[2]

        # create a label for each dimension
        x_label = np.zeros(8)
        y_label = np.zeros(8)
        z_label = np.zeros(8)

        # get the id of each spatial translation, and assign the label at this id to 1
        x_label[self.labels_per_d_dict[str(x_disp)]] = 1
        y_label[self.labels_per_d_dict[str(y_disp)]] = 1
        z_label[self.labels_per_d_dict[str(z_disp)]] = 1

        # concatenate the labels
        label = np.concatenate((x_label, y_label, z_label), axis=0)

        return label

    def __getitem__(self, idx):
        """
        The label id is stored in the file name, from which the label can be extracted
        :param idx:
        :return:
        """

        path = self.patch_file_path_list[idx]

        label_id = os.path.basename(path).split('_')[1]  # second part of the name is the id as a binary number

        patch = np.load(path)

        label = self.get_triple_label(label_id)

        return patch, label


def get_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    return train_transforms, val_transforms, test_transforms


def get_train_and_val_loaders(params):

    data_path = params.train_and_val_dir
    batch_size = params.batch_size

    # get paths to the patches
    patch_file_path_list = glob.glob(os.path.join(data_path, "*_patch.npy"))
    patch_file_path_list.sort()

    # reduce the dataset size if necessary
    if params.dataset_size >= len(patch_file_path_list):
        params.dataset_size = len(patch_file_path_list)

    patch_file_path_list = patch_file_path_list[0:params.dataset_size]

    # train and val split
    if params.validate is True:
        train_list, val_list = train_test_split(patch_file_path_list, test_size=0.2, random_state=params.seed)
    else:
        train_list, val_list = patch_file_path_list, patch_file_path_list

    # get transforms
    train_transforms, val_transforms, _ = get_transforms()

    # create the datasets
    train_dataset = PatchDataset(train_list, train_transforms)
    val_dataset = PatchDataset(val_list, val_transforms)

    # pass the dataset to the dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def get_test_loader(data_path):
    test_list = glob.glob(os.path.join(data_path, "*_patch.npy"))

    _, _, transform = get_transforms()

    test_dataset = PatchDataset(test_list, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader
