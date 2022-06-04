import glob
import os
import numpy as np
import ast

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import get_label_from_label_id


class PatchDataset(Dataset):
    def __init__(self, patch_file_path_list, transform=None):
        self.patch_file_path_list = patch_file_path_list
        self.transform = transform

    def __len__(self):
        return len(self.patch_file_path_list)

    def __getitem__(self, idx):
        """
        The label id is stored in the file name, from which the label can be extracted
        :param idx:
        :return:
        """

        path = self.patch_file_path_list[idx]
        label_id = os.path.basename(path).split('_')[1]  # second part of the name is the id as a binary number

        patch = np.load(path)
        label = get_label_from_label_id(label_id)

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
