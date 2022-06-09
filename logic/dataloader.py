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


def get_loader(data_path, batch_size, dataset_size, loader_type='train'):

    shuffle = True

    if loader_type == 'train':
        data_transforms, _, _ = get_transforms()
    elif loader_type == 'val':
        _, data_transforms, _ = get_transforms()
        dataset_size = int(0.2 * dataset_size)
        if dataset_size < 2:  # that's the minimum dataset size, because 2 is the minimum batch size, because we have
            # batch norm layers
            dataset_size = 2

    elif loader_type == 'test':
        _, _, data_transforms = get_transforms()
        batch_size = 1
        shuffle = False
    else:
        raise ValueError("loader_type must be either 'train', 'val' or 'test'")

    # get paths to the patches
    patch_file_path_list = glob.glob(os.path.join(data_path, "*_patch.npy"))
    patch_file_path_list.sort()

    # reduce the dataset size if necessary
    if dataset_size >= len(patch_file_path_list):
        dataset_size = len(patch_file_path_list)

    patch_file_path_list = patch_file_path_list[0:dataset_size]

    # create the dataset
    dataset = PatchDataset(patch_file_path_list, transform=data_transforms)

    if len(dataset) == 0:
        raise ValueError("No patches found in the dataset")

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle)

    return loader
