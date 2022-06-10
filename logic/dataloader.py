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
