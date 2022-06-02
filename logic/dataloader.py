import nibabel as nib
import glob
import os
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from logic.patcher import Patcher


class PatchDataset(Dataset):
    def __init__(self, patch_file_path_list, label_path, transform=None):
        self.patch_file_path_list = patch_file_path_list
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.patch_file_path_list)

    def __getitem__(self, idx):
        patch = np.load(self.patch_file_path_list[idx])
        label = self.labels[idx]

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


def get_data_loader(data_path="/Users/fryderykkogl/Data/temp/data_npy", batch_size=1):

    # get transforms
    train_transforms, _, _ = get_transforms()

    # create the dataset
    train_dataset = PatchDataset(data_path, train_transforms)

    # pass the dataset to the dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
