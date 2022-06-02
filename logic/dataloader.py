import glob
import os
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PatchDataset(Dataset):
    def __init__(self, patch_file_path_list, label_path, transform=None):
        self.patch_file_path_list = patch_file_path_list
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.patch_file_path_list)

    def __getitem__(self, idx):
        """
        Label ID can be different if we have a train and val split, because all labels are stored in one file
        """

        path = self.patch_file_path_list[idx]
        label_id = path.split('/')[-1]
        label_id = int(label_id.split('_')[0])

        patch = np.load(path)
        label = self.labels[label_id]

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


def get_data_loaders(params):

    data_path = params.train_and_val_dir
    batch_size = params.batch_size

    # get paths to the patches and labels
    patch_file_path_list = glob.glob(os.path.join(data_path, "*_fixed_and_moving.npy"))
    patch_file_path_list.sort()

    # reduce the dataset size if necessary
    if params.dataset_size >= len(patch_file_path_list):
        params.dataset_size = len(patch_file_path_list)

    patch_file_path_list = patch_file_path_list[0:params.dataset_size]
    label_path = os.path.join(data_path, "labels.npy")

    # train and val split
    if params.validate is True:
        train_list, val_list = train_test_split(patch_file_path_list, test_size=0.2, random_state=params.seed)
    else:
        train_list, val_list = patch_file_path_list, patch_file_path_list

    # get transforms
    train_transforms, val_transforms, _ = get_transforms()

    # create the datasets
    train_dataset = PatchDataset(train_list, label_path, train_transforms)
    val_dataset = PatchDataset(val_list, label_path, val_transforms)

    # pass the dataset to the dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, None
