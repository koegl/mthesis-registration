import random
import os
import numpy as np
import glob

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from dataloader import PatchDataset


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    return train_transforms, val_transforms, test_transforms


def get_labels(params):

    train_and_val_dir = params.train_and_val_dir
    train_and_val_list = glob.glob(os.path.join(train_and_val_dir, '*.jpg'))

    # get the labels which are the first part of each file name
    labels = [path.split('/')[-1].split('.')[0] for path in train_and_val_list]

    return labels


def get_data_loaders(params):

    # convert params to correct types
    batch_size = int(params.batch_size)
    seed = int(params.seed)

    # get directories
    train_and_val_dir = params.train_and_val_dir
    test_dir = params.test_dir

    # get a list of all files (.jpegs)
    train_and_val_list = glob.glob(os.path.join(train_and_val_dir, '*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    # get train and val split -> here 'test' refers to validation
    train_list, val_list = train_test_split(train_and_val_list, test_size=0.2, random_state=seed)

    # get transforms
    train_transforms, val_transforms, test_transforms = get_transforms()

    train_data = PatchDataset(train_list, transform=train_transforms)
    val_data = PatchDataset(val_list, transform=test_transforms)
    test_data = PatchDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
