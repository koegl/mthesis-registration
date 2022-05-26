import random
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from logic.dataloader import PatchDataset
from architectures.vit_standard import ViTStandard
from architectures.vit_for_small_datasets import ViTForSmallDatasets


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    val_data = PatchDataset(val_list, transform=test_transforms)
    test_data = PatchDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=params["batch_size"], shuffle=True)

    return train_loader, val_loader, test_loader


def plot_accuracies_and_losses(array_of_arrays_to_plot, array_of_sub_titles, title="Training"):

    assert len(array_of_arrays_to_plot) == len(array_of_sub_titles), "Number of arrays to plot and sub titles must be the same"

    amount = len(array_of_arrays_to_plot)
    epochs = np.arange(1, len(array_of_arrays_to_plot[0]) + 1)

    sns.set_style("dark")
    sns.set_style("darkgrid")
    palette = sns.color_palette("dark")

    fig, (axes) = plt.subplots(amount, figsize=(12, 2*amount))

    axes[0].set_title(title)

    for i in range(amount):
        p = sns.lineplot(
            x=epochs,
            y=array_of_arrays_to_plot[i],
            ax=axes[i],
            color=palette[1])

        p.set_ylabel(array_of_sub_titles[i])
        p.set(ylim=(-0.05, 1.05))

    p.set_xlabel("Epochs")



    plt.show()


def initialise_wandb(params, len_train, len_val, project="Classification", entity="fryderykkogl"):
    """
    Initialise everything for wand
    :param params: the user arguments
    :param len_train: the length of the training set
    :param len_val: the length of the validation set
    :param project: the project name
    :param entity: the entity name
    :return:
    """

    wandb.init(project=project, entity=entity)
    os.environ["WANDB_NOTEBOOK_NAME"] = "Classification"

    config_dict = {
        "learning_rate": params["learning_rate"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "training_data": params["train_and_val_dir"],
        "test_data": params["test_dir"],
        "architecture_type": params["architecture_type"],
        "device": params["device"],
    }
    wandb.config = config_dict
    wandb.log(config_dict)
    wandb.log({"Training size": len_train,
               "Validation size": len_val})


def get_architecture(architecture_type, device):

    if architecture_type.lower() == "vitstandard":
        model = ViTStandard(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=2,
            channels=3,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            device=device
        ).to(device)

    elif architecture_type.lower() == "vitforsmalldatasets":
        model = ViTForSmallDatasets(
            image_size=256,
            patch_size=16,
            num_classes=2,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    else:
        raise NotImplementedError("Architecture not supported. Only ViTStandard and ViTForSmallDatasets are supported.")

    return model


def convert_cmd_args_to_correct_type(params):
    """
    All args are stored in string, but some of them need to be converted - will be saved as a dict
    :param params: the cmd line args
    :return: a dict with them stored correctly
    """

    params_dict = {"batch_size": int(params.batch_size),
                   "epochs": int(params.epochs),
                   "learning_rate": float(params.batch_size),
                   "seed": int(params.seed),
                   "train_and_val_dir": params.train_and_val_dir,
                   "test_dir": params.test_dir,
                   "mode": params.mode,
                   "model_path": params.model_path,
                   "architecture_type": params.architecture_type,
                   "device": params.device,
                   }

    return params_dict
