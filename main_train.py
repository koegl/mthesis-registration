

# PLAN
# 1. get a list of all files (.niftis)
# 2. for each file create a list of centres - combine into list of lists
# 3. create and save the patches as numpy arrays in a new folder
# 3. pass this big list of lists to the dataloader

# https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html

import argparse

import torch
import torch.optim as optim
import torch.nn as nn

from utils import seed_everything, initialise_wandb, get_architecture
from logic.train import train
from logic.dataloader import get_loader


def main(params):

    params.batch_size = int(params.batch_size)
    params.epochs = int(params.epochs)
    params.seed = int(params.seed)
    params.dataset_size = int(params.dataset_size)

    if params.device == "cpu":
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    seed_everything(params.seed)

    # get train and data loaders
    train_loader = get_loader(params.train_dir, params.batch_size, params.dataset_size, loader_type="train")
    val_loader = get_loader(params.val_dir, params.batch_size, params.dataset_size, loader_type="val")

    # get the model
    model = get_architecture(params).to(params.device)

    # set-up loss-function
    criterion = nn.CrossEntropyLoss()

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(params.learning_rate))

    # logging
    if params.logging == "wandb":
        # set up logging with wandb
        initialise_wandb(params, len(train_loader.dataset), len(val_loader.dataset),
                         project="Classification", entity="fryderykkogl")

    # train the model
    train(params, train_loader, model, criterion, optimizer, val_loader, params.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs", "--batch_size", default=800)
    parser.add_argument("-e", "--epochs", default=15)
    parser.add_argument("-lr", "--learning_rate", default=0.001)
    parser.add_argument("-s", "--seed", default=42, help="For seeding eveyrthing")
    parser.add_argument("-td", "--train_dir", default="/Users/fryderykkogl/Data/patches/train_npy",
                        help="Directory of the training data")
    parser.add_argument("-vd", "--val_dir", default="/Users/fryderykkogl/Data/patches/val_npy",
                        help="Directory of the validation data")
    parser.add_argument("-dv", "--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("-ds", "--dataset_size", default=100000000, type=int, help="Amount of images used for training")
    parser.add_argument("-v", "--validate", default=True, type=bool, help="Choose whether to validate or not")
    parser.add_argument("-lg", "--logging", default="wandb", choices=["print", "wandb"])
    parser.add_argument("-at", "--architecture_type", default="densenet264", choices=["densenet121", "densenet169",
                                                                                      "densenet201", "densenet264",
                                                                                      "vit"])
    parser.add_argument("-dp", "--dropout", default=0.1, type=float,
                        help="Dropout probability")
    parser.add_argument("-es", "--early_stopping", default=True, type=bool)

    args = parser.parse_args()

    main(args)

