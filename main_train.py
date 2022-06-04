

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
from logic.dataloader import get_data_loaders


def main(params):

    params.batch_size = int(params.batch_size)

    if params.device == "cpu":
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    seed_everything(params.seed)

    # get train, val, and test data loaders
    train_loader, val_loader, test_loader = get_data_loaders(params)

    # get the model
    model = get_architecture(params.architecture_type).to(params.device)

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

    parser.add_argument("-bs", "--batch_size", default=100)
    parser.add_argument("-e", "--epochs", default=100)
    parser.add_argument("-lr", "--learning_rate", default=0.001)
    parser.add_argument("-s", "--seed", default=42, help="For seeding eveyrthing")
    parser.add_argument("-tvd", "--train_and_val_dir", default="/Users/fryderykkogl/Data/temp/data_npy",
                        help="Directory of the training data (and validation")
    parser.add_argument("-dv", "--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("-ds", "--dataset_size", default=100, type=int, help="Amount of images used for training")
    parser.add_argument("-v", "--validate", default=True, type=bool, help="Choose whether to validate or not")
    parser.add_argument("-lg", "--logging", default="wandb", choices=["print", "wandb"])
    parser.add_argument("-at", "--architecture_type", default="densenet", choices=["densenet"])

    args = parser.parse_args()

    main(args)

