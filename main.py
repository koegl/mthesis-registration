import argparse
import os

import torch.optim as optim
import torch.nn as nn
import torch
import wandb

from utils import seed_everything, get_data_loaders, initialise_wandb
from network import get_network
from train import train
from test import test_model


# based on https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb
# https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html

# todo should the logging happen after each batch?
# todo why is DenseNet loss so high?
# todo implement early stopping
# todo implement sweep of hyper-parameters

def main(params):
    # define training parameters
    lr = float(params.learning_rate)
    seed = int(params.seed)

    # set seed
    seed_everything(seed)

    # get train, val, and test data loaders
    train_loader, val_loader, test_loader = get_data_loaders(params)

    # get the model
    device = params.device
    model = get_network(params.network_type, device=device)

    # set-up loss-function
    criterion = nn.CrossEntropyLoss()

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # set up logging with wandb
    initialise_wandb(params, len(train_loader.dataset), len(val_loader.dataset),
                     project="Classification", entity="fryderykkogl")

    # train or test the model (or both)
    if params.mode == "train":
        train(params.epochs, train_loader, model, criterion, optimizer, val_loader, device, save_path=params.model_path)

    elif params.mode == "test":
        model_test_accuracy = test_model(model_path=params.model_path, test_loader=test_loader)
        print(f"\n\nTest set accuracy: {model_test_accuracy}")

    elif params.mode == "both":
        train(params.epochs, train_loader, model, criterion, optimizer, val_loader, device, save_path=params.model_path)
        model_test_accuracy = test_model(model_path=params.model_path, test_loader=test_loader)
        print(f"\n\nTest set accuracy: {model_test_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs", "--batch_size", default=2)
    parser.add_argument("-e", "--epochs", default=200)
    parser.add_argument("-lr", "--learning_rate", default=0.001)
    parser.add_argument("-s", "--seed", default=42, help="For seeding eveyrthing")
    parser.add_argument("-tvd", "--train_and_val_dir", default="/Users/fryderykkogl/Data/ViT_training_data/data/train",
                        help="Directory of the training data (and validation")
    parser.add_argument("-vd", "--test_dir", default="/Users/fryderykkogl/Data/ViT_training_data/data/test",
                        help="Directory of the test data")
    parser.add_argument("-m", "--mode", default="train", choices=["both", "test", "both"],
                        help="train or test the model")
    parser.add_argument("-mp", "--model_path", default="models/model.pt",
                        help="Path to the model to be loaded/saved")
    parser.add_argument("-nt", "--network_type", default="ViT", choices=["ViT", "DenseNet"])
    parser.add_argument("-dv", "--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("-ps", "--patch_size", default=32)
    parser.add_argument("-d", "--depth", default=6)
    parser.add_argument("-hd", "--heads", default=8)
    parser.add_argument("-nd", "--network_dim", default=128)
    parser.add_argument("-md", "--mlp_dim", default=2048)

    args = parser.parse_args()

    main(args)
