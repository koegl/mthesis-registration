import argparse
import os

import torch.optim as optim
import torch.nn as nn
import torch
import wandb

from utils import seed_everything, get_data_loaders
from network import get_network
from train import train
from test import test_model


# todo why is DenseNet loss so high?
# todo implement early stopping

def main(params):
    # define training parameters
    lr = float(params.learning_rate)
    seed = int(params.seed)

    # set seed
    seed_everything(seed)

    # get train, val, and test data loaders
    train_loader, val_loader, test_loader = get_data_loaders(params)

    # get the model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_network(params.network_type, device=device)

    # set-up loss-function
    criterion = nn.CrossEntropyLoss()

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # set up logging with wandb
    wandb.init(project="ViT", entity="fryderykkogl")
    os.environ["WANDB_NOTEBOOK_NAME"] = "ViT"

    wandb.config = {
        "learning_rate": lr,
        "epochs": int(params.epochs),
        "batch_size": int(params.batch_size),
        "training_data": params.train_and_val_dir,
        "test_data": params.test_dir,
        "network_type": params.network_type,
    }

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

    parser.add_argument("-bs", "--batch_size", default=1)
    parser.add_argument("-e", "--epochs", default=20)
    parser.add_argument("-lr", "--learning_rate", default=3e-5)
    parser.add_argument("-s", "--seed", default=42, help="For seeding eveyrthing")
    parser.add_argument("-tvd", "--train_and_val_dir", default="/Users/fryderykkogl/Data/ViT_training_data/data_overfit/train",
                        help="Directory of the training data (and validation")
    parser.add_argument("-vd", "--test_dir", default="/Users/fryderykkogl/Data/ViT_training_data/data_overfit/test",
                        help="Directory of the test data")
    parser.add_argument("-m", "--mode", default="both", choices=["train", "test", "both"],
                        help="train or test the model")
    parser.add_argument("-mp", "--model_path", default="model.pt",
                        help="Path to the model to be loaded/saved")
    parser.add_argument("-nt", "--network_type", default="DenseNet", choices=["ViT", "DenseNet"])

    args = parser.parse_args()

    main(args)
