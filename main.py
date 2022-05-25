# https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html

import argparse

import torch.optim as optim
import torch.nn as nn

from utils import seed_everything, get_data_loaders, initialise_wandb, get_architecture, convert_cmd_args_to_correct_type
from logic.train import train
from logic.test import test_model


def main(params):

    params = convert_cmd_args_to_correct_type(params)

    # set seed
    seed_everything(params["seed"])

    # get train, val, and test data loaders
    train_loader, val_loader, test_loader = get_data_loaders(params)

    # get the model
    model = get_architecture(params["architecture_type"], device=params["device"])

    # set-up loss-function
    criterion = nn.CrossEntropyLoss()

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # set up logging with wandb
    initialise_wandb(params, len(train_loader.dataset), len(val_loader.dataset),
                     project="Classification", entity="fryderykkogl")

    # train or test the model (or both)
    if params["mode"] == "train":
        train(params["epochs"], train_loader, model, criterion, optimizer, val_loader, params["device"],
              save_path=params["model_path"])

    elif params["mode"] == "test":
        model_test_accuracy = test_model(model_path=params["model_path"], test_loader=test_loader)
        print(f"\n\nTest set accuracy: {model_test_accuracy}")

    elif params["mode"] == "both":
        train(params["epochs"], train_loader, model, criterion, optimizer, val_loader, params["device"],
              save_path=params["model_path"])
        model_test_accuracy = test_model(model_path=params["model_path"], test_loader=test_loader)
        print(f"\n\nTest set accuracy: {model_test_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs", "--batch_size", default=512)
    parser.add_argument("-e", "--epochs", default=20)
    parser.add_argument("-lr", "--learning_rate", default=0.004)
    parser.add_argument("-s", "--seed", default=42, help="For seeding eveyrthing")
    parser.add_argument("-tvd", "--train_and_val_dir", default="/Users/fryderykkogl/Data/ViT_training_data/data/train",
                        help="Directory of the training data (and validation")
    parser.add_argument("-vd", "--test_dir", default="/Users/fryderykkogl/Data/ViT_training_data/data/test",
                        help="Directory of the test data")
    parser.add_argument("-m", "--mode", default="train", choices=["both", "test", "both"],
                        help="train or test the model")
    parser.add_argument("-mp", "--model_path", default="models/model.pt",
                        help="Path to the model to be loaded/saved")
    parser.add_argument("-at", "--architecture_type", default="ViTStandard", choices=["ViTStandard", "ViTForSmallDatasets"])
    parser.add_argument("-dv", "--device", default="mps", choices=["cpu", "mps"])

    args = parser.parse_args()

    main(args)
