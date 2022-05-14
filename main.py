import argparse
from PIL import Image

from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch

from utils import seed_everything, get_data_loaders
from network import get_network
from train import train


def main(params):
    # define training parameters
    lr = params.learning_rate
    seed = params.seed

    # set seed
    seed_everything(seed)

    # get train, val, and test data loaders
    train_loader, train_list, val_loader, test_loader = get_data_loaders(params)

    # get the model
    model = get_network(device="cpu")

    # set-up loss-function
    criterion = nn.CrossEntropyLoss()

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train or test the model (or both)
    if params.mode == "train":
        train(params.epochs, train_loader, model, criterion, optimizer, val_loader, save_path=params.model_path)

    elif params.mode == "test":
        model_test_accuracy = test_model(model_path=params.model_path, test_loader=test_loader)
        print(f"\n\nTest set accuracy: {model_test_accuracy}")

    elif params.mode == "both":
        train(params.epochs, train_loader, model, criterion, optimizer, val_loader, save_path=params.model_path)
        model_test_accuracy = test_model(model_path=params.model_path, test_loader=test_loader)
        print(f"\n\nTest set accuracy: {model_test_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs", "--batch_size", default=1)
    parser.add_argument("-e", "--epochs", default=60)
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
    # parser.add_argument("-pu", "--print_updated", default="cpu", choices=["cpu", "gpu"],

    args = parser.parse_args()

    main(args)
