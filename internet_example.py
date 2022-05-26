# https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import initialise_wandb
import wandb
import numpy as np
from architectures.vit_standard import ViTStandard
from logic.train import train
import torch.optim as optim
from utils import get_architecture, get_data_loaders


def plot_accuracies(historyy):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in historyy]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def plot_losses(historyy):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in historyy]
    val_losses = [x['val_loss'] for x in historyy]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(modell, val_loader):
    modell.eval()
    outputs = [modell.validation_step(batch) for batch in val_loader]
    return modell.validation_epoch_end(outputs)


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)
        return {'train_loss': loss, 'train_acc': acc}

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, xb):
        return self.network(xb)


class MyClassifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(5, 5))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(6400, 1024)
        self.fc2 = nn.Linear(1024, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # x = self.softmax(x)

        return x


def fit(epochs, lrr, modell, train_loader, val_loader, opt_funcc=torch.optim.SGD):
    historyy = []
    optimizer = opt_funcc(modell.parameters(), lrr)
    for epoch in range(epochs):

        modell.train()
        train_losses = []
        train_accs = []
        for batch in train_loader:
            loss_acc = modell.training_step(batch)
            train_losses.append(loss_acc['train_loss'])
            train_accs.append(loss_acc['train_acc'])
            loss_acc['train_loss'].backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(modell, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        modell.epoch_end(epoch, result)
        historyy.append(result)

    return historyy


def main(args=None):
    # train and test data directory
    data_dir = "/Users/fryderykkogl/Data/ViT_training_data/two_folders_small/"

    # load the train and test data
    dataset = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor()
    ]))

    batch_size = 1
    num_epochs = 30
    lr = 0.001
    dataset_size = 4

    params = {"epochs": num_epochs,
              "validate": False,
              "train_and_val_dir": "/Users/fryderykkogl/Data/ViT_training_data/data_overfit/train",
              "dataset_size": dataset_size,
              "test_dir": "/Users/fryderykkogl/Data/ViT_training_data/data_overfit/train",
              "batch_size": batch_size,
              }

    # load the train and validation into batches.
    train_dl = DataLoader(dataset, batch_size)
    train_dl_my, _, _ = get_data_loaders(params)

    model = get_architecture("CNNSmall", "cpu")

    opt_func = optim.Adam(model.parameters(), lr=lr)

    # fitting the model on training data and record the result after each epoch
    # history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    criterion = nn.CrossEntropyLoss()
    train(params, train_dl_my, model, criterion, opt_func, None, "cpu")


if __name__ == '__main__':
    main()
