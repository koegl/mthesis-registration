import torch
import wandb

import matplotlib.pyplot as plt

from utils import plot_accuracies_and_losses


def train(epochs, train_loader, model, criterion, optimizer, val_loader, device="cpu", interval=1, save_path="model.pt"):
    """
    Train the model for a given number of epochs and save the model at the end of training.
    :param epochs:
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param val_loader:
    :param device:
    :param interval:
    :param save_path:
    :return:
    """

    train_accuracy_array = []
    train_loss_array = []
    val_accuracy_array = []
    val_loss_array = []

    # convert params to the correct types
    epochs = int(epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_val_accuracy = 0
        epoch_val_loss = 0

        # train step
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        # eval step
        with torch.no_grad():
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

        train_accuracy_array.append(float(epoch_accuracy.detach().numpy()))
        train_loss_array.append(float(epoch_loss.detach().numpy()))
        val_accuracy_array.append(float(epoch_val_accuracy.detach().numpy()))
        val_loss_array.append(float(epoch_val_loss.detach().numpy()))

        wandb.log({"Train loss": epoch_loss, "Val loss": epoch_val_loss})

    plot_accuracies_and_losses(
        [train_accuracy_array, train_loss_array, val_accuracy_array, val_loss_array],
        ['Training accuracy', 'Training loss', 'Validation accuracy', 'Validation loss'],
    )

    torch.save(model, "model_full.pt")

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
