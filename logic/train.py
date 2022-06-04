import os
from time import perf_counter
from datetime import datetime, timedelta
import numpy as np
import wandb

import torch
import torch.nn

from visualisations import display_tensor_and_label, display_volume_slice
from utils import calculate_accuracy


def save_model(model, optimizer, epoch, train_array, val_array, start_datetime, save_path="models/model.pt"):
    """
    Function to save models only if the validation accuracy is higher than in all the previous ones
    :param model:
    :param optimizer:
    :param epoch:
    :param train_array:
    :param val_array:
    :param start_datetime:
    :param save_path:
    """

    if epoch == 0:
        return

    # save only when validation accuracy of current epoch is higher than the previous ones
    previous_validation_accuracies = val_array[:epoch, 1]  # this doesn't include the current epoch
    current_validation_accuracy = val_array[epoch, 1]

    # check if we currently have the best validation accuracy
    if current_validation_accuracy > np.max(previous_validation_accuracies):
        try:
            # we have a better val accuracy, so first delete all previous models
            old_models = [file for file in os.listdir(f"./models/{start_datetime}") if file.endswith(".pt")]
            for old_model in old_models:
                os.remove(f"./models/{start_datetime}/{old_model}")

            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_array[epoch, 0]},
                       f"./models/{start_datetime}/model_epoch{epoch}_valacc{val_array[epoch, 1]:.3f}.pt"
                       )
        except Exception as e:
            print(f"Error while saving model at epoch {epoch}: {e}")


def train_step(train_loader, device, model, criterion, optimizer, epoch, logging):

    epoch_accuracy = 0
    epoch_loss = 0
    now = perf_counter()

    for data, label in train_loader:
        data = data.to(device).to(torch.float32)
        label = label.to(device).to(torch.float32)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(output, label)

        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    if logging == "wandb":
        wandb.log({"Train loss": epoch_loss,
                   "Train accuracy": epoch_accuracy,
                   "Epoch": epoch})
    else:
        print(f"\n{epoch}:\tTime: {perf_counter() - now:.2f}s;\t Train loss: {epoch_loss:.2f};\tTrain accuracy: {epoch_accuracy:.2f}",
              end='\t')

    return epoch_loss.detach().cpu().numpy(), epoch_accuracy.detach().cpu().numpy()


def val_step(val_loader, device, model, criterion, epoch, logging):

    epoch_val_accuracy = 0
    epoch_val_loss = 0

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device).to(torch.float32)
            label = label.to(device).to(torch.float32)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = calculate_accuracy(val_output, label)
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    if logging == "wandb":
        wandb.log({"Val loss": epoch_val_loss,
                   "Val accuracy": epoch_val_accuracy,
                   "Epoch": epoch})
    else:
        print(f"Val loss: {epoch_val_loss:.2f};\tVal accuracy: {epoch_val_accuracy:.2f}")

    return epoch_val_loss.detach().cpu().numpy(), epoch_val_accuracy.detach().cpu().numpy()


def train(params, train_loader, model, criterion, optimizer, val_loader, device, save_path="models/model.pt"):
    """
    Train the model for a given number of epochs and save the model at each epoch.
    """
    val_array = np.zeros((params.epochs, 2))
    train_array = np.zeros((params.epochs, 2))
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace(" ", "_").replace("/", "").replace(":", "")
    os.mkdir(f"./models/{start_datetime}")

    time_idx = False

    for epoch in range(params.epochs):

        now = perf_counter()

        # train step
        train_array[epoch] = train_step(train_loader, device, model, criterion, optimizer, epoch, params.logging)

        # eval step
        if params.validate is True:
            val_array[epoch] = val_step(val_loader, device, model, criterion, epoch, params.logging)

        # save model
        save_model(model, optimizer, epoch, train_array, val_array, start_datetime, save_path)

        if time_idx is False and params.logging == "wandb":
            time = perf_counter() - now

            wandb.log({"Time per epoch": f'{"{:0>8}".format(str(timedelta(seconds=time)))}',
                       "Expected total time": f'{"{:0>8}".format(str(timedelta(seconds=time * params.epochs)))}'})
            time_idx = True
