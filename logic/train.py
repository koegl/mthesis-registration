import os
from time import perf_counter
from datetime import datetime, timedelta
import numpy as np
import wandb
from ast import literal_eval
import torch
import torch.nn

from visualisations import display_tensor_and_label, display_volume_slice
from utils import calculate_accuracy, get_label_id_from_label
from logic.patcher import Patcher
from utils import calculate_accuracy, EarlyStopping, mean_var_loss


# todo first save the model, then if the saving was successful, delete the old ones
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

    for data, offset in train_loader:

        data = data.to(device).to(torch.float32)
        offset = offset.to(device).to(torch.float32)

        output = model(data)

        loss = mean_var_loss(output, offset)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(output, offset)

        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    if logging == "wandb":
        wandb.log({"Train loss": epoch_loss,
                   "Train accuracy": epoch_accuracy,
                   "Epoch": epoch})
    else:
        print(f"\n{epoch}:\tTime: {perf_counter() - now:.2f}s;\t Train loss: {epoch_loss:.2f};\tTrain accuracy: {epoch_accuracy:.3f}",
              end='\t')

    return epoch_loss.detach().cpu().numpy(), epoch_accuracy


def val_step(val_loader, device, model, criterion, epoch, logging):

    epoch_val_accuracy = 0
    epoch_val_loss = 0

    with torch.no_grad():
        for data, offset in val_loader:
            data = data.to(device).to(torch.float32)
            offset = offset.to(device).to(torch.float32)

            val_output = model(data)

            val_loss = mean_var_loss(val_output, offset)

            acc = calculate_accuracy(val_output, offset)

            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    if logging == "wandb":
        wandb.log({"Val loss": epoch_val_loss,
                   "Val accuracy": epoch_val_accuracy,
                   "Epoch": epoch})
    else:
        print(f"Val loss: {epoch_val_loss:.2f};\tVal accuracy: {epoch_val_accuracy:.2f}")

    return epoch_val_loss.detach().cpu().numpy(), epoch_val_accuracy


def train(params, train_loader, model, criterion, optimizer, val_loader, device, save_path="models/model.pt"):
    """
    Train the model for a given number of epochs and save the model at each epoch.
    """
    val_array = np.zeros((params.epochs, 2))
    train_array = np.zeros((params.epochs, 2))
    start_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace(" ", "_").replace("/", "").replace(":", "")
    os.mkdir(f"./models/{start_datetime}")
    early_stopping = EarlyStopping()

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

        # log time
        if time_idx is False and params.logging == "wandb":
            time = perf_counter() - now

            wandb.log({"Time per epoch": f'{"{:0>8}".format(str(timedelta(seconds=time)))}',
                       "Expected total time": f'{"{:0>8}".format(str(timedelta(seconds=time * params.epochs)))}'})
            time_idx = True

        # early stopping
        if params.early_stopping is True:
            early_stopping(val_array[epoch, 0])
            if early_stopping.early_stop is True:
                wandb.log({"Early stopping": "True"})
                break

