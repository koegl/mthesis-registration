import torch
import wandb
from utils import display_tensor_and_label
from architectures.cnn_small import
import torch.nn

softmax = torch.nn.Softmax(dim=1)


def train_step(train_loader, device, model, criterion, optimizer, epoch):

    epoch_accuracy = 0
    epoch_loss = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device).to(torch.float32)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        wandb.log({"Train loss": epoch_loss,
                   "Train accuracy": epoch_accuracy,
                   "Epoch": epoch})


def val_step(val_loader, device, model, criterion, epoch):

    epoch_val_accuracy = 0
    epoch_val_loss = 0

    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device).reshape(-1, 1)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    wandb.log({"Val loss": epoch_val_loss,
               "Val accuracy": epoch_val_accuracy,
               "Epoch": epoch})
    # print(f"{epoch}: Train loss: {epoch_val_loss};\tTrain accuracy: {epoch_val_accuracy}")


def train(params, train_loader, model, criterion, optimizer, val_loader, device, save_path="model.pt"):
    """
    Train the model for a given number of epochs and save the model at the end of training.
    """

    for epoch in range(params["epochs"]):

        # train step
        train_step(train_loader, device, model, criterion, optimizer, epoch)

        # eval step
        if params["validate"] is True:
            val_step(val_loader, device, model, criterion, epoch)

        # save model
        split_save_path = save_path.split('.')
        new_save_path = split_save_path[0] + '_' + str(epoch) + '.' + split_save_path[1]

        # torch.save(model, new_save_path)
