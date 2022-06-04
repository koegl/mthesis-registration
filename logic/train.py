import torch
import wandb
from utils import display_tensor_and_label, display_volume_slice
from utils import calculate_accuracy
import torch.nn
from time import perf_counter


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

    return epoch_loss.detach().numpy(), epoch_accuracy.detach().numpy()


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

    return epoch_val_loss.detach().numpy(), epoch_val_accuracy.detach().numpy()


def train(params, train_loader, model, criterion, optimizer, val_loader, device, save_path="models/model.pt"):
    """
    Train the model for a given number of epochs and save the model at the end of training.
    """
    val_loss = 0

    for epoch in range(params.epochs):

        # train step
        train_loss = train_step(train_loader, device, model, criterion, optimizer, epoch, params.logging)

        # eval step
        if params.validate is True:
            val_loss = val_step(val_loader, device, model, criterion, epoch, params.logging)

        # save model
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss},
                   f"/Users/fryderykkogl/Documents/university/master/thesis/code.nosync/mthesis-registration/models/model_epoch{epoch}_loss{train_loss}.pt"
                   )

    torch.save(model, save_path)

