import torch
import wandb


def train(epochs, train_loader, model, criterion, optimizer, val_loader, device, save_path="model.pt"):
    """
    Train the model for a given number of epochs and save the model at the end of training.
    """

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

        wandb.log({"Train loss": epoch_loss,
                   "Train accuracy": epoch_accuracy,
                   "Val loss": epoch_val_loss,
                   "Val accuracy": epoch_val_accuracy,
                   "Epoch": epoch})

        split_save_path = save_path.split('.')
        new_save_path = split_save_path[0] + '_' + str(epoch) + '.' + split_save_path[1]

        torch.save(model, new_save_path)
