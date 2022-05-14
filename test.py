import torch


def test_model(model_path, test_loader):
    model = torch.load(model_path)

    model_accuracy = 0

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to("cpu")
            label = label.to("cpu")

            model_output = model(data)

            accuracy = (model_output.argmax(dim=1) == label).float().mean()

            model_accuracy += accuracy / len(test_loader)

    return model_accuracy
