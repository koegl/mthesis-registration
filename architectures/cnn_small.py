from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_vals(vals):
    tensor1 = vals.detach().numpy().squeeze()[0, :]

    tensor2 = vals.detach().numpy().squeeze()[1, :]

    batch = vals.detach().numpy().squeeze()

    arrays = [batch[i, :] for i in range(batch.shape[0])]

    # create figure with two subplots
    fig, axes = plt.subplots(1, 2)

    # plot the first subplot
    for i in range(len(arrays)):
        axes[i].plot(arrays[i])

    plt.show()


class CNNSmall(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(5, 5))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(25600, 1024)
        self.fc2 = nn.Linear(1024, 2)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
        # x = self.sigmoid(x)

        return x
