import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def display_volume_slice(volumes, label=None):
    """
    Displays a slice of a 3D volume in a matplotlib figure
    :param volumes: the volume(s)
    :param label: the displacement label
    """

    if len(volumes.shape) == 3:
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        ax.imshow(volumes[volumes.shape[0] // 2, :, :], cmap='gray')

        ax_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
        slice = Slider(ax_slider, 'Slice', 0, volumes.shape[0] - 1, valinit=volumes.shape[0] // 2)

        def update(val):
            ax.clear()
            ax.imshow(volumes[int(slice.val), :, :], cmap='gray')
            fig.canvas.draw_idle()

        slice.on_changed(update)

        if label is not None:
            plt.title(label)

        plt.show()

        return slice

    elif len(volumes.shape) == 4:
        volume_0 = volumes[0, :, :, :]
        volume_1 = volumes[1, :, :, :]

        fig, ax = plt.subplots(1, 2)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        ax[0].imshow(volume_0[volume_0.shape[0] // 2, :, :], cmap='gray')
        ax[1].imshow(volume_1[volume_1.shape[0] // 2, :, :], cmap='gray')

        ax_slider_left = plt.axes([0.32, 0.2, 0.15, 0.03])
        ax_slider_right = plt.axes([0.7, 0.2, 0.15, 0.03])

        # add two sliders

        slice_left = Slider(ax_slider_left, 'Slice', 0, volume_0.shape[0] - 1, valinit=volume_0.shape[0] // 2)
        slice_right = Slider(ax_slider_right, 'Slice', 0, volume_1.shape[0] - 1, valinit=volume_1.shape[0] // 2)

        def update_left(val):
            ax[0].clear()
            ax[0].imshow(volume_0[int(slice_left.val), :, :], cmap='gray')
            fig.canvas.draw_idle()

        def update_right(val):
            ax[1].clear()
            ax[1].imshow(volume_1[int(slice_right.val), :, :], cmap='gray')
            fig.canvas.draw_idle()

        slice_left.on_changed(update_left)
        slice_right.on_changed(update_right)

        if label is not None:
            plt.suptitle(label)

        plt.show()

        return slice_left, slice_right


def display_tensor_and_label(tensor, label):
    """
    Display a tensor and its label
    :param tensor: the tensor
    :param label: the label
    :return:
    """
    tensor = tensor.squeeze()
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    label = label.numpy()

    label = "dog" if all(label == [1.0, 0.0]) else "cat"

    plt.imshow(tensor)
    plt.title(label)
    plt.show()


def visualise_per_class_accuracies(accuracies, class_names, title="Accuracies"):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.2)

    p = sns.barplot(class_names, accuracies, palette="crest")

    p.set_xlabel("Offsets", fontsize=15)
    p.set_xticklabels(p.get_xticklabels(), rotation=90)

    p.set_ylabel("Accuracies per class", fontsize=15)
    p.set(ylim=(0, 1))

    plt.title(title, fontsize=15, fontweight="bold")
