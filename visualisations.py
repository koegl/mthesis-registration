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

    volumes /= 255

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
            plt.title(label, fontweight="bold")

        plt.show()

        return slice

    elif len(volumes.shape) == 4:
        volume_0 = volumes[0, :, :, :]
        volume_1 = volumes[1, :, :, :]

        fig, ax = plt.subplots(1, 3, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 1, 1.1]})
        #plt.subplots_adjust(top=0.2)
        ax[0].imshow(volume_0[volume_0.shape[0] // 2, :, :], cmap='gray')#, vmin=0, vmax=0.5)
        ax[0].set_title("Fixed patch")
        ax[1].imshow(volume_1[volume_1.shape[0] // 2, :, :], cmap='gray')#, vmin=0, vmax=0.5)
        ax[1].set_title("Moving patch")
        im = ax[2].imshow(np.abs(volume_0[volume_0.shape[0] // 2, :, :] - volume_1[volume_0.shape[0] // 2, :, :]), cmap='inferno')
        ax[2].set_title("Difference")

        # [left, bottom, width, height]
        ax_slider = plt.axes([0.3, 0.1, 0.4, 0.03])

        # add sliders
        slice = Slider(ax_slider, 'Slice', 0, volume_0.shape[0] - 1, valinit=volume_0.shape[0] // 2)

        def update_left(val):
            ax[0].clear()
            ax[0].imshow(volume_0[int(slice.val), :, :], cmap='gray')#, vmin=0, vmax=1)

            ax[1].clear()
            ax[1].imshow(volume_1[int(slice.val), :, :], cmap='gray')#, vmin=0, vmax=1)

            ax[2].clear()
            im = ax[2].imshow(np.abs(volume_0[int(slice.val), :, :] - volume_1[int(slice.val), :, :]), cmap='inferno')

            fig.canvas.draw_idle()

            ax[0].set_title("Fixed patch")
            ax[1].set_title("Moving patch")
            ax[2].set_title("Difference")

        slice.on_changed(update_left)

        if label is not None:
            plt.suptitle(label, fontweight="bold")

        fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

        # plt.tight_layout()
        plt.show()

        return slice


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


def visualise_per_class_accuracies(accuracies, class_names, title="Accuracies", lim=True):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.2)

    class_names[0] = "unrelated"
    p = sns.barplot(class_names, accuracies, palette="crest")

    p.set_xlabel("Offsets", fontsize=15)

    p.set_xticklabels(p.get_xticklabels(), rotation=90)

    p.set_ylabel("Accuracies per class", fontsize=15)
    if lim is True:
        p.set(ylim=(0, 1))

    plt.title(title, fontsize=15, fontweight="bold")
