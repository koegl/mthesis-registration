import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D


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
        im = ax[2].imshow(np.abs(volume_0[volume_0.shape[0] // 2, :, :] - volume_1[volume_0.shape[0] // 2, :, :]), cmap='inferno', vmin=0, vmax=1)
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


def visualise_per_class_accuracies(accuracies, class_names, title="Accuracies", lim=(0, 1)):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.2)

    class_names[0] = "unrelated"
    p = sns.barplot(class_names, accuracies, palette="crest")

    p.set_xlabel("Offsets", fontsize=15)

    p.set_xticklabels(p.get_xticklabels(), rotation=90)

    p.set_ylabel("Accuracies per class", fontsize=15)
    p.set(ylim=lim)

    plt.title(title, fontsize=15, fontweight="bold")


def plot_offsets(true_offset, predicted_offsets=None):
    """
    Works only for not compounded offsets in x or y
    :param true_offset:
    :param predicted_offsets:
    :return:
    """

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # set axis limits
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    plt.xticks([], [])
    plt.yticks([], [])

    plt.gca().set_aspect('equal', adjustable='box')

    plt.text(19.5, 1, 'x', fontsize=10)
    plt.text(19.5, -0.5, '>', fontsize=10)
    plt.text(-1.5, 19.5, 'y', fontsize=10)
    plt.text(-0.8, 19.5, '>', fontsize=10, rotation=90)
    plt.legend(loc='upper right')

    custom_lines = [Line2D([0], [0], color='k', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='b', lw=4)]

    ax.legend(custom_lines, ['True offset', 'Predicted x-offset', 'Predicted y-offset'], loc='upper right')

    plt.suptitle("True and predicted offsets in x-y.\n", fontsize=15, fontweight="bold")
    plt.title("The bigger the opacity of green or blue, the bigger the predicted probability\n"
              "The length of the bars corresponds to the predicted offset", style="italic")

    if true_offset[2] != 0:
        return

    # display true offset
    if true_offset[0] == 0:
        starting_point = (-0.25, 0)
        width = 0.5
        height = true_offset[1]
    else:
        starting_point = (0, -0.25)
        width = true_offset[0]
        height = 0.5

    rect = patches.Rectangle(starting_point, width, height, linewidth=0, edgecolor='k', facecolor='0.0')
    ax.add_patch(rect)  # Add the patch to the Axes

    # get amount of offsets in x, y and z
    amounts_xy = [0, 0]
    for packet in predicted_offsets:
        offset = packet[0]

        if offset[0] == 0:
            amounts_xy[0] += 1
        elif offset[1] == 0:
            amounts_xy[1] += 1

    counter_xy = [0, 0]
    # display predicted offsets
    starts_in_x = np.linspace(-amounts_xy[0], amounts_xy[0], amounts_xy[0])
    starts_in_y = np.linspace(-amounts_xy[1], amounts_xy[1], amounts_xy[1])
    for packet in predicted_offsets:
        offset = packet[0]
        prob = packet[1]

        if offset[0] == 0:
            width = 2
            height = offset[1]
            starting_point = (starts_in_x[counter_xy[0]] - width/2, 0)
            if amounts_xy[0] == 1:
                starting_point = (starts_in_x[counter_xy[0]], 0)
            counter_xy[0] += 1
            color = 'b'
        else:
            width = offset[0]
            height = 2
            starting_point = (0, starts_in_y[counter_xy[1]] - height/2)
            if amounts_xy[1] == 1:
                starting_point = (starts_in_y[counter_xy[1]], 0)
            counter_xy[1] += 1
            color = 'g'

        rect = patches.Rectangle(starting_point, width, height, linewidth=0, facecolor=color, alpha=prob)
        ax.add_patch(rect)


    plt.show()
