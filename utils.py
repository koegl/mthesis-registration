import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys


# todo extend to 3-D
def create_deformation_grid(list_of_vectors, grid_shape, dim=3):
    """
    This function takes a list of vectors and creates a grid of deformation vectors for the package 'elasticdeform'.
    The length of list_of_vectors must be equal to the product of all elements in grid_shape.
    The first vector is th edeformation of the top right point, the second vector is the next point to the right, etc
    until all rows are filled. The last vector is the deformation of the bottom right point.

    Example 1:
    shape: (2, 4, 3);
    shape[0]: spatial dimensions of the input image -> 2D (top row are y-coordinates, bottom x-coordinates)
    shape[1]: number of rows (of control points)    -> four rows of control points
    shape[2]: number of columns (of control points) -> three columns of control points

    Example 2:
    for shape (2, 3, 3)
    [[top-left-inY, top-center-inY, top-right-inY], [center-left-inY, center-center-inY, center-right-inY], [down-left-inY, down-center-inY, down-right-inY]],
    [[top-left-inX, top-center-inX, top-right-inX], [center-left-inX, center-center-inX, center-right-inX], [down-left-inX, down-center-inX, down-right-inX]]

    Example 3:
    (input list_of_vectors as a list but written with new lines to make it easier to read)
    [[0, -16], [0, 0], [0, 0], [0, -16],
     [0, 0],   [0, 0], [0, 0], [0, 0],
     [0, 16],  [0, 0], [0, 0], [0, 16]]
     this corresponds to a grid with control points 'p', where active ones are marked with A
     A = = p = = p = = A
     = = = = = = = = = =
     = = = = = = = = = =
     p = = p = = p = = p
     = = = = = = = = = =
     = = = = = = = = = =
     A = = p = = p = = A

    :param list_of_vectors: A list containing all displacement vectors
    :param grid_shape: The shape of the grid (how many vectors in each dimension)
    :param dim: The dimension of the volume
    :return: The deformation grid
    """

    assert isinstance(list_of_vectors, list), 'list_of_vectors must be a list'
    assert isinstance(grid_shape, list), 'grid_shape must be a list'
    assert len(list_of_vectors) == np.prod(grid_shape), 'The length of list_of_vectors must be equal to the product of' \
                                                        ' all elements in grid_shape'

    assert dim == 2 or dim == 3, 'The dimension of the volume must be 2 or 3.'

    # convert the list of vectors to a numpy array
    coors = np.asarray(list_of_vectors).transpose()

    # reshape the vectors so that they match the elasticdeform requirements
    deformation_grid = np.reshape(coors, [dim] + grid_shape)

    return deformation_grid


def display_volume_slice(volume, title=None):
    """
    Displays a slice of a 3D volume in a matplotlib figure
    :param volume: the volume
    :param title: the title of the plot
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    ax.imshow(volume[volume.shape[0] // 2, :, :], cmap='gray')
    ax.set_xlabel('y')
    ax.set_ylabel('z')

    global axis
    axis = 'x'

    if title is not None:
        plt.title(title, fontweight="bold")
    else:
        ax.set_title(f"Axis {axis}")

    ax_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=volume.shape[0] // 2)

    def on_press(event):
        global axis

        sys.stdout.flush()
        if event.key == 'x':
            axis = 'x'
            update(int(slider.val))
        elif event.key == 'y':
            axis = 'y'
            update(int(slider.val))
        elif event.key == 'z':
            axis = 'z'
            update(int(slider.val))

    def update(val):
        ax.clear()
        global axis

        if axis == 'x':
            ax.imshow(volume[int(slider.val), :, :], cmap='gray')
            ax.set_xlabel('y')
            ax.set_ylabel('z')
        elif axis == 'y':
            ax.imshow(volume[:, int(slider.val), :], cmap='gray')
            ax.set_xlabel('z')
            ax.set_ylabel('x')
        elif axis == 'z':
            ax.imshow(volume[:, :, int(slider.val)], cmap='gray')
            ax.set_xlabel('y')
            ax.set_ylabel('x')

        if title is not None:
            ax.set_title(title, fontweight="bold")
        else:
            ax.set_title(f"Axis {axis}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.show()

    return slider


