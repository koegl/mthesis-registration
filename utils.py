import numpy as np
import nibabel as nib
import warnings
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# todo add way of encoding of patches with offset 0
# todo add way of encoding of patches with offset bigger than patch (unrelated)
# todo patches in pixel data actually shouldn't be cube, because they are stretched in world space
# todo would just reversing the order of the patches be enough to be treated as 'augmented data'?
# to each patch add the label containing the offset


def save_np_array_as_nifti(array, path, affine, header=None):
    """
    Save an nd array as a nifti file.
    :param array: the nd array to save
    :param path: the path to save the nifti file
    :param header: the header of the nifti file
    :param affine: the affine of the nifti file
    """

    img = nib.Nifti1Image(array, affine=affine, header=header)

    nib.save(img, path)


def create_radial_gradient(width, height, depth):
    """
    Create a radial gradient.
    :param width: width of the volume
    :param height: height of the volume
    :param depth: depth of the volume
    :return: the gradient volume as a nd array
    """

    x, y, z = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height), np.linspace(-1, 1, depth))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    r = r / np.max(r)

    return r


def generate_list_of_patch_offsets(offsets):
    """
    Generate a list of offsets. Symmetric in all three (x,y,z) dimensions. The offset is always only in one dimension.
    :param offsets: a list of offsets (will be used for all directions)
    :return: a list of offsets in random order
    """

    dimensions = 3  # spatial dimensions

    offset_list = []
    zero_offset = False  # we want to skip once we added one zero offset

    # loop through the offsets and for each one three times for the three dimensions
    for offset in offsets:
        for j in range(dimensions):

            offset_vector = [0, 0, 0]
            offset_vector[j] = offset

            if offset_vector == [0, 0, 0]:
                if zero_offset is True:
                    continue

                zero_offset = True

            offset_list.append(offset_vector)

    # append unrelated patch - encoded with negative 7
    offset_list.append([-7, -7, -7])

    # randomise the order of the offsets
    # random.shuffle(offset_list)

    return offset_list


def crop_volume_borders(volume):
    """
    Removes as much surrounding black pixels from a volume as possible
    :param volume: The entire volume
    :return: volume_cropped
    """

    shape = volume.shape

    assert len(shape) == 3, "Volume must be 3D"

    # set maximum and minimum boundaries in case the volume touches the sides of the image
    min_x = 0
    min_y = 0
    min_z = 0
    max_x = shape[0] - 1
    max_y = shape[1] - 1
    max_z = shape[2] - 1

    # find first plane in the x-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[0]):
        if np.count_nonzero(volume[i, :, :]) > 0:
            min_x = i
            break
    # find first plane in the x-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[0] - 1, -1, -1):
        if np.count_nonzero(volume[i, :, :]) > 0:
            max_x = i
            break

    # find first plane in the y-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[1]):
        if np.count_nonzero(volume[:, i, :]) > 0:
            min_y = i
            break
    # find first plane in the y-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[1] - 1, -1, -1):
        if np.count_nonzero(volume[:, i, :]) > 0:
            max_y = i
            break

    # find first plane in the z-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[2]):
        if np.count_nonzero(volume[:, :, i]) > 0:
            min_z = i
            break
    # find first plane in the z-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[2] - 1, -1, -1):
        if np.count_nonzero(volume[:, :, i]) > 0:
            max_z = i
            break

    # crop the volume
    volume_cropped = volume[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

    return volume_cropped





def display_volume_slice(volume):
    """
    Displays a slice of a 3D volume in a matplotlib figure
    :param volume: the volume
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    ax.imshow(volume[volume.shape[0] // 2, :, :], cmap='gray')

    ax_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
    slice = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=volume.shape[0] // 2)

    def update(val):
        ax.clear()
        ax.imshow(volume[int(slice.val), :, :], cmap='gray')
        fig.canvas.draw_idle()

    slice.on_changed(update)

    plt.show()



























