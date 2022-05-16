import numpy as np
import nibabel as nib
import warnings
import random


# todo add way of encoding of patches with offset 0
# todo add way of encoding of patches with offset bigger than pacth (unrelated)


def save_np_array_as_nifti(array, path):
    """
    Save an nd array as a nifti file.
    :param array: the nd array to save
    :param path: the path to save the nifti file
    """

    img = nib.Nifti1Image(array, np.eye(4))

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


def extract_cubical_patch_offset(image, center, size, offset=None):
    """
    Extract a cubical patch from the image.
    :param image: the volume as an nd array
    :param center: the center of the cubical patch
    :param size: the size of the cubical patch
    :param offset: the offset of the cubical patch
    :return: the cubical patch as an nd array
    """

    if offset is None:
        offset = [0, 0, 0]

    assert len(offset) == 3, "Offset must be a 3D vector"
    assert len(center) == 3, "Center must be a 3D vector"
    assert isinstance(size, int), "Size must be a scalar integer"

    x_min = center[0] - size // 2 + offset[0]
    x_max = center[0] + size // 2 + offset[0]
    y_min = center[1] - size // 2 + offset[1]
    y_max = center[1] + size // 2 + offset[1]
    z_min = center[2] - size // 2 + offset[2]
    z_max = center[2] + size // 2 + offset[2]

    # check if the patch is out of bounds
    if x_min < 0 or x_max >= image.shape[0] or \
       y_min < 0 or y_max >= image.shape[1] or \
       z_min < 0 or z_max >= image.shape[2]:
        warnings.warn("The patch is out of bounds.")
        return np.zeros((1, 1))

    return image[x_min:x_max, y_min:y_max, z_min:z_max]


def extract_overlapping_patches(image_fixed, image_offset, centre, size, offset=None):
    """
    Extract overlapping patches from the two volumes. One of the volume patches will be offset by 'offset'
    :param image_fixed: the volume with the standard patch
    :param image_offset: the volume with the offset patch
    :param centre: centre of the patch
    :param size: size of the patch
    :param offset: offset of the image_offset patch
    :return:
    """
    
    assert image_fixed.shape == image_offset.shape, "The two volumes must have the same shape"

    patch_fixed = extract_cubical_patch_offset(image_fixed, centre, size, offset=None)

    patch_offset = extract_cubical_patch_offset(image_offset, centre, size, offset=offset)

    return patch_fixed, patch_offset


def generate_list_of_patch_offsets(offsets):
    """
    Generate a list of offsets. Symmetric in all three (x,y,z) dimensions. The offset is always only in one dimension.
    :param offsets: a list of offsets (will be used for all directions)
    :return: a list of offsets in random order
    """

    dimensions = 3  # spatial dimensions

    offset_list = []
    zero_offset = False  # we want to skip once we added one zero offser

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

    # randomise the order of the offsets
    # random.shuffle(offset_list)

    return offset_list


# todo make the centres per dimension different for eachd dimension
def generate_list_of_patch_centres(centres_per_dimension, volume_size, patch_size):
    """
    Returns a list of patch centres that follow a grid based on centres_per_dimension
    :param centres_per_dimension: Amount of centres per x,y and z dimension (symmetric)
    :param volume_size: the size of the volume from which the centres will be taken
    :param patch_size: Size of the patch has to be provided so that there won't be any out of bounds problems (cubical)
    :return: a list of patch centres in randomised order
    """

    centres_list = []

    # get maximum dimension
    max_dim = np.max(volume_size)
    dimension_factor = volume_size / max_dim  # so we have a uniform grid in all dimensions

    # loop through the amount of centres per dimension with three nesting loops (one for each dimension)
    for i in range(int(centres_per_dimension * dimension_factor[0])):
        for j in range(int(centres_per_dimension * dimension_factor[1])):
            for k in range(int(centres_per_dimension * dimension_factor[2])):

                factor_x = (volume_size[0] // centres_per_dimension)
                factor_y = (volume_size[1] // centres_per_dimension)
                factor_z = (volume_size[2] // centres_per_dimension)

                if centres_per_dimension > volume_size[0]:
                    factor_x = 1
                if centres_per_dimension > volume_size[1]:
                    factor_y = 1
                if centres_per_dimension > volume_size[2]:
                    factor_z = 1

                centre_x = i * factor_x
                centre_y = j * factor_y
                centre_z = k * factor_z

                # check for out of bounds
                if centre_x < patch_size // 2 or centre_x > volume_size[0] - patch_size // 2:
                    continue
                if centre_y < patch_size // 2 or centre_y > volume_size[1] - patch_size // 2:
                    continue
                if centre_z < patch_size // 2 or centre_z > volume_size[2] - patch_size // 2:
                    continue

                centre = [centre_y, centre_x, centre_z]

                centres_list.append(centre)

    # randomise the order of the centres
    # random.shuffle(centres_list)

    return centres_list


# def generate_corresponding_patches_and_offsets


























