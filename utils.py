import numpy as np
import nibabel as nib
import warnings


# todo generalise everything to 3D


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


def extract_overlapping_patches(image_fixed, image_moving, centre, size, offset=None):
    """
    Extract overlapping patches from the two volumes. One of the volume patches will be offset by 'offset'
    :param image_fixed: the volume with the standard patch
    :param image_moving: the volume with the offset patch
    :param centre: centre of the patch
    :param size: size of the patch
    :param offset: offset of the image_moving patch
    :return:
    """

    patch_fixed = extract_cubical_patch_offset(image_fixed, centre, size, offset=None)

    patch_moving = extract_cubical_patch_offset(image_moving, centre, size, offset=offset)

    return patch_fixed, patch_moving
