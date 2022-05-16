import numpy as np
import warnings


def create_radial_gradient(width, height):
    """
    Create a radial gradient.
    :param width: width of the image
    :param height: height of the image
    :return: the gradient image as an nd array
    """

    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    r = np.sqrt(x ** 2 + y ** 2)

    return r


def extract_square_patch_offset(image, center, size, offset=None):
    """
    Extract a square patch from the image.
    :param image: the image as an nd array
    :param center: the center of the square patch
    :param size: the size of the square patch
    :param offset: the offset of the square patch
    :return: the square patch as an nd array
    """

    if offset is None:
        offset = [0, 0]

    x_min = center[0] - size // 2 + offset[0]
    x_max = center[0] + size // 2 + offset[0]
    y_min = center[1] - size // 2 + offset[1]
    y_max = center[1] + size // 2 + offset[1]

    # check if the patch is out of bounds
    if x_min < 0 or x_max >= image.shape[0] or y_min < 0 or y_max >= image.shape[1]:
        warnings.warn("The patch is out of bounds.")
        return np.zeros((1, 1))

    return image[x_min:x_max, y_min:y_max]


def extract_overlapping_patches(image_fixed, image_moving, centre, size, offset=None):
    """
    Extract overlapping patches from the two images. One of the image patches will be offset by 'offset'
    :param image_fixed: the image with the standard patch
    :param image_moving: the image with the offset patch
    :param centre: centre of the patch
    :param size: size of the patch
    :param offset: offset of the image_moving patch
    :return:
    """

    patch_fixed = extract_square_patch_offset(image_fixed, centre, size, offset=None)

    patch_moving = extract_square_patch_offset(image_moving, centre, size, offset=offset)

    return patch_fixed, patch_moving
