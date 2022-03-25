import numpy as np
import cv2


def scale_image(image, sx, sy):
    """
    Returns the image scaled by sx and sy
    :param image: numpy image to scale
    :param sx: scale in x direction
    :param sy: scale in y direction
    :return: The scaled image
    """

    rows, cols = image.shape

    # create the transformation matrix (3x3)
    transform = np.float32([[1, sx, 0], [0, sy, 0]])

    return cv2.warpAffine(image, transform, (cols, rows))


def translate_image(image, dx, dy):
    """
    Returns the image translated along x and y by dx and dy, respectively
    :param image: numpy image to translate
    :param dx: displacement in x
    :param dy: displacement in y
    :return: The translated image
    """

    cols, rows = image.shape

    # create the transformation matrix (3x3)
    transform = np.float32([[1, 0, dx], [0, 1, dy]])

    return cv2.warpAffine(image, transform, (cols, rows))


def rotate_image(image, angle):
    """
    Returns the image rotated counter-clockwise by angle
    :param image: numpy image to translate
    :param angle: counter-clockwise rotation angle
    :return: The rotated image
    """

    cols, rows = image.shape

    # create the transformation matrix (3x3)
    transform = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    return cv2.warpAffine(image, transform, (cols, rows))


def rigid_transform(image, angle, dx, dy):
    """
    Returns the image rotated by angle and translated by dx and dy
    :param image: numpy image to transform
    :param angle: counter-clockwise rotation angle
    :param dx: displacement in x
    :param dy: displacement in y
    :return: The transformed image
    """

    return translate_image(rotate_image(image, angle), dx, dy)


def affine_transform(image, a, dx, dy, sx, sy):
    """
    Returns the image trasnformed with an affine transformation matrix
    :param image: numpy image to transform
    :param a: counter-clockwise rotation angle
    :param dx: displacement in x
    :param dy: displacement in y
    :param sx: x scale
    :param sy: y scale
    :return: The transformed image
    """

    scale = scale_image(image, sx, sy)
    scale_rot = rotate_image(scale, a)
    scale_rot_tran = translate_image(scale_rot, dx, dy)

    return scale_rot_tran

