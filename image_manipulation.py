import numpy as np
import cv2


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
