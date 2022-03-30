import cv2
import numpy as np


def translate_image(x, dx, dy):
    ''' Returns the matrix [x] translated along x and y of [dx] and [dy] respectively.
    :param x: numpy matrix to translate
    :param dx: displacement in x
    :param dy: displacement in y
    :return: the translated matrix
    '''

    cols, rows = x.shape  # size of the matrix.

    # A way to build a transformation is to manually enter its values.
    # Here we only need to fill the translational part of a 3x3 matrix.
    transform = np.float32([[1, 0, dx], [0, 1, dy]])

    # Transforms the image with the given transformation.
    # The last parameter gives the size of the output, we want it to be the same of the input.
    return cv2.warpAffine(x, transform, (cols, rows))


def rotate_image(x, angle):
    ''' Returns the matrix [x] rotated counter-clock wise by [angle].
    :param x: numpy matrix to rotate
    :param angle: angle of rotation in DEGREES
    :return: the rotated matrix
    '''

    cols, rows = x.shape  # size of the matrix.

    # Creates a rotation matrix to rotate around a rotation center of a certain angle.
    # In this case we rotate around the center of the image (cols / 2, rows / 2) by the given angle.
    # The last paramters is a scale factor, 1 means no scaling.
    transform = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Transforms the image with the given transformation.
    # The last parameter gives the size of the output, we want it to be the same of the input.
    return cv2.warpAffine(x, transform, (cols, rows))


def rigid_transform(x, angle, dx, dy):
    ''' Returns the matrix [x] rotated counter-clock wise by [angle] and translated by [dx] and [dy] in x and y respectively.
    :param x: numpy matrix to transform
    :param angle: angle of rotation in DEGREES
    :param dx:  displacement in x
    :param dy: displacement in y
    :return: the transformed matrix
    '''

    # We just concatenate the functions to rotate and translate, rotation is always done first.
    return translate_image(rotate_image(x, angle), dx, dy)


def transform_image(image, t):
    """
    Combine the above transformations to be called by one function
    :param image: The image to be transformed
    :param t: The transformation parameters
    :return:
    """
    if t.size == 3:
        return rigid_transform(image, t[0], t[1], t[2])
    else:
        raise NotImplementedError("Wrong number of transformation parameters in transform()")
