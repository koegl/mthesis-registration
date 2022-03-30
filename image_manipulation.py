import cv2
import numpy as np


def transform_image(image, t):
    """
    Combine the above transformations to be called by one function
    :param image: The image to be transformed
    :param t: The transformation parameters
    :return:
    """
    if t.size == 3:
        return rigid_transform(image, t[0], t[1], t[2])
    elif t.size == 5:
        return affine_transform(image, t[0], t[1], t[2], t[3], t[4])
    elif t.size == 9:
        return perspective_transform(image, t)
    else:
        raise NotImplementedError("Wrong number of transformation parameters in transform()")


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
    transform = np.float32([[sx, 0, 0], [0, sy, 0]])

    return cv2.warpAffine(image, transform, (cols, rows))


def translate_image(image, dx, dy):
    """
    Returns the image translated along x and y by dx and dy, respectively
    :param image: numpy image to translate
    :param dx: displacement in x
    :param dy: displacement in y
    :return: The translated image
    """

    rows, cols = image.shape  # size of the matrix.

    # A way to build a transformation is to manually enter its values.
    # Here we only need to fill the translational part of a 3x3 matrix.
    transform = np.float32([[1, 0, dx], [0, 1, dy]])

    # Transforms the image with the given transformation.
    # The last parameter gives the size of the output, we want it to be the same of the input.
    return cv2.warpAffine(image, transform, (cols, rows))


def rotate_image(image, angle):
    """
    Returns the image rotated counter-clockwise by angle
    :param image: numpy image to translate
    :param angle: counter-clockwise rotation angle
    :return: The rotated image
    """

    rows, cols = image.shape  # size of the matrix.

    # Creates a rotation matrix to rotate around a rotation center of a certain angle.
    # In this case we rotate around the center of the image (cols / 2, rows / 2) by the given angle.
    # The last parameters is a scale factor, 1 means no scaling.
    transform = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Transforms the image with the given transformation.
    # The last parameter gives the size of the output, we want it to be the same of the input.
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

    # We just concatenate the functions to rotate and translate, rotation is always done first.
    return translate_image(rotate_image(image, angle), dx, dy)


def affine_transform(image, a, dx, dy, sx, sy):
    """
    Returns the image transformed with an affine transformation matrix
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


def perspective_transform(image, parameters):
    """
    Returns the image transformed with a perspective transformation matrix
    :param image: numpy image to transform
    :param parameters: the transformation 3x3 matrix
    :return: The transformed image
    """

    parameters = np.reshape(parameters, (3, 3))

    result = cv2.warpPerspective(image, parameters, (image.shape[1], image.shape[0]))

    return result
