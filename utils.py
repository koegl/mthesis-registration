import numpy as np


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
