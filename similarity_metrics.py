import numpy as np

from LC2_similarity_metric_3D import lc2_3d, gradient_magnitude


def compute_similarity_metric(im1, im2, metric="ssd", patchsize=None):
    """
    Copute the given similarity metric between images m1 and m2
    :param im1: First image
    :param im2: Second image
    :param metric: Third image
    :param patchsize: Patchsize for LC2
    :return: The calculated similarity
    """

    if metric.lower() == "ssd":
        s = ssd(im1, im2)
    elif metric.lower() == "ncc":
        raise NotImplementedError("NCC not implemented yet")
        # s = -ncc(im1, im2)  # we want to maximise it, so '-'
    elif metric.lower() == "mi":
        raise NotImplementedError("MI not implemented yet")
        # s = -mi(im1, im2)  # we want to maximise it, so '-'
    elif metric.lower() == "lc2":
        s = -lc2_3d(im1, im2, gradient_magnitude(im2), patchsize)
    else:
        raise NotImplementedError("Wrong similarity metric.")

    return s


def ssd(x, y):
    """
    Computes the sum of the squared differences between the given matrices
    :param x: first image
    :param y: second image
    :return: sum of the squared differences
    """
    result = (1/(np.prod(y.shape))) * np.sum(np.square(x - y))

    return result


def ncc(x, y):
    # todo adapt for 3D

    """
    Computes the normalised cross correlation between the given matrices
    :param x: first image
    :param y: second image
    :return: normalised cross correlation
    """
    n, m = y.shape

    # Demean the images
    x_demean = x - np.mean(x)
    y_demean = y - np.mean(y)

    # calculate ncc
    s = (1/(n*m)) * np.sum((x_demean * y_demean) / (x_demean.std() * y_demean.std()))

    return s


def mi(x, y):
    # todo adapt for 3D
    """
    Computes the mutual information between the given matrices
    :param x: first image
    :param y: second image
    :return: mutual information
    """

    # compute joint histogram
    jh = np.histogram2d(x.ravel(), y.ravel())[0]
    jh = jh + np.finfo(float).eps # add eps for stability

    # Normalize.
    sh = np.sum(jh)
    jh /= sh

    # get individual distributions
    s1 = np.sum(jh, axis=0, keepdims=True)
    s2 = np.sum(jh, axis=1, keepdims=True)

    # Compute MI
    mi_result = np.sum(-s1 * np.log2(s1)) + np.sum(-s2 * np.log2(s2)) - np.sum(-jh * np.log2(jh))

    return mi_result
