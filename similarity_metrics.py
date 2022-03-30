import numpy as np


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
        s = -ncc(im1, im2)  # we want to maximise it, so '-'
    elif metric.lower() == "mi":
        s = -mi(im1, im2)  # we want to maximise it, so '-'
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

    n, m = y.shape

    result = (1/(n*m)) * np.sum(np.square(x - y))

    return result


def ncc(x, y):
    """
    Computes the normalised cross correlation between the given matrices
    :param x: first image
    :param y: second image
    :return: normalised cross correlation
    """
    n, m = y.shape

    # Demean the images
    x -= np.mean(x)
    y -= np.mean(y)

    # calculate ncc
    s = (1/(n*m)) * np.sum((x * y) / (x.std() * y.std()))

    return s


def mi(x, y):
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
