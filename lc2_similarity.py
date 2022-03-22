# adapted from http://campar.in.tum.de/Main/LC2Code

import numpy as np
from scipy import ndimage


def lc2_similarity(us=None, mr=None):
    """
    Calculates the LC2 similarity between a 2D US image and the corresponding 2D MR image. The images have to be of
    equal size. Based on  http://campar.in.tum.de/Main/LC2Code -> however, this implementation gives slightly different
    results, because the matlab gradient() function gives different results than the scipy sobel() function.
    :param us: The US image (one channel image of size n*m)
    :param mr: The MR image (one channel image of size n*m)
    :return: similarity, measure, weight
    """
    if us.shape != mr.shape:
        raise ValueError("US and MR images have different dimensions! (they have to be equal)")

    shape = us.shape

    # create an MR+gradient matrix
    mr_and_grad = np.zeros((shape[0], shape[1], 2))
    mr_and_grad[:, :, 0] = mr
    mr_and_grad[:, :, 1] = np.absolute(ndimage.sobel(mr, axis=1))  # matlab gradient is a bit different

    # get amount of pixels
    pixels_amount = shape[0] * shape[1]

    # find indices of elements > 0
    buf = us.copy()
    buf[buf < 0] = 0  # change negatives to zero
    ids = np.flatnonzero(buf)

    # get non-zero elements in a flat array
    us_non_zero = us.flatten()[ids]

    # get variance of non-zero elements
    us_variance = np.var(us_non_zero)  # slightly different from matlab var

    # if the variance is 'significant'
    if us_variance > 10 ** - 12 and len(ids) > pixels_amount / 2:
        # flatten and reshape img2
        mr_and_grad_reshaped = np.reshape(mr_and_grad, (pixels_amount, mr_and_grad.shape[2]))

        # concatenate with ones
        ones = np.ones((pixels_amount, 1))
        mr_and_grad_and_ones = np.concatenate((mr_and_grad_reshaped, ones), 1)

        # get the pseudo-inverse of the array with only non-zero elements
        mr_pseudo_inverse = np.linalg.pinv(mr_and_grad_and_ones[ids, :])

        parameter = np.dot(mr_pseudo_inverse, us_non_zero)

        similarity = 1 - (np.var(us_non_zero - np.dot(mr_and_grad_and_ones[ids, :], parameter)) / us_variance)
        weight = np.sqrt(us_variance)

        measure = weight * similarity

    else:
        similarity = 0
        weight = 0
        measure = 0

    return similarity, measure, weight
