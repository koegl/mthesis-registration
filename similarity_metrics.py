import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import sys


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


def lc2_similarity(us, mr_and_grad):
    """
    Calculates the LC2 similarity between a 2D US image and the corresponding 2D MR+grad image. The images have to be of
    equal size. Based on  http://campar.in.tum.de/Main/LC2Code (LC2Similarity)
    :param us: The US image (one channel image of size n*m)
    :param mr_and_grad: The MR+grad image (c channel image of size n*m*c)
    :return: similarity, measure, weight
    """
    if us.shape != mr_and_grad.shape[0:2]:
        raise ValueError("US and MR images have different dimensions! (they have to be equal)")

    shape = us.shape

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


def lc2_inner_loop(y, max_y, max_x, img1, img2, patchsize, total_size):
    """
    The inner loop of the double for loop in lc2_similarity_patch
    """
    b = "\033[1;31;31mProgress: " + str(y) + " / " + str(max_y)
    sys.stdout.write('\r' + b)

    measure_list = []
    weights_list = []

    for x in range(max_x):

        # extract patches from us and mr+grad
        patch1 = img1[
                 max(0, x - patchsize):min(max_x, x + patchsize),
                 max(0, y - patchsize):min(max_y, y + patchsize)
                 ]

        patch2 = img2[
                 max(0, x - patchsize):min(max_x, x + patchsize),
                 max(0, y - patchsize):min(max_y, y + patchsize),
                 :
                 ]

        # if a patch is bigger than half the maximal size of the patch calculate the similarity
        # patches that are too small (too close to the border get ignored)
        if patch1.size > total_size:
            _, measure, weights = lc2_similarity(patch1, patch2)
            measure_list.append(measure)
            weights_list.append(weights)

    return sum(measure_list), sum(weights_list)


def lc2(img1, img2, patchsize=9):
    """
    Calculates the LC2 similarity patch-wise between a 2D US image and the corresponding 2D MR image. The images have to
    be of equal size. Based on http://campar.in.tum.de/Main/LC2Code (LC2SimilarityPatch)
    :param img1: The US image (one channel image of size n*m)
    :param img2: The MR image (one channel image of size n*m)
    :param patchsize: Size of the patch
    :return: similarity
    """
    # calculate gradient of MR
    # create an MR+gradient matrix
    mr_and_grad = np.zeros((img1.shape[0], img1.shape[1], 2))
    mr_and_grad[:, :, 0] = img2
    mr_and_grad[:, :, 1] = np.absolute(np.gradient(img2, axis=1))

    img2 = mr_and_grad

    # set parameters
    max_x = img1.shape[0]
    max_y = img1.shape[1]

    # half of the maximal size of the patch
    total_size = ((2*patchsize + 1)**2) / 2

    # loop through all pixels
    num_cores = multiprocessing.cpu_count()
    return_list = Parallel(n_jobs=num_cores)(delayed(lc2_inner_loop)(i, max_y, max_x, img1, img2, patchsize, total_size)
                                             for i in range(max_y))

    measure_sum = 0
    weights_sum = 0

    for elem in return_list:
        measure_sum += elem[0]
        weights_sum += elem[1]

    if weights_sum == 0:
        return measure_sum

    similarity = measure_sum / weights_sum

    return similarity
