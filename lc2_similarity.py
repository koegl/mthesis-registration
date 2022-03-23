import numpy as np
import tensorflow as tf
from utils import set_up_progressbar


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
    mask = tf.greater(us, 0)
    us_non_zero = tf.boolean_mask(us, mask)

    # get non-zero elements in a flat array

    # get variance of non-zero elements
    _, us_variance = tf.nn.moments(tf.reshape(us_non_zero, [-1]))  # slightly different from matlab var

    # if the variance is 'significant'
    if us_variance > 10 ** - 12 and len(us_non_zero) > pixels_amount / 2:
        # flatten and reshape img2
        mr_and_grad_reshaped = tf.reshape(mr_and_grad, (pixels_amount, mr_and_grad.shape[2]))

        # concatenate with ones
        ones = tf.ones((pixels_amount, 1))
        mr_and_grad_and_ones = tf.stack([mr_and_grad_reshaped, ones])

        # get the pseudo-inverse of the array with only non-zero elements
        mr_pseudo_inverse = tf.linalg.pinv(mr_and_grad_and_ones[ids, :])

        parameter = tf.multiply(mr_pseudo_inverse, us_non_zero)

        buf_var = tf.nn.moments(us_non_zero - np.dot(mr_and_grad_and_ones[ids, :], parameter))
        similarity = 1 - (buf_var / us_variance)
        weight = tf.sqrt(us_variance)

        measure = weight * similarity

    else:
        similarity = 0
        weight = 0
        measure = 0

    return similarity, measure, weight


def lc2_similarity_patch(img1, img2, patchsize=9):
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
    # mr_and_grad = tf.zeros((img1.shape[0], img1.shape[1], 2))

    buf0 = img2
    buf1 = tf.math.abs(tf.image.image_gradients(tf.expand_dims(tf.expand_dims(img2, axis=0), axis=0)))  # , axis=1))
    buf1 = tf.squeeze(buf1[0, :, :, :])
    mr_and_grad = tf.stack([buf0, buf1], axis=2)
    img2 = mr_and_grad

    # set parameters
    max_x = img1.shape[0]
    max_y = img1.shape[1]

    # half of the maximal size of the patch
    total_size = ((2*patchsize + 1)**2) / 2

    measure = tf.zeros(img1.shape)
    weights = tf.zeros(img1.shape)

    # set up progressbar
    progress_bar = set_up_progressbar(max_x * max_y)
    counter = 1
    progress_bar.start()

    # loop through all pixels
    for y in range(max_y):
        for x in range(max_x):

            progress_bar.update(counter)
            counter += 1

            # extract patches from us and mr+grad
            patch1 = img1[
                            max(1, x-patchsize):min(max_x, x+patchsize),
                            max(1, y-patchsize):min(max_y, y+patchsize)
                     ]

            patch2 = img2[
                            max(1, x-patchsize):min(max_x, x+patchsize),
                            max(1, y-patchsize):min(max_y, y+patchsize),
                            :
                     ]

            # if a patch is bigger than half the maximal size of the patch calculate the similarity
            # patches that are too small (too close to the border get ignored)
            if patch1.shape[0] * patch1.shape[1] > total_size:
                _, measure[x, y], weights[x, y] = lc2_similarity(patch1, patch2)

    if sum(weights.flatten()) == 0:
        return 0

    similarity = sum(measure.flatten()) / sum(weights.flatten())

    return similarity
