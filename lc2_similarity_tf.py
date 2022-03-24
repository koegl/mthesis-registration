import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from utils import set_up_progressbar


def lc2_similarity_tf(us, mr_and_grad):
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
    buf = tf.reshape(tf.maximum(us, 0), [-1])
    ids = tf.experimental.numpy.nonzero(buf)

    # get non-zero elements in a flat array
    us_non_zero = tf.squeeze(tf.gather(tf.reshape(us, [-1]), ids))

    # get variance of non-zero elements
    _, us_variance = tf.nn.moments(us_non_zero, axes=[0])  # slightly different from matlab var

    # if the variance is 'significant'
    if us_variance > 10 ** - 12 and len(us_non_zero) > pixels_amount / 2:
        # flatten and reshape img2
        mr_and_grad_reshaped = tf.reshape(mr_and_grad, (pixels_amount, mr_and_grad.shape[2]))

        # concatenate with ones
        ones = tf.ones((pixels_amount, 1), dtype=dtypes.float64)
        mr_and_grad_and_ones = tf.concat([mr_and_grad_reshaped, ones], axis=1)

        # get the pseudo-inverse of the array with only non-zero elements
        constant_mr_and_grad_and_ones = tf.constant(mr_and_grad_and_ones)
        mr_pseudo_inverse = tf.squeeze(tf.linalg.pinv(tf.gather(constant_mr_and_grad_and_ones, ids)))

        parameter = tf.matmul(mr_pseudo_inverse, tf.expand_dims(us_non_zero, axis=1))

        buf = tf.squeeze(tf.gather(constant_mr_and_grad_and_ones, ids))
        _, buf_var = tf.nn.moments(us_non_zero - tf.squeeze(tf.matmul(buf, parameter)), axes=[0])
        similarity = 1 - (buf_var / us_variance)
        weight = tf.sqrt(us_variance)

        measure = weight * similarity

    else:
        similarity = 0
        weight = 0
        measure = 0

    return similarity, measure, weight


def lc2_similarity_patch_tf(us, mr, patchsize=9):
    """
    Calculates the LC2 similarity patch-wise between a 2D US image and the corresponding 2D MR image. The images have to
    be of equal size. Based on http://campar.in.tum.de/Main/LC2Code (LC2SimilarityPatch)
    :param us: The US image (one channel image of size n*m)
    :param mr: The MR image (one channel image of size n*m)
    :param patchsize: Size of the patch
    :return: similarity
    """
    # calculate gradient of MR
    # create an MR+gradient matrix

    # todo convert gradient to tensorflow method - problem: couldn't find equivalent function in tensorflow
    mr_and_grad = np.zeros((us.shape[0], us.shape[1], 2))
    mr_and_grad[:, :, 0] = mr
    mr_and_grad[:, :, 1] = np.absolute(np.gradient(mr, axis=1))
    mr_and_grad = tf.convert_to_tensor(mr_and_grad)
    # gradx, grady = tf.math.abs(tf.image.image_gradients(tf.expand_dims(tf.expand_dims(mr, axis=0), axis=3)))
    # grad = tf.squeeze(grad[0, :, :, :])
    # mr_and_grad = tf.stack([mr, grad], axis=2)

    # set parameters
    max_x = us.shape[0]
    max_y = us.shape[1]

    # half of the maximal size of the patch
    total_size = ((2*patchsize + 1)**2) / 2

    # measure = tf.Variable(tf.zeros(us.shape))
    # weights = tf.Variable(tf.zeros(us.shape))
    measure_sum = 0
    weights_sum = 0


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
            patch1 = us[
                            max(1, x-patchsize):min(max_x, x+patchsize),
                            max(1, y-patchsize):min(max_y, y+patchsize)
                     ]

            patch2 = mr_and_grad[
                            max(1, x-patchsize):min(max_x, x+patchsize),
                            max(1, y-patchsize):min(max_y, y+patchsize),
                            :
                     ]

            # if a patch is bigger than half the maximal size of the patch calculate the similarity
            # patches that are too small (too close to the border get ignored)
            if patch1.shape[0] * patch1.shape[1] > total_size:
                _, measure, weights = lc2_similarity_tf(patch1, patch2)
                measure_sum += measure
                weights_sum += weights

    if sum(tf.reshape(weights_sum, [-1])) == 0:
        return 0

    similarity = sum(tf.reshape(measure_sum, [-1])) / sum(tf.reshape(weights_sum, [-1]))

    return similarity
