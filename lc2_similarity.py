import numpy as np
from numba import jit, vectorize

@vectorize
def clip_below(value, threshold):
    if value < threshold:
        value = threshold
    return value

@jit(nopython=True)
def lc2_similarity(us, mr_and_grad):
    """
    Calculates the LC2 similarity between a 2D US image and the corresponding 2D MR+grad image. The images have to be of
    equal size. Based on  http://campar.in.tum.de/Main/LC2Code (LC2Similarity)
    :param us: The US image (one channel image of size n*m)
    :param mr_and_grad: The MR+grad image (c channel image of size n*m*c)
    :return: similarity, measure, weight
    """
    # check the shape of the input images
    if us.shape != mr_and_grad.shape[0:2]:
        raise ValueError("US and MR images have different dimensions! (they have to be equal)")

    # define output variables
    similarity = 0
    weight = 0
    measure = 0

    # get amount of pixels
    pixels_amount = us.size

    # find indices of elements > 0
    us_clipped = clip_below(us, 0.0)  # change negatives to zero
    ids = np.flatnonzero(us_clipped)

    # get non-zero elements in a flat array
    us_non_zero = us.flatten()[ids]

    # get variance of non-zero elements
    us_variance = np.var(us_non_zero)  # slightly different from matlab var

    # if the variance is 'significant'
    if len(ids) > pixels_amount / 2:
        if us_variance > 10 ** - 12:

            # make array contiguous
            mr_and_grad = np.ascontiguousarray(mr_and_grad)
            
            # flatten and reshape img2
            mr_and_grad_reshaped = mr_and_grad.reshape((pixels_amount, mr_and_grad.shape[2]))
            
            # concatenate with ones
            ones = np.ones((pixels_amount, 1))
            mr_and_grad_and_ones = np.concatenate((mr_and_grad_reshaped, ones), 1)

            # get the pseudo-inverse of the array with only non-zero elements
            mr_pseudo_inverse = np.linalg.pinv(mr_and_grad_and_ones[ids, :])

            parameter = np.dot(mr_pseudo_inverse, us_non_zero)

            similarity = 1 - (np.var(us_non_zero - np.dot(mr_and_grad_and_ones[ids, :], parameter)) / us_variance)
            weight = np.sqrt(us_variance)

            measure = weight * similarity

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
    # calculate gradient of MR and create an MR+gradient matrix
    img2_grad = np.absolute(np.gradient(img2, axis=1))
    img2 = np.concatenate((img2[..., None], img2_grad[..., None]), -1)

    # set parameters
    max_x = img1.shape[0]
    max_y = img1.shape[1]

    # half of the maximal size of the patch
    total_size = ((2*patchsize + 1)**2) / 2

    measure = np.zeros(img1.shape)
    weights = np.zeros(img1.shape)

    # loop through all pixels
    for y in range(max_y):
        for x in range(max_x):

            # extract patches from us and mr+grad
            patch1 = img1[
                            max(0, x-patchsize):min(max_x, x+patchsize),
                            max(0, y-patchsize):min(max_y, y+patchsize)
                     ]

            patch2 = img2[
                            max(0, x-patchsize):min(max_x, x+patchsize),
                            max(0, y-patchsize):min(max_y, y+patchsize),
                            :
                     ]

            # if a patch is bigger than half the maximal size of the patch calculate the similarity
            # patches that are too small (too close to the border get ignored)
            if patch1.size > total_size:
                _, measure[x, y], weights[x, y] = lc2_similarity(patch1, patch2)

    if np.sum(weights) == 0:
        return 0

    similarity = np.sum(measure) / np.sum(weights)

    return similarity