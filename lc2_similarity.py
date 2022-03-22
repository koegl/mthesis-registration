import numpy as np


def lc2_similarity(us, mr):
    """
    Calculates the LC2 similarity between a 2D US image and the corresponding 2D MR image. The images have to be of
    equal size. Based on  http://campar.in.tum.de/Main/LC2Code (LC2Similarity)
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
    mr_and_grad[:, :, 1] = np.absolute(np.gradient(mr, axis=1))

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
    mr_and_grad = np.zeros((img1.shape[0], img1.shape[1], 2))
    mr_and_grad[:, :, 0] = img2
    mr_and_grad[:, :, 1] = np.absolute(np.gradient(img2, axis=1))

    img2 = mr_and_grad

    # set parameters
    max_x = img1.shape[0]
    max_y = img1.shape[1]

    # half of the maximal size of the patch
    total_size = ((2*patchsize + 1)**2) / 2

    sim = np.zeros(img1.shape)
    measure = np.zeros(img1.shape)
    weights = np.zeros(img1.shape)

    # loop through all pixels
    for y in range(max_y):
        for x in range(max_x):

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
            # todo what happens with border pixels?
            if patch1.size > total_size:
                sim[x, y], measure[x, y], weights[x, y] = lc2_similarity(patch1, patch2)

    if sum(weights.flatten()) == 0:
        return 0

    similarity = sum(measure.flatten()) / sum(weights.flatten())

    return similarity
