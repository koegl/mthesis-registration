# adapted from http://campar.in.tum.de/Main/LC2Code

import numpy as np


def lc2_similarity(img1 = None,img2 = None):
    # img1: one channel image of size n*m
    # img2: multi-channel image of size n*m*c

    # get amount of pixels
    pixels_amount = img2.shape[0] * img2.shape[1]

    # find indices of non-zero elements
    ids = np.flatnonzero(img1)

    # get non-zero elements in a flat array
    img1_non_zero = img1.flatten()[ids]

    # get variance of non-zero elements
    v1 = np.var(img1_non_zero)

    # if the variance is 'significant'
    if v1 > 10 ** - 12:
        if len(ids) > pixels_amount / 2:
            # flatten and reshape img2
            img2_flat_reshape = np.reshape(img2, (pixels_amount, img2.shape[2]))

            # concatenate with ones
            ones = np.ones((pixels_amount, 1))
            img2r = np.concatenate((img2_flat_reshape, ones), 1)

            # get the pseudo-inverse of the array with only non-zero elements
            pimg2r = np.linalg.pinv(img2r[ids, :])

            parameter = np.dot(pimg2r, img1_non_zero)

            similarity = 1 - (np.var(img1_non_zero - np.dot(img2r[ids, :], parameter)) / v1)
            weight = np.sqrt(v1)

            measure = weight * similarity

        else:
            similarity = 0
            weight = 0
            measure = 0
    
    else:
        similarity = 0
        weight = 0
        measure = 0

    return similarity, measure, weight
