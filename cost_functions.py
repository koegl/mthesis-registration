import sys
from scipy.ndimage import affine_transform
import numpy as np
from similarity_metrics import compute_similarity_metric


step = 0


def cost_function(transformation, fixed_image, moving_image, deformation_type, similarity="ssd", patchsize=None):
    """
    Cost function for several transformation types.
    Computes a similarity measure between the given images using the given similarity metric
    :param transformation: the transformation parameters
    :param fixed_image: the reference image
    :param moving_image: the image which will be transformed to match the reference image
    :param deformation_type: the type of deformation
    :param similarity: the name of the similarity metric
    :param patchsize: the patchsize for lc2
    :return: the computed similarity metric
    """

    if deformation_type.lower() != "affine":
        raise NotImplementedError("Wrong deformation type, only affine is implemented")

    # reshape transformation to 4x4 matrix (and add homogeneous coordinate row)
    transformation = transformation.reshape(3, 4)
    homogeneous_vec = np.zeros((1, 4))
    homogeneous_vec[0, 3] = 1
    transformation = np.concatenate((transformation, homogeneous_vec), axis=0)

    # transform the moving image with the parameters
    transformed_moving_image = affine_transform(moving_image, transformation)

    # compute the similarity value
    s = compute_similarity_metric(fixed_image, transformed_moving_image, similarity, patchsize)

    # write out progress
    global step
    step += 1
    b = f"\033[1;31;31mStep: {step}; \tSimilarity = {s:0.2f}"
    sys.stdout.write('\r' + b)

    return s
