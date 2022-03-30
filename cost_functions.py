import cv2
import numpy as np

from similarity_metrics import compute_similarity_metric ,ssd, ncc, mi, lc2
from image_manipulation import rigid_transform, affine_transform


def rigid_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity="lc2"):
    """
    Cost function for a rigid transformation.
    Computes a similarity measure between the given images using the given similarity metric
    :param transform_parameters: angle, dx, dy
    :param fixed_image: the reference image
    :param moving_image: the image which will be transformed to match the reference image
    :param similarity: the name of the similarity metric
    :return: the computed similarity metric
    """

    # get parameters
    angle = transform_parameters[0]
    dx = transform_parameters[1]
    dy = transform_parameters[2]

    # transform the moving image with the parameters
    transformed_moving_image = rigid_transform(moving_image, angle, dx, dy)

    # compute the similarity value
    s = compute_similarity_metric(transformed_moving_image, fixed_image, similarity)

    return s, transformed_moving_image


def affine_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity="lc2"):
    """
    Cost function for a rigid transformation.
    Computes a similarity measure between the given images using the given similarity metric
    :param transform_parameters: angle, dx, dy
    :param fixed_image: the reference image
    :param moving_image: the image which will be transformed to match the reference image
    :param similarity: the name of the similarity metric
    :return: the computed similarity metric
    """

    # get parameters
    angle = transform_parameters[0]
    dx = transform_parameters[1]
    dy = transform_parameters[2]
    sx = transform_parameters[3]
    sy = transform_parameters[4]

    # transform the moving image with the parameters
    transformed_moving_image = affine_transform(moving_image, angle, dx, dy, sx, sy)

    # compute the similarity value
    s = compute_similarity_metric(transformed_moving_image, fixed_image, similarity)

    return s, transformed_moving_image


def perspective_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity="lc2"):
    """
    Cost function for a rigid transformation.
    Computes a similarity measure between the given images using the given similarity metric
    :param transform_parameters: 3x3 matrix
    :param fixed_image: the reference image
    :param moving_image: the image which will be transformed to match the reference image
    :param similarity: the name of the similarity metric
    :return: the computed similarity metric
    """
    assert transform_parameters.shape == (9,), "Transform matrix for perspective_transformation_cost_function has to " \
                                               "be 3x3 - or length 9 as passed through the optimiser"

    transform_parameters = np.reshape(transform_parameters, (3, 3))

    # transform the moving image with the parameters
    transformed_moving_image = cv2.warpPerspective(moving_image, transform_parameters,
                                                   (moving_image.shape[1], moving_image.shape[0]))

    # compute the similarity value
    s = compute_similarity_metric(transformed_moving_image, fixed_image, similarity)

    return s, transformed_moving_image


def cost_function(transform_parameters, fixed_image, moving_image, similarity="ssd", symmetry=False):
    """
    Cost function for several transformation types.
    Computes a similarity measure between the given images using the given similarity metric
    :param transform_parameters: the transformation parameters
    :param fixed_image: the reference image
    :param moving_image: the image which will be transformed to match the reference image
    :param similarity: the name of the similarity metric
    :param symmetry: Regularisation to enforce symmetry
    :return: the computed similarity metric
    """

    if len(transform_parameters) == 3:  # rotate + translate
        s, transformed = rigid_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity)
    elif len(transform_parameters) == 5:  # scale + rotate + translate
        s, transformed = affine_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity)
    elif len(transform_parameters) == 9:  # scale + rotate + translate + skew
        s, transformed = perspective_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity)
    else:
        raise NotImplementedError("Wrong number of transformation parameters in cost_function()")

    if symmetry:
        reg_val = symmetry_regulariser(transformed, 1.0)
        s += reg_val

    return s


def extract_mask(mask):
    """
    crops the image so that it only contains the mask
    :param mask: the mask to crop
    :return: cropped mask
    """

    x, y = mask.shape
    x_left = 0
    x_right = x + 0
    y_top = 0
    y_bottom = y + 0

    # new left index
    for i in range(x):
        if np.sum(mask[i, :]) > 0:
            x_left = i - 1
            break

    # new right index
    for i in reversed(range(x)):
        if np.sum(mask[i, :]) > 0:
            x_right = i + 1
            break

    # new top index
    for i in range(y):
        if np.sum(mask[:, i]) > 0:
            y_top = i - 1
            break

    # new bottom index
    for i in reversed(range(y)):
        if np.sum(mask[:, i]) > 0:
            y_bottom = i + 1
            break

    if x_left < 0:
        x_left = 0
    if x_right > x:
        x_right = x
    if y_top < 0:
        y_top = 0
    if y_bottom > y:
        y_bottom = y

    new_mask = mask[x_left:x_right, y_top:y_bottom]

    return new_mask


def symmetry_regulariser(image, factor=0.5):
    """
    Checks if an image is qpproximately symmetric. if Yes it returns True.
    :param image: The image
    :param factor: regulariser factor by which the similarity metric will be multiplied
    :return: 1 if symmetric, 1.5 otherwise
    """

    mask = extract_mask(image)

    if mask.shape[1] % 2 == 0:
        left = mask[:, :int(mask.shape[1]/2)]
    else:
        left = mask[:, :int(mask.shape[1]/2)+1]
    right = mask[:, int(mask.shape[1]/2):]

    right_flip = cv2.flip(right, 1)

    reg = ssd(left, right_flip)

    return reg * factor
