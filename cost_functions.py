from image_manipulation import rigid_transform
from similarity_metrics import compute_similarity_metric


def cost_function(transform_parameters, fixed_image, moving_image, similarity="ssd"):
    """
    Cost function for several transformation types.
    Computes a similarity measure between the given images using the given similarity metric
    :param transform_parameters: the transformation parameters
    :param fixed_image: the reference image
    :param moving_image: the image which will be transformed to match the reference image
    :param similarity: the name of the similarity metric
    :return: the computed similarity metric
    """

    if len(transform_parameters) == 3:  # rotate + translate
        s = rigid_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity)
    elif len(transform_parameters) == 5:  # scale + rotate + translate
        s = affine_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity)
    else:
        raise NotImplementedError("Wrong number of transformation parameters in cost_function()")

    return s


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

    return s


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

    return s

