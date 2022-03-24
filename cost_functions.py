from similarity_metrics import mi, lc2
from image_manipulation import rigid_transform


def rigid_transformation_cost_function(transform_parameters, fixed_image, moving_image, similarity="lc2"):
    """
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
    if similarity.lower() == "lc2":
        s = lc2(transformed_moving_image, fixed_image)
    elif similarity.lower() == "mi":
        s = -mi(transformed_moving_image, fixed_image)
    else:
        raise NotImplementedError("Wrong similarity metric. Only lc2 is implemented at the moment.")

    return s