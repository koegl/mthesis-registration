import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform_scipy_2d(image, alpha=0.5, sigma=2, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.


    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    # shape = (shape[0]//10, shape[1]//10)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


# todo extend to 3-D
def create_deformation_grid(list_of_vectors, grid_shape, dim=2):
    """
    This function takes a list of vectors and creates a grid of deformation vectors for the package 'elasticdeform'.
    The length of list_of_vectors must be equal to the product of all elements in grid_shape.
    The first vector is th edeformation of the top right point, the second vector is the next point to the right, etc
    until all rows are filled. The last vector is the deformation of the bottom right point.

    Example 1:
    shape: (2, 4, 3);
    shape[0]: spatial dimensions of the input image -> 2D (top row are y-coordinates, bottom x-coordinates)
    shape[1]: number of rows (of control points)    -> four rows of control points
    shape[2]: number of columns (of control points) -> three columns of control points

    Example 2:
    for shape (2, 3, 3)
    [[top-left-inY, top-center-inY, top-right-inY], [center-left-inY, center-center-inY, center-right-inY], [down-left-inY, down-center-inY, down-right-inY]],
    [[top-left-inX, top-center-inX, top-right-inX], [center-left-inX, center-center-inX, center-right-inX], [down-left-inX, down-center-inX, down-right-inX]]

    Example 3:
    (input list_of_vectors as a list but written with new lines to make it easier to read)
    [[0, -16], [0, 0], [0, 0], [0, -16],
     [0, 0],   [0, 0], [0, 0], [0, 0],
     [0, 16],  [0, 0], [0, 0], [0, 16]]
     this corresponds to a grid with control points 'p', where active ones are marked with A
     A = = p = = p = = A
     = = = = = = = = = =
     = = = = = = = = = =
     p = = p = = p = = p
     = = = = = = = = = =
     = = = = = = = = = =
     A = = p = = p = = A

    :param list_of_vectors: A list containing all displacement vectors
    :param grid_shape: The shape of the grid (how many vectors in each dimension)
    :param dim: The dimension of the volume
    :return: The deformation grid
    """

    assert isinstance(list_of_vectors, list), 'list_of_vectors must be a list'
    assert isinstance(grid_shape, list), 'grid_shape must be a list'
    assert len(list_of_vectors) == np.prod(grid_shape), 'The length of list_of_vectors must be equal to the product of' \
                                                        ' all elements in grid_shape'

    assert dim == 2, 'The dimension of the volume must be 2. 3D will be implemented later'

    # convert the list of vectors to a numpy array
    vectors = np.asarray(list_of_vectors)

    coors = np.stack((vectors[:, 1], vectors[:, 0]))

    # reshape the vectors so that they match the elasticdeform requirements
    deformation_grid = np.reshape(coors, [dim] + grid_shape)

    return deformation_grid
