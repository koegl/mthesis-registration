import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import PIL.Image as Image
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys


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


def get_image(path, im_size):
    im = Image.open(path)
    im = im.convert("L")
    min_shape = np.min(np.asarray(im).shape)
    im = im.crop((0, 0, min_shape, min_shape))
    im = im.resize(im_size, Image.ANTIALIAS)
    im = np.asarray(im)

    return im


def generate_bspline_deformation(displacements, shape):
    """
    Creates a b-spline deformation. Its inputs are the shape of the volume and an array of displacement vectors
    following the elasticdeform package convention.
    https://elasticdeform.readthedocs.io/en/latest/
    https://www.programcreek.com/python/example/96384/SimpleITK.BSplineTransformInitializer
    :param displacements: nd array of displacements
    :param shape: shape of the volume
    :return: the bspline deformation
    """
    assert isinstance(displacements, np.ndarray), 'displacements must be a numpy array'
    assert displacements.shape[0] == len(shape), "The dimension of the displacement array must match the dimension of "\
                                                 "the volume"

    # get the shape of the grid
    grid_shape = np.asarray(displacements.shape[1:])  # we skip the first dimension because it is the dim of the volume

    # Initialize bspline transform
    args = shape+(sitk.sitkFloat32,)
    ref_volume = sitk.Image(*args)

    params_shape = list(grid_shape - 3)
    params_shape = [int(x) for x in params_shape]
    bst = sitk.BSplineTransformInitializer(ref_volume, params_shape)

    # Transform displacements so that they can be used by the bspline transform
    p = displacements.flatten('A')

    # Set bspline transform parameters to the above shifts
    bst.SetParameters(p)

    return bst


def generate_deformation_field(bspline_deformation, shape):
    """
    Generates a deformation field from a bspline deformation.
    :param bspline_deformation: the deformation
    :param shape: the shape of the volume
    :return:
    """
    assert isinstance(bspline_deformation, sitk.BSplineTransform), 'bspline_deformation must be a bspline transform'

    args = shape + (sitk.sitkFloat32,)
    ref_volume = sitk.Image(*args)

    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_volume)
    displacement_field = displacement_filter.Execute(bspline_deformation)

    field = np.asarray(displacement_field).reshape(shape + (len(shape),))

    return field


def transform_volume(volume, bspline_deformation):
    """
    Transforms a volume using a sitk bspline deformation field.
    :param volume: the volume to transform
    :param bspline_deformation: the deformation field
    """
    # check if the volume is a numpy array
    assert isinstance(volume, np.ndarray), 'volume must be a numpy array'
    assert isinstance(bspline_deformation, sitk.BSplineTransform), 'bspline_deformation must be a bspline transform'

    # create sitk volume from numpy array
    sitk_volume = sitk.GetImageFromArray(volume, isVector=False)

    # create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_volume)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(bspline_deformation)
    resampler.SetDefaultPixelValue(0)

    # deform the volume
    deformed_volume = resampler.Execute(sitk_volume)
    deformed_volume = sitk.GetArrayFromImage(deformed_volume)
    deformed_volume = deformed_volume.astype(dtype=np.float32)

    return deformed_volume


def create_checkerboard(dimension, shape):
    assert dimension == 2 or dimension == 3, 'dimension must be 2 or 3'
    assert len(shape) == dimension, 'shape must be of length dimension'

    for sh in shape:
        assert sh % 10 == 0, 'shape must be divisible by 10'

    # create 2D checkerboard
    x = np.ones((shape[0]//10, shape[1]//10), dtype=float)
    x[::2] = 0
    x[:, ::2] = 1 - x[:, ::2]
    checkerboard_2d = x.repeat(10, axis=0).repeat(10, axis=1)

    if dimension == 2:
        return checkerboard_2d

    assert shape[2] % 20 == 0, 'shape must be divisible by 20'
    assert shape[0] == shape[1], 'first 2 shapes must be a square'

    # create 3D checkerboard
    checkerboard_2d_r = np.rot90(checkerboard_2d)

    checkerboard_2d_3 = np.tile(checkerboard_2d, (10, 1, 1))
    checkerboard_2d_r_3 = np.tile(checkerboard_2d_r, (10, 1, 1))

    packet = np.concatenate((checkerboard_2d_3, checkerboard_2d_r_3))
    checkerboard_3d = np.tile(packet, (shape[2]//20, 1, 1))

    return checkerboard_3d.T





def generate_grid_coordinates(grid_shape, volume_shape):
    """
    This function generates a grid (that is uniform along each axis separately). The 0th grid point lies in the corner
    of the volume (not in the center of the corresponding patch)
    :param grid_shape: shape of the grid
    :param volume_shape: shape of the volume
    :return: the grid coordinates
    """

    assert len(grid_shape) == len(volume_shape), 'grid_shape and volume_shape must be of same length'
    assert len(grid_shape) in [2, 3], 'grid_shape must be of length 2 or 3'

    if len(volume_shape) == 2:
        x, y = np.mgrid[0:volume_shape[0]:complex(0, grid_shape[0]),
                        0:volume_shape[1]:complex(0, grid_shape[1])]

        coordinates = np.stack((x, y))

    else:
        x, y, z = np.mgrid[0:volume_shape[0]:complex(0, grid_shape[0]),
                           0:volume_shape[1]:complex(0, grid_shape[1]),
                           0:volume_shape[2]:complex(0, grid_shape[2])]

        coordinates = np.stack((x, y, z))

    return coordinates
