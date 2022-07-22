import numpy as np
from numpy import sin, cos
import nibabel as nib
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import PIL.Image as Image
import SimpleITK as sitk
import matplotlib.pyplot as plt


def save_np_array_as_nifti(array, path, affine, header=None):
    """
    Save an nd array as a nifti file.
    :param array: the nd array to save
    :param path: the path to save the nifti file
    :param header: the header of the nifti file
    :param affine: the affine of the nifti file
    """

    img = nib.Nifti1Image(array, affine=affine, header=header)

    nib.save(img, path)


def create_radial_gradient(width, height, depth):
    """
    Create a radial gradient.
    :param width: width of the volume
    :param height: height of the volume
    :param depth: depth of the volume
    :return: the gradient volume as a nd array
    """

    x, y, z = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height), np.linspace(-1, 1, depth))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    r = r / np.max(r)

    return r


def crop_volume_borders(volume):
    """
    Removes as much surrounding black pixels from a volume as possible
    :param volume: The entire volume
    :return: volume_cropped
    """

    shape = volume.shape

    assert len(shape) == 3, "Volume must be 3D"

    # set maximum and minimum boundaries in case the volume touches the sides of the image
    min_x = 0
    min_y = 0
    min_z = 0
    max_x = shape[0] - 1
    max_y = shape[1] - 1
    max_z = shape[2] - 1

    # find first plane in the x-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[0]):
        if np.count_nonzero(volume[i, :, :]) > 0:
            min_x = i
            break
    # find first plane in the x-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[0] - 1, -1, -1):
        if np.count_nonzero(volume[i, :, :]) > 0:
            max_x = i
            break

    # find first plane in the y-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[1]):
        if np.count_nonzero(volume[:, i, :]) > 0:
            min_y = i
            break
    # find first plane in the y-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[1] - 1, -1, -1):
        if np.count_nonzero(volume[:, i, :]) > 0:
            max_y = i
            break

    # find first plane in the z-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[2]):
        if np.count_nonzero(volume[:, :, i]) > 0:
            min_z = i
            break
    # find first plane in the z-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[2] - 1, -1, -1):
        if np.count_nonzero(volume[:, :, i]) > 0:
            max_z = i
            break

    # crop the volume
    volume_cropped = volume[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

    return volume_cropped


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


def mark_patch_borders(volume: 'np.ndarray', centre: list, border_value: float, halp_patch_size: int):

    assert isinstance(volume, np.ndarray), 'volume must be a numpy array'
    assert isinstance(centre, list), 'centre must be a list'
    assert len(centre) == len(volume.shape), 'centre and volume must be of same length'

    d = halp_patch_size

    bounds = [centre[0] - d, centre[0] + d, centre[1] - d, centre[1] + d, centre[2] - d, centre[2] + d]

    # plane 1
    volume[bounds[0]:bounds[1], bounds[2]:bounds[3], centre[2] - d] = border_value
    # plane 2
    volume[bounds[0]:bounds[1], bounds[2]:bounds[3], centre[2] + d] = border_value
    # plane 3
    volume[bounds[0]:bounds[1], centre[1] - d, bounds[4]:bounds[5]] = border_value
    # plane 4
    volume[bounds[0]:bounds[1], centre[1] + d, bounds[4]:bounds[5]] = border_value
    # plane 5
    volume[centre[0] - d, bounds[2]:bounds[3], bounds[4]:bounds[5]] = border_value
    # plane 6
    volume[centre[0] + d, bounds[2]:bounds[3], bounds[4]:bounds[5]] = border_value

    return volume


def rot_vector_to_matrix(vector):
    """
    Rotation vector in radians gets converted to rotation matrix
    :param vector:
    :return:
    """
    a = vector[0]
    b = vector[1]
    c = vector[2]

    rot = np.array(
        [[cos(a) * cos(b), cos(a) * sin(b) * sin(c) - sin(a) * cos(c), cos(a) * sin(b) * cos(c) + sin(a) * sin(c)],
         [sin(a) * cos(b), sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * sin(b) * cos(c) - cos(a) * sin(c)],
         [-sin(b), cos(b) * sin(c), cos(b) * cos(c)]]
    )

    return rot


def create_transform_matrix(alpha, beta, gamma, dx, dy, dz):

    rotation_matrix = rot_vector_to_matrix([alpha, beta, gamma])

    transform = np.eye(4)

    transform[0:3, 0:3] = rotation_matrix
    transform[0:3, 3] = [dx, dy, dz]

    transform[transform < 0] = 0

    return transform
