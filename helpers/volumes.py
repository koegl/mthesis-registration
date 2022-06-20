import numpy as np
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
