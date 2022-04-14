import matplotlib.pyplot as plt
import numpy as np
import cv2
import nibabel as nib
from matplotlib.widgets import Slider
import functools
from skimage import data, color, io, img_as_float


def plot_images(x, y=None, z=None, main_title=""):
    """
    Display two images and their difference
    :param x: first image
    :param y: second image
    :param z: third image
    :param main_title: The main title over all the sub-plots
    """

    if len(x.shape) == 3:
        plot_slice(x, main_title)
    elif y is None and z is None:
        plot_one_image(x, main_title)

    elif x is not None and y is not None and z is not None:
        plot_fixed_moving_overlap(x, y, z, main_title)

    else:
        raise NotImplementedError("wrong number of images or shapes")


def plot_slice(volume, title=None):
    """
    Plot a 2d slice of a 3d volume
    :param volume: the 3d volume
    :param title: the title of the plot
    """
    image = plt.imshow(volume[0, :, :], vmin=0.0, vmax=volume.max(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(title)
    plt.subplots_adjust(bottom=0.25)

    # make a horizontal slider to control the slice
    slice_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(slice_slider_ax, label='Slice', valmin=0, valmax=volume.shape[0]-1, valinit=0, valstep=1)

    # register the callback function with the slider (package the image and volume with functools - parameter passing)
    slice_slider.on_changed(functools.partial(update_slice, image, volume))

    plt.show()


def update_slice(image, volume, value):
    image.set_data(volume[int(value), :, :])


def plot_one_image(image, title="", cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()


def plot_fixed_moving_overlap(x, y, z, main_title=""):
    # Creating a figure with subplots of a certain size.
    fig, ((plot1, plot2, plot3), (plot4, plot5, plot6)) = plt.subplots(2, 3, figsize=(10, 8))

    fig.suptitle(r"$\bf{" + main_title + "}$", fontsize=16)

    # Display the two images.
    plot1.imshow(x, cmap=plt.cm.gray)
    plot1.axis('off')
    plot1.set_title("Fixed image")

    plot2.imshow(y, cmap=plt.cm.gray)
    plot2.axis('off')
    plot2.set_title("Initial moving image")

    plot3.imshow(x-y, cmap=plt.cm.gray)
    plot3.axis('off')
    plot3.set_title("Initial overlap")

    plot4.imshow(x, cmap=plt.cm.gray)
    plot4.axis('off')
    plot4.set_title("Fixed image")

    plot5.imshow(z, cmap=plt.cm.gray)
    plot5.axis('off')
    plot5.set_title("Transformed moving image")

    plot6.imshow(x-z, cmap=plt.cm.gray)
    plot6.axis('off')
    plot6.set_title("Final overlap")

    fig.tight_layout()

    plt.show()


def load_jpgs_or_pngs(params):
    """
    Load jpegs or pngs
    :param params: contains the paths to the two images
    :return: the fixed and moving image
    """

    # load fixed image
    fixed_image = cv2.imread(params.fixed_path)

    if fixed_image.max() > 1:
        fixed_divisor = 255
    else:
        fixed_divisor = 1

    try:
        fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY).astype('float64') / fixed_divisor
    except:
        fixed_image = fixed_image.astype('float64') / fixed_divisor

    # laod moving image
    moving_image = cv2.imread(params.moving_path)

    if moving_image.max() > 1:
        moving_divisor = 255
    else:
        moving_divisor = 1

    try:
        moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY).astype('float64') / moving_divisor
    except:
        moving_image = moving_image.astype('float64') / moving_divisor

    fixed_image, moving_image = pad_images_to_same_size(fixed_image, moving_image)

    return fixed_image, moving_image


def load_nii(params):
    """
    Function to load niftii images
    :param params: the user parameters
    :return: fixed and moving images
    """
    fixed_image = nib.load(params.fixed_path)
    moving_image = nib.load(params.moving_path)

    fixed_image_data = np.asarray(fixed_image.get_data())
    moving_image_data = np.asarray(moving_image.get_data())

    fixed_image_data /= fixed_image_data.max()
    moving_image_data /= moving_image_data.max()

    return fixed_image_data, moving_image_data


def load_images(params):
    """
    Overachring image loading function that invokes the correct loading function based on the extension of the files
    :param params: The user input parameters
    :return: the fixed and moving images
    """

    if params.fixed_path.lower().endswith(".jpg") or params.fixed_path.lower().endswith(".png"):
        return load_jpgs_or_pngs(params)
    elif params.fixed_path.lower().endswith(".nii"):
        return load_nii(params)
    else:
        raise NotImplementedError("Only loading jpg/png and .nii is implemented at the moment")


def pad_images_to_same_size(im1, im2):
    """
    Function to pad two images so that they have the same size
    :param im1: image 1
    :param im2: image 2
    :return: im1, im2 (one is padded)
    """

    if len(im1.shape) != 2 or len(im2.shape) != 2:
        raise ValueError("Images must be 2D")

    if im1.shape == im2.shape:
        return im1, im2

    # x padding
    if im1.shape[0] < im2.shape[0]:
        smaller = im1
        smaller_id = "im1"
        bigger = im2
    else:
        smaller = im2
        smaller_id = "im2"
        bigger = im1

    dx = abs(im1.shape[0] - im2.shape[0])
    y = smaller.shape[1]
    zeros = np.zeros((dx, y))

    smaller = np.concatenate((smaller, zeros), 0)

    if smaller_id == "im1":
        im1 = smaller
        im2 = bigger
    else:
        im2 = smaller
        im1 = bigger

    if im1.shape == im2.shape:
        return im1, im2

    # y padding
    if im1.shape[1] < im2.shape[1]:
        smaller = im1
        smaller_id = "im1"
        bigger = im2
    else:
        smaller = im2
        smaller_id = "im2"
        bigger = im1

    dy = abs(im1.shape[1] - im2.shape[1])
    x = smaller.shape[0]
    zeros = np.zeros((x, dy))

    smaller = np.concatenate((smaller, zeros), 1)

    if smaller_id == "im1":
        im1 = smaller
        im2 = bigger
    else:
        im2 = smaller
        im1 = bigger

    return im1, im2


def create_two_image_overlay(image, overlay, alpha=0.4, cmap="plasma"):
    """
    Takes in two 2D numpy arrays and overlays the second with a colour map and transparency on the first
    :param image: the image to overlay on
    :param overlay: the overlay image
    :param alpha: alpha value for the overlay
    :param cmap: the colour map to use for the overlay
    :return: the combined image
    """

    assert len(image.shape) == 2 and len(overlay.shape) == 2, "Image(s) must be 2D"
    assert image.shape == overlay.shape, "Images must be the same size"

    # create RGB images from image and variance
    image1_color = np.dstack((image, image, image))

    cm = plt.get_cmap(cmap)
    overlay_color = cm(overlay)
    overlay_color = overlay_color[:, :, :3]

    # convert to HSV
    image1_hsv = color.rgb2hsv(image1_color)
    overlay_hsv = color.rgb2hsv(overlay_color)

    # combine
    image1_hsv[..., 0] = overlay_hsv[..., 0]
    image1_hsv[..., 1] = overlay_hsv[..., 1] * alpha

    combined = color.hsv2rgb(image1_hsv)

    return combined


def create_gird(image_shape, min_resolution: int):
    """
    Creates a grid on an image with a minimum amount of points in x and y min_resolution
    :param image_shape: Shape of the image on which the grid will be calculated
    :param min_resolution: minimal resolution in x and y
    :return: grid, points
    """

    assert image_shape[0] == image_shape[1], "Image must be square"
    assert min_resolution > 0, "Minimal resolution must be greater than 0"
    assert image_shape[0] > min_resolution, "Image must be larger than minimal resolution"

    # create points on x and y
    min_step = int(image_shape[0] / min_resolution)

    x_points = np.arange(0, image_shape[0], min_step)
    y_points = np.arange(0, image_shape[0], min_step)

    # create grid out of x_points and y_points
    grid = np.meshgrid(x_points, y_points)

    # extract all points from the grid (we use ones because we use homogeneous coordinates)
    points = np.ones((len(x_points)**2, 3))

    counter = 0

    for i in range(len(x_points)):
        for j in range(len(y_points)):
            points[counter, 0:2] = grid[0][i][j], grid[1][i][j]
            counter += 1

    return grid, points


def mark_points_in_image(image, points):
    """
    Mark points as pixels in an image
    :param image: The image to be marked
    :param points: points to be marked
    :param size: size in pixels of the points
    :return: image with marked points
    """

    assert len(image.shape) == 2, "Image must be 2D"
    assert len(points.shape) == 3, "Points must be 2D"
    assert points.shape[2] >= 2, "Points must have 2 columns"

    # create a copy of the image
    image_copy = image.copy()

    # mark points
    for point in points:
        image_copy[int(point[0][0]), int(point[0][1])] = 1

    return image_copy


def calculate_distance_between_points(point_array1, point_array2):
    """
    Calculate the distance between each point pair from the two arrays
    :param point_array1: First array of shape (n, 1, 2)
    :param point_array2: Second array of shape (n, 1, 2)
    :return: Array of points and distances of shape (n, 1)
    """
    assert len(point_array1.shape) == 3 and len(point_array2.shape) == 3, "Point arrays must be fake 3D"

    # squeeze arrays
    point_array1 = np.squeeze(point_array1)
    point_array2 = np.squeeze(point_array2)

    # calculate the distances between all point pairs from point_array1 and point_array2
    distances = np.sqrt(np.sum(np.square(point_array1 - point_array2), axis=1))

    return distances

