import matplotlib.pyplot as plt
import numpy as np
import cv2
import nibabel as nib
from matplotlib.widgets import Slider
import functools


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


def plot_one_image(image, title=""):
    plt.imshow(image, cmap=plt.cm.gray)
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


def load_nifti_image(path):
    """
    Loads NIFTI Image
    """

    img = nib.load(path)
    canonical_img = nib.as_closest_canonical(img)

    return canonical_img

