import matplotlib.pyplot as plt
import numpy as np
import progressbar


def set_up_progressbar(max_value):
    """
    Set up progress bar.
    Example usage:
        bar = set_up_progressbar(200)
        bar.start()
        for i in range(200):
            bar.update(i)
    :param max_value: maximum amount of iterations
    :return: the bar object
    """

    widgets = [' [', progressbar.Timer(format='elapsed time: %(elapsed)s'), '] ',
               progressbar.Bar('*'),
               ' (', progressbar.ETA(), ') ',
               ]
    return progressbar.ProgressBar(max_value=max_value, widgets=widgets)


def plot(array, cmap=None):
    """
    Plots a numpy array
    :param array: The array to be displayed
    :param cmap: The cmap for the plot
    """
    plt.imshow(array, cmap=cmap)
    plt.show()


def plot_images(x, y, z):
    """
    Display two images and their difference
    :param x: first image
    :param y: second image
    :param z: third image
    """

    # Creating a figure with subplots of a certain size.
    fig, (plot1, plot2, plot3, plot4, plot5, plot6) = plt.subplots(2, 3, figsize=(10, 8))

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

    # Computing the difference of the two images and display it.
    plot4.imshow(x, cmap=plt.cm.gray)
    plot4.axis('off')
    plot4.set_title("Fixed image")

    plot4.imshow(z, cmap=plt.cm.gray)
    plot4.axis('off')
    plot4.set_title("Transformed moving image")

    plot4.imshow(x-z, cmap=plt.cm.gray)
    plot4.axis('off')
    plot4.set_title("Final overlap")

    plt.show()


def pad_smaller_image(im1, im2):
    """
    Function to pad two images so that they have the same size
    :param im1: image 1
    :param im2: image 2
    :return: im1, im2 (one is padded)
    """

    smaller_id = ""

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



