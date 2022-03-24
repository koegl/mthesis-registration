import matplotlib.pyplot as plt
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


def plot_images(x, y):
    """
    Display two images and their difference
    :param x: first image
    :param y: second image
    """

    # Creating a figure with subplots of a certain size.
    fig, (plot1, plot2, plot3) = plt.subplots(1, 3, figsize=(10, 3))

    # Display the two images.
    plot1.imshow(x, cmap=plt.cm.gray)
    plot1.axis('off')
    plot2.imshow(y, cmap=plt.cm.gray)
    plot2.axis('off')

    # Computing the difference of the two images and display it.
    diff = x - y
    plot3.imshow(diff, cmap=plt.cm.gray)
    plot3.axis('off')

    plt.show()
