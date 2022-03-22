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
