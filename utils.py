import matplotlib.pyplot as plt


def plot(array, cmap=None):
    """
    Plots a numpy array
    :param array: The array to be displayed
    :param cmap: The cmap for the plot
    """
    plt.imshow(array, cmap=cmap)
    plt.show()
