import matplotlib.pyplot as plt


def plot_images(x, y):

    '''Function to display two images and their difference on screen.
    :param x: first image to display
    :param y: second image to display
    :return: void
    '''

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
