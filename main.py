from lc2_similarity import lc2_similarity

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


def plot(array):
    plt.imshow(array)
    plt.show()


def main():
    # load image
    phantom = Image.open("/Users/fryderykkogl/Documents/university/master/thesis/code/mthesis-registration/phantom.png")
    phantom = np.asarray(phantom).astype('float64')

    # add noise
    # gauss_noise = np.random.normal(0, 1, phantom.size)
    # gauss_noise = gauss_noise.reshape(phantom.shape[0], phantom.shape[1]) * 10
    # phantom_noise = cv2.add(phantom, gauss_noise)

    phantom2 = np.zeros((phantom.shape[0], phantom.shape[1], 2))
    phantom2[:, :, 0] = phantom
    phantom2[:, :, 1] = np.absolute(ndimage.sobel(phantom, axis=1))

    plot(1-phantom)
    plot(phantom2[:, :, 0])
    plot(phantom2[:, :, 1])

    similarity, measure, weight = lc2_similarity(1 - phantom, phantom2)

    print(f"{similarity=}\n{measure=},\n{weight=}")


if __name__ == "__main__":
    main()
