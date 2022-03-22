from lc2_similarity import lc2_similarity

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle


def plot(array):
    plt.imshow(array)
    plt.show()


def main():
    # load images
    # phantom = Image.open("/Users/fryderykkogl/Documents/university/master/thesis/code/mthesis-registration/phantom.png")
    # phantom = np.asarray(phantom).astype('float64')
    # normal phantom
    with open("/Users/fryderykkogl/Downloads/similarity_LC2/p.pkl", 'rb') as f:
        p = pickle.load(f)
    # phantom with noise
    with open("/Users/fryderykkogl/Downloads/similarity_LC2/img1.pkl", 'rb') as f:
        img1 = pickle.load(f)

    img2 = np.zeros((p.shape[0], p.shape[1], 2))
    img2[:, :, 0] = p
    # phantom2[:, :, 1] = np.absolute(ndimage.sobel(p, axis=1))  # this gives similar results to matlab gradient (!=)

    # load gradient
    with open("/Users/fryderykkogl/Downloads/similarity_LC2/img2_second_layer.pkl", 'rb') as f:
        img2_second_layer = pickle.load(f)
    img2[:, :, 1] = img2_second_layer

    similarity, measure, weight = lc2_similarity(img1, img2)

    print(f"{similarity=}\n{measure=},\n{weight=}")


if __name__ == "__main__":
    main()
