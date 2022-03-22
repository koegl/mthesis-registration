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
    mr = Image.open("/Users/fryderykkogl/Documents/university/master/thesis/code/mthesis-registration/misc/test_mr.png")
    mr = np.asarray(mr).astype('float64') / 255
    us = Image.open("/Users/fryderykkogl/Documents/university/master/thesis/code/mthesis-registration/misc/test_us.png")
    us = np.asarray(us).astype('float64') / 255

    # create a combined image of the MR and MR-gradient
    mr_and_grad = np.zeros((us.shape[0], us.shape[1], 2))
    mr_and_grad[:, :, 0] = mr
    mr_and_grad[:, :, 1] = np.absolute(ndimage.sobel(mr, axis=1))  # matlab gradient is a bit different

    similarity, measure, weight = lc2_similarity(us, mr_and_grad)

    print(f"{similarity=}\n{measure=},\n{weight=}")


if __name__ == "__main__":
    main()
