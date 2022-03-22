import os.path

from lc2_similarity import lc2_similarity

import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import pathlib


def plot(array):
    plt.imshow(array)
    plt.show()


def main(params):
    # load images
    mr = Image.open(params.mr_path)
    mr = np.asarray(mr).astype('float64') / 255
    us = Image.open(params.ultrasound_path)
    us = np.asarray(us).astype('float64') / 255

    similarity, measure, weight = lc2_similarity(us, mr)

    print(f"{similarity=}\n{measure=},\n{weight=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-up", "--ultrasound_path", default=os.path.join(current_directory, "misc/test_us.png"),
                        help="Path to the Ultrasound image")
    parser.add_argument("-mp", "--mr_path", default=os.path.join(current_directory, "misc/test_mr.png"),
                        help="Path to the MR image")

    args = parser.parse_args()

    main(args)
