import os.path

from lc2_similarity import lc2_similarity_patch

import numpy as np
from PIL import Image
import argparse
import pathlib
import time


def main(params):
    # load images
    mr = Image.open(params.mr_path)
    mr = np.asarray(mr).astype('float64') / 255
    us = Image.open(params.ultrasound_path)
    us = np.asarray(us).astype('float64') / 255

    start_time_parallel = time.time()
    similarity = lc2_similarity_patch(us, mr, params.patch_size)
    compute_time = time.time() - start_time_parallel

    print(f"\033[0;0m\n{similarity=}"
          f"\n{compute_time=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-up", "--ultrasound_path", default=os.path.join(current_directory, "misc/test_us.png"),
                        help="Path to the Ultrasound image")
    parser.add_argument("-mp", "--mr_path", default=os.path.join(current_directory, "misc/test_mr.png"),
                        help="Path to the MR image")
    parser.add_argument("-ps", "--patch_size", default=9, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
