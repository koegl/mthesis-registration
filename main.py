import os.path
import pathlib
import argparse

import numpy as np
from PIL import Image
from time import perf_counter

from lc2_similarity import lc2_similarity_patch

def main(params):
    # load images
    mr = Image.open(params.mr_path)
    mr = np.asarray(mr).astype('float64') / 255
    us = Image.open(params.ultrasound_path)
    us = np.asarray(us).astype('float64') / 255

    start_time = perf_counter()
    similarity = lc2_similarity_patch(us, mr, params.patch_size)
    end_time = perf_counter()
    
    print(f"Similarity = {similarity}, time = {end_time-start_time:0.2f}s")


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
