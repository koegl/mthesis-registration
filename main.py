import os.path
import pathlib
import argparse

import numpy as np
from PIL import Image
from time import perf_counter

from lc2_similarity_2 import LC2_similarity_patch, gradient_magnitude

def main(params):
    # load images
    us = Image.open(params.ultrasound_path)
    us = np.asarray(us).astype('float64') / 255
    mr = Image.open(params.mr_path)
    mr = np.asarray(mr).astype('float64') / 255
    mr_gm = gradient_magnitude(mr)

    # first time executing is not representative because of JIT compilation of Numba function
    LC2_similarity_patch(us, mr, mr_gm, params.patch_size)

    # we therefore time the second function call
    start_time = perf_counter()
    similarity, _, _ = LC2_similarity_patch(us, mr, mr_gm, params.patch_size)
    end_time = perf_counter()
    
    print(f"Similarity = {similarity}, time = {end_time-start_time:0.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-up", "--ultrasound_path", default=os.path.join(current_directory, "misc/test_us.png"),
                        help="Path to the Ultrasound image")
    parser.add_argument("-mp", "--mr_path", default=os.path.join(current_directory, "misc/test_mr.png"),
                        help="Path to the MR image")
    parser.add_argument("-ps", "--patch_size", default=19, help="The patch size for calculating LC2")

    args = parser.parse_args()

    main(args)
