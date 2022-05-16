import matplotlib.pyplot as plt

from utils import *
import numpy as np

# image_1 = create_radial_gradient(100, 100, 100)
# image_2 = create_radial_gradient(100, 100, 100)
#
# center = [50, 50, 50]
# size = 20
# offset = [5, 0, 0]
#
# patches = extract_overlapping_patches(image_1, image_2, center, size, offset)
#
# save_np_array_as_nifti(patches[0], "/Users/fryderykkogl/Desktop/temp/patch_fixed.nii")
# save_np_array_as_nifti(patches[1], "/Users/fryderykkogl/Desktop/temp/patch_offset.nii")
# save_np_array_as_nifti(image_1, "/Users/fryderykkogl/Desktop/temp/image.nii")

us_size = [800, 600, 150]

offsets = generate_list_of_patch_offsets([-8, -4, -2, 0, 2, 4, 8])

centres = generate_list_of_patch_centres(centres_per_dimension=40, volume_size=us_size, patch_size=17)

print(f"Patches per image: {len(centres) * len(offsets)}")
image_pairs_in_dataset = 100 * 2

print(f"Patches in dataset: {image_pairs_in_dataset * len(centres) * len(offsets)}")

# for center in centres:
#     print(center)

