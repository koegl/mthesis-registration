import matplotlib.pyplot as plt
import nibabel as nib

from utils import *
import numpy as np

# image_1 = create_radial_gradient(100, 100, 100)
# image_2 = create_radial_gradient(100, 100, 100)
ds_us1 = nib.load("/Users/fryderykkogl/Data/temp/conversion/US1_in_US3_pad.nii.gz")
image_1 = ds_us1.get_fdata()
ds_us2 = nib.load("/Users/fryderykkogl/Data/temp/conversion/US2_in_US3_pad.nii.gz")
ds_us2 = ds_us1
image_2 = ds_us2.get_fdata()

header_copy = ds_us1.header.copy()
affine_copy = ds_us1.affine.copy()
pixel_spacing = ds_us1.header["pixdim"][1:4]

center = [650, 615, 100]
size = 200
offset = [50, 0, 0]

patches = extract_overlapping_patches(image_1, image_2, center, size, pixel_spacing, offset)

save_np_array_as_nifti(patches[0], "/Users/fryderykkogl/Data/temp/patch_fixed_us1.nii", affine=affine_copy, header=header_copy)
save_np_array_as_nifti(patches[1], "/Users/fryderykkogl/Data/temp/patch_offset_us1.nii", affine=affine_copy, header=header_copy)
# save_np_array_as_nifti(image_1, "/Users/fryderykkogl/Desktop/temp/image.nii")

# us_size = [800, 600, 150]
#
# offsets = generate_list_of_patch_offsets([-8, -4, -2, 0, 2, 4, 8])
#
# centres = generate_list_of_patch_centres(centres_per_dimension=40, volume_size=us_size, patch_size=17)
#
# print(f"Patches per image: {len(centres) * len(offsets)}")
# image_pairs_in_dataset = 100 * 2
#
# print(f"Patches in dataset: {image_pairs_in_dataset * len(centres) * len(offsets)}")

print("\n\nDONE")

