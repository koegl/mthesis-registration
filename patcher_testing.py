import matplotlib.pyplot as plt
import nibabel as nib

from utils import *
from logic.patcher import Patcher
import numpy as np

# image_1 = create_radial_gradient(100, 100, 100)
# image_2 = create_radial_gradient(100, 100, 100)
ds_us = nib.load("/Users/fryderykkogl/Data/temp/US1_resmapled.nii.gz")
volume = ds_us.get_fdata()

header_copy = ds_us.header.copy()
affine_copy = ds_us.affine.copy()

patch_center = [300, 300, 100]
patch_size = 32
patch_offset = [4, 0, 0]  # allowed offset: 4, 8, 16
centres_per_dimension = 10

# patches = extract_overlapping_patches(volume, volume, patch_center, patch_size, patch_offset)
#
# nib.save(nib.Nifti1Image(patches[0], affine=affine_copy, header=header_copy),
#          f"/Users/fryderykkogl/Data/temp/us1_patch_centre_{str(center)}.nii")
# nib.save(nib.Nifti1Image(patches[1], affine=affine_copy, header=header_copy),
#          f"/Users/fryderykkogl/Data/temp/us1_patch_offset_{str(offset)}.nii")

patcher = Patcher()

centres = patcher.generate_list_of_patch_centres(centres_per_dimension=centres_per_dimension,
                                                 volume=volume, patch_size=patch_size)

packet = patcher.get_patch_and_label(volume, centres[0], patch_size, patch_offset)
print(centres[0])

nib.save(nib.Nifti1Image(packet["patch"][:, :, :, 0], affine=affine_copy, header=header_copy),
         f"/Users/fryderykkogl/Data/temp/us1_patch_centre_{str(centres[0])}.nii")
nib.save(nib.Nifti1Image(packet["patch"][:, :, :, 1], affine=affine_copy, header=header_copy),
         f"/Users/fryderykkogl/Data/temp/us1_patch_offset_{str(patch_offset)}.nii")
