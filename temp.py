import nibabel as nib
from time import perf_counter
from logic.dataloader import get_data_loader
from logic.patcher import Patcher
import numpy as np


patcher = Patcher()

ds_us = nib.load("/Users/fryderykkogl/Data/temp/data_nii/US1_resmapled.nii.gz")
volume = ds_us.get_fdata()

header_copy = ds_us.header.copy()
affine_copy = ds_us.affine.copy()

loader = get_data_loader(batch_size=1)

x = perf_counter()
patch, label = next(iter(loader))
print(perf_counter() - x)

patch = patch.squeeze().numpy()
label = label.squeeze().numpy()
offset = patcher.label_to_offset_dict[str(label)]


nib.save(nib.Nifti1Image(patch[:, :, :, 0], affine=affine_copy, header=header_copy),
         f"/Users/fryderykkogl/Data/temp/us1_patch_centre.nii")
nib.save(nib.Nifti1Image(patch[:, :, :, 1], affine=affine_copy, header=header_copy),
         f"/Users/fryderykkogl/Data/temp/us1_patch_offset_{offset}.nii")
