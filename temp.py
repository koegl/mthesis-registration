import helpers.visualisations as visualisations
from helpers.volumes import mark_patch_borders
import nibabel as nib
import numpy as np


fixed = np.load("/Users/fryderykkogl/Data/patches/offset_volumes/49.npy")
offset = np.load("/Users/fryderykkogl/Data/patches/offset_volumes/50_49.npy")
# _ = visualisations.display_volume_slice(skin, 0)

fixed = np.swapaxes(fixed, 0, 2)
offset = np.swapaxes(offset, 0, 2)

# load patch centres
patch_centres = np.load("/Users/fryderykkogl/Data/patches/val_npy/centres.npy")
patch_centres = list(patch_centres)
patch_centres = [list(x) for x in patch_centres]

for centre in patch_centres:

    temp = centre.copy()

    centre[0] = temp[2]
    centre[2] = temp[0]

    fixed = mark_patch_borders(fixed, centre, 1.0, 16)
    offset = mark_patch_borders(offset, centre, 1.0, 16)

_ = visualisations.display_two_volume_slices(np.stack((fixed, offset), 0))

print(5)
