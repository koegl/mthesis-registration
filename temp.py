import numpy as np

import helpers.visualisations as visualisations


volume_fixed = np.load("/Users/fryderykkogl/Data/patches/offset_volumes/49.npy")
volume_offset = np.load("/Users/fryderykkogl/Data/patches/offset_volumes/50_49.npy")

# pad volume_offset with 16 pixels on the left side of x
volume_offset = np.pad(volume_offset, ((12, 0), (0, 0), (0, 0)), constant_values=0)
volume_offset = volume_offset.astype(np.float16)[:-12, :, :]

np.save("/Users/fryderykkogl/Data/patches/offset_volumes/50_49_pad_x12.npy", volume_offset)

# combined = np.stack((volume_fixed, volume_offset), 0)
#
# visualisations.display_two_volume_slices(combined)

print(5)
