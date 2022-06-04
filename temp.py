from logic.patcher import Patcher


patcher = Patcher("/Users/fryderykkogl/Data/patches/test_data/20211129_craini_golby/resliced",
                  "/Users/fryderykkogl/Data/patches/test_data/20211129_craini_golby/resliced",
                  "nii.gz", 10, False)

patcher.create_and_save_all_patches_and_labels()

print(5)

# import nibabel as nib
#
# from utils import display_volume_slice
#
# volume_fixed = nib.load("/Users/fryderykkogl/Data/patches/test_data/20211129_craini_golby/2.nii.gz")
# volume_fixed = volume_fixed.get_fdata()
# volume_moving = nib.load("/Users/fryderykkogl/Data/patches/test_data/20211129_craini_golby/1_2.nii.gz")
# volume_moving = volume_moving.get_fdata()
#
# _ = display_volume_slice(volume_moving, 'Slicer resampled volumes')
