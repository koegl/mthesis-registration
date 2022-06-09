import os
import glob
import tqdm

import nibabel as nib

x = nib.load("/Users/fryderykkogl/Dropbox (Partners HealthCare)/share/IXI_sedghi_subset_2upscaled/test/1.nii.gz").get_fdata()
print(x.shape)
print(x.dtype)

# folder = "/Users/fryderykkogl/Data/temp"
#
# files = glob.glob(os.path.join(folder, "*.nii*"))
# files.sort()
#
# index = 41
#
# for file in tqdm.tqdm(files):
#     command = f"/Applications/Convert3DGUI.app/Contents/bin/c3d {file} -resample 200% -o {file}"
#     os.system(command)

    # file_name_1 = f"{index}"
    # file_name_2 = f"{index+1}_{index}"
    #
    # # # rename file
    # new_file_path_1 = os.path.join(folder, file_name_1 + ".nii.gz")
    # new_file_path_2 = os.path.join(folder, file_name_2 + ".nii.gz")
    #
    # os.rename(files[2*i], new_file_path_1)
    # os.rename(files[2*i+1], new_file_path_2)
    #
    # index += 2

