import os
import glob
import tqdm

folder = "/Users/fryderykkogl/Data/patches/mr_nii"

files = glob.glob(os.path.join(folder, "*.nii*"))
files.sort()
for file in tqdm.tqdm(files):
    command = f"/Applications/Convert3DGUI.app/Contents/bin/c3d {file} -resample 200% -o {file}"
    os.system(command)

print("Done up-scaling")
