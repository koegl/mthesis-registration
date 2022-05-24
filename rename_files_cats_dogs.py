import os
import PIL

# rename datset files to be numbered

path1 = "/Users/fryderykkogl/Data/ViT_training_data/data_overfit/temp/cat/1_cat.0.jpg"

p1 = path1.split("/")
p2 = p1[-1]
p3 = p2.split(".")
p4 = p3[0]

label = path1.split('/')[-1].split('.')[0]

directory = "/Users/fryderykkogl/Data/ViT_training_data/renamed_data/train"
# list all files in directory
files = os.listdir(directory)
files = [os.path.join(directory, file) for file in files if file.endswith(".jpg")]
files.sort()

cats = files[0:12500]
dogs = files[12500:]

i = 0
idx = 0

while i < len(dogs):

    new_name = dogs[i].split("/")[-1]
    new_name = str(idx).zfill(5) + "_" + new_name
    new_name = os.path.join(directory, new_name)
    os.rename(dogs[i], new_name)

    idx += 1

    new_name = cats[i].split("/")[-1]
    new_name = str(idx).zfill(5) + "_" + new_name
    new_name = os.path.join(directory, new_name)
    os.rename(cats[i], new_name)

    i += 1
    idx += 1


print(5)