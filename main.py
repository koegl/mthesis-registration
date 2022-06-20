import numpy as np
import SimpleITK as sitk
import PIL.Image as Image
import matplotlib.pyplot as plt

from utils import get_image, generate_bspline_deformation, generate_deformation_field, transform_image


imshape = (50, 50)

image = get_image("/Users/fryderykkogl/Downloads/cat.jpg", imshape)

# set center pixel to 0
image[24:26, 24:26] = 0
plt.figure()
plt.imshow(image, cmap='gray')

# create deformation vectors
grid = np.zeros((2, 5, 5))
grid[1, 1:4, 1:4] = 20

# create deformation field
deformation = generate_bspline_deformation(grid, imshape)

# transform image
out_img = transform_image(image, deformation)

# display image
plt.figure()
plt.imshow(out_img, cmap='gray')

px, py = deformation.TransformPoint((25, 25))

py = imshape[1] - py
px = imshape[0] - px

field = generate_deformation_field(deformation, imshape)

plt.figure()
plt.imshow(field[:, :, 0])
plt.figure()
plt.imshow(field[:, :, 1])

print(5)