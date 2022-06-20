import numpy as np
import SimpleITK as sitk
import PIL.Image as Image
import matplotlib.pyplot as plt

from helpers.volumes import get_image, generate_bspline_deformation, generate_deformation_field, transform_volume, create_checkerboard
from helpers.visualisations import  display_volume_slice


dim = 3

assert dim in [2, 3], "Dimension must be 2 or 3"

if dim == 2:
    volume_shape = (60, 60)
    point = (30, 30)
else:
    volume_shape = (60, 60, 60)
    point = (30, 30, 30)

volume = create_checkerboard(dim, volume_shape)

# set center pixel to 0 and make grid
if dim == 2:
    volume[28:32, 28:32] = 0.5
    grid = np.zeros((2, 5, 5))
    grid[0, 2, 2] = 20
else:
    volume[28:32, 28:32, 28:32] = 0.5
    grid = np.zeros((3, 5, 5, 5))
    grid[0, 2, 2, 2] = 20

# create deformation field
deformation = generate_bspline_deformation(grid, volume_shape)

# transform volume
transformed_volume = transform_volume(volume, deformation)

point_transformed = deformation.TransformPoint(point)

point_transformed = np.asarray(volume_shape) - point_transformed

field = generate_deformation_field(deformation, volume_shape)

if dim == 2:
    plt.imshow(transformed_volume, cmap='gray')
else:
    display_volume_slice(transformed_volume, 0)

print(5)