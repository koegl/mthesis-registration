import numpy as np
import SimpleITK as sitk
import PIL.Image as Image
import matplotlib.pyplot as plt

from helpers.volumes import get_image, create_checkerboard
from helpers.visualisations import display_volume_slice, get_cmap
from logic.deformer import Deformer

dim = 2

assert dim in [2, 3], "Dimension must be 2 or 3"

deformer = Deformer()

if dim == 2:
    volume_shape = (60, 60)
    point = (30, 30)
else:
    volume_shape = (60, 60, 60)
    point = (30, 30, 30)

volume = create_checkerboard(dim, volume_shape)

# set center pixel to 0 and make grid
if dim == 2:
    # volume[28:32, 28:32] = 0.5
    grid = np.zeros((2, 4, 4))
    grid_x = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    grid[0, :, :] = grid_x

else:
    # volume[28:32, 28:32, 28:32] = 0.5
    grid = np.zeros((3, 7, 7, 7))
    grid[0, 2:5, 2:5, 2:5] = 15
    # grid[1, 2:5, 2:5, 2:5] = 15
    # grid[2, 2:5, 2:5, 2:5] = 15

# create deformation field
deformation = deformer.generate_bspline_deformation(grid, volume_shape)

# transform volume
transformed_volume = deformer.transform_volume(volume, deformation)

point_transformed = deformation.TransformPoint(point)

point_transformed = np.asarray(volume_shape) - point_transformed

field = deformer.generate_deformation_field(deformation, volume_shape)

if dim == 2:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(transformed_volume, cmap='gray')
    axs[0].set_title('Transformed volume')
    # subsample deformation field
    sub_fac = 4
    field_subsampled = field[::sub_fac, ::sub_fac, :]
    x, y = np.meshgrid(np.linspace(volume_shape[0], 0, volume_shape[0]//sub_fac), np.linspace(volume_shape[1], 0, volume_shape[1]//sub_fac))

    # calculate length of deformation vectors - color
    color = np.sqrt(field_subsampled[:, :, 0]**2 + field_subsampled[:, :, 1]**2)

    axs[1].quiver(x, y, field_subsampled[:, :, 0] * -1, field_subsampled[:, :, 1], color, cmap=get_cmap(['cyan', 'red']))
    axs[1].set_title('Deformation field')

else:
    sub_fac = 20
    field_subsampled = field[::sub_fac, ::sub_fac, ::sub_fac, :]
    x, y, z = np.meshgrid(np.linspace(volume_shape[0], 0, volume_shape[0]//sub_fac),
                          np.linspace(volume_shape[1], 0, volume_shape[1]//sub_fac),
                          np.linspace(volume_shape[2], 0, volume_shape[2]//sub_fac))
    color = np.sqrt(field_subsampled[:, :, :, 0] ** 2 + field_subsampled[:, :, :, 1] ** 2 + field_subsampled[:, :, :, 2] ** 2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.quiver(x, y, z, field_subsampled[:, :, :, 0], field_subsampled[:, :, :, 1], field_subsampled[:, :, :, 2], color)  # , cmap=get_cmap(['cyan', 'red']))
    plt.show()

    display_volume_slice(transformed_volume, 0)

print(5)
