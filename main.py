import numpy
import imageio
import elasticdeform
import matplotlib.pyplot as plt
import numpy as np

from utils import create_grid_of_control_points

# x = numpy.zeros((200, 300))
# x[::10, ::10] = 1
#
# # apply deformation with a random 3 x 3 grid
# x_deformed = elasticdeform.deform_random_grid(x, sigma=25, points=5)
# x_deformed = elasticdeform.deform_grid(x, np.asarray([[3], [3]]))
#
# ax1 = plt.subplot(1, 2, 1)
# ax1.imshow(x, cmap='gray')
# ax1.set_title('Original')
# ax1.axis('off')
#
# ax2 = plt.subplot(1, 2, 2)
# ax2.imshow(x_deformed, cmap='gray')
# ax2.set_title('Deformed')
# ax2.axis('off')

x = np.ones((10, 10), dtype=float)
x[::2] = 0
x[:, ::2] = 1 - x[:, ::2]
x = x.repeat(10, axis=0).repeat(5, axis=1)
# plt.imshow(x, cmap='gray')
# plt.title('Original')

# shape of displacement matrix:
#   [target dimension, grid points dim 1, grid points dim 2, ...]
displacement_old = np.array([
                        [[0, -32, 0], [0, 0, 0], [0, 32, 0], [0, 0, 0]],
                        [[0, 0, 0], [-32, 0, 32], [0, 0, 0], [0, 0, 0]]
                        ])
# shape: (2, 4, 3);
#   shape[0]: spatial dimensions of the input image -> 2D (top row are y coordinates)
#   shape[1]: number of rows (of control points)    -> four rows of control points
#   shape[2]: number of columns (of control points) -> three columns of control points

# first column of arrays are 2D vectors corresponding to control points on the top line
# second column of arrays are 2D vectors corresponding to control points on the top center line
# third column of arrays are 2D vectors corresponding to control points on the bottom center line
# fourth column of arrays are 2D vectors corresponding to control points on the bottom line

# let's say we have a 3x3 grid of control points, so we will have shape (2 (spatial dimensions),
#                                                                        3 (number of rows),
#                                                                        3 (number of columns)):
# p = = p = = p
# = = = = = = =
# = = = = = = =
# p = = p = = p
# = = = = = = =
# = = = = = = =
# p = = p = = p
# the center of the coordinate system is in the bottom right

displacement = np.zeros((2, 3, 3))
# let's say we want to move the center point diagonally upwards to the right, then:
displacement[0][1][1] = 32
displacement[1][1][1] = -32

shape = (100, 50)
grid, coordinates = create_grid_of_control_points([3, 4], shape)

print(grid.shape)
print(coordinates)

new_x = np.zeros((shape[0] + 2, shape[1] + 2))
new_x[1:-1, 1:-1] = x

for xy in coordinates:
    xy = (xy[0] + 1, xy[1] + 1)
    new_x[xy[0]-1:xy[0]+1, xy[1]-1:xy[1]+1] = 0.5

x = new_x[1:-1, 1:-1]

plt.imshow(x)
plt.title('Original')

Y = elasticdeform.deform_grid(X=x, displacement=displacement, order=0)
plt.figure()
plt.imshow(Y)

print(4)

"""
[[top-left-inY, top-center-inY, top-right-inY], [center-left-inY, center-center-inY, center-right-inY], [down-left-inY, down-center-inY, down-right-inY]],
[[top-left-inX, top-center-inX, top-right-inX], [center-left-inX, center-center-inX, center-right-inX], [down-left-inX, down-center-inX, down-right-inX]]
"""