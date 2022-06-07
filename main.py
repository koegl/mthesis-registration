import numpy
import imageio
import elasticdeform
import matplotlib.pyplot as plt
import numpy as np

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
x = x.repeat(10, axis=0).repeat(10, axis=1)
plt.imshow(x, cmap='gray')
plt.title('Original')

# shape of displacement matrix:
#   [target dimension, grid points dim 1, grid points dim 2, ...]
displacement = np.array([
                        [[0, -32, 0], [0, 0, 0], [0, 32, 0]],
                        [[0, 0, 0], [-32, 0, 32], [0, 0, 0]]
                        ])
Y = elasticdeform.deform_grid(X=x, displacement=displacement, order=0)
plt.figure()
plt.imshow(Y, cmap='gray')

print(4)


"""
[[top-left-inY, top-center-inY, top-right-inY], [center-left-inY, center-center-inY, center-right-inY], [down-left-inY, down-center-inY, down-right-inY]],
[[top-left-inX, top-center-inX, top-right-inX], [center-left-inX, center-center-inX, center-right-inX], [down-left-inX, down-center-inX, down-right-inX]]
"""