import numpy
import imageio
import elasticdeform
import matplotlib.pyplot as plt
import numpy as np

from utils import create_deformation_grid, elastic_transform_scipy_2d

x = np.ones((10, 10), dtype=float)
x[::2] = 0
x[:, ::2] = 1 - x[:, ::2]
checkerboard = x.repeat(10, axis=0).repeat(10, axis=1)
plt.imshow(x, cmap='gray')
plt.title('Original')

checkerboard_deformed = elastic_transform_scipy_2d(checkerboard, alpha=40, sigma=2, random_state=None)


plt.figure()
plt.imshow(checkerboard_deformed, cmap='gray')

print(4)

