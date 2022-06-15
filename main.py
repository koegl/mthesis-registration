import numpy
import imageio
import elasticdeform
import matplotlib.pyplot as plt
import numpy as np

from utils import create_deformation_grid

x = np.ones((10, 10), dtype=float)
x[::2] = 0
x[:, ::2] = 1 - x[:, ::2]
x = x.repeat(10, axis=0).repeat(10, axis=1)
plt.imshow(x, cmap='gray')
plt.title('Original')

deformation = create_deformation_grid([[0, -16], [0, 0], [0, 0], [0, -16],
                                       [0, 0],   [0, 0], [0, 0], [0, 0],
                                       [0, 16],  [0, 0], [0, 0], [0, 16]],
                                      [3, 4], dim=2)

Y = elasticdeform.deform_grid(X=x, displacement=deformation, order=0)
plt.figure()
plt.imshow(Y)

print(4)

