# could be used to show how the similarity metrics behaves with respect to x-y translation (also shows capture range)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

# create x-y grid
size = 50
x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
dst = np.sqrt(x * x + y * y)

# Calculating Gaussian array
sigma = 1
muu = 0.000
gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))

# Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, gauss, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
