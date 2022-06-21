from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

def plot_3d_quiver(x, y, z, u, v, w):
    c = np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2 + np.abs(w) ** 2)
    c = (c.ravel() - c.min()) / c.ptp()
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    c = plt.cm.jet(c)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x, y, z, u, v, w, colors=c, length=0.1)
    plt.show()

field = np.random.rand(3, 3, 3, 3)

x, y, z = np.meshgrid(np.linspace(60, 0, 3),
                      np.linspace(60, 0, 3),
                      np.linspace(60, 0, 3))

color = np.sqrt(field[:, :, :, 0] ** 2 + field[:, :, :, 1] ** 2 + field[:, :, :, 2] ** 2)

# fig = plt.figure(figsize=(5, 5))
# ax = fig.gca(projection='3d')
# ax.quiver(x, y, z, field[:, :, :, 0], field[:, :, :, 1], field[:, :, :, 2], color)
# plt.show()

