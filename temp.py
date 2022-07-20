import numpy as np
from sklearn.linear_model import LinearRegression

import helpers.utils as utils


primary_n = np.array([[40., 20., 20.],
                    [20., 40., 20.],
                    [20., 20., 40.],
                    [20., 20., 20.],
                    [30., 20., 20.]
                    ])
secondary_n = np.array([[65., 10., 30.],  # (20, -10, 10)
                      [40., 30., 30.],
                      [40., 10., 50.],
                      [40., 10., 30.],
                      [50., 10., 30.]
                      ])



primary = np.array([[40., 20., 20., 1],
                    [20., 40., 20., 1],
                    [20., 20., 40., 1],
                    [20., 20., 20., 1],
                    [30., 20., 20., 1]
                    ])

secondary = np.array([[65., 10., 30., 1],  # (20, -10, 10)
                      [40., 30., 30., 1],
                      [40., 10., 50., 1],
                      [40., 10., 30., 1],
                      [50., 10., 30., 1]
                      ])

np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.0f}'.format})


def least_squares_transform():

    # Pad the data with ones, so that our transformation can do translations too
    n = primary.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    # X = pad(primary)
    # Y = pad(secondary)
    X = primary
    Y = secondary

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    print("Target:")
    print(secondary)
    print("Result:")
    print(transform(primary))
    print("Max error:", np.abs(secondary - transform(primary)).max())

    A[np.abs(A) < 1e-10] = 0 # set really small values to zero
    print(A)


# least_squares_transform()

# A, res, rank, s = np.linalg.lstsq(primary, secondary)
# print(A)
#
#
#

# W = [1, 1, 1, 1, 1]
# # W = [1,1,1]
# W = np.sqrt(np.diag(W))
# prim_w = np.dot(W, primary)
# sec_w = np.dot(secondary.T, W)
# A_w, _, _, _ = np.linalg.lstsq(prim_w, sec_w.T)
# print(A_w.T)
# print("\n")


W = [1., 1.0, 1, 1.0, 1.0]
W = np.sqrt(np.diag(W))
prim_w = np.dot(W, primary)
sec_w = np.dot(secondary.T, W)
A_w, _, _, _ = np.linalg.lstsq(prim_w, sec_w.T)
print(A_w.T)
print("\n")


print(utils.calculate_affine_transform(primary_n, secondary_n, [0.01, 1.0, 1.0, 1.0, 1.0]))


X = primary
Y = secondary
W = [0.1, 0.9, 0.9, 0.9, 0.9]
W = np.sqrt(np.diag(W))
a = np.linalg.inv(np.matmul(X.T, np.matmul(W, X)))
c = np.matmul(X.T, np.matmul(W, Y))
beta = np.matmul(a, c)

print(beta.T)


# wls = LinearRegression()
# W = [1, 1, 1, 1]
# wls.fit(primary, secondary)  # , sample_weight=W)
# print(wls.coef_)

print(5)
