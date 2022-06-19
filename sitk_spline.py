# https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb

import SimpleITK as sitk
import numpy as np
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt


def display_displacement_scaling_effect(s, original_x_mat, original_y_mat, tx, original_control_point_displacements):
    if tx.GetDimension() != 2:
        raise ValueError("display_displacement_scaling_effect only works in 2D")

    plt.scatter(
        original_x_mat,
        original_y_mat,
        marker="o",
        color="blue",
        label="original points",
    )
    pointsX = []
    pointsY = []
    tx.SetParameters(s * original_control_point_displacements)

    for index, value in np.ndenumerate(original_x_mat):
        px, py = tx.TransformPoint((value, original_y_mat[index]))
        pointsX.append(px)
        pointsY.append(py)

    plt.scatter(pointsX, pointsY, marker="^", color="red", label="transformed points")
    plt.legend(loc=(0.25, 1.01))
    plt.xlim((-2.5, 2.5))
    plt.ylim((-2.5, 2.5))


# Create the transformation (when working with images it is easier to use the BSplineTransformInitializer function
# or its object oriented counterpart BSplineTransformInitializerFilter).
dimension = 2
spline_order = 3
direction_matrix_row_major = [1.0, 0.0, 0.0, 1.0]  # identity, mesh is axis aligned
origin = [-1.0, -1.0]
domain_physical_dimensions = [2, 2]
mesh_size = [4, 3]

bspline = sitk.BSplineTransform(dimension, spline_order)
bspline.SetTransformDomainOrigin(origin)
bspline.SetTransformDomainDirection(direction_matrix_row_major)
bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
bspline.SetTransformDomainMeshSize(mesh_size)

# Random displacement of the control points, specifying the x and y
# displacements separately allows us to play with these parameters,
# just multiply one of them with zero to see the effect.
x_displacement = np.random.random(len(bspline.GetParameters()) // 2)*10
y_displacement = np.random.random(len(bspline.GetParameters()) // 2)*10
original_control_point_displacements = np.concatenate([x_displacement, y_displacement])
bspline.SetParameters(original_control_point_displacements)

# Apply the BSpline transformation to a grid of points
# starting the point set exactly at the origin of the BSpline mesh is problematic as
# these points are considered outside the transformation's domain,
# remove epsilon below and see what happens.
numSamplesX = 10
numSamplesY = 20

coordsX = np.linspace(
    origin[0] + np.finfo(float).eps,
    origin[0] + domain_physical_dimensions[0],
    numSamplesX,
)
coordsY = np.linspace(
    origin[1] + np.finfo(float).eps,
    origin[1] + domain_physical_dimensions[1],
    numSamplesY,
)
XX, YY = np.meshgrid(coordsX, coordsY)

interact(
    display_displacement_scaling_effect,
    s=(-1.5, 1.5),
    original_x_mat=fixed(XX),
    original_y_mat=fixed(YY),
    tx=fixed(bspline),
    original_control_point_displacements=fixed(original_control_point_displacements),
)
