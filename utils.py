import numpy as np
from itertools import product


# todo extend to 3-D
def create_grid_of_control_points(control_points_per_dimension, volume_size):
    """
    Function that creates a grid of control points and their coordinates
    :param control_points_per_dimension: list of integers of how many control points there are per dimension
    :param volume_size: volume size
    :return: grid, centers
    """
    assert isinstance(control_points_per_dimension, list), 'control_points_per_dimension must be a list of integers'
    assert len(control_points_per_dimension) == len(volume_size), "Bost must have the same amount of dimensions"

    # reverse control_points_per_dimension to get the correct order for the for loops
    # control_points_per_dimension = control_points_per_dimension[::-1]

    for p in control_points_per_dimension:
        assert p > 1, 'There must be more than one control point for every dimension'

    spatial_dimension_of_the_volume = [len(volume_size)]

    grid_shape = spatial_dimension_of_the_volume + control_points_per_dimension[::-1]

    grid = np.zeros(grid_shape)

    volume_dims = len(volume_size)

    coordinates = []

    for dim in range(volume_dims):

        distance = volume_size[dim] // (control_points_per_dimension[dim] - 1)

        points = []

        for i in range(control_points_per_dimension[dim]):
            points.append(distance * i)


        points[-1] = volume_size[dim]

        coordinates.append(points)

    all_coordinates = list(product(coordinates[0], coordinates[1]))

    return grid, all_coordinates
