import numpy as np

from cost_functions import cost_function
import scipy.optimize
import pybobyqa


def optimise(optimiser, deformation_type, initial_transform, fixed_image, moving_image, similarity_metric,
             patch_size=None):
    """
    Optimise a cost function for registration with a given optimiser. The cost function is chosen automatically based on
    the amount of parameters in the initial_transform
    :param optimiser: scipy or bobyqa
    :param deformation_type: the type of deformtion to use
    :param initial_transform: initial transformation (rigid: 3 params; affine: 5 params; perspective: 9 params)
    :param fixed_image: The fixed image in the registration
    :param moving_image: The moving image in the registration
    :param similarity_metric: Any of the available similarity metrics
    :param patch_size: only relevant for LC2
    :return: resulting parameters
    """

    if optimiser.lower() == "scipy":
        parameters = scipy.optimize.fmin(cost_function, initial_transform, maxiter=1,
                                         args=(fixed_image, moving_image, deformation_type, similarity_metric, patch_size))
    elif optimiser.lower() == "bobyqa":
        bobyqa_result = pybobyqa.solve(cost_function, initial_transform.flatten(),
                                       args=(fixed_image, moving_image, deformation_type, similarity_metric, patch_size),
                                       print_progress=False)
        parameters = bobyqa_result.x
    else:
        raise NotImplementedError("Only scipy and bobyqa optimisers are implemented")

    return parameters
