import numpy as np
import matplotlib.pyplot as plt
from numba import njit, vectorize, prange

def gradient_magnitude(img):
    """
    Computes the gradient magnitude of an image.
    :param img:  2D image
    :return: gradient_magnitude
    """
    grad_y, grad_x = np.gradient(img)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

@njit
def least_squares_solution(A, b):
    """
    Find the least-squares solution for a linear system of equations Ax = b.
    :param A: Matrix A with equations
    :param b: Vector b with targets
    :return: x
    """
    x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
    return x

@njit
def combine_MR_channels(us, mr, mr_gm, ids):
    """
    Reshape the images and find the least-squares solution between the US image and the MR channels.
    :param us: US image
    :param mr: MR image
    :param mr_gm: Gradient magnitude of MR image
    :param ids: Indices of all non-zero elements
    :return: M, c, U
    """
    M = np.concatenate((mr.reshape(-1, 1), mr_gm.reshape(-1, 1), np.zeros_like(mr).reshape(-1, 1)), 1)[ids,:]
    U = us.reshape(-1, 1)[ids,:]
    c = least_squares_solution(M, U)
    return M, c, U

@njit
def LC2_metric(us, mr, mr_gm):
    """
    Computes the LC2 similarity metric for the entire image. 
    :param us: US image
    :param mr: MR image
    :param mr_gm: Gradient magnitude of MR image
    :return: LC2, weight
    """    
    # initialize variables
    LC2 = 0.0
    weight = 0.0
    
    # find all non-zero elements in the ultrasound image
    ids = np.flatnonzero(us)

    # at least two non-zero elements are required for calculating the variance 
    if len(ids) > 0:
        
        # calculate the variance of all non-zero elements in the ultrasound image
        U_var = np.var(us.flatten()[ids])
        
        # check if the variance is non-zero to prevent division by zero errors
        if U_var > 1e-12:
    
            # make sure the arrays are contiguous
            us = np.ascontiguousarray(us)
            mr = np.ascontiguousarray(mr)
            mr_gm = np.ascontiguousarray(mr_gm)
    
            # flatten the images and find the weights for combining the MR channels
            M, c, U = combine_MR_channels(us, mr, mr_gm, np.arange(us.size))
            
            # calculate the LC2
            LC2 = 1 - (np.sum((U-np.dot(M, c))**2) / (len(ids)*U_var))
            weight = np.sqrt(U_var)
    
    return LC2, weight

@vectorize
def clip_below(value, threshold):
    """
    Clips value if below threshold.
    :value: numeric value
    :threshold: numeric threshold
    :returns: value (after clipping)
    """
    if value < threshold:
        value = threshold
    return value

@njit(parallel=True)
def LC2_similarity_patch(us, mr, mr_gm, patch_size):
    """
    Computes the patch-wise LC2 similarity metric for the entire image. 
    :param us: US image
    :param mr: MR image
    :param mr_gm: Gradient magnitude of MR image
    :patch_size: size of patch
    :return: LC2_similarity, LC2_map, weight_map
    """
    if patch_size%2 == 0 or patch_size < 1:
        raise ValueError('Invalid patch size')
    
    # define patch-related variable
    offset = patch_size//2

    # define image size related parameters
    max_rows, max_cols = us.shape

    # set all intensity values in the US image below zero to zero 
    us = clip_below(us, 0.0) 

    # allocate memory to store results
    LC2_map = np.zeros(us.shape)
    weight_map = np.zeros(us.shape)

    # loop through all pixels
    for col in prange(max_cols):
        for row in range(max_rows):

            # extract patches from us and mr+grad
            us_patch = us[
                max(0, row-offset):min(max_rows, row+offset+1),
                max(0, col-offset):min(max_cols, col+offset+1)
            ]

            mr_patch = mr[
                max(0, row-offset):min(max_rows, row+offset+1),
                max(0, col-offset):min(max_cols, col+offset+1)
            ]
            
            mr_gm_patch = mr_gm[
                max(0, row-offset):min(max_rows, row+offset+1),
                max(0, col-offset):min(max_cols, col+offset+1)
            ]

            LC2_map[row, col], weight_map[row, col] = LC2_metric(us_patch, mr_patch, mr_gm_patch)
    
    if np.sum(weight_map) == 0:
        return 0, LC2_map, weight_map
    else:
        return np.sum(LC2_map*weight_map) / np.sum(weight_map), LC2_map, weight_map