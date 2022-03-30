import numpy as np
from PIL import Image
import argparse
import pathlib
import os
import matplotlib.pyplot as plt
import cv2
import scipy.optimize

from utils import plot_images
from image_manipulation import transform_image


def ssd(x, y):
    ''' Returns the sum of the squared differences of the given matrices
    '''
    n, m = y.shape
    return (1/(n*m)) * np.sum(np.square(x - y))


def sad(x, y):
    ''' Returns the sum of the absolute differences of the given matrices
    '''
    n, m = y.shape
    return (1/(n*m)) * np.sum(np.abs(x - y))


def ncc(x, y):
    ''' Computes the normalized cross correlation between the given matrices
    '''
    n, m = y.shape

    # Demeaning the matrices.
    x -= np.mean(x)
    y -= np.mean(y)

    # This is the formula as simple as it is.
    # The computation of the standard deviation might not be computationally optimal,
    # but for this exercises we do not care.
    s = (1/(n*m)) * np.sum((x * y) / (x.std() * y.std()))
    return s


def mi(x, y):
    ''' Computes the mutual information between the given matrices
    '''

    # Compute joint histogram. You may want to use np.histogram2d for this.
    # histogram2d takes 1D vectors, ravel() reshape the matrix to a 1D vector.
    # Don not forget to normalize the histogram in order to get a joint distribution.
    jh = np.histogram2d(x.ravel(), y.ravel())[0]
    jh = jh + np.finfo(float).eps # add eps for stability

    # Normalize.
    sh = np.sum(jh)
    jh /= sh

    # You can get the individual distributions by marginalization of the joint distribution jh.
    # We have two random variables x and y whose joint distribution is known,
    # the marginal distribution of X is the probability distribution of X,
    # when the value of Y is not known. And vice versa.
    s1 = np.sum(jh, axis=0, keepdims=True)
    s2 = np.sum(jh, axis=1, keepdims=True)

    # Compute the MI.
    MI = np.sum(-s1 * np.log2(s1)) + np.sum(-s2 * np.log2(s2)) - np.sum(-jh * np.log2(jh))

    return MI


def cost_function(transform_params, fixed_image, moving_image, similarity):
    ''' Computes a similarity measure between the given images using the given similarity metric
    :param transform_params: 3 element array with values for rotation, displacement along x axis, dislplacement along y axis.
                             The moving_image will be transformed using these values before the computation of similarity.
    :param fixed_image: the reference image for registration
    :param moving_image: the image to register to the fixed_image
    :param similarity: a string naming the similarity metric to use. e.g. SSD, SAD, ...
    :return: the compute similarity
    '''

    # Transform the moving_image with the current parameters (We already have code for this)
    transformed_moving_img = transform_image(moving_image, transform_params)

    # Compute the similarity value using the given method.
    #
    if similarity == "SSD":
        s = ssd(transformed_moving_img, fixed_image)
    elif similarity == "SAD":
        s = sad(transformed_moving_img, fixed_image)
    elif similarity == "NCC":  # Since we want to maximize NCC, we can minimize its negative
        s = -ncc(transformed_moving_img, fixed_image)
    elif similarity == "MI":
        s = -mi(transformed_moving_img, fixed_image)  # Since we want to maximize MI, we can minimize its negative
    else:
        print("Wrong similarity measure given.")
        return -1
    return s


def main(params):
    # load images
    fixed_image = Image.open(params.fixed_path)
    fixed_image = np.asarray(fixed_image).astype('float64') / 255
    moving_image = Image.open(params.moving_path)
    moving_image = np.asarray(moving_image).astype('float64') / 255

    # Give some initial values to the transformation parameters
    x0 = [75, -15, -15]

    # Choose with similarity metric to use
    # similarity_metric = "SSD"
    # similarity_metric = "SAD"
    # similarity_metric = "NCC"
    similarity_metric = "MI"

    # Call fmin : https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.fmin.html
    # In short fmin needs a function (cost_function) and initial values for the parameters to minimize for (x0),
    # what is in args are additional parameters that are needed for our specific function.
    # fmin will return the values that after running the algorithm are minimazing the function
    result_params = scipy.optimize.fmin(cost_function, x0, args=(fixed_image, moving_image, similarity_metric))

    # Transform the moving images with the found parameters
    result_image = transform_image(moving_image, result_params)
    # Let's have a look at the result!
    plot_images(result_image, fixed_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    current_directory = pathlib.Path(__file__).parent.resolve()

    parser.add_argument("-fp", "--fixed_path", default=os.path.join(current_directory, "misc/ct_fixed.png"),
                        help="Path to the fixed image")
    parser.add_argument("-mp", "--moving_path", default=os.path.join(current_directory, "misc/ct_moving.png"),
                        help="Path to the moving image")

    args = parser.parse_args()

    main(args)
