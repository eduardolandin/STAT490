import numpy as np


def constant_func(input_vector, constant):
    """
    :param input_vector: a numpy vector in [0, 1]^d
    :param constant: the constant to return
    :return: constant
    """
    return constant


def linear_func(input_vector, weights, intercept):
    """
    :param input_vector: a numpy vector in [0, 1]^d
    :param weights: a numpy vector of integers with length d
    :param intercept: a number
    :return: (w.T * input_vector) + intercept
    """
    return np.sum(np.multiply(input_vector, weights)) + intercept


def x_square(input_vector, weights, scale, intercept):
    """
    :param input_vector: a numpy vector in [0, 1]^d
    :param weights: a numpy vector of integers with length d
    :param scale: a number
    :param intercept: a number
    :return: (scale * sum(w.T * input_vector)^2) + intercept
    """
    linear_comb = np.sum(np.multiply(input_vector, weights))
    return np.power(linear_comb, 2)


def sinusoidal(input_vector, amp, ang_freq, x_off, y_off):
    """
    :param input_vector: a numpy vector in [0, 1]^d
    :param amp:
    :param ang_freq:
    :param x_off:
    :param y_off:
    :return: a sinusoidal function of the sum of the input vector
    """
    input_sum = np.sum(input_vector)
    return (amp * np.sin((ang_freq * input_sum) - x_off)) + y_off


def generate_regressor_mat(dimension, num_obs):
    """
    Generate a regressor matrix with "dimension" regressors variables and
    "num_obs" observed vectors. Each of matrix's columns is a vector in the
    "dimension"-dimensional hyper-cube.

    :param dimension: the dimension of the vectors in the matrix
    :param num_obs: the number of observed vectors
    :return: matrix of regressor variables
    """

    reg_mat = np.zeros((dimension, num_obs))
    for obs in range(num_obs):
        reg_mat[:, obs] = np.random.uniform(size=dimension)
    return reg_mat


def generate_data(input_matrix, func):
    """
    Generate observations by passing the columns of "input_matrix" as inputs to "func".

    :param input_matrix: a matrix where the columns are the realizations of i.i.d random predictor vectors in the unit
                         hypercube
    :param func: the function that will be used to generate the observed quantities
    :return: matrix of observed y values
    """

    num_observations = input_matrix.shape[1]
    noise = np.random.normal(size=(num_observations, 1), scale=0.5)  # generate N(0, 0.5^2) noise
    observations = noise
    for i in range(num_observations):
        observations[i] += func(input_matrix[:, i])  # add noise to the true observed values
    return observations
