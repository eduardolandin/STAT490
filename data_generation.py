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


def sinusoidal(input_vector, weights, amp, ang_freq, x_off, y_off):
    """
    :param input_vector: a numpy vector in [0, 1]^d
    :param weights:
    :param amp:
    :param ang_freq:
    :param x_off:
    :param y_off:
    :return: a sinusoidal function of the sum of the input vector
    """
    input_sum = np.sum(np.multiply(input_vector, weights))
    return (amp * np.sin((ang_freq * input_sum) - x_off)) + y_off


def generate_regressor_mat(dimension, num_obs):
    """
    Generate a regressor matrix with "dimension" regressors variables and
    "num_obs" observed vectors. Each of matrix's rows is a vector in the
    "dimension"-dimensional hyper-cube.

    :param dimension: an int, the dimension of the vectors in the matrix
    :param num_obs: an int, the number of observed vectors
    :return: matrix of regressor variables
    """

    reg_mat = np.zeros((num_obs, dimension))
    for obs in range(num_obs):
        reg_mat[obs, :] = np.random.uniform(size=dimension)
    return reg_mat


def generate_data(input_matrix, func, sd_noise):
    """
    Generate observations by passing the columns of "input_matrix" as inputs to "func".

    :param input_matrix: a matrix where the rows are the i.i.d random predictor vectors in the unit hypercube
    :param func: the function that will be used to generate the observed quantities
    :param sd_noise: a float, the SD of the noise term that will be added to the observations
    :return: matrix of observed y values
    """

    num_observations = input_matrix.shape[0]
    observations = np.random.normal(size=(num_observations, 1), scale=sd_noise)  # generate N(0, 0.5^2) noise
    for i in range(num_observations):
        observations[i] += func(input_matrix[i, :])  # add noise to the true observed values
    return observations
