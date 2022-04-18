import data_generation as data_gen
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from functools import partial
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam


class RegFunc:
    """
    A class that represents a regression function and its properties

    :param func: the underlying regression function to use
    :param beta: a numpy array, the beta holder coefficients of the function
    :param t: a numpy array,
    :param K: a numpy array,
    """

    def __init__(self, func, beta, t, K):
        self.func = func
        self.beta = beta
        self.t = t
        self.K = K

    def eval(self, input_vector):
        return self.func(input_vector)


def gen_train_data(dim, num_obs, reg_func, sd_noise):
    """
    :param dim: an integer, the dimension of the regression problem
    :param num_obs: an integer, the number of observations to generate for the training set
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param sd_noise: the SD of the noise term added to the observations
    :param verbose: a boolean, enables print statements
    """

    # Generate a matrix of random regressor vectors and corresponding observations
    input_reg = data_gen.generate_regressor_mat(dim, num_obs)
    obs = data_gen.generate_data(input_reg, reg_func.eval, sd_noise)
    return input_reg, obs


def gen_test_data(dim, num_test, reg_func):
    """
    :param dim: an integer, the dimension of the regression problem
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param num_test: an integer, the size of the training set
    :return: A dataloader for the train data set and a dataloader for the test set
    """
    # Generate test data
    test_reg = data_gen.generate_regressor_mat(dim, num_test)
    # Generate true test data y values
    true_obs = np.zeros((num_test, 1))
    for i in range(num_test):
        true_obs[i] += reg_func.eval(test_reg[i, :])
    # Convert test data to torch tensor and Dataset
    return test_reg, true_obs


def keras_model_test_train(x_train, y_train):
    num_obs = y_train.shape[0]
    max_while = 3
    msemse = 50 * np.ones(max_while)
    iter_num = 0

    activ_func = relu

    # increasing Glorot initialization
    wgt1 = RandomUniform(minval=0., maxval=1., seed=None)
    wgt2 = RandomUniform(minval=-1., maxval=0., seed=None)
    wgt3 = RandomUniform(minval=-np.sqrt(6.) / np.sqrt(10.), maxval=0., seed=None)
    wgt4 = RandomUniform(minval=0., maxval=np.sqrt(6.) / np.sqrt(10.), seed=None)

    # Adam optimizer
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00021)

    # earlystop if MSE does not decrease
    earlystop = keras.callbacks.EarlyStopping(monitor='loss', patience=400, min_delta=0.00000001, verbose=1,
                                              mode='auto')
    callbacks = [earlystop]

    # DNN
    model = Sequential()
    while iter_num < max_while:
        print("iter_num: " + str(iter_num))

        model = Sequential()
        model.add(Dense(5, input_dim=1, kernel_initializer=wgt1, bias_initializer=wgt2, activation=activ_func))
        model.add(Dense(5, kernel_initializer=wgt4, bias_initializer=wgt3, activation=activ_func))
        model.add(Dense(5, kernel_initializer=wgt4, bias_initializer=wgt3, activation=activ_func))
        model.add(Dense(1, kernel_initializer=wgt1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.fit(x_train, y_train, epochs=1500, callbacks=callbacks, verbose=0)
        predicted = model.predict(x_train)
        msemse[iter_num] = sum((y_train - predicted) ** 2)

        print("train mse: " + str(msemse[iter_num] / num_obs))
        if msemse[iter_num] < 10:  # <0.000002 to reproduce simulations from paper
            iter_num = max_while
        iter_num = iter_num + 1
        print("----------------------------------------- END OF ITER -----------------------------------------")
    train_mse = min(msemse) / num_obs
    print("Train MSE DNN: " + str(train_mse))
    return model


def main_obs(dim, obs_array, reps, num_test, reg_func, sd_noise, verbose):
    """
    Runs the NN vs. the world experiments at different number of observations.
    Plots the results.

    :param dim: an int, the dimension of the regression problemx
    :param obs_array: an array containing the number of observations to use in the training set in each iteration
    :param reps: an int, how many reps per each number of observation
    :param num_test: an int, the size of the testing set
    :param reg_func: a RegFunc object, the underlying regression function
    :param sd_noise: a float, the SD of the noise term that is added to the training observations
    :param verbose: a boolean, controls print statements
    :return: none
    """

    # initialize MSE tracking arrays
    train_mse_nn = np.zeros(obs_array.size)
    test_mse_nn = np.zeros(obs_array.size)
    # train_mse_lr = np.zeros(len(obs_array))
    test_mse_lr = np.zeros(obs_array.size)
    # train_mse_ag = np.zeros(len(obs_array))
    test_mse_ag = np.zeros(obs_array.size)

    # Generate test data
    x_test, y_test = gen_test_data(dim, num_test, reg_func)
    pd_test = pd.DataFrame(x_test)
    if verbose:
        print("x_test:------------------------")
        print("  Shape of x_test: " + str(x_test.shape))
        print("  Avg of x_test: " + str(np.mean(x_test)))
        print(x_test.T)
        print("y_test:------------------------")
        print("  Shape of y_test: " + str(y_test.shape))
        print("  Avg of y_test: " + str(np.mean(y_test)))
        print(y_test.T)

    for index in range(len(obs_array)):
        num_obs = obs_array[index]
        print("num_obs: " + str(num_obs) + "-------------------------------------------------------------------------")
        for rep in range(reps):
            print("rep: " + str(rep) + "-----------------------------------------------")
            # Generate train data
            x_train, y_train = gen_train_data(dim, num_obs, reg_func, sd_noise)
            if verbose:
                print("x_train: ")
                print("  Shape of x_train: " + str(x_train.shape))
                # print("  input_reg: " + str(input_reg))
                print("  Avg of x_train: " + str(np.mean(x_train)))
                print("y_train: ")
                print("  Shape of y_train: " + str(y_train.shape))
                # print("  obs: " + str(obs))
                print("  Avg of y_train: " + str(np.mean(y_train)))

            pd_train = pd.DataFrame(np.concatenate((x_train, y_train), axis=1))

            # NN
            print("NN------------------------------------------------------------------")
            nn_model = keras_model_test_train(x_train, y_train)
            nn_predicted = nn_model.predict(x_test)
            if verbose:
                print("nn_predicted: " + str(nn_predicted.T))
                plt.scatter(x_test, y_test, color='b')
                plt.scatter(x_test, nn_predicted, color='r')
                plt.ylim((-0.1, 1.1))
                plt.title('${DNN}$')
                plt.show()
            test_mse_nn[index] += (np.sum(np.power(y_test - nn_predicted, 2)) / num_test)

            # Linear regression
            print("Linear Regression------------------------------------------------------------------")
            reg = LinearRegression().fit(x_train, y_train)
            reg_predicted = reg.predict(x_test)
            test_mse_lr[index] += (np.sum(np.power(y_test - reg_predicted, 2)) / num_test)
            if verbose:
                print("reg_predidcted: " + str(reg_predicted.T))
                plt.scatter(x_test, y_test, color='b')
                plt.scatter(x_test, reg_predicted, color='r')
                plt.ylim((-0.1, 1.1))
                plt.title('${Linear Regression}$')
                plt.show()

            # Autogluon
            path = "agModels-predictClass"
            train_data = TabularDataset(pd_train)
            predictor = TabularPredictor(label=train_data.columns[len(train_data.columns) - 1], path=path).fit(train_data)
            ag_predicted = predictor.predict(pd_test, as_pandas=False)
            test_mse_ag[index] += (np.sum(np.power(y_test - ag_predicted, 2)) / num_test)
            if verbose:
                print("ag_pred: " + str(ag_predicted.T))
                plt.scatter(x_test, y_test, color='b')
                plt.scatter(x_test, reg_predicted, color='r')
                plt.ylim((-0.1, 1.1))
                plt.title('${AutoGluon}$')
                plt.show()

        train_mse_nn[index] = train_mse_nn[index] / reps
        test_mse_nn[index] = test_mse_nn[index] / reps
        test_mse_lr[index] = test_mse_lr[index] / reps
        test_mse_ag[index] = test_mse_ag[index] / reps
        if verbose:
            print("train_mse_nn (index = " + str(index) + "): " + str(train_mse_nn[index]))
            print("test_mse_nn (index = " + str(index) + "): " + str(test_mse_nn[index]))
            print("test_mse_lr (index = " + str(index) + "): " + str(test_mse_lr[index]))
            print("test_mse_ag (index = " + str(index) + "): " + str(test_mse_ag[index]))

    fig, ax = plt.subplots()
    l1, = ax.plot(obs_array, test_mse_nn, 'b')
    l2, = ax.plot(obs_array, test_mse_lr, 'r')
    l3, = ax.plot(obs_array, test_mse_ag, 'k')
    ax.set(xlabel='Number of Training Observations', ylabel='Test Set MSE', title='Comparison of Regression Performance (# Noise SD: ' + str(sd_noise) + ', # Testing Obs: ' + str(num_test) + ')')
    ax.legend([l1, l2, l3], ['NN loss', 'Regression loss', "AG Loss"])
    ax.grid()
    plt.show()


def main_noise(dim, num_obs, reps, num_test, reg_func, sd_noise_arr, verbose):
    """
    Runs the NN vs. the world experiments at different SDs for the noise term in the model.
    Plots the results.

    :param dim: an int, the dimension of the regression problemx
    :param obs: an int, the number of observations to use in the training set
    :param reps: an int, how many reps per each number of observation
    :param num_test: an int, the size of the testing set
    :param reg_func: a RegFunc object, the underlying regression function
    :param sd_noise_arr: an array containing the SD of the noise term that is added to the training observations in each iter
    :param verbose: a boolean, controls print statements
    :return: none
    """

    # initialize MSE tracking arrays
    train_mse_nn = np.zeros(num_obs)
    test_mse_nn = np.zeros(num_obs)
    # train_mse_lr = np.zeros(len(obs_array))
    test_mse_lr = np.zeros(num_obs)
    # train_mse_ag = np.zeros(len(obs_array))
    test_mse_ag = np.zeros(num_obs)

    # Generate test data
    x_test, y_test = gen_test_data(dim, num_test, reg_func)
    pd_test = pd.DataFrame(x_test)
    if verbose:
        print("x_test:------------------------")
        print("  Shape of x_test: " + str(x_test.shape))
        print("  Avg of x_test: " + str(np.mean(x_test)))
        print(x_test.T)
        print("y_test:------------------------")
        print("  Shape of y_test: " + str(y_test.shape))
        print("  Avg of y_test: " + str(np.mean(y_test)))
        print(y_test.T)

    for index in range(len(sd_noise_arr)):
        sd_noise = sd_noise_arr[index]
        print("num_obs: " + str(num_obs) + "-------------------------------------------------------------------------")
        for rep in range(reps):
            print("rep: " + str(rep) + "-----------------------------------------------")
            # Generate train data
            x_train, y_train = gen_train_data(dim, num_obs, reg_func, sd_noise)
            if verbose:
                print("x_train: ")
                print("  Shape of x_train: " + str(x_train.shape))
                # print("  input_reg: " + str(input_reg))
                print("  Avg of x_train: " + str(np.mean(x_train)))
                print("y_train: ")
                print("  Shape of y_train: " + str(y_train.shape))
                # print("  obs: " + str(obs))
                print("  Avg of y_train: " + str(np.mean(y_train)))

            pd_train = pd.DataFrame(np.concatenate((x_train, y_train), axis=1))

            # NN
            print("NN------------------------------------------------------------------")
            nn_model = keras_model_test_train(x_train, y_train)
            nn_predicted = nn_model.predict(x_test)
            if verbose:
                print("nn_predicted: " + str(nn_predicted.T))
                plt.scatter(x_test, y_test, color='b')
                plt.scatter(x_test, nn_predicted, color='r')
                plt.ylim((-0.1, 1.1))
                plt.title('${DNN}$')
                plt.show()
            test_mse_nn[index] += (np.sum(np.power(y_test - nn_predicted, 2)) / num_test)

            # Linear regression
            print("Linear Regression------------------------------------------------------------------")
            reg = LinearRegression().fit(x_train, y_train)
            reg_predicted = reg.predict(x_test)
            test_mse_lr[index] += (np.sum(np.power(y_test - reg_predicted, 2)) / num_test)
            if verbose:
                print("reg_predicted: " + str(reg_predicted.T))
                plt.scatter(x_test, y_test, color='b')
                plt.scatter(x_test, reg_predicted, color='r')
                plt.ylim((-0.1, 1.1))
                plt.title('${Linear Regression}$')
                plt.show()

            # Autogluon
            path = "agModels-predictClass"
            train_data = TabularDataset(pd_train)
            predictor = TabularPredictor(label=train_data.columns[len(train_data.columns) - 1], path=path).fit(
                train_data)
            ag_predicted = predictor.predict(pd_test, as_pandas=False)
            test_mse_ag[index] += (np.sum(np.power(y_test - ag_predicted, 2)) / num_test)
            if verbose:
                print("ag_predicted: " + str(ag_predicted.T))
                plt.scatter(x_test, y_test, color='b')
                plt.scatter(x_test, reg_predicted, color='r')
                plt.ylim((-0.1, 1.1))
                plt.title('${AutoGluon}$')
                plt.show()

        train_mse_nn[index] = train_mse_nn[index] / reps
        test_mse_nn[index] = test_mse_nn[index] / reps
        test_mse_lr[index] = test_mse_lr[index] / reps
        test_mse_ag[index] = test_mse_ag[index] / reps
        if verbose:
            print("train_mse_nn (index = " + str(index) + "): " + str(train_mse_nn[index]))
            print("test_mse_nn (index = " + str(index) + "): " + str(test_mse_nn[index]))
            print("test_mse_lr (index = " + str(index) + "): " + str(test_mse_lr[index]))
            print("test_mse_ag (index = " + str(index) + "): " + str(test_mse_ag[index]))

    fig, ax = plt.subplots()
    l1, = ax.plot(sd_noise_arr, test_mse_nn, 'b')
    l2, = ax.plot(sd_noise_arr, test_mse_lr, 'r')
    l3, = ax.plot(sd_noise_arr, test_mse_ag, 'k')
    ax.set(xlabel='Number of Training Observations', ylabel='Test Set MSE',
           title='Comparison of Regression Performance (# Training Obs: ' + str(num_obs) + ', # Testing Obs: ' + str(num_test) + ')')
    ax.legend([l1, l2, l3], ['NN loss', 'Regression loss', "AG Loss"])
    ax.grid()
    plt.show()


# partial_func = partial(data_gen.constant_func, constant=100)
# partial_func = partial(data_gen.linear_func, weights=np.ones(dim_prob), intercept=10)
# partial_func = partial(data_gen.linear_func, weights=np.full(dim_prob, [10, -20]), intercept=0)
# partial_func = partial(data_gen.x_square, weights=np.full(dim_prob, [2, -20]), scale=1, intercept=0)
# func = RegFunc(partial_func, beta=np.ones(dim_prob), t=np.ones(dim_prob), K=1)

num_obs_arr = np.arange(100, 500, step=50)
dim_prob = 1
partial_func = partial(data_gen.x_square, weights=np.ones(1), scale=1, intercept=0)
func = RegFunc(partial_func, beta=np.ones(1), t=np.ones(1), K=1)
main_obs(dim_prob, num_obs_arr, 2, 50, func, 0.5, True)


