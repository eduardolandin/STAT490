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


def gen_train_data(dim, num_obs, reg_func, verbose):
    """
    :param dim: an integer, the dimension of the regression problem
    :param num_obs: an integer, the number of observations to generate for the training set
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param verbose: a boolean, enables print statements
    """

    # Generate a matrix of random regressor vectors and corresponding observations
    input_reg = data_gen.generate_regressor_mat(dim, num_obs)
    obs = data_gen.generate_data(input_reg, reg_func.eval)
    if verbose:
        print("input_reg: ")
        print("  Shape of input_reg: " + str(input_reg.size()))
        # print("  input_reg: " + str(input_reg))
        print("  Avg of input_reg: " + str(np.mean(input_reg)))
        print("obs: ")
        print("  Shape of obs: " + str(obs.size()))
        # print("  obs: " + str(obs))
        print("  Avg of observations: " + str(np.mean(obs)))
    return input_reg, obs


def gen_test_data(dim, num_test, reg_func, verbose):
    """
    :param dim: an integer, the dimension of the regression problem
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param num_test: an integer, the size of the training set
    :param verbose: a boolean, enables print statements
    :return: A dataloader for the train data set and a dataloader for the test set
    """
    # Generate test data
    test_reg = data_gen.generate_regressor_mat(dim, num_test)
    # Generate true test data y values
    true_obs = np.zeros((num_test, 1))
    for i in range(num_test):
        true_obs[i] += reg_func.eval(test_reg[:, i])
    # Convert test data to torch tensor and Dataset
    if verbose:
        print("test_reg: ")
        print("  Shape of test_reg: " + str(test_reg.size()))
        # print("  test_reg: " + str(train_dataset.input_reg))
        print("  Avg of test_reg: " + str(test_reg))
        print("true_obs: ")
        print("  Shape of true_obs: " + str(true_obs.size()))
        # print("  true_obs: " + str(train_dataset.obs))
        print("  Avg of observations: " + str(np.mean(true_obs)))
    return test_reg, true_obs


def keras_model_test_train(x_train, y_train, x_test, y_test):
    num_obs = x_test.size[0]
    num_test = y_test.size[0]
    max_while = 1
    msemse = 50 * np.ones(max_while)
    test_msemse = 50 * np.ones(max_while)
    iter_num = 0
    while iter_num < max_while:
        print("iter_num: " + str(iter_num))
        z = relu
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
        model.add(Dense(5, input_dim=1, kernel_initializer=wgt1, bias_initializer=wgt2, activation=z))
        model.add(Dense(5, kernel_initializer=wgt4, bias_initializer=wgt3, activation=z))
        model.add(Dense(5, kernel_initializer=wgt4, bias_initializer=wgt3, activation=z))
        model.add(Dense(1, kernel_initializer=wgt1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.fit(x_train, y_train, epochs=1500, callbacks=callbacks, verbose=0)
        predicted = model.predict(x_train)
        msemse[iter_num] = sum((y_train - predicted) ** 2)

        predicted_test = model.predict(x_test)
        test_msemse[iter_num] = sum((y_test - predicted_test) ** 2)
        print("x_test:------------------------")
        print(x_test)
        print("y_test:------------------------")
        print(y_test)
        print("predicted_test:------------------------")
        print(predicted_test)
        print("-----------------------------------------")
        print("train mse: " + str(msemse[iter_num] / num_obs))
        print("test mse: " + str(test_msemse[iter_num] / num_test))
        plt.scatter(x_test, y_test, color='b')
        plt.scatter(x_test, predicted_test, color='r')
        plt.ylim((-0.1, 1.1))
        plt.title('${DNN}$')
        plt.show()
        print("----------------------------------------- END OF ITER -----------------------------------------")
        if msemse[iter_num] < 50:  # <0.000002 to reproduce simulations from paper
            iter_num = max_while
        iter_num = iter_num + 1
    train_mse = min(msemse) / num_obs
    print("Train MSE DNN: " + str(train_mse))

    test_mse = min(test_msemse) / num_test
    print("Test MSE DNN: " + str(test_mse))
    return train_mse, test_mse


def main(dim, obs_array, num_test, reg_func, verbose):
    # initialize MSE tracking arrays
    train_mse_nn = np.zeros(len(obs_array))
    test_mse_nn = np.zeros(num_test)
    train_mse_lr = np.zeros(len(obs_array))
    test_mse_lr = np.zeros(num_test)
    train_mse_ag = np.zeros(len(obs_array))
    test_mse_ag = np.zeros(num_test)

    for index in range(len(obs_array)):
        # Generate data
        num_obs = obs_array[index]
        x_train, y_train = gen_train_data(dim, num_obs, reg_func, verbose)
        x_test, y_test = gen_test_data(dim, num_test, reg_func, verbose)
        pd_train = pd.DataFrame(np.concatenate((x_train.T, y_train), axis=1))
        pd_test = pd.DataFrame(x_test.T)

        # NN
        train_mse_nn[index], test_mse_nn[index] = keras_model_test_train(x_train, y_train, x_test, y_test)
        if verbose:
            print("train_mse_nn (index = " + str(index) + "): " + str(train_mse_nn[index]))
            print("test_mse_nn (index = " + str(index) + "): " + str(test_mse_nn[index]))

        # Linear regression
        reg = LinearRegression().fit(x_train.T, y_train)
        reg_pred = reg.predict(x_test.T)
        test_mse_lr[index] = np.sum(np.power(y_test - reg_pred, 2)) / num_test
        if verbose:
            print("reg_pred: " + str(reg_pred))
            print("test_mse_lr (index = " + str(index) + "): " + str(test_mse_lr[index]))

        # Autogluon
        path = "agModels-predictClass"
        train_data = TabularDataset(pd_train)
        predictor = TabularPredictor(label=train_data.columns[len(train_data.columns) - 1], path=path).fit(train_data)
        ag_pred = predictor.predict(pd_test, as_pandas=False)
        test_mse_ag[index] = np.sum(np.power(y_test - ag_pred, 2)) / num_test
        if verbose:
            print("ag_pred: " + str(ag_pred))
            print("test_mse_ag (index = " + str(index) + "): " + str(test_mse_ag[index]))


num_obs_arr = np.arange(300, 3000, step=300)
dim_prob = 2
# partial_func = partial(data_gen.constant_func, constant=100)
# partial_func = partial(data_gen.linear_func, weights=np.ones(dim_prob), intercept=10)
# partial_func = partial(data_gen.linear_func, weights=np.full(dim_prob, [10, -20]), intercept=0)
#partial_func = partial(data_gen.x_square, weights=np.full(dim_prob, [2, -20]), scale=1, intercept=0)
#func = RegFunc(partial_func, beta=np.ones(dim_prob), t=np.ones(dim_prob), K=1)
#main(num_obs_arr, reps=2, dim=dim_prob, reg_func=func, c_inv=10, batch_size=5, num_epochs=10, num_test=20, learning_rate=0.001, weight_decay=0.0001, verbose=True)


partial_func = partial(data_gen.x_square, weights=np.ones(1), scale=1, intercept=0)
func = RegFunc(partial_func, beta=np.ones(1), t=np.ones(1), K=1)