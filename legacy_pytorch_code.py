import data_generation as data_gen
import nn_helpers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from functools import partial
from sklearn.linear_model import LinearRegression
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class NeuralNetwork(nn.Module):
    """
    A class that represents a feed-forward neural network

    :param nn_layers: the number of layers in the network
    """

    def __init__(self, nn_layers):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(nn_layers)

    def forward(self, x):
        """
        :param x: input vector
        :return: feeds x through the network
        """
        output = self.linear_relu_stack(x)
        return output


class GenDataset(Dataset):
    """
    A class that represents a dataset. Used to train the network.
    """

    def __init__(self, reg_mat, obs):
        self.reg_mat = reg_mat
        self.obs = obs

    def __len__(self):
        return self.obs.size(dim=0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.reg_mat[idx, :], self.obs[idx]


def gen_train_data(dim, num_obs, reg_func, batch_size, verbose):
    """
    :param dim: an integer, the dimension of the regression problem
    :param num_obs: an integer, the number of observations to generate for the training set
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param batch_size: an integer, how many samples per batch to load
    :param verbose: a boolean, enables print statements
    :return: A dataloader for the train data set and a dataloader for the test set
    """

    # Generate a matrix of random regressor vectors and corresponding observations
    input_reg = data_gen.generate_regressor_mat(dim, num_obs)
    obs = data_gen.generate_data(input_reg, reg_func.eval)
    tens_obs = torch.tensor(obs, dtype=torch.float)
    tens_input_reg = torch.tensor(input_reg.T, dtype=torch.float)
    train_dataset = GenDataset(tens_input_reg, tens_obs)
    if verbose:
        print("input_reg: ")
        print("  Shape of input_reg: " + str(train_dataset.reg_mat.size()))
        # print("  input_reg: " + str(train_dataset.input_reg))
        print("  Avg of input_reg: " + str(torch.mean(train_dataset.reg_mat)))
        print("obs: ")
        print("  Shape of obs: " + str(train_dataset.obs.size()))
        # print("  obs: " + str(train_dataset.obs))
        print("  Avg of observations: " + str(torch.mean(train_dataset.obs)))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, input_reg, obs


def gen_test_data(dim, reg_func, batch_size, num_test, verbose):
    """
    :param dim: an integer, the dimension of the regression problem
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param batch_size: an integer, how many samples per batch to load
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
    tens_test_reg = torch.from_numpy(test_reg.T).float()
    tens_true_obs = torch.from_numpy(true_obs).float()
    test_dataset = GenDataset(tens_test_reg, tens_true_obs)
    if verbose:
        print("test_reg: ")
        print("  Shape of test_reg: " + str(test_dataset.reg_mat.size()))
        # print("  test_reg: " + str(train_dataset.input_reg))
        print("  Avg of test_reg: " + str(torch.mean(test_dataset.reg_mat)))
        print("true_obs: ")
        print("  Shape of true_obs: " + str(test_dataset.obs.size()))
        # print("  true_obs: " + str(train_dataset.obs))
        print("  Avg of observations: " + str(torch.mean(test_dataset.obs)))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_dataloader, test_reg, true_obs


def create_network(dim, num_obs, reg_func, c_inv, verbose):
    """
    :param dim: an integer, the dimension of the regression problem
    :param num_obs: an integer, the number of observations to generate for the training set
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param c_inv: a float, regulates the width of the network
    :param verbose: boolean, should information be printed to console?
    :return:
    """
    # Calculate the network parameters
    min_F, min_layers, min_nodes, s = nn_helpers.network_parameters(reg_func.beta, reg_func.t, reg_func.K, num_obs, c_inv)
    if verbose:
        print("min_F: " + str(min_F))
        print("min_layers: " + str(min_layers))
        print("min_nodes: " + str(min_nodes))
        print("s: " + str(s))

    # Create the network architecture
    nn_layers = nn_helpers.create_network_graph(min_layers, min_nodes, dim, 1, False)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print("Using {} device".format(device))

    # Move neural network to device and print its structure
    model = NeuralNetwork(nn_layers).to(device)
    if verbose:
        print(model)
    return model


def train_test_net(model, train_dataloader, test_dataloader, num_epochs, learning_rate, weight_decay, verbose):
    """
    :param model: a NeuralNetwork object, the model used
    :param train_dataloader: a Dataloader, the training set
    :param test_dataloader: a Dataloader, the testing set
    :param num_epochs: an integer, the number of epochs to train for
    :param learning_rate: a float, the learning rate of the model
    :param weight_decay: a float, the weight decay parameter
    :param verbose: a boolean, enables print statements
    :return: the average test error
    """

    initial_params = torch.nn.utils.parameters_to_vector(model.parameters())
    if verbose:
        print("Initial network parameters: " + str(initial_params))

    # Loss function to evaluate model performance
    loss_func = nn.MSELoss()
    # Optimizer to update network parameters
    # optimizer_func = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_func = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model on the generated data
    model.train(True)
    for t in range(num_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        nn_helpers.train_loop(dataloader=train_dataloader, model=model, loss_fn=loss_func, optimizer=optimizer_func)
    if verbose:
        print("Final network params = initial params: " + str(initial_params == torch.nn.utils.parameters_to_vector(model.parameters())))
        print("Final network parameters: " + str(torch.nn.utils.parameters_to_vector(model.parameters())))

    # number of non-zero parameters:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    num_non_zero_params = torch.count_nonzero(torch.nn.utils.parameters_to_vector(model.parameters()))
    if verbose:
        print("Number of parameters: " + str(num_params))
        print("Number of non-zero parameters: " + str(num_non_zero_params))

    # Evaluate model performance on test data
    model.eval()
    test_loss = nn_helpers.test_loop(test_dataloader, model, loss_func)
    return test_loss


def main(obs, reps, dim, reg_func, c_inv, batch_size, num_epochs, num_test, learning_rate, weight_decay, verbose):
    """
    :param obs: an numpy array of observations
    :param reps: an integer, how many times to train/test at a given observation
    :param dim: an integer, the dimension of the regression problem
    :param reg_func: a RegFunc, the underlying regression function and its properties
    :param c_inv: a float, regulates the width of the network
    :param batch_size: an integer, how many samples per batch to load
    :param num_epochs: an integer, how many training epochs to use
    :param num_test: an integer, the size of the test set
    :param learning_rate: a float, the learning rate
    :param weight_decay: a float,  the weight decay regularization parameter
    :param verbose: boolean, enables print statements
    :return: none
    """

    np.random.seed(2022)
    tot_loss_nn = np.zeros(obs.size)
    tot_loss_reg = np.zeros(obs.size)
    tot_loss_ag = np.zeros(obs.size)
    for index in range(obs.size):
        num_obs = obs[index]
        model = create_network(dim, num_obs, reg_func, c_inv, verbose)
        for rep in range(reps):
            train_dataloader, x_train, y_train = gen_train_data(dim, num_obs, reg_func, batch_size, verbose)
            test_dataloader, x_test, y_test = gen_test_data(dim, reg_func, 1, num_test, verbose)
            pd_train = pd.DataFrame(np.concatenate((x_train.T, y_train), axis=1))
            pd_test = pd.DataFrame(x_test.T)

            # NN
            tot_loss_nn[index] += train_test_net(model, train_dataloader, test_dataloader, num_epochs, learning_rate, weight_decay, verbose)

            # regression
            reg = LinearRegression().fit(x_train.T, y_train)
            reg_pred = reg.predict(x_test.T)
            reg_loss = np.sum(np.power(y_test - reg_pred, 2)) / num_test
            if verbose:
                print("regression coefficients: " + str(reg.coef_))
                print("regression intercept: " + str(reg.intercept_))
                for j in range(len(reg_pred)):
                    print("true obs: " + str(np.around(y_test[j], 3)) + "    |    pred: " + str(np.around(reg_pred[j], 3)))
                print("reg_loss: " + str(reg_loss))
            tot_loss_reg[index] += reg_loss

            # autogluon
            path = "agModels-predictClass"
            train_data = TabularDataset(pd_train)
            predictor = TabularPredictor(label=train_data.columns[len(train_data.columns)-1], path=path).fit(train_data)
            y_pred = predictor.predict(pd_test, as_pandas=False)
            ag_loss = np.sum(np.power(y_test - y_pred, 2)) / num_test
            if verbose:
                for j in range(len(reg_pred)):
                    print("true obs: " + str(np.around(y_test[j], 3)) + "    |    pred: " + str(np.around(reg_pred[j], 3)))
                print("ag_loss: " + str(ag_loss))
            tot_loss_ag[index] += ag_loss

        tot_loss_nn[index] = tot_loss_nn[index] / reps
        tot_loss_reg[index] = tot_loss_reg[index] / reps
        tot_loss_ag[index] = tot_loss_ag[index] / reps
        if verbose:
            print("--------------------------------------------------------------------------------------------------")
    print("tot_loss_nn: " + str(tot_loss_nn))
    print("tot_loss_reg: " + str(tot_loss_reg))
    print("tot_loss_ag: " + str(tot_loss_ag))

    # Plot the error across different sample sizes
    fig, ax = plt.subplots()
    l1, = ax.plot(obs, tot_loss_nn, 'b')
    l2, = ax.plot(obs, tot_loss_reg, 'r')
    l3, = ax.plot(obs, tot_loss_ag, 'k')
    ax.set(xlabel='Number of Observations', ylabel='Total Loss', title='Convergence of NN function estimate')
    ax.legend([l1, l2, l3], ['NN loss', 'Regression loss', "AG Loss"])
    ax.grid()
    plt.show()
