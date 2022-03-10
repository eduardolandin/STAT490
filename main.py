import data_generation as data_gen
import nn_helpers
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class RegFunc:
    """
    A class that represents a regression function and its properties

    :param func: the underlying regression function to use
    :param beta: the beta holder coefficients of the function
    :param t:
    :param K:
    """

    def __init__(self, func, beta, t, K):
        self.func = func
        self.beta = beta
        self.t = t
        self.K = K

    def eval(self, input_vector):
        return self.func(input_vector)


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


def main_helper(dim, num_obs, reg_func, c_inv, batch_size, num_test, learning_rate, weight_decay, verbose):
    """
    :param dim: the dimension of the regression problem
    :param num_obs: the number of observations to generate
    :param reg_func: a RegFunc object representing the underlying regression function and its properties
    :param c_inv: a constant that regulates the width of the network
    :param batch_size: number of samples in a batch
    :param num_test: size of testing set
    :param learning_rate: the learning rate of the model
    :param weight_decay: the weight decay parameter
    :param verbose:
    :return:
    """

    # Generate a matrix of random regressor vectors and corresponding observations
    input_reg = data_gen.generate_regressor_mat(dim, num_obs)
    obs = torch.tensor(data_gen.generate_data(input_reg, reg_func.eval), dtype=torch.float)
    input_reg = torch.tensor(input_reg.T, dtype=torch.float)
    train_dataset = GenDataset(input_reg, obs)
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

    # Generate test data
    test_reg = data_gen.generate_regressor_mat(dim, num_test)
    # Generate true test data y values
    true_obs = np.zeros((num_test, 1))
    for i in range(num_test):
        true_obs[i] += reg_func.eval(test_reg[:, i])
    # Convert test data to torch tensor and Dataset
    test_reg = torch.from_numpy(test_reg.T).float()
    true_obs = torch.from_numpy(true_obs).float()
    test_dataset = GenDataset(test_reg, true_obs)
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
    initial_params = torch.nn.utils.parameters_to_vector(model.parameters())

    # Loss function to evaluate model performance
    loss_func = nn.MSELoss()
    # Optimizer to update network parameters
    # optimizer_func = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_func = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model on the generated data
    model.train(True)
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        nn_helpers.train_loop(dataloader=train_dataloader, model=model, loss_fn=loss_func, optimizer=optimizer_func)
    print("Initial network parameters: " + str(initial_params))
    print("Final network params = initial params: " + str(initial_params == torch.nn.utils.parameters_to_vector(model.parameters())))
    print("Final network parameters: " + str(torch.nn.utils.parameters_to_vector(model.parameters())))

    # number of non-zero parameters:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: " + str(params))
    num_non_zero_params = torch.count_nonzero(torch.nn.utils.parameters_to_vector(model.parameters()))
    print("Number of non-zero parameters: " + str(num_non_zero_params))

    # Evaluate model performance on test data
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for index in range(test_reg.size(0)):
            pred = model.forward(test_reg[index, :])
            print("input: " + str(test_reg[index, :]) + "    |    " + "pred: " + str(pred))
            test_loss += loss_func(pred, true_obs).item()
    return test_loss / num_test


def main(obs, dim, reg_func, c_inv, batch_size, num_test, learning_rate, weight_decay, verbose):
    """
    Evaluating the model at different observations
    :param obs: an array of observations
    :param dim: the dimension of the regression problem
    :param reg_func: a RegFunc object representing the underlying regression function and its properties
    :param c_inv: a constant that regulates the width of the network
    :param batch_size: number of batches used in training
    :param num_test: the number of tests to run
    :param learning_rate: the learning rate
    :param weight_decay: the weight decay regularization parameter
    :param verbose:
    :return: none
    """
    tot_loss = np.zeros(obs.size)
    for index in range(obs.size):
        num_obs = obs[index]
        tot_loss[index] = main_helper(dim, num_obs, reg_func, c_inv, batch_size, num_test, learning_rate, weight_decay, verbose)
        print("------------------------------------------------------------------------------------------------------")
    # Plot the error across different sample sizes
    fig, ax = plt.subplots()
    ax.plot(obs, tot_loss, 'b')
    ax.set(xlabel='Number of Observations', ylabel='Total Loss', title='Convergence of NN function estimate')
    ax.grid()
    plt.show()


num_obs_arr = np.arange(500, 2000, step=200)
main(num_obs_arr, dim=1, batch_size=20, num_test=20, learning_rate=0.001, weight_decay=0, verbose=True)


