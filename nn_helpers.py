from collections import OrderedDict
import numpy as np
import torch
from torch import nn


def network_parameters(beta, t, K, num_obs, c_inv):
    """
    Calculate network parameters based on Theorem 1 of Schmidt-Hieber (2020).

    :param beta: vector of length q + 1
    :param t: vector of length q + 1
    :param K: constant
    :param num_obs: number of observations
    :param c_inv: a constant that regulates the width of the neural network
    :return:
        - min_F - minimum possible value of F
        - min_layers - minimum number of layers
        - min_nodes - minimum number of nodes per layer
        - s - number of non-zero network parameters
    """

    # calculate smoothness indices
    q = beta.shape[0] - 1
    beta_and_ones = np.column_stack((beta, np.ones(q + 1)))
    beta_and_ones = beta_and_ones.min(axis=1)
    smoothness = np.zeros(q + 1)
    for i in range(q + 1):
        beta_interest = beta[i:]
        smoothness[i] = np.prod(beta_interest)
    denominator = (2 * smoothness) + t
    numerator = -2 * smoothness
    smoothness = np.divide(numerator, denominator)

    # calculate rate
    possible_rates = num_obs * np.ones(q + 1)
    possible_rates = np.power(possible_rates, smoothness)
    rate = possible_rates.max()

    # calculate min_F
    min_F = max(K, 1)

    # calculate min_layers
    t_and_beta = 4 * np.column_stack((t, beta))
    t_and_beta = t_and_beta.max(axis=1)
    min_layers = np.log2(t_and_beta).sum()
    min_layers = min_layers * np.log2(num_obs)
    min_layers = np.int64(np.ceil(min_layers))

    # calculate min_nodes
    min_nodes = num_obs * rate
    min_nodes = np.int64(np.ceil(min_nodes / c_inv))

    # calculate s
    s = num_obs * rate * np.log(num_obs)
    s = np.int64(np.ceil(s))

    return min_F, min_layers, min_nodes, s


def create_network_graph(num_layers, num_nodes, num_inputs, num_outputs, last_bias=False):
    """
    Creates a neural network graph with "num_layers" hidden layers each of which
    has "num_nodes" neurons and uses a ReLU activation function.

    - nn_layers:
    :param num_layers: the number of hidden layers in the network
    :param num_nodes: the number of nodes in each hidden layer of the network
    :param num_inputs: the number of network inputs
    :param num_outputs: the number of network outputs
    :param last_bias: should the output layer have a bias? Defaults to false
    :return: an OrderedDict describing the network graph
    """

    nn_layers = OrderedDict()
    for i in range(num_layers):
        linear_name = "linear" + str(i)
        if i == 0:
            nn_layers[linear_name] = nn.Linear(num_inputs, num_nodes)
        else:
            nn_layers[linear_name] = nn.Linear(num_nodes, num_nodes)
        torch.nn.init.xavier_uniform_(nn_layers[linear_name].weight, gain=1)
        relu_name = "relu" + str(i + 1)
        nn_layers[relu_name] = nn.ReLU()
    last_layer = "linear" + str(num_layers)
    nn_layers[last_layer] = nn.Linear(num_nodes, num_outputs, bias=last_bias)
    torch.nn.init.xavier_uniform_(nn_layers[last_layer].weight, gain=1)
    return nn_layers


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model.forward(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n")