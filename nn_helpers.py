from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor, Lambda, Compose


def network_parameters(beta, t, K, num_obs, C_inv):
    """
    Calculate network parameters based on Theorem 1 of Schmidt-Hieber (2020).

    :param beta: vector of length q + 1
    :param t: vector of length q + 1
    :param K: constant
    :param num_obs: number of observations
    :param C_inv
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
    # print("min nodes: " + str(min_nodes))
    min_nodes = np.int64(np.ceil(min_nodes / C_inv))
    # print("min nodes: " + str(min_nodes))
    # min_nodes = max(int(np.int64(np.ceil(min_nodes)) / C_inv), 1)

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


def get_batch(regressors, observed, tot_batches, batch_num):
    """
    Batch the data.

    :param regressors: a torch.Tensor matrix of n columns where each column is the observed values of a random vector
                       in the unit hypercube
    :param observed: a torch.Tensor vector of n observed values. The ith entry corresponds to the ith vector in the
                     "regressors" matrix
    :param batch_size: the total number of batches desired
    :param batch_num: the current batch number
    :return:
        - reg_batch - a subset of the columns in the "regressors" matrix corresponding to the current batch of data.
        - obs_batch - a subset of the entries in the "observed" vector corresponding to the current batch of data.
        - current - an integer representing the greatest index of "observed" included in "obs_batch"
    """

    size = len(observed)
    per_batch = int(np.ceil(size / tot_batches))  # calculate the number of observations in each batch
    start_index = np.int64(batch_num * per_batch)
    remaining = size - 1 - start_index

    # determine the end point of the current batch based on the number of remaining entries
    if (remaining < per_batch):
        end_index = size
    else:
        end_index = start_index + per_batch

    reg_batch = regressors[start_index:end_index, :]
    obs_batch = observed[start_index:end_index]
    current = end_index + 1
    return reg_batch, obs_batch, current


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


def train(regressors, observed, tot_batches, model, loss_fn, optimizer, device):
    """
    Train the model using the specified number of batches, model, loss function,
    and optimizer.

    :param regressors: a matrix of n columns where each column is the observed value of a random vector in the unit
                       hypercube
    :param observed: a vector of n observed values. The ith entry corresponds to the ith vector in the "regressors"
                     matrix
    :param tot_batches: the total number of batches desired
    :param model: the network model
    :param loss_fn: the loss function used to evaluate the model
    :param optimizer: the optimizer to update the network parameters
    :param device: where the computations will take place
    :return: none
    """

    size = len(observed)
    # print("regressors size: " + str(regressors.size()))
    # print(str(regressors))
    # print("observed size: " + str(observed.size()))
    # print(str(observed))

    if tot_batches > size:
        print("Error: total number of batches exceeds number of observations.")
        return

    # Iterate through the batches in the dataloader
    tot_batches = size
    for batch in range(tot_batches):
        reg_batch, obs_batch, current = get_batch(regressors, observed, tot_batches, batch)
        # print("reg batch size: " + str(reg_batch.size()))
        # print("obs batch size: " + str(obs_batch.size()))
        reg_batch, obs_batch = reg_batch.to(device), obs_batch.to(device)

        # Compute prediction error
        pred = model(reg_batch)
        loss = loss_fn(pred, obs_batch)

        # Backpropagation
        optimizer.zero_grad()  # reset grad value
        loss.backward()  # backprop
        optimizer.step()  # update parameter values based on grads

        if batch % 100 == 0:
            loss, current = loss.item(), current
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

