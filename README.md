# STAT 490: Nonparametric Regression With Neural Networks

The goal of this project is to compare the empirical convergence rates of neural network estimators and classical statistical methods in the context of nonparametric regression problems.

## Statistical Methods Used

- Linear regression `sklearn.linear_model.LinearRegression`
- Autogluon `TabularPredictor`
- Neural network with the following architecture and properties
  - 4 hidden layers
  - 5 nodes in each hidden layer
  - ReLU activation function
  - Trained using Adam optimizer
  - Weights initialized using Glorot initialization   

## Points of comparison

- Performance across different types of function
- Performance across different input sizes
- Performance across different signal to noise ratios

## Procedure


## Code Overview

- `data_generation` implements regression functions and generates simulated data
- `main` performs experiments using `data generation` and a reconfigured version of `SH_Code`

## Legacy Code
- `nn_helpers` old neural network implementation
- `SH_Code` example code from https://arxiv.org/abs/1804.02253
- `legacy_pytorch_code` numerical experiments code with PyTorch implementation

## Future Work

