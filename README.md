# STAT 490: Nonparametric Regression With Neural Networks

The goal of this project is to compare the empirical convergence rates of neural network estimators and classical statistical methods in the context of nonparametric regression problems.

## Statistical Methods Used

- [Linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) `sklearn.linear_model.LinearRegression`
- [AutoGluon](https://auto.gluon.ai/stable/index.html) `TabularPredictor`
- [Neural network](https://arxiv.org/abs/1804.02253) with the following architecture and properties
  - 4 hidden layers
  - 5 nodes in each hidden layer
  - ReLU activation function
  - Trained using Adam optimizer
  - Weights initialized using Glorot initialization   

## Points of comparison and procedure

- Performance across different functions. The methods above were tested on constant, linear, quadratic and sinusoidal functions. These functions are implemented in `data_generation.py`. After fitting a model using the methods above, we examined the models' testing set accuracy to evaluate the methods' performance for each of the functions.
- Performance across different training set sizes. Training sets of different sizes were generated. For each training set, a linear regression model, a tabular predictor model and a neural network model were fit. The testing set accuracy of the resulting models was then used to evaluate the methods' performance at each of the training set sizes.
- Performance across different signal to noise ratios. Firstly, training sets (of the same size) were generated. A random noise term was added to each observation in the training set. The variance of this noise term was changed from training set to training set. This allowed us to give each training set a different signal-to-noise ratio. Each of the methods mentioned above was used to fit a model using the aforementioned training sets. The testing set accuracy of the resulting models was then used to evaluate the methods' performance at different signal to noise ratios.

## Code Overview

- `data_generation` implements regression functions and generates simulated data
- `main` performs experiments using `data generation` and a reconfigured version of `SH_Code`

## Legacy Code
- `nn_helpers` old neural network implementation
- `SH_Code` example code from https://arxiv.org/abs/1804.02253
- `legacy_pytorch_code` numerical experiments code with PyTorch implementation

## Initial observations

## Future Work
Wrap `main_noise` and `main_obs` in `main` in a parallel for loop so that the experiments discussed above can be run multiple times in a time-efficient manner. This will help validate the initial observations described above.
