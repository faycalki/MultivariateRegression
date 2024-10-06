# Author: Faycal Kilali
# Date: 2024-09-10
# Goal: Multivariate regression

import numpy as np
import random
import torch as torch

import matplotlib.pyplot as plt


def multivariate_hypothesis_space_take_element(X, w):
    """
    Takes an element from the defined Hypothesis Space and returns it. Does so by computing the hypothesis function out of the hypothesis space for multivariate regression.

    Parameters:
        X (torch.Tensor): The input tensor of shape (N, D) representing the data points.
        w (torch.Tensor): The weight tensor of shape (D, 1) representing the coefficients.

    Returns:
        torch.Tensor: Hypothesis function from the hypothesis space, which is the result of multiplying the input tensor X with the weight tensor w.

    Note:
        The hypothesis space is the set of all possible linear combinations of the input tensor X and the weight tensor w. In other words, it is the set of all elements of form w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_D * x_D, for any x_1, x_2, ..., x_D and w_0, w_1, ..., w_D.
        The first operation can be optimized by adding the intercept term w_0 to the input tensor X once rather than each time we pick from the hypothesis space. This is achieved by inserting ones at index 0 along columns (axis=1) of the input tensor X.
    """
    # Add the intercept term w_0 so that it works properly with matrix multiplication. Performs this through inserting ones at index 0 along columns (axis=1)

    # Create the intercept term (a column of ones)
    intercept_term = torch.ones(X.shape[0], 1)  # Shape (N, 1)

    # Concatenate the intercept term with the input tensor
    X_with_intercept = torch.cat((intercept_term, X), dim=1)  # Shape (N, D+1)

    # Compute the hypothesis function
    element_from_hypothesis_space = torch.matmul(X_with_intercept, w)

    # Print out current hypothesis function
    print("Obtained element from hypothesis space: ", element_from_hypothesis_space)

    return element_from_hypothesis_space


def generate_estimated_labels(X, w):
    pass


def l2_norm_error(Y, Y_hat):
    """
    Computes the L2 norm error (squared error) between labels and predicted labels

    Parameters:
    - Y: True labels (N, 1)
    - Y_hat: Predicted labels (N, 1)

    Returns:
    L2 norm error (squared error)
    """
    # Return the sum of squared errors between the true and predicted labels
    return np.sum((Y - Y_hat) ** 2)


def l2_regularization(w, lambda_reg=0.01):
    """
    Computes the L2 regularization term (Ridge regularization).

    Parameters:
    - w: Weights (D, 1)
    - lambda_reg: Regularization parameter

    Returns:
    L2 regularization term
    """
    # Compute the L2 regularization term as the sum of squared weights and returns it.
    return lambda_reg * np.sum(w ** 2)


def total_l2_norm_loss_with_regularization(Y, Y_hat, w, lambda_reg=0.01):
    """
    Combines the L2 norm error and L2 regularization to compute the total loss with regularization

    Parameters:
    - Y: True values (N, 1)
    - Y_hat: Predicted values (N, 1)
    - w: Weights (D, 1)
    - lambda_reg: Regularization parameter (default: 0.01)

    Returns:
    - total_loss: Total loss (L2 norm error + L2 regularization)
    """
    # Compute the L2 norm error (squared error)
    MSE = l2_norm_error(Y, Y_hat)  # Also called the Mean Squared Error

    # Compute the L2 regularization term
    reg_term = l2_regularization(w,
                                 lambda_reg)  # Also called the Ridge Regression Term, applied as penalty to the weights

    total_loss = MSE + reg_term

    return total_loss


def gradient_descent(X):
    """
    Compute the gradient of a function using gradient descent.

    Parameters:
        X (torch.Tensor): The input tensor for which to compute the gradient.

    Returns:
        torch.Tensor: The gradient of the function with respect to X.
    """
    grad = torch.gradient(X)
    return grad


def begin_learning_gradient_descent(X, Y, w, learning_rate, epochs):
    for epoch in range(epochs):
        # Update the weights

        w = w - gradient_descent(X)
    return w


# Function: closed_form_multivariate_solution
# Parameters:
#   X:            (N, D) matrix
#   y:            (N, 1) vector
# Returns:
#   wstar:        (D, 1) vector
# Purpose:
#   Solves the closed form multivariate regression
def closed_form_multivariate_solution(X, y):
    wstar = torch.linalg.inv(X.T @ X) @ torch.transpose(X, 0, 1) @ y
    return wstar


# Function: simple_f
# Author: Dr. Michael Cormier
# Parameters:
#   x: Dependent variable
# Returns:
#   4.2x - 6
# Purpose:
#   Function for generating simple test data
def simple_f(x):
    return 4.2 * x - 6.0


# Function: make_data
# Author: Dr. Michael Cormier
# Parameters:
#   true_fn:     Function of one floating point value that returns a floating
#                point value
#   domain:      Tuple representing the minimum and maximum values of the
#                independent variable
#   noise_sigma: Standard deviation for Gaussian noise
#   n_pts:       Number of points to generate
# Returns:
#   data_x: Independent variable values
#   data_y: Dependent variable values (what we want to predict)
def make_data(true_fn, domain, noise_sigma, n_pts):
    # Create Numpy matrices
    data_x = np.zeros((n_pts, 1))
    data_y = np.zeros((n_pts, 1))
    # Generate each data point
    for i in range(n_pts):
        data_x[i, 0] = random.uniform(domain[0], domain[1])
        data_y[i, 0] = true_fn(data_x[i, 0]) + random.normalvariate(0.0, noise_sigma)
    return data_x, data_y


def generalized_machine_learning_initialization():
    """
    Initializes the machine learning process by generating a dataset and initializing weights.

    Returns:
        data_x (torch.Tensor): The input tensor of shape (N, D) representing the data points.
        data_y (torch.Tensor): The label tensor of shape (N, 1) representing the corresponding labels.
        w (torch.Tensor): The initial weights tensor of shape (D + 1, 1) where D is the number of features.

    This function generates a dataset by calling the `initialize_dataset` function, which generates data using the `make_data` function. It then converts the data to Tensors using `torch.tensor`. The function also generates initial weights by calling the `generate_initial_weights` function, which generates weights with relevant dimensions based on the number of features in the dataset. The initial weights are also converted to Tensors using `torch.tensor`.

    Note:
        - `simple_f` is a function that generates simple test data.
        - `make_data` is a function that generates data using the `simple_f` function.
        - The existence of this function is for the purpose of modularity between this and the supervised and or unsupervised machine learning function. This is important to make the code more modular and reusable.
    """

    def initialize_dataset():
        data_x, data_y = make_data(simple_f, (0, 10), 2.5, 25)
        return data_x, data_y

    def generate_initial_weights():
        # Generate initial weights with relevant dimensions as to the number of features
        return np.random.uniform(low=-0.01, high=0.01, size=(data_x.shape[1] + 1, 1))  # +1 for intercept

    data_x, data_y = initialize_dataset()

    data_x, data_y = torch.tensor(data_x), torch.tensor(data_y)  # Convert to Tensors

    w = generate_initial_weights()
    # Convert to Tensors
    w = torch.tensor(w)

    return data_x, data_y, w


def generalized_supervised_machine_learning_initialization(data_x, data_y, w):
    """
    Initializes the supervised machine learning process by generating training and validation subsets for supervised learning.

    Args:
        data_x (torch.Tensor): The input tensor of shape (N, D) representing the data points.
        data_y (torch.Tensor): The label tensor of shape (N, 1) representing the corresponding labels.
        w (torch.Tensor): The initial weights tensor of shape (D + 1, 1) where D is the number of features.

    Returns:
        training_set (torch.utils.data.TensorDataset): The training set consisting of input tensors and label tensors.
        validation_set (torch.utils.data.TensorDataset): The validation set consisting of input tensors and label tensors.
        hypothesis_function_initial (torch.Tensor): The initial hypothesis function computed using the training features and weights.

    Raises:
        AssertionError: If there is a mismatch in the size of the training set inputs and labels.
        AssertionError: If there is a mismatch in the size of the validation set inputs and labels.

    Prints:
        Training set size: The number of inputs in the training set.
        Validation set size: The number of inputs in the validation set.

    This function generates training and validation subsets for supervised learning by splitting the input data into training and validation sets. It then computes the initial hypothesis function using the training features and weights. Finally, it displays relevant information about the dataset, training set, validation set, and initial weights.
    """
    def generate_supervised_learning_subsets(X, Y):
        """
        Generates training and validation subsets for supervised learning.

        Args:
            X (torch.Tensor): The input tensor of shape (N, D) representing the data points.
            Y (torch.Tensor): The label tensor of shape (N, 1) representing the corresponding labels.

        Returns:
            Two Tensor Dataset objects consisting of two tensors. The first Tensor Dataset object represents the training set,
            and the second object represents the validation set. Each dataset consists of input tensors and label tensors.
        """

        training_size_percentage = 0.7  # 70% for training
        training_size = int(training_size_percentage * len(X))
        indices = torch.randperm(len(X))  # Generate a random permutation of indices

        # Split the indices into training and validation
        train_indices = indices[:training_size]
        val_indices = indices[training_size:]

        # Use the split indices to create the training and validation datasets
        training_set_inputs = X[train_indices]
        training_set_labels = Y[train_indices]
        validation_set_inputs = X[val_indices]
        validation_set_labels = Y[val_indices]

        # Create TensorDatasets for training and validation
        training_set = torch.utils.data.TensorDataset(training_set_inputs, training_set_labels)
        validation_set = torch.utils.data.TensorDataset(validation_set_inputs, validation_set_labels)

        # Check to ensure the order is maintained between labels and features
        assert training_set_inputs.size(0) == training_set_labels.size(0), "Mismatch in training set size"
        assert validation_set_inputs.size(0) == validation_set_labels.size(0), "Mismatch in validation set size"

        print(f"Training set size: {training_set_inputs.size(0)}")
        print(f"Validation set size: {validation_set_inputs.size(0)}")

        return training_set, validation_set

    training_set, validation_set = generate_supervised_learning_subsets(data_x, data_y)

    # Compute initial hypothesis function
    unzipped_training_features, unzipped_training_labels = zip(*training_set)
    training_features = torch.stack(unzipped_training_features)
    hypothesis_function_initial = multivariate_hypothesis_space_take_element(training_features, w)

    display_relevant_information(data_x, data_y, training_set, validation_set, w, hypothesis_function_initial)

    return training_set, validation_set, hypothesis_function_initial


def display_relevant_information(data_x, data_y, training_set, validation_set, w, hypothesis_function_initial):
    # Print the shapes and values in a readable format
    print("=== Dataset Information ===")
    print(f"Shape of data_x: {data_x.shape}  # (Samples, Features)")
    print(f"Shape of data_y: {data_y.shape}  # (Samples, Output)")

    print("\n=== Data Samples ===")
    print("data_x (Features):")
    print(data_x)
    print("\ndata_y (Labels):")
    print(data_y)

    # Display training and validation sets
    print("\n=== Training and Validation Sets ===")

    # Extracting features and labels from TensorDataset
    unzipped_training_features, unzipped_training_labels = zip(*training_set)  # Unzip the training set
    unzipped_validation_features, unzipped_validation_labels = zip(*validation_set)  # Unzip the validation set

    # Convert the tuples back to tensors for shape and value display
    training_features = torch.stack(unzipped_training_features)  # Shape (N, D)
    training_labels = torch.stack(unzipped_training_labels)  # Shape (N, 1)
    validation_features = torch.stack(unzipped_validation_features)  # Shape (M, D)
    validation_labels = torch.stack(unzipped_validation_labels)  # Shape (M, 1)

    print("Training Set (Features):")
    print(training_features)
    print("\nTraining Set (Labels):")
    print(training_labels)

    print("\nValidation Set (Features):")
    print(validation_features)
    print("\nValidation Set (Labels):")
    print(validation_labels)

    print("\n=== Initialized Weights ===")
    print(f"Initialized weights (w):\n{w}\n")
    print(f"Shape of weights: {w.shape}  # (Features + 1, 1)")

    print("\n=== Initial Hypothesis Function ===")
    print(f"Hypothesis (predictions) based on initial weights:\n{hypothesis_function_initial}\n")


data_x, data_y, w = generalized_machine_learning_initialization()
training_set, validation_set, hypothesis_function_initial = generalized_supervised_machine_learning_initialization(data_x, data_y, w)

# Agent measures loss from estimated labels by feeding the estimated labels from the validation set to the loss function
l2_norm_error(validation_set, hypothesis_function_initial)
