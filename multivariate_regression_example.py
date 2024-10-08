# Author: Faycal Kilali
# Date: 2024-09-10
# Goal: Multivariate regression

import numpy as np
import random
import torch as torch

import matplotlib.pyplot as plt


def generalized_creation_of_hypothesis_function(w):
    """
        Takes an element from the defined Hypothesis Space and returns it. The hypothesis space consists of element functions.
        The returned function can be used to compute the predicted values (y_hat) for any input tensor X.

    Parameters:
        w (torch.Tensor): The weight tensor of shape (D+1, 1) representing the coefficients,
                          including the intercept term (w_0).

    Returns:
        function: A hypothesis function that takes an input tensor X of shape (N, D) and computes
                  the corresponding predicted values y_hat of shape (N, 1).

    Note:
        The hypothesis space is defined by the inner function.
        This function is generalized enough to be adapt-able easily for any hypothesis space through inner child functions.
    """

    def multivariate_hypothesis_function(X, num_parameters=1):
        """
        Defines the hypothesis space and returns an element hypothesis function from the hypothesis space.

        Parameters:
            X (torch.Tensor): The input tensor of shape (N, D) representing the data points (N samples, D features).
            num_parameters (int): The number of parameters in the hypothesis space.

        Returns:
            torch.Tensor: Hypothesis function from the hypothesis space, which is the result of multiplying the input tensor X with the weight tensor w. This produces corresponding y_hat's for each data point in the input tensor X. Hence, this tensor can also be observed as Y_hat.

        Note:
            This hypothesis space is the set of all possible linear combinations of the input tensor X and the weight tensor w. In other words, it is the set of all elements of form w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_D * x_D, for any x_1, x_2, ..., x_D and w_0, w_1, ..., w_D.
            The first operation can be optimized by adding the intercept term w_0 to the input tensor X once rather than each time we pick from the hypothesis space. This is achieved by inserting ones at index 0 along columns (axis=1) of the input tensor X.
            If lambda is zero during training or for wstar, then this is equivalent to OLS.
        """

        # Create the intercept term (a column of ones to be the first column in the matrix, so that the features, x_i, can come after it)
        intercept_term = torch.ones(X.shape[0], 1)  # Shape (N, 1)

        # Concatenate the intercept term with the input tensor
        X_with_intercept = torch.cat((intercept_term, X), dim=1)  # Shape (N, D+1)

        # Create the degree parameters, so that we can have w_2, w_3, .... w_d up to the num_parameters
        original_num_columns = X.shape[1]  # D = number of features in the original matrix that we rae looking through
        for degree in range(original_num_columns + 1, original_num_columns + num_parameters):
            X_poly = X ** (degree)  # Raise each feature to the new degree
            X_with_intercept = torch.cat((X_with_intercept, X_poly), dim=1)  # Concatenate polynomial terms into a new column to the right of the current column that we are iterating through

        # Compute the hypothesis function output (predicted values y_hat)
        Y_hat = torch.matmul(X_with_intercept, w)

        return Y_hat

    # Return the hypothesis function for later use
    return multivariate_hypothesis_function


def l2_norm_error(Y_hat, Y):
    """
    Computes the L2 norm error (squared error) between labels and predicted labels. This doesn't compute the mean value, only the sum of the errors.

    This is also known as the Mean Squared Error (MSE)

    Parameters:
    - Y_hat: Predicted labels (N, 1)
    - Y: True labels (N, 1)

    Returns:
    L2 norm error (squared error). MSE.
    """
    # Return the sum of squared errors between the true and predicted labels
    return torch.mean((Y - Y_hat) ** 2)


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
    return lambda_reg * torch.sum(w ** 2)


def total_l2_norm_loss_with_regularization(Y, Y_hat, w, lambda_reg=0.01):
    """
    Combines the L2 norm error and L2 regularization to compute the total loss with regularization. MSE + L2 regularization

    Parameters:
    - Y: True values (N, 1)
    - Y_hat: Predicted values (N, 1)
    - w: Weights (D, 1)
    - lambda_reg: Regularization parameter (default: 0.01)

    Returns:
    - total_loss: Total loss (L2 norm error + L2 regularization)
    """
    # Compute the L2 norm error (squared error)
    MSE = l2_norm_error(Y_hat, Y)  # compute the MSE

    # Compute the L2 regularization term
    reg_term = l2_regularization(w,
                                 lambda_reg)  # Also called the Ridge Regression Term, applied as penalty to the weights

    total_loss = MSE + reg_term

    return total_loss


def machine_learning(w, learning_rate, epochs, X, Y, hypothesis_function, lambda_reg=0.00):
    """
    This function is used to initialize the machine learning part of the project. Specifically, it is used to attempt to find better weights.
    The way this works is by minimizing a cost function. This only requires the current weights, a learning rate, and which cost function to use.
    The cost function is minimized by gradient descent because each y_hat involves terms with weights in them, the coefficients.

    Parameters:
        w (torch.Tensor): The weights to begin training from.
        learning_rate (float): The learning rate for gradient descent.
        epochs (int): The number of epochs for gradient descent.
        X (torch.Tensor): Input features (N, D+1) where N is number of samples, D is the number of features.
        Y (torch.Tensor): True labels (N, 1).
        hypothesis_function (function): The hypothesis function to use for the machine learning part of the project.
        lambda_reg (float, optional): The regularization parameter. Defaults to 0.


    Returns:
        w (torch.Tensor): The optimized weights.

    TODO:
        - Add documentation
        - Add additional methods for machine learning
        - Add additional methods for gradient descent
        - Add an interactive ability to choose which machine learning form to select
    """

    # Add an intercept term (column of ones) to X
    intercept = torch.ones((X.shape[0], 1))  # shape (N, 1)
    X_with_intercept = torch.cat((intercept, X), dim=1)  # shape (N, D+1)

    # Function to perform the gradient descent update
    def gradient_descent(X, Y, hypothesis_function, lambda_reg):
        """
        Perform a forward pass and compute the gradient of the loss w.r.t. weights.

        Parameters:
            X (torch.Tensor): Input features (N, D+1) where N is number of samples, D is the number of features.
            Y (torch.Tensor): True labels (N, 1).
            hypothesis_function (function): The hypothesis function to use for the machine learning part of the project.
            lambda_reg (float, optional): The regularization parameter. Defaults to 0.

        Returns:
            torch.Tensor: The gradient of the loss with respect to the weights.
        """
        # Forward pass: Calculate predicted labels y_hat = X @ w
        Y_hat = hypothesis_function(X)  # Shape (N, 1)
        print(Y_hat)

        # Compute the loss. Uses lambda regularization if not passed as a neutral number parameter, otherwise not.
        if (lambda_reg != 0):
            loss = total_l2_norm_loss_with_regularization(Y, Y_hat, w, lambda_reg)
        else:
            loss = l2_norm_error(Y_hat, Y)

        # Backward pass: Compute gradient of loss with respect to weights
        loss.backward()  # Populates w.grad with gradients d(loss)/dw

        # Now we'll return the computed gradient
        return w.grad

    # Function to begin the gradient descent process
    def begin_learning_gradient_descent(X, Y, w, learning_rate, epochs, hypothesis_function):
        """
        Train the model by updating the weights using gradient descent over a number of epochs.

        Parameters:
            X (torch.Tensor): Input features (N, D+1).
            Y (torch.Tensor): True labels (N, 1).
            w (torch.Tensor): Initial weights (D+1, 1).
            learning_rate (float): Learning rate for weight updates.
            epochs (int): Number of training iterations (epochs).
            hypothesis_function (function): The hypothesis function to use for the machine learning part of the project.

        NOTE:
            The utilized hypothesis function will have a gradient attached to it. You must re-generate the hypothesis function using the returned weights from this function.

        Returns:
            torch.Tensor: Updated weights after gradient descent.
        """
        # We must ensure that weights require gradients to compute the gradient
        w.requires_grad_(True)

        for epoch in range(epochs):
            # Perform a forward and backward pass to compute the gradient
            gradient = gradient_descent(X, Y, hypothesis_function)

            # Update the weights using gradient descent
            with torch.no_grad():
                w -= learning_rate * gradient  # w = w - learning_rate * gradient
                w.grad.zero_()  # Zero out gradients after the update

            # Pick another element hypothesis function from the hypothesis space
            hypothesis_function = generalized_creation_of_hypothesis_function(w)

            # Detach the tensor from the computations
        w = w.detach()

        # Return the optimized weights
        return w

    # Return the inner function to start training
    return begin_learning_gradient_descent(X, Y, w, learning_rate, epochs, hypothesis_function)


def closed_form_multivariate_solution(X, Y, num_parameters=1, regularization_lambda = 0.0):
    """
    Compute the closed-form solution for multivariate regression using linear algebra.

    Parameters:
        X (torch.Tensor): The input tensor of shape (N, D) representing the data points.
        Y (torch.Tensor): The label tensor of shape (N, 1) representing the corresponding labels.
        num_parameters (int, optional): The number of parameters in the hypothesis function. Defaults to 1.
        regularization_lambda (float, optional): The regularization parameter. Defaults to 0.0.

    Returns:
        torch.Tensor: The closed-form solution for the weights wstar of shape (D+1, 1).

    The closed-form solution for multivariate regression with regularization lambda 0 is obtained by attaining the matrix wstar through the following linear algebra operations:
        wstar = (X.T @ X)^-1 @ X.T @ Y

    The closed-form solution for multivariate regression with regularization lambda greater than 0 is obtained through the following linear algebra operations
        wstar = (X^T X + λI)^−1 X^T Y

    This function takes in the input tensor X and the label tensor Y, and returns the closed-form solution for the weights wstar.
    """
    # Create the intercept term (a column of ones to be the first column in the matrix, so that the features, x_i, can come after it)
    intercept_term = torch.ones(X.shape[0], 1)  # Shape (N, 1)

    # Concatenate the intercept term with the input tensor
    X_with_intercept = torch.cat((intercept_term, X), dim=1)  # Shape (N, D+1)

    # Create the degree parameters, so that we can have w_2, w_3, .... w_d up to the num_parameters
    original_num_columns = X.shape[1]  # D = number of features in the original matrix that we rae looking through
    for degree in range(original_num_columns + 1, original_num_columns + num_parameters):
        X_poly = X ** (degree)  # Raise each feature to the new degree
        X_with_intercept = torch.cat((X_with_intercept, X_poly),
                                     dim=1)  # Concatenate polynomial terms into a new column to the right of the current column that we are iterating through

    # Compute the closed-form solution for the multivariate linear regression

    if regularization_lambda > 0.0:
        # wstar = (X^T X + λI)^−1 X^T Y generates the closed-form solution wstar with lambda regularization
        regularized_wstar = torch.pinverse(X_with_intercept.T @ X_with_intercept + regularization_lambda * torch.eye(X_with_intercept.shape[1])) @ X_with_intercept.T @ Y
        return regularized_wstar
    else:
        wstar = torch.pinverse(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ Y)
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
    return -6.0 + x * 4.2


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
    data_x = torch.zeros((n_pts, 1))
    data_y = torch.zeros((n_pts, 1))
    # Generate each data point
    for i in range(n_pts):
        data_x[i, 0] = random.uniform(domain[0], domain[1])
        data_y[i, 0] = true_fn(data_x[i, 0]) + random.normalvariate(0.0, noise_sigma)
    return data_x, data_y


def generalized_machine_learning_initialization(generative_function, domain, noise_sigma, n_pts):
    """
    Initializes the machine learning process by generating a dataset and initializing weights.

    Returns:
        data_x (torch.Tensor): The input tensor of shape (N, D) representing the data points.
        data_y (torch.Tensor): The label tensor of shape (N, 1) representing the corresponding labels.
        w (torch.Tensor): The initial weights tensor of shape (D + 1, 1) where D is the number of features.

    This function generates a dataset by calling the `initialize_dataset` function, which generates data using the `make_data` function. It then converts the data to Tensors using `torch.tensor`. The function also generates initial weights by calling the `generate_initial_weights` function, which generates weights with relevant dimensions based on the number of features in the dataset. The initial weights are also converted to Tensors using `torch.tensor`.

    Note:
        - `simple_f` is a function that generates simple test data.
        - `larger_f` is a function that generates more complex test data.
        - `make_data` is a function that generates data using the `simple_f` function.
        - The existence of this function is for the purpose of modularity between this and the supervised and or unsupervised machine learning function. This is important to make the code more modular and reusable.
    """

    def initialize_dataset():
        data_x, data_y = make_data(generative_function, domain, noise_sigma, n_pts)
        return data_x, data_y

    def initialize_weights(data_x):
        """
        Initialize the initial weights using a uniform distribution between -0.01 and 0.01.

        Parameters:
            data_x (torch.Tensor): Input tensor (N, D) representing the data points.

        Returns:
            torch.Tensor: Randomly initialized weights of shape (D+1, 1), where +1 is for the intercept term.

        Note:
            This should only be used at the start for an initial weights tensor. Any involved learning process should adjust the weights as necessary.
        """
        # Initialize weights with shape (D+1, 1) for the intercept term
        weights = torch.empty((data_x.shape[1] + 1, 1))

        # Initialize weights with values from uniform distribution [-0.01, 0.01]
        torch.nn.init.uniform_(weights, a=-0.01, b=0.01)

        return weights

    data_x, data_y = initialize_dataset()

    data_x, data_y = data_x, data_y  # Convert to Tensors

    w = initialize_weights(data_x)

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

        The data is split into two subsets:
        – A training and validation set used only to find the right predictor

        Note that  A test set used to report the prediction error of the algorithm should be generated completely independently without the need to generate subsets from it.

        All sets must be disjoint.

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

        return training_set, validation_set

    training_set, validation_set = generate_supervised_learning_subsets(data_x, data_y)

    # Generate initial hypothesis function
    hypothesis_function_initial = generalized_creation_of_hypothesis_function(w)

    display_datasets(data_x, data_y, training_set, validation_set)

    display_relevant_information(data_x, data_y, training_set, validation_set, w, hypothesis_function_initial)

    return training_set, validation_set, hypothesis_function_initial


def display_datasets(data_x, data_y, training_set, validation_set):
    print("data_x:", data_x)
    print("data_y:", data_y)
    print("training_set:", training_set)
    print("validation_set:", validation_set)


def display_relevant_information(data_x, data_y, training_set, validation_set, w, hypothesis_function):
    # Print the shapes and values in a readable format
    print("=== Dataset Information ===")
    print(f"Shape of data_x: {data_x.shape}  # (Samples, Features)")
    print(f"Shape of data_y: {data_y.shape}  # (Samples, Output)")
    print("Features matrix has:", data_x.shape[0], "samples (points) x", data_x.shape[1], "features")
    print("Labels matrix has:", data_y.shape[0], "samples (points) y", data_y.shape[1], "features")

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

    print(f"Training set size: {training_features.size(0) + training_labels.size(0)}")
    print(f"Validation set size: {validation_features.size(0) + validation_labels.size(0)}")

    print("\n=== Current Weights ===")
    print(f"Initialized weights (w):\n{w}\n")
    print(f"Shape of weights: {w.shape}  # (Features + 1, 1)")
    print("Weights tensor has:", data_x.shape[0], "coefficients x")

    print("\n=== Hypothesis Function ===")
    print(f"Hypothesis function defined with weights at memory address:\n{hypothesis_function}\n")


def regressive_experiments():
        # Setup for experiment 1

        # Same hypothesis class and training set size, less noise
        data_x, data_y, w = generalized_machine_learning_initialization(simple_f, (0, 10), 2.5, 25)
        training_set, validation_set, hypothesis_function = generalized_supervised_machine_learning_initialization(
            data_x, data_y, w)

        # Unzipping training and validation data
        unzipped_training_features, unzipped_training_labels = zip(*training_set)
        unzipped_validation_features, unzipped_validation_labels = zip(*validation_set)

        # Convert tuples back to tensors
        validation_features = torch.stack(unzipped_validation_features)
        validation_labels = torch.stack(unzipped_validation_labels)
        training_features = torch.stack(unzipped_training_features)
        training_labels = torch.stack(unzipped_training_labels)

        # Measure initial error and perform training
        prev_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)
        w = machine_learning(w, 0.01, 1000, training_features, training_labels, hypothesis_function)
        hypothesis_function = generalized_creation_of_hypothesis_function(w)
        post_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)

        # Closed form solution
        wstar = closed_form_multivariate_solution(data_x, data_y)
        closed_solution_hypothesis_function_multivariate = generalized_creation_of_hypothesis_function(wstar)

        print("==== FINAL RESULTS ====")
        print("Initial error:", prev_error)
        print("Final error from learned Hypothesis Function:", post_error)
        print(f"Weights after update: {w}")
        print(f"Weight shapes: {w.shape}")
        print("w* Hypothesis Function Error:",
              l2_norm_error(closed_solution_hypothesis_function_multivariate(validation_features), validation_labels))
        print(f"w* weights: {wstar}")

        loss_rate_hypothesis_function = post_error
        loss_rate_wstar_hypothesis_function = l2_norm_error(
            closed_solution_hypothesis_function_multivariate(validation_features), validation_labels)
        testing_features, testing_labels, _ = generalized_machine_learning_initialization(simple_f, (0, 10), 0, 2500)

        # Now plot with testing features
        plot_experiments(hypothesis_function, testing_features, testing_labels, loss_rate_hypothesis_function,
                         loss_rate_wstar_hypothesis_function)

        # Now for more noise, but same hypothesis class and training set size experiment 1.

        # Same hypothesis class and training set size, less noise
        data_x, data_y, w = generalized_machine_learning_initialization(simple_f, (0, 10), 25, 25)
        training_set, validation_set, hypothesis_function = generalized_supervised_machine_learning_initialization(
            data_x, data_y, w)

        # Unzipping training and validation data
        unzipped_training_features, unzipped_training_labels = zip(*training_set)
        unzipped_validation_features, unzipped_validation_labels = zip(*validation_set)

        # Convert tuples back to tensors
        validation_features = torch.stack(unzipped_validation_features)
        validation_labels = torch.stack(unzipped_validation_labels)
        training_features = torch.stack(unzipped_training_features)
        training_labels = torch.stack(unzipped_training_labels)

        # Measure initial error and perform training
        prev_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)
        w = machine_learning(w, 0.01, 1000, training_features, training_labels, hypothesis_function)
        hypothesis_function = generalized_creation_of_hypothesis_function(w)
        post_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)

        # Closed form solution
        wstar = closed_form_multivariate_solution(data_x, data_y)
        closed_solution_hypothesis_function_multivariate = generalized_creation_of_hypothesis_function(wstar)

        print("==== FINAL RESULTS ====")
        print("Initial error:", prev_error)
        print("Final error from learned Hypothesis Function:", post_error)
        print(f"Weights after update: {w}")
        print(f"Weight shapes: {w.shape}")
        print("w* Hypothesis Function Error:",
              l2_norm_error(closed_solution_hypothesis_function_multivariate(validation_features), validation_labels))
        print(f"w* weights: {wstar}")

        loss_rate_hypothesis_function = post_error
        loss_rate_wstar_hypothesis_function = l2_norm_error(
            closed_solution_hypothesis_function_multivariate(validation_features), validation_labels)
        testing_features, testing_labels, _ = generalized_machine_learning_initialization(simple_f, (0, 10), 0, 2500)

        # Now plot with testing features
        plot_experiments(hypothesis_function, testing_features, testing_labels, loss_rate_hypothesis_function,
                         loss_rate_wstar_hypothesis_function)


        # Experiment 2, larger training set provides a better learned function than a smalelr one, given teh same level of noise and hypothesis class

        # Same hypothesis class and training set size, less noise
        data_x, data_y, w = generalized_machine_learning_initialization(simple_f, (0, 10), 25, 250)
        training_set, validation_set, hypothesis_function = generalized_supervised_machine_learning_initialization(
            data_x, data_y, w)

        # Unzipping training and validation data
        unzipped_training_features, unzipped_training_labels = zip(*training_set)
        unzipped_validation_features, unzipped_validation_labels = zip(*validation_set)

        # Convert tuples back to tensors
        validation_features = torch.stack(unzipped_validation_features)
        validation_labels = torch.stack(unzipped_validation_labels)
        training_features = torch.stack(unzipped_training_features)
        training_labels = torch.stack(unzipped_training_labels)

        # Measure initial error and perform training
        prev_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)
        w = machine_learning(w, 0.01, 1000, training_features, training_labels, hypothesis_function)
        hypothesis_function = generalized_creation_of_hypothesis_function(w)
        post_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)

        # Closed form solution
        wstar = closed_form_multivariate_solution(data_x, data_y)
        closed_solution_hypothesis_function_multivariate = generalized_creation_of_hypothesis_function(wstar)

        print("==== FINAL RESULTS ====")
        print("Initial error:", prev_error)
        print("Final error from learned Hypothesis Function:", post_error)
        print(f"Weights after update: {w}")
        print(f"Weight shapes: {w.shape}")
        print("w* Hypothesis Function Error:",
              l2_norm_error(closed_solution_hypothesis_function_multivariate(validation_features), validation_labels))
        print(f"w* weights: {wstar}")

        loss_rate_hypothesis_function = post_error
        loss_rate_wstar_hypothesis_function = l2_norm_error(
            closed_solution_hypothesis_function_multivariate(validation_features), validation_labels)
        testing_features, testing_labels, _ = generalized_machine_learning_initialization(simple_f, (0, 10), 0, 2500)

        # Now plot with testing features
        plot_experiments(hypothesis_function, testing_features, testing_labels, loss_rate_hypothesis_function,
                         loss_rate_wstar_hypothesis_function)


        # Experiment 3, a function with too many parameters (e.g., a polynomial of very high degree) has a tendency to overfit, especially on small datasets

        data_x, data_y, w = generalized_machine_learning_initialization(simple_f, (0, 10), 2.5, 250)
        training_set, validation_set, hypothesis_function = generalized_supervised_machine_learning_initialization(
            data_x, data_y, w)

        # Unzipping training and validation data
        unzipped_training_features, unzipped_training_labels = zip(*training_set)
        unzipped_validation_features, unzipped_validation_labels = zip(*validation_set)

        # Convert tuples back to tensors
        validation_features = torch.stack(unzipped_validation_features)
        validation_labels = torch.stack(unzipped_validation_labels)
        training_features = torch.stack(unzipped_training_features)
        training_labels = torch.stack(unzipped_training_labels)

        # Measure initial error and perform training
        prev_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)
        w = machine_learning(w, 0.01, 1000, training_features, training_labels, hypothesis_function)
        hypothesis_function = generalized_creation_of_hypothesis_function(w)
        post_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)

        # Closed form solution
        wstar = closed_form_multivariate_solution(data_x, data_y)
        closed_solution_hypothesis_function_multivariate = generalized_creation_of_hypothesis_function(wstar)

        print("==== FINAL RESULTS ====")
        print("Initial error:", prev_error)
        print("Final error from learned Hypothesis Function:", post_error)
        print(f"Weights after update: {w}")
        print(f"Weight shapes: {w.shape}")
        print("w* Hypothesis Function Error:",
              l2_norm_error(closed_solution_hypothesis_function_multivariate(validation_features), validation_labels))
        print(f"w* weights: {wstar}")

        loss_rate_hypothesis_function = post_error
        loss_rate_wstar_hypothesis_function = l2_norm_error(
            closed_solution_hypothesis_function_multivariate(validation_features), validation_labels)
        testing_features, testing_labels, _ = generalized_machine_learning_initialization(simple_f, (0, 10), 0, 2500)

        # Now plot with testing features
        plot_experiments(hypothesis_function, testing_features, testing_labels, loss_rate_hypothesis_function,
                         loss_rate_wstar_hypothesis_function)

        # Experiment 4, L_2 norm regularization can help avoid overfitting

        data_x, data_y, w = generalized_machine_learning_initialization(simple_f, (0, 10), 2.5, 250)
        training_set, validation_set, hypothesis_function = generalized_supervised_machine_learning_initialization(
            data_x, data_y, w)

        # Unzipping training and validation data
        unzipped_training_features, unzipped_training_labels = zip(*training_set)
        unzipped_validation_features, unzipped_validation_labels = zip(*validation_set)

        # Convert tuples back to tensors
        validation_features = torch.stack(unzipped_validation_features)
        validation_labels = torch.stack(unzipped_validation_labels)
        training_features = torch.stack(unzipped_training_features)
        training_labels = torch.stack(unzipped_training_labels)

        # Measure initial error and perform training
        prev_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)
        w = machine_learning(w, 0.01, 1000, training_features, training_labels, hypothesis_function)
        hypothesis_function = generalized_creation_of_hypothesis_function(w)
        post_error = l2_norm_error(hypothesis_function(validation_features), validation_labels)

        # Closed form solution
        wstar = closed_form_multivariate_solution(data_x, data_y)
        closed_solution_hypothesis_function_multivariate = generalized_creation_of_hypothesis_function(wstar)

        print("==== FINAL RESULTS ====")
        print("Initial error:", prev_error)
        print("Final error from learned Hypothesis Function:", post_error)
        print(f"Weights after update: {w}")
        print(f"Weight shapes: {w.shape}")
        print("w* Hypothesis Function Error:",
              l2_norm_error(closed_solution_hypothesis_function_multivariate(validation_features), validation_labels))
        print(f"w* weights: {wstar}")



        regularizer = l2_regularization(w)

        loss_rate_hypothesis_function = post_error
        loss_rate_wstar_hypothesis_function = total_l2_norm_loss_with_regularization(
            closed_solution_hypothesis_function_multivariate(validation_features), validation_labels, regularizer)
        testing_features, testing_labels, _ = generalized_machine_learning_initialization(simple_f, (0, 10), 0, 2500)

        # Now plot with testing features
        #NOTE: I manually changed the loss_function call from the gradient to use l2 with regularization in order to
        plot_experiments(hypothesis_function, testing_features, testing_labels, loss_rate_hypothesis_function,
                         loss_rate_wstar_hypothesis_function)



def plot_experiments(hypothesis_function, testing_features, testing_labels, loss_rate_hypothesis_function,
                     loss_rate_wstar_hypothesis_function):
    """
    Testing labels are the true function values at the testing features.
    :param hypothesis_function: The hypothesis function to evaluate.
    :param testing_features: The features to test the hypothesis function.
    :param testing_labels: The true labels corresponding to the testing features.
    :param loss_rate_hypothesis_function: Loss rate of the learned hypothesis function.
    :param loss_rate_wstar_hypothesis_function: Loss rate of the closed-form hypothesis function.
    #TODO: Add documentation and start using :param, etc. Look up python documentation for better documenting.
    """

    plt.figure(figsize=(10, 5))

    # Calculate the hypothesis values at the testing features
    Y_hypothesis = hypothesis_function(testing_features)  # Hypothesis function values at testing points

    # Closed form hypothesis values
    wstar = closed_form_multivariate_solution(testing_features, testing_labels)
    closed_solution_hypothesis_function = generalized_creation_of_hypothesis_function(wstar)
    Y_wstar = closed_solution_hypothesis_function(testing_features)

    # Plot true function values at testing features
    plt.scatter(testing_features, testing_labels, label='True Function', color='green', s=30, alpha=0.5)

    # Plot hypothesis function values at testing features
    plt.scatter(testing_features, Y_hypothesis, label='Learned Hypothesis Function', color='red', s=30, marker='x')

    # Plot w* hypothesis function values
    plt.scatter(testing_features, Y_wstar, label='w* Hypothesis Function', color='blue', s=30, marker='o')

    # Add loss annotations with colors
    plt.text(0.05, 0.95, f'Loss (Learned Hypothesis): {loss_rate_hypothesis_function:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='red')
    plt.text(0.05, 0.90, f'Loss (w* Hypothesis): {loss_rate_wstar_hypothesis_function:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='blue')

    # Add labels and legend
    plt.title('Comparison of Learned Hypothesis Function and w*')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # Set axis limits to ensure all points are visible
    all_y_values = torch.cat((testing_labels, Y_hypothesis, Y_wstar), dim=0)  # Combine all y-values
    all_x_values = testing_features

    x_margin = (torch.max(all_x_values) - torch.min(all_x_values)) * 0.05
    y_margin = (torch.max(all_y_values) - torch.min(all_y_values)) * 0.05

    plt.xlim(torch.min(all_x_values) - x_margin, torch.max(all_x_values) + x_margin)
    plt.ylim(torch.min(all_y_values) - y_margin, torch.max(all_y_values) + y_margin)

    plt.show()



regressive_experiments()