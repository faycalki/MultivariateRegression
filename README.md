# Machine Learning Framework & Experimental Analysis
*Author: Faycal Kilali*  
*With contributions from Dr. Michael Cormier*

## Overview
This repository provides a comprehensive implementation of a machine learning framework with a focus on linear regression analysis. It includes both theoretical components (gradient descent, regularization) and practical implementations (data generation, model training, and analysis).

## Features

### Core Machine Learning Components
1. **Generalized Hypothesis Function Creation**
   - Dynamic parameter handling
   - Support for multivariate hypothesis functions
   - Flexible polynomial degree adaptation
   - Automatic intercept term handling

2. **Loss Functions & Regularization**
   - L2 norm error (MSE) implementation
   - L2 regularization (Ridge) support
   - Combined loss function with regularization
   - Configurable regularization parameter (λ)

3. **Training Framework**
   - Gradient descent optimization
   - Configurable learning rate and epochs
   - Support for batch processing
   - Automatic gradient computation and weight updates

4. **Data Management**
   - Dataset generation with controlled noise
   - Training/validation split functionality (70/30)
   - Support for multivariate data
   - Data sorting and preprocessing capabilities

### Advanced Features

1. **Closed-Form Solutions**
   - Implementation of analytical solutions
   - Support for regularized and non-regularized cases
   ```python
   # Without regularization:
   w* = (X^T · X)^{-1} · X^T · Y
   
   # With regularization (λ > 0):
   w* = (X^T X + λI)^{-1} X^T Y
   ```

2. **Supervised Learning Pipeline**
   - Automated dataset splitting
   - Training set management
   - Validation set management
   - Performance metrics calculation

3. **Visualization & Debugging**
   - Comprehensive dataset information display
   - Training progress monitoring
   - Model performance visualization
   - Parameter tracking

## Mathematical Foundation

### Key Components

1. **Hypothesis Space**
   - Linear combinations of input features
   - Support for polynomial transformations
   - Intercept term handling
   - Multivariate capabilities

2. **Loss Functions**
   ```python
   MSE = mean((Y - Y_hat)^2)
   L2_reg = λ * sum(w^2)
   Total_Loss = MSE + L2_reg
   ```

## Implementation Details

### Default Parameters
- Number of points: 25
- Noise Sigma: 2.5
- Domain: R(0, 10)
- Training split: 70%
- Validation split: 30%
- Default learning epochs: 1000

### Key Functions

```python
def generalized_creation_of_hypothesis_function(w):
    """Creates a hypothesis function from the defined hypothesis space"""

def machine_learning(w, learning_rate, epochs, X, Y, hypothesis_function, lambda_reg=0.00):
    """Main training loop with gradient descent optimization"""

def closed_form_multivariate_solution(X, Y, num_parameters_excluding_intercept_term=1, regularization_lambda=0.0):
    """Computes the closed-form solution for multivariate regression"""
```

## Experimental Results

### 1. Impact of Noise on Model Performance
![Default Noise Level](./assets/plot_experiment_1_4.png)
![Increased Noise Level](./assets/plot_experiment_2_3.png)

### 2. Training Set Size Impact
![Large Training Set](./assets/plot_experiment_3_3.png)

### 3. Parameter Complexity and Overfitting
![Overfitting Example](./assets/plot_experiment_4_5.png)

### 4. L2 Regularization Effects
![Regularization Effects](./assets/plot_experiment_5_3.png)

## Directory Structure
```
.
├── README.md
├── assets/
│   ├── plot_experiment_1_4.png
│   ├── plot_experiment_2_3.png
│   ├── plot_experiment_3_3.png
│   ├── plot_experiment_4_5.png
│   └── plot_experiment_5_3.png
└── src/
    ├── __init__.py
    ├── machine_learning.py
    ├── hypothesis.py
    ├── loss_functions.py
    └── data_utils.py
```

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Usage

```python
# Initialize data and weights
data_x, data_y, w = generalized_machine_learning_initialization(
    true_fn=simple_f,
    domain=(0, 10),
    noise_sigma=2.5,
    n_pts=25
)

# Create supervised learning setup
training_set, validation_set, hypothesis_fn = generalized_supervised_machine_learning_initialization(
    data_x, data_y, w
)

# Train model with regularization
trained_weights = machine_learning(
    w=w,
    learning_rate=0.01,
    epochs=1000,
    X=data_x,
    Y=data_y,
    hypothesis_function=hypothesis_fn,
    lambda_reg=0.01
)
```

## Acknowledgments
Special thanks to Dr. Michael Cormier for contributing core data generation functions.

## Future Development
- Integration of additional optimization methods
- Support for non-linear kernel methods
- Implementation of cross-validation
- Extension to classification tasks

