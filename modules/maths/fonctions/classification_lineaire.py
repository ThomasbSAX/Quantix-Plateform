"""
Quantix – Module classification_lineaire
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# perceptron
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_loss(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute specified loss metric."""
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'logloss':
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif callable(metric):
        return metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _update_weights(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                    learning_rate: float, margin: float) -> np.ndarray:
    """Update weights using perceptron rule."""
    for i in range(X.shape[0]):
        if y[i] * np.dot(X[i], weights) <= margin:
            weights += learning_rate * y[i] * X[i]
    return weights

def perceptron_fit(X: np.ndarray, y: np.ndarray,
                   learning_rate: float = 0.01,
                   max_iter: int = 1000,
                   margin: float = 1.0,
                   normalize_method: str = 'standard',
                   metric: Union[str, Callable] = 'mse') -> Dict:
    """
    Fit a perceptron model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,) with values in {-1, 1}
    learning_rate : float
        Learning rate for weight updates
    max_iter : int
        Maximum number of iterations
    margin : float
        Margin for classification
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Metric to evaluate performance

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, -1, 1])
    >>> result = perceptron_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize weights with zeros
    n_features = X.shape[1]
    weights = np.zeros(n_features)

    # Normalize data
    X_normalized = _normalize_data(X, normalize_method)

    # Add bias term
    X_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]
    y = y.astype(float)

    # Training loop
    for _ in range(max_iter):
        weights = _update_weights(X_bias, y, weights, learning_rate, margin)

    # Compute predictions
    y_pred = np.sign(np.dot(X_bias, weights))

    # Calculate metrics
    loss = _compute_loss(y, y_pred, metric)

    return {
        'result': weights,
        'metrics': {'loss': loss},
        'params_used': {
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'margin': margin,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

################################################################################
# logistic_regression
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def logistic_regression_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = 'logloss',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    penalty: float = 1.0,
    custom_metric: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit a logistic regression model to the given data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, 1).
    normalizer : Optional[Callable]
        Function to normalize the input features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate the model. Can be 'logloss', 'accuracy', or a custom callable.
    solver : str
        Solver to use. Options are 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str]
        Type of regularization. Options are 'l1', 'l2', or None.
    tol : float
        Tolerance for stopping criteria. Default is 1e-4.
    max_iter : int
        Maximum number of iterations. Default is 1000.
    learning_rate : float
        Learning rate for gradient descent. Default is 0.01.
    penalty : float
        Regularization strength. Default is 1.0.
    custom_metric : Optional[Callable]
        Custom metric function. Default is None.
    verbose : bool
        Whether to print progress information. Default is False.

    Returns:
    --------
    Dict
        A dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize parameters
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    # Choose solver
    if solver == 'gradient_descent':
        weights, bias = _gradient_descent(
            X, y, weights, bias,
            learning_rate=learning_rate,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization,
            penalty=penalty
        )
    elif solver == 'newton':
        weights, bias = _newton_method(
            X, y, weights, bias,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization,
            penalty=penalty
        )
    elif solver == 'coordinate_descent':
        weights, bias = _coordinate_descent(
            X, y, weights, bias,
            tol=tol,
            max_iter=max_iter,
            regularization=regularization,
            penalty=penalty
        )
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    metrics = _calculate_metrics(X, y, weights, bias, metric, custom_metric)

    # Prepare output
    result = {
        'weights': weights,
        'bias': bias,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate,
            'penalty': penalty
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.

    Raises:
    -------
    ValueError
        If the inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    penalty: float
) -> tuple:
    """
    Perform gradient descent to optimize the logistic regression model.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    weights : np.ndarray
        Initial weights.
    bias : float
        Initial bias.
    learning_rate : float
        Learning rate.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    regularization : Optional[str]
        Type of regularization.
    penalty : float
        Regularization strength.

    Returns:
    --------
    tuple
        Optimized weights and bias.
    """
    n_samples = X.shape[0]
    for _ in range(max_iter):
        # Compute predictions
        z = np.dot(X, weights) + bias
        y_pred = _sigmoid(z)

        # Compute gradients
        gradient_weights = np.dot(X.T, (y_pred - y)) / n_samples
        gradient_bias = np.mean(y_pred - y)

        # Apply regularization if specified
        if regularization == 'l1':
            gradient_weights += penalty * np.sign(weights)
        elif regularization == 'l2':
            gradient_weights += 2 * penalty * weights

        # Update parameters
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

        # Check for convergence
        if np.linalg.norm(gradient_weights) < tol and abs(gradient_bias) < tol:
            break

    return weights, bias

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    penalty: float
) -> tuple:
    """
    Perform Newton's method to optimize the logistic regression model.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    weights : np.ndarray
        Initial weights.
    bias : float
        Initial bias.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    regularization : Optional[str]
        Type of regularization.
    penalty : float
        Regularization strength.

    Returns:
    --------
    tuple
        Optimized weights and bias.
    """
    n_samples = X.shape[0]
    for _ in range(max_iter):
        # Compute predictions
        z = np.dot(X, weights) + bias
        y_pred = _sigmoid(z)

        # Compute Hessian and gradient
        hessian = np.dot(X.T * (y_pred * (1 - y_pred)), X) / n_samples
        gradient_weights = np.dot(X.T, (y_pred - y)) / n_samples
        gradient_bias = np.mean(y_pred - y)

        # Apply regularization if specified
        if regularization == 'l1':
            gradient_weights += penalty * np.sign(weights)
        elif regularization == 'l2':
            hessian += 2 * penalty * np.eye(X.shape[1])
            gradient_weights += 2 * penalty * weights

        # Update parameters
        delta_weights = np.linalg.solve(hessian, -gradient_weights)
        weights += delta_weights
        bias -= gradient_bias

        # Check for convergence
        if np.linalg.norm(delta_weights) < tol and abs(gradient_bias) < tol:
            break

    return weights, bias

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    penalty: float
) -> tuple:
    """
    Perform coordinate descent to optimize the logistic regression model.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    weights : np.ndarray
        Initial weights.
    bias : float
        Initial bias.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    regularization : Optional[str]
        Type of regularization.
    penalty : float
        Regularization strength.

    Returns:
    --------
    tuple
        Optimized weights and bias.
    """
    n_samples = X.shape[0]
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            # Compute predictions
            z = np.dot(X, weights) + bias
            y_pred = _sigmoid(z)

            # Compute gradient for the i-th feature
            gradient_i = np.dot(X[:, i], (y_pred - y)) / n_samples

            # Apply regularization if specified
            if regularization == 'l1':
                gradient_i += penalty * np.sign(weights[i])
            elif regularization == 'l2':
                gradient_i += 2 * penalty * weights[i]

            # Update the i-th weight
            weights[i] -= gradient_i

        # Compute bias gradient and update
        gradient_bias = np.mean(y_pred - y)
        bias -= gradient_bias

        # Check for convergence
        if np.linalg.norm(gradient_i) < tol and abs(gradient_bias) < tol:
            break

    return weights, bias

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.

    Parameters:
    -----------
    z : np.ndarray
        Input values.

    Returns:
    --------
    np.ndarray
        Sigmoid of the input values.
    """
    return 1 / (1 + np.exp(-z))

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """
    Calculate the metrics for the logistic regression model.

    Parameters:
    -----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    weights : np.ndarray
        Model weights.
    bias : float
        Model bias.
    metric : Union[str, Callable]
        Metric to evaluate the model.
    custom_metric : Optional[Callable]
        Custom metric function.

    Returns:
    --------
    Dict
        A dictionary containing the calculated metrics.
    """
    z = np.dot(X, weights) + bias
    y_pred = _sigmoid(z)

    metrics = {}
    if metric == 'logloss':
        metrics['logloss'] = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    elif metric == 'accuracy':
        y_pred_class = (y_pred > 0.5).astype(int)
        metrics['accuracy'] = np.mean(y_pred_class == y)
    elif callable(metric):
        metrics['custom_metric'] = metric(y, y_pred)

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y, y_pred)

    return metrics

################################################################################
# svm_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_kernel(X: np.ndarray, y: np.ndarray, kernel_func: Callable) -> np.ndarray:
    """Compute kernel matrix."""
    return np.array([[kernel_func(x, y_i) for y_i in y] for x in X])

def closed_form_solver(X: np.ndarray, y: np.ndarray,
                      penalty: str = 'none',
                      C: float = 1.0) -> np.ndarray:
    """Solve SVM using closed form solution."""
    if penalty == 'none':
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    elif penalty == 'l1':
        # Using coordinate descent for L1 penalty
        raise NotImplementedError("L1 penalty not implemented in closed form")
    elif penalty == 'l2':
        return np.linalg.inv(X.T @ X + C * np.eye(X.shape[1])) @ X.T @ y
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

def gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                           penalty: str = 'none',
                           C: float = 1.0,
                           learning_rate: float = 0.01,
                           max_iter: int = 1000) -> np.ndarray:
    """Solve SVM using gradient descent."""
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        gradient = -2 * X.T @ (y - X @ w)
        if penalty == 'l1':
            gradient += C * np.sign(w)
        elif penalty == 'l2':
            gradient += 2 * C * w
        w -= learning_rate * gradient
    return w

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    return {name: func(y_true, y_pred) for name, func in metric_funcs.items()}

def svm_lineaire_fit(X: np.ndarray,
                    y: np.ndarray,
                    normalize_method: str = 'standard',
                    penalty: str = 'none',
                    C: float = 1.0,
                    solver: str = 'closed_form',
                    metric_funcs: Optional[Dict[str, Callable]] = None,
                    kernel_func: Optional[Callable] = None) -> Dict:
    """
    Fit a linear SVM model.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalize_method: Normalization method ('none', 'standard', 'minmax', 'robust')
    - penalty: Penalty type ('none', 'l1', 'l2')
    - C: Regularization strength
    - solver: Solver type ('closed_form', 'gradient_descent')
    - metric_funcs: Dictionary of metric functions to compute
    - kernel_func: Custom kernel function

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_normalized = normalize_data(X, normalize_method)

    # Use kernel if provided
    if kernel_func is not None:
        X_normalized = compute_kernel(X_normalized, y, kernel_func)

    # Solve for weights
    if solver == 'closed_form':
        w = closed_form_solver(X_normalized, y, penalty, C)
    elif solver == 'gradient_descent':
        w = gradient_descent_solver(X_normalized, y, penalty, C)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions
    y_pred = X_normalized @ w

    # Compute metrics if provided
    metrics = {}
    warnings = []
    if metric_funcs is not None:
        try:
            metrics = compute_metrics(y, y_pred, metric_funcs)
        except Exception as e:
            warnings.append(f"Metric computation failed: {str(e)}")

    return {
        'result': {'weights': w},
        'metrics': metrics,
        'params_used': {
            'normalize_method': normalize_method,
            'penalty': penalty,
            'C': C,
            'solver': solver
        },
        'warnings': warnings
    }

# Example usage:
"""
from sklearn.metrics import mean_squared_error, r2_score

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

result = svm_lineaire_fit(
    X,
    y,
    normalize_method='standard',
    penalty='l2',
    C=1.0,
    solver='closed_form',
    metric_funcs={
        'mse': mean_squared_error,
        'r2': r2_score
    }
)
"""

################################################################################
# decision_boundary
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def decision_boundary_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'closed_form',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'accuracy',
    distance: str = 'euclidean',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    penalty: float = 1.0,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the decision boundary for linear classification.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,) or (n_samples, n_classes).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the input features. Default is None.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
        Default is 'closed_form'.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model ('accuracy', 'mse', 'mae', 'r2', 'logloss').
        Default is 'accuracy'.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski').
        Default is 'euclidean'.
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet'). Default is None.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for gradient-based solvers. Default is 0.01.
    penalty : float, optional
        Regularization penalty strength. Default is 1.0.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Computed parameters (weights and bias).
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used during fitting.
        - "warnings": List of warnings encountered.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 0])
    >>> result = decision_boundary_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose solver
    if solver == 'closed_form':
        weights = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        weights = _solve_gradient_descent(X_normalized, y, tol, max_iter, learning_rate)
    elif solver == 'newton':
        weights = _solve_newton(X_normalized, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        weights = _solve_coordinate_descent(X_normalized, y, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        weights = _apply_regularization(weights, X_normalized, y, regularization, penalty)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, weights, metric, custom_metric)

    # Prepare output
    result = {
        "result": {"weights": weights},
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "solver": solver,
            "metric": metric if isinstance(metric, str) else None,
            "distance": distance,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "penalty": penalty
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the input features."""
    if normalizer is not None:
        X_normalized = normalizer(X)
    else:
        X_normalized = X
    return X_normalized

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for weights using closed-form solution."""
    X_with_bias = _add_bias(X)
    weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return weights

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    learning_rate: float
) -> np.ndarray:
    """Solve for weights using gradient descent."""
    X_with_bias = _add_bias(X)
    weights = np.zeros(X_with_bias.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X_with_bias.T @ (X_with_bias @ weights - y) / len(y)
        new_weights = weights - learning_rate * gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return weights

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for weights using Newton's method."""
    X_with_bias = _add_bias(X)
    weights = np.zeros(X_with_bias.shape[1])
    for _ in range(max_iter):
        gradient = 2 * X_with_bias.T @ (X_with_bias @ weights - y) / len(y)
        hessian = 2 * X_with_bias.T @ X_with_bias / len(y)
        new_weights = weights - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return weights

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve for weights using coordinate descent."""
    X_with_bias = _add_bias(X)
    weights = np.zeros(X_with_bias.shape[1])
    for _ in range(max_iter):
        for i in range(len(weights)):
            X_i = X_with_bias[:, i]
            residuals = y - (X_with_bias @ weights)
            numerator = X_i.T @ residuals
            denominator = X_i.T @ X_i
            new_weight = numerator / denominator if denominator != 0 else weights[i]
            weights[i] = new_weight
        if np.linalg.norm(weights - np.roll(weights, 1)) < tol:
            break
    return weights

def _apply_regularization(
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str,
    penalty: float
) -> np.ndarray:
    """Apply regularization to the weights."""
    X_with_bias = _add_bias(X)
    if regularization == 'l1':
        weights -= penalty * np.sign(weights)
    elif regularization == 'l2':
        weights -= 2 * penalty * weights
    elif regularization == 'elasticnet':
        weights -= (penalty * np.sign(weights) + 2 * penalty * weights)
    return weights

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    X_with_bias = _add_bias(X)
    predictions = X_with_bias @ weights

    metrics_dict = {}
    if isinstance(metric, str):
        if metric == 'accuracy':
            metrics_dict['accuracy'] = np.mean((predictions > 0.5) == y)
        elif metric == 'mse':
            metrics_dict['mse'] = np.mean((predictions - y) ** 2)
        elif metric == 'mae':
            metrics_dict['mae'] = np.mean(np.abs(predictions - y))
        elif metric == 'r2':
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics_dict['r2'] = 1 - (ss_res / ss_tot)
        elif metric == 'logloss':
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            metrics_dict['logloss'] = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    else:
        metrics_dict['custom_metric'] = metric(predictions, y)

    if custom_metric is not None:
        metrics_dict['custom_metric'] = custom_metric(predictions, y)

    return metrics_dict

def _add_bias(X: np.ndarray) -> np.ndarray:
    """Add a bias term to the input features."""
    return np.hstack([X, np.ones((X.shape[0], 1))])

################################################################################
# hyperplan_separateur
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hyperplan_separateur_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Fit a linear separator hyperplane to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,) or (n_samples, n_outputs).
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    distance_metric : str or callable
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularisation : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function for evaluation.

    Returns:
    --------
    Dict containing:
        - 'result': Fitted hyperplane parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([0, 1])
    >>> result = hyperplan_separateur_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalisation(X, normalisation)

    # Choose distance metric
    distance_func = _get_distance_metric(distance_metric, **kwargs)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularisation if specified
    if regularisation:
        params = _apply_regularisation(params, X_normalized, y, regularisation)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, custom_metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularisation': regularisation,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim not in (1, 2):
        raise ValueError("y must be a 1D or 2D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalisation(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalisation to the data."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

def _get_distance_metric(metric: Union[str, Callable], **kwargs) -> Callable:
    """Get distance metric function."""
    if callable(metric):
        return metric
    elif metric == 'euclidean':
        return lambda a, b: np.linalg.norm(a - b)
    elif metric == 'manhattan':
        return lambda a, b: np.sum(np.abs(a - b))
    elif metric == 'cosine':
        return lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif metric == 'minkowski':
        p = kwargs.get('p', 2)
        return lambda a, b: np.sum(np.abs(a - b) ** p) ** (1 / p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for hyperplane parameters using closed-form solution."""
    X_t = np.transpose(X)
    return np.linalg.inv(X_t @ X) @ X_t @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve for hyperplane parameters using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve for hyperplane parameters using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y) / len(y)
        hessian = 2 * X.T @ X / len(y)
        new_params = params - np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve for hyperplane parameters using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - np.dot(X, params) + params[j] * X_j
            params[j] = np.sum(X_j * residual) / np.sum(X_j ** 2)

        if np.linalg.norm(params - np.zeros_like(params)) < tol:
            break

    return params

def _apply_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply regularisation to the parameters."""
    if method == 'l1':
        return _apply_l1_regularisation(params, X, y)
    elif method == 'l2':
        return _apply_l2_regularisation(params, X, y)
    elif method == 'elasticnet':
        return _apply_elasticnet_regularisation(params, X, y)
    else:
        raise ValueError(f"Unknown regularisation method: {method}")

def _apply_l1_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L1 regularisation."""
    alpha = 0.1
    gradient = 2 * X.T @ (X @ params - y) / len(y)
    regularisation = alpha * np.sign(params)
    return params - gradient - regularisation

def _apply_l2_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply L2 regularisation."""
    alpha = 0.1
    gradient = 2 * X.T @ (X @ params - y) / len(y)
    regularisation = 2 * alpha * params
    return params - gradient - regularisation

def _apply_elasticnet_regularisation(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Apply elasticnet regularisation."""
    alpha = 0.1
    l1_ratio = 0.5
    gradient = 2 * X.T @ (X @ params - y) / len(y)
    l1_penalty = alpha * l1_ratio * np.sign(params)
    l2_penalty = 2 * alpha * (1 - l1_ratio) * params
    return params - gradient - l1_penalty - l2_penalty

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute metrics for the fitted hyperplane."""
    y_pred = X @ params
    metrics = {
        'mse': np.mean((y - y_pred) ** 2),
        'mae': np.mean(np.abs(y - y_pred)),
        'r2': 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    }

    if custom_metric:
        metrics['custom'] = custom_metric(y, y_pred)

    return metrics

################################################################################
# fonction_perte_logistique
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _logistic_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute logistic loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _gradient_descent(X: np.ndarray, y: np.ndarray,
                      learning_rate: float = 0.01,
                      max_iter: int = 1000,
                      tol: float = 1e-4) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        linear_model = np.dot(X, weights) + bias
        y_pred = 1 / (1 + np.exp(-linear_model))

        dw = np.dot(X.T, (y_pred - y)) / n_samples
        db = np.mean(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        if np.linalg.norm(dw) < tol and np.abs(db) < tol:
            break

    return np.append(weights, bias)

def _newton_method(X: np.ndarray, y: np.ndarray,
                   max_iter: int = 100,
                   tol: float = 1e-4) -> np.ndarray:
    """Perform Newton's method optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        linear_model = np.dot(X, weights) + bias
        y_pred = 1 / (1 + np.exp(-linear_model))

        gradient = np.dot(X.T, (y_pred - y)) / n_samples
        hessian = np.dot(X.T * (y_pred * (1 - y_pred)), X) / n_samples

        update = np.linalg.solve(hessian, -gradient)
        weights += update[:-1]
        bias += update[-1]

        if np.linalg.norm(gradient) < tol:
            break

    return np.append(weights, bias)

def fonction_perte_logistique_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = 'standard',
    solver: str = 'gradient_descent',
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit logistic regression model and compute loss.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str, optional
        Optimization algorithm ('gradient_descent', 'newton')
    learning_rate : float, optional
        Learning rate for gradient descent
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping criteria
    custom_metric : callable, optional
        Custom metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': fitted parameters
        - 'metrics': computed metrics
        - 'params_used': used parameters
        - 'warnings': any warnings

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([0, 1])
    >>> result = fonction_perte_logistique_fit(X, y)
    """
    _validate_inputs(X, y)

    X_normalized = _normalize_data(X, normalization)
    params_used = {
        'normalization': normalization,
        'solver': solver,
        'learning_rate': learning_rate,
        'max_iter': max_iter,
        'tol': tol
    }

    if solver == 'gradient_descent':
        weights = _gradient_descent(X_normalized, y,
                                   learning_rate=learning_rate,
                                   max_iter=max_iter,
                                   tol=tol)
    elif solver == 'newton':
        weights = _newton_method(X_normalized, y,
                                max_iter=max_iter,
                                tol=tol)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    linear_model = np.dot(X_normalized, weights[:-1]) + weights[-1]
    y_pred = 1 / (1 + np.exp(-linear_model))

    metrics = {
        'log_loss': _logistic_loss(y, y_pred)
    }

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y, y_pred)

    return {
        'result': {'weights': weights},
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

################################################################################
# gradient_descent
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for gradient descent."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / ((max_val - min_val) + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: str = 'mse',
                  custom_metric: Optional[Callable] = None) -> float:
    """Compute specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def gradient_descent_step(X: np.ndarray, y: np.ndarray,
                         weights: np.ndarray,
                         learning_rate: float,
                         penalty: str = 'none',
                         alpha: float = 1.0) -> np.ndarray:
    """Perform a single gradient descent step."""
    residuals = y - X @ weights
    gradient = -X.T @ residuals / len(y)

    if penalty == 'l1':
        gradient += alpha * np.sign(weights)
    elif penalty == 'l2':
        gradient += 2 * alpha * weights
    elif penalty == 'elasticnet':
        gradient += alpha * (np.sign(weights) + 2 * weights)

    return weights - learning_rate * gradient

def gradient_descent_fit(X: np.ndarray, y: np.ndarray,
                        learning_rate: float = 0.01,
                        n_iter: int = 1000,
                        tol: float = 1e-4,
                        penalty: str = 'none',
                        alpha: float = 1.0,
                        normalize_method: str = 'standard',
                        metric: str = 'mse',
                        custom_metric: Optional[Callable] = None,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    Perform gradient descent for linear classification.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - learning_rate: Step size for gradient descent
    - n_iter: Maximum number of iterations
    - tol: Tolerance for stopping criterion
    - penalty: Type of regularization ('none', 'l1', 'l2', 'elasticnet')
    - alpha: Regularization strength
    - normalize_method: Data normalization method
    - metric: Evaluation metric
    - custom_metric: Custom evaluation function
    - verbose: Whether to print progress

    Returns:
    Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_normalized = normalize_data(X, method=normalize_method)
    n_features = X.shape[1]

    # Initialize weights
    weights = np.zeros(n_features)

    # Store metrics history
    metric_history = []
    best_weights = None
    best_metric = float('inf')

    for i in range(n_iter):
        # Perform gradient descent step
        weights = gradient_descent_step(X_normalized, y, weights,
                                      learning_rate, penalty, alpha)

        # Compute predictions and metric
        y_pred = X_normalized @ weights
        current_metric = compute_metric(y, y_pred, metric, custom_metric)
        metric_history.append(current_metric)

        # Check for early stopping
        if i > 0 and abs(metric_history[-2] - current_metric) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        # Track best weights
        if current_metric < best_metric:
            best_metric = current_metric
            best_weights = weights.copy()

    # Prepare results
    result = {
        'weights': best_weights,
        'metric_history': metric_history,
        'final_metric': current_metric
    }

    metrics = {
        'metric_name': metric,
        'final_value': current_metric
    }

    params_used = {
        'learning_rate': learning_rate,
        'n_iter': i + 1,
        'tol': tol,
        'penalty': penalty,
        'alpha': alpha,
        'normalize_method': normalize_method
    }

    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
"""
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 1])
result = gradient_descent_fit(X, y)
"""

################################################################################
# regularisation_l1
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_l1_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'coordinate_descent',
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a linear classification model with L1 regularization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    alpha : float, optional
        Regularization strength.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criteria.
    random_state : int or None, optional
        Random seed for reproducibility.
    custom_metric : callable, optional
        Custom metric function.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings generated.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> result = regularisation_l1_fit(X, y, normalisation='standard', solver='coordinate_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalisation(X, normalisation)

    # Initialize parameters
    n_features = X_normalized.shape[1]
    coef_ = np.zeros(n_features)
    intercept_ = 0.0

    # Choose solver
    if solver == 'coordinate_descent':
        coef_, intercept_ = _coordinate_descent_solver(
            X_normalized, y, alpha=alpha, max_iter=max_iter, tol=tol
        )
    elif solver == 'gradient_descent':
        coef_, intercept_ = _gradient_descent_solver(
            X_normalized, y, alpha=alpha, max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(
        X_normalized, y, coef_, intercept_,
        metric=metric, custom_metric=custom_metric
    )

    # Prepare output
    result = {
        'result': {'coef': coef_, 'intercept': intercept_},
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalisation(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the feature matrix."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unsupported normalisation method: {method}")

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple:
    """Coordinate descent solver for L1 regularization."""
    n_samples, n_features = X.shape
    coef_ = np.zeros(n_features)
    intercept_ = 0.0

    for _ in range(max_iter):
        old_coef = coef_.copy()

        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - intercept_ - np.dot(X, coef_) + coef_[j] * X_j

            if np.all(X_j == 0):
                continue

            corr = np.dot(X_j, residuals)
            rss = np.sum(residuals**2)

            if corr < -alpha / 2:
                coef_[j] = (corr + alpha / 2) / np.sum(X_j**2)
            elif corr > alpha / 2:
                coef_[j] = (corr - alpha / 2) / np.sum(X_j**2)
            else:
                coef_[j] = 0

        intercept_ = np.mean(y - np.dot(X, coef_))

        if np.linalg.norm(coef_ - old_coef) < tol:
            break

    return coef_, intercept_

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> tuple:
    """Gradient descent solver for L1 regularization."""
    n_samples, n_features = X.shape
    coef_ = np.zeros(n_features)
    intercept_ = 0.0
    learning_rate = 0.1

    for _ in range(max_iter):
        old_coef = coef_.copy()

        # Compute gradients
        residuals = y - intercept_ - np.dot(X, coef_)
        grad_coef = -np.dot(X.T, residuals) / n_samples
        grad_intercept = -np.mean(residuals)

        # Update parameters with L1 regularization
        coef_ -= learning_rate * (grad_coef + alpha * np.sign(coef_))
        intercept_ -= learning_rate * grad_intercept

        if np.linalg.norm(coef_ - old_coef) < tol:
            break

    return coef_, intercept_

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute specified metrics."""
    y_pred = np.dot(X, coef) + intercept
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((y - y_pred)**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    elif metric == 'logloss':
        y_pred = 1 / (1 + np.exp(-y_pred))
        metrics['logloss'] = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
    elif callable(metric):
        metrics['custom'] = metric(y, y_pred)
    if custom_metric is not None:
        metrics['custom'] = custom_metric(y, y_pred)

    return metrics

################################################################################
# regularisation_l2
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularisation_l2_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit a linear classifier with L2 regularization.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate ('mse', 'mae', 'r2', 'logloss') or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton')
    alpha : float, optional
        Regularization strength (must be >= 0)
    max_iter : int, optional
        Maximum number of iterations for iterative solvers
    tol : float, optional
        Tolerance for stopping criteria
    random_state : int or None, optional
        Random seed for reproducibility
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = regularisation_l2_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if requested
    X_normalized = _apply_normalization(X, normalisation)

    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(X_normalized.shape[0]), X_normalized])

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_with_intercept, y, alpha)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            X_with_intercept, y, alpha,
            max_iter=max_iter, tol=tol,
            random_state=random_state
        )
    elif solver == 'newton':
        params = _solve_newton(
            X_with_intercept, y, alpha,
            max_iter=max_iter, tol=tol
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    y_pred = X_with_intercept @ params
    metrics = _calculate_metrics(y, y_pred, metric=metric, custom_metric=custom_metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'solver': solver,
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to features."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve using closed-form solution with L2 regularization."""
    n_features = X.shape[1]
    identity = np.eye(n_features)
    identity[-1, -1] = 0  # Don't regularize the intercept
    return np.linalg.inv(X.T @ X + alpha * identity) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Solve using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)

    n_features = X.shape[1]
    params = np.random.randn(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = (2/n_features) * X.T @ (X @ params - y)
        gradient[-1] = (2/n_features) * X.T[-1] @ (X @ params - y)  # No regularization for intercept
        gradient[1:-1] += 2 * alpha * params[1:-1]

        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break

        params = new_params

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        residuals = X @ params - y
        gradient = (2/n_features) * X.T @ residuals
        gradient[-1] = (2/n_features) * X.T[-1] @ residuals  # No regularization for intercept

        hessian = (2/n_features) * X.T @ X
        hessian[1:-1, 1:-1] += 2 * alpha * np.eye(n_features-2)

        delta = np.linalg.solve(hessian, -gradient)
        params += delta

        if np.linalg.norm(delta) < tol:
            break

    return params

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate specified metrics."""
    metrics = {}

    if metric == 'mse' or custom_metric is None:
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    if metric == 'mae' or custom_metric is None:
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    if metric == 'r2' or custom_metric is None:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(y_true, y_pred)

    return metrics

################################################################################
# bias_variance_tradeoff
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def bias_variance_tradeoff_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    metric: Union[str, Callable],
    n_splits: int = 5,
    normalizer: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute the bias-variance tradeoff for a linear classification model.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model : Callable
        A callable that implements the linear classification model.
    metric : Union[str, Callable]
        The metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    n_splits : int, optional
        Number of splits for cross-validation. Default is 5.
    normalizer : Optional[Callable], optional
        A callable that normalizes the input features. Default is None.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        A dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = bias_variance_tradeoff_fit(X, y, LinearRegression(), 'mse')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize results dictionary
    results = {
        "result": None,
        "metrics": {},
        "params_used": {
            "n_splits": n_splits,
            "normalizer": normalizer.__name__ if normalizer else None
        },
        "warnings": []
    }

    # Perform cross-validation
    cv_results = _cross_validate(X, y, model, metric, n_splits, random_state)

    # Compute bias and variance
    bias, variance = _compute_bias_variance(cv_results)

    # Store results
    results["result"] = {
        "bias": bias,
        "variance": variance
    }

    # Compute and store metrics
    results["metrics"]["bias"] = bias
    results["metrics"]["variance"] = variance

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate the input features and target values.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    metric: Union[str, Callable],
    n_splits: int,
    random_state: Optional[int]
) -> Dict[str, np.ndarray]:
    """
    Perform cross-validation and return the results.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model : Callable
        A callable that implements the linear classification model.
    metric : Union[str, Callable]
        The metric to evaluate the model. Can be 'mse', 'mae', 'r2', or a custom callable.
    n_splits : int
        Number of splits for cross-validation.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the predictions and true values for each split.
    """
    # Initialize results dictionary
    cv_results = {
        "predictions": [],
        "true_values": []
    }

    # Perform cross-validation
    for i in range(n_splits):
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = _train_test_split(X, y, random_state, i)

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_val)

        # Store results
        cv_results["predictions"].append(y_pred)
        cv_results["true_values"].append(y_val)

    return cv_results

def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    random_state: Optional[int],
    split_index: int
) -> tuple:
    """
    Split the data into training and validation sets.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    random_state : Optional[int]
        Random seed for reproducibility.
    split_index : int
        Index of the current split.

    Returns
    -------
    tuple
        A tuple containing the training and validation sets.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    val_indices = np.arange(split_index * (n_samples // n_splits), (split_index + 1) * (n_samples // n_splits))
    train_indices = np.setdiff1d(np.arange(n_samples), val_indices)

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

def _compute_bias_variance(
    cv_results: Dict[str, np.ndarray]
) -> tuple:
    """
    Compute the bias and variance from cross-validation results.

    Parameters
    ----------
    cv_results : Dict[str, np.ndarray]
        A dictionary containing the predictions and true values for each split.

    Returns
    -------
    tuple
        A tuple containing the bias and variance.
    """
    predictions = np.array(cv_results["predictions"])
    true_values = np.array(cv_results["true_values"])

    # Compute bias (average error)
    bias = np.mean((predictions - true_values) ** 2)

    # Compute variance (average squared difference between predictions)
    variance = np.mean((predictions - np.mean(predictions, axis=0)) ** 2)

    return bias, variance

################################################################################
# marge_separation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def marge_separation_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Calculate the separation margin for linear classification.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,) with values in {-1, 1}
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize features. If None, no normalization is applied.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton')
    regularization : str, optional
        Regularization type ('l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Callable[[np.ndarray, np.ndarray], float], optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _apply_normalization(X, normalizer)

    # Get distance function
    distance_func = _get_distance_function(distance_metric, **kwargs)

    # Solve for parameters
    params = _solve_linear_classifier(
        X_normalized, y,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate separation margin
    margin = _calculate_separation_margin(X_normalized, y, params['weights'])

    # Calculate metrics
    metrics = _calculate_metrics(
        X_normalized, y,
        params['weights'],
        custom_metric=custom_metric
    )

    return {
        'result': {'margin': margin},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(X_normalized, y)
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y must contain only -1 and 1")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _get_distance_function(
    metric: str,
    **kwargs
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on metric name."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        p = kwargs.get('p', 2)
        return lambda x, y: np.sum(np.abs(x - y)**p)**(1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _solve_linear_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, np.ndarray]:
    """Solve for linear classifier parameters."""
    if solver == 'closed_form':
        return _solve_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(X, y, tol, max_iter)
    elif solver == 'newton':
        return _solve_newton(X, y, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str]
) -> Dict[str, np.ndarray]:
    """Solve linear classifier using closed form solution."""
    if regularization is None:
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
    elif regularization == 'l2':
        alpha = 1.0
        XtX = np.dot(X.T, X) + alpha * np.eye(X.shape[1])
        Xty = np.dot(X.T, y)
    else:
        raise ValueError(f"Unsupported regularization: {regularization}")

    weights = np.linalg.solve(XtX, Xty)
    return {'weights': weights}

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, np.ndarray]:
    """Solve linear classifier using gradient descent."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = -2 * np.dot(X.T, y) + 2 * np.dot(X.T, X @ weights)
        new_weights = weights - learning_rate * gradient

        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return {'weights': weights}

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int
) -> Dict[str, np.ndarray]:
    """Solve linear classifier using Newton's method."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    hessian = 2 * np.dot(X.T, X)

    for _ in range(max_iter):
        gradient = -2 * np.dot(X.T, y) + 2 * np.dot(X.T, X @ weights)
        new_weights = weights - np.linalg.solve(hessian, gradient)

        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return {'weights': weights}

def _calculate_separation_margin(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray
) -> float:
    """Calculate the separation margin."""
    scores = X @ weights
    correct_class_scores = np.abs(scores[y == 1])
    incorrect_class_scores = scores[y == -1]

    if len(correct_class_scores) == 0 or len(incorrect_class_scores) == 0:
        return 0.0

    min_correct = np.min(correct_class_scores)
    max_incorrect = np.max(incorrect_class_scores)

    if min_correct <= 0 or max_incorrect >= 0:
        return 0.0

    margin = min_correct - max_incorrect
    return margin / np.linalg.norm(weights)

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate classification metrics."""
    scores = X @ weights
    predictions = np.sign(scores)
    accuracy = np.mean(predictions == y)

    metrics = {'accuracy': float(accuracy)}

    if custom_metric is not None:
        metrics['custom'] = custom_metric(scores, y)

    return metrics

def _check_warnings(
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, str]:
    """Check for potential warnings."""
    warnings = {}

    if np.any(np.isnan(X)):
        warnings['nan_in_features'] = 'NaN values found in features'
    if np.any(np.isinf(X)):
        warnings['inf_in_features'] = 'Infinite values found in features'
    if np.any(np.isnan(y)):
        warnings['nan_in_target'] = 'NaN values found in target'
    if np.any(np.isinf(y)):
        warnings['inf_in_target'] = 'Infinite values found in target'

    return warnings

################################################################################
# kernel_lineaire
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def kernel_lineaire_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: x @ y.T,
    normalisation: str = 'standard',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularisation: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Fit a linear kernel classifier with configurable parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,) or (n_samples, n_classes).
    kernel : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Kernel function to compute the kernel matrix.
    normalisation : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the model: 'mse', 'mae', 'r2', or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', or 'newton'.
    regularisation : Optional[str]
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    learning_rate : float
        Learning rate for gradient-based solvers.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": Fitted model parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": List of warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = kernel_lineaire_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalisation(X, normalisation)

    # Compute kernel matrix
    K = kernel(X_normalized, X_normalized)

    # Solve for parameters based on solver choice
    if solver == 'closed_form':
        params = _solve_closed_form(K, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(K, y, tol=tol, max_iter=max_iter, learning_rate=learning_rate)
    elif solver == 'newton':
        params = _solve_newton(K, y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularisation if specified
    if regularisation is not None:
        params = _apply_regularisation(params, regularisation)

    # Compute metrics
    metrics = _compute_metrics(y, params, metric=metric, custom_metric=custom_metric)

    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "kernel": kernel.__name__ if hasattr(kernel, '__name__') else str(kernel),
            "normalisation": normalisation,
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver,
            "regularisation": regularisation,
            "tol": tol,
            "max_iter": max_iter,
            "learning_rate": learning_rate
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalisation(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalisation to the input data."""
    if method == 'none':
        return X
    elif method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

def _solve_closed_form(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for parameters using closed-form solution."""
    return np.linalg.pinv(K) @ y

def _solve_gradient_descent(
    K: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Solve for parameters using gradient descent."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    for _ in range(max_iter):
        gradient = 2 * K @ (K @ params - y) / n_samples
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _solve_newton(
    K: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Solve for parameters using Newton's method."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    for _ in range(max_iter):
        gradient = 2 * K @ (K @ params - y) / n_samples
        hessian = 2 * K.T @ K / n_samples
        params -= np.linalg.pinv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _apply_regularisation(params: np.ndarray, method: str) -> np.ndarray:
    """Apply specified regularisation to the parameters."""
    if method == 'none':
        return params
    elif method == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - 1, 0)
    elif method == 'l2':
        return params / (1 + np.linalg.norm(params))
    elif method == 'elasticnet':
        l1_params = np.sign(params) * np.maximum(np.abs(params) - 0.5, 0)
        l2_params = params / (1 + np.linalg.norm(params))
        return 0.5 * l1_params + 0.5 * l2_params
    else:
        raise ValueError(f"Unknown regularisation method: {method}")

def _compute_metrics(
    y_true: np.ndarray,
    params: np.ndarray,
    *,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Compute specified metrics."""
    y_pred = params @ X  # Assuming X is available in the scope

    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((y_true - y_pred) ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(y_true - y_pred))}
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return {'r2': 1 - ss_res / (ss_tot + 1e-8)}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif callable(metric):
        return {'custom_metric': metric(y_true, y_pred)}
    else:
        raise TypeError("metric must be a string or callable")

    if custom_metric is not None:
        return {'custom_metric': custom_metric(y_true, y_pred)}

################################################################################
# fonction_activation_sigmoid
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs contain infinite values")

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid activation."""
    return 1 / (1 + np.exp(-z))

def _compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_func: str = "logloss",
    custom_loss: Optional[Callable] = None
) -> float:
    """Compute loss between true and predicted values."""
    if custom_loss is not None:
        return custom_loss(y_true, y_pred)

    if loss_func == "logloss":
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    elif loss_func == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif loss_func == "mae":
        return np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

def _normalize(
    X: np.ndarray,
    method: str = "standard"
) -> tuple[np.ndarray, Dict[str, Union[float, np.ndarray]]]:
    """Normalize input data."""
    params = {}
    if method == "none":
        return X, params
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
        params["mean"] = mean
        params["std"] = std
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
        params["min"] = min_val
        params["max"] = max_val
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - median) / iqr
        params["median"] = median
        params["iqr"] = iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return X_normalized, params

def fonction_activation_sigmoid_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalize_method: str = "standard",
    loss_func: str = "logloss",
    custom_loss: Optional[Callable] = None,
    solver: str = "gradient_descent",
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit a linear classifier with sigmoid activation.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,) in [0, 1]
    normalize_method : str
        Normalization method for features ('none', 'standard', 'minmax', 'robust')
    loss_func : str
        Loss function to optimize ('logloss', 'mse', 'mae')
    custom_loss : Callable, optional
        Custom loss function taking (y_true, y_pred) and returning a scalar
    solver : str
        Optimization algorithm ('gradient_descent', 'newton')
    learning_rate : float
        Learning rate for gradient descent
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion

    Returns
    -------
    Dict containing:
        - "weights": Estimated weights
        - "metrics": Dictionary of computed metrics
        - "params_used": Parameters used during fitting
        - "warnings": Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = fonction_activation_sigmoid_fit(X, y)
    """
    _validate_inputs(X, y)

    # Normalize data
    X_normalized, norm_params = _normalize(X, normalize_method)

    # Initialize weights
    n_features = X_normalized.shape[1]
    weights = np.zeros(n_features)

    # Optimization loop
    for _ in range(max_iter):
        z = np.dot(X_normalized, weights)
        y_pred = _sigmoid(z)

        # Compute loss
        current_loss = _compute_loss(y, y_pred, loss_func, custom_loss)

        # Gradient descent update
        if solver == "gradient_descent":
            gradient = np.dot(X_normalized.T, (y_pred - y)) / len(y)
            weights -= learning_rate * gradient

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    # Compute final metrics
    z_final = np.dot(X_normalized, weights)
    y_pred_final = _sigmoid(z_final)

    metrics = {
        "loss": current_loss,
        "accuracy": np.mean((y_pred_final > 0.5) == y)
    }

    return {
        "result": weights,
        "metrics": metrics,
        "params_used": {
            "normalize_method": normalize_method,
            "loss_func": loss_func,
            "solver": solver,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

################################################################################
# coefficient_pente
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'standard') -> tuple:
    """Normalize input data."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_norm = (X - mean_X) / std_X
        mean_y = np.mean(y)
        std_y = np.std(y)
        y_norm = (y - mean_y) / std_y
    elif normalization == 'minmax':
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        X_norm = (X - min_X) / (max_X - min_X + 1e-8)
        min_y = np.min(y)
        max_y = np.max(y)
        y_norm = (y - min_y) / (max_y - min_y + 1e-8)
    elif normalization == 'robust':
        median_X = np.median(X, axis=0)
        iqr_X = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median_X) / (iqr_X + 1e-8)
        median_y = np.median(y)
        iqr_y = np.subtract(*np.percentile(y, [75, 25]))
        y_norm = (y - median_y) / (iqr_y + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return X_norm, y_norm

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear coefficients."""
    X_tx = X.T @ X
    if np.linalg.cond(X_tx) < 1/np.finfo(float).eps:
        coeff = np.linalg.solve(X_tx, X.T @ y)
    else:
        coeff = np.linalg.pinv(X_tx) @ X.T @ y
    return coeff

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                            learning_rate: float = 0.01,
                            n_iter: int = 1000,
                            tol: float = 1e-4) -> np.ndarray:
    """Gradient descent solver for linear coefficients."""
    n_samples, n_features = X.shape
    coeff = np.zeros(n_features)
    prev_coeff = coeff.copy()
    for _ in range(n_iter):
        gradient = -2 * X.T @ (y - X @ coeff) / n_samples
        coeff -= learning_rate * gradient
        if np.linalg.norm(coeff - prev_coeff) < tol:
            break
        prev_coeff = coeff.copy()
    return coeff

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric: Union[str, Callable]) -> Dict:
    """Compute evaluation metrics."""
    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
    elif callable(metric):
        metrics['custom'] = metric(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics

def coefficient_pente_fit(X: np.ndarray, y: np.ndarray,
                         normalization: str = 'standard',
                         solver: str = 'closed_form',
                         metric: Union[str, Callable] = 'mse',
                         **solver_kwargs) -> Dict:
    """
    Fit linear coefficients (pente) with various options.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    solver : str
        Solver method ('closed_form', 'gradient_descent')
    metric : str or callable
        Evaluation metric
    solver_kwargs : dict
        Additional arguments for the solver

    Returns
    -------
    Dict containing:
        - 'result': fitted coefficients
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = coefficient_pente_fit(X, y)
    """
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Add intercept term
    X_norm = np.column_stack([np.ones(X_norm.shape[0]), X_norm])

    # Solve for coefficients
    if solver == 'closed_form':
        coeff = _closed_form_solver(X_norm, y_norm)
    elif solver == 'gradient_descent':
        coeff = _gradient_descent_solver(X_norm, y_norm, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions and metrics
    y_pred = X_norm @ coeff
    metrics = _compute_metrics(y_norm, y_pred, metric)

    return {
        'result': coeff,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'solver': solver,
            'metric': metric.__name__ if callable(metric) else metric
        },
        'warnings': []
    }

################################################################################
# intercept_ordonnee_origine
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def intercept_ordonnee_origine_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    alpha: float = 1.0,
    beta: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict[str, float], str]]:
    """
    Fit a linear classification model and compute the intercept at origin.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the features, by default None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the model, by default 'mse'.
    solver : str, optional
        Solver to use for optimization, by default 'closed_form'.
    regularization : Optional[str], optional
        Type of regularization, by default None.
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-4.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    learning_rate : float, optional
        Learning rate for gradient descent, by default 0.01.
    alpha : float, optional
        Regularization strength for L1/L2, by default 1.0.
    beta : float, optional
        Mixing parameter for elastic net, by default 1.0.
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict[str, float], str]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 1])
    >>> result = intercept_ordonnee_origine_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose solver
    if solver == 'closed_form':
        coefficients = _closed_form_solver(X_normalized, y, regularization, alpha, beta)
    elif solver == 'gradient_descent':
        coefficients = _gradient_descent_solver(
            X_normalized, y, tol, max_iter, learning_rate,
            regularization, alpha, beta, random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute intercept at origin
    intercept = _compute_intercept_at_origin(X_normalized, y, coefficients)

    # Compute metrics
    metrics = _compute_metrics(y, _predict(X_normalized, coefficients), metric)

    # Prepare output
    result = {
        'result': intercept,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'beta': beta,
            'random_state': random_state
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("X and y must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("X and y must not contain infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the feature matrix."""
    if normalizer is not None:
        X = normalizer(X)
    return X

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Solve the linear classification problem using closed-form solution."""
    if regularization is None:
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    elif regularization == 'l1':
        coefficients = _l1_regularized_solver(X, y, alpha)
    elif regularization == 'l2':
        coefficients = _l2_regularized_solver(X, y, alpha)
    elif regularization == 'elasticnet':
        coefficients = _elasticnet_regularized_solver(X, y, alpha, beta)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    return coefficients

def _l1_regularized_solver(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve the linear classification problem with L1 regularization."""
    # This is a placeholder for actual L1 solver implementation
    return np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y

def _l2_regularized_solver(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve the linear classification problem with L2 regularization."""
    # This is a placeholder for actual L2 solver implementation
    return np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y

def _elasticnet_regularized_solver(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    beta: float
) -> np.ndarray:
    """Solve the linear classification problem with elastic net regularization."""
    # This is a placeholder for actual elastic net solver implementation
    return np.linalg.inv(X.T @ X + alpha * beta * np.eye(X.shape[1])) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    tol: float,
    max_iter: int,
    learning_rate: float,
    regularization: Optional[str],
    alpha: float,
    beta: float,
    random_state: Optional[int]
) -> np.ndarray:
    """Solve the linear classification problem using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)
    coefficients = np.random.randn(X.shape[1])
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, coefficients, regularization, alpha, beta)
        coefficients -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return coefficients

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    regularization: Optional[str],
    alpha: float,
    beta: float
) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    residuals = y - _predict(X, coefficients)
    gradient = -(X.T @ residuals) / X.shape[0]
    if regularization == 'l1':
        gradient += alpha * np.sign(coefficients)
    elif regularization == 'l2':
        gradient += 2 * alpha * coefficients
    elif regularization == 'elasticnet':
        gradient += alpha * (beta * np.sign(coefficients) + (1 - beta) * 2 * coefficients)
    return gradient

def _compute_intercept_at_origin(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray
) -> float:
    """Compute the intercept at origin."""
    return y.mean() - (X @ coefficients).mean()

def _predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict the target values."""
    return X @ coefficients

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute the metrics."""
    if callable(metric):
        return {'custom_metric': metric(y_true, y_pred)}
    elif metric == 'mse':
        return {'mse': np.mean((y_true - y_pred) ** 2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y_true - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return {'r2': 1 - ss_res / ss_tot}
    elif metric == 'logloss':
        return {'logloss': -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))}
    else:
        raise ValueError(f"Unknown metric: {metric}")
