"""
Quantix – Module hyperparametres
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# learning_rate
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def learning_rate_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the optimal learning rate for a given model and dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features.
    metric : str
        Metric to optimize. Options: 'mse', 'mae', 'r2', 'logloss'.
    distance : str
        Distance metric for the solver. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str
        Solver to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    learning_rate : float
        Initial learning rate.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Choose metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Choose distance function
    distance_func = _get_distance_function(distance, custom_distance)

    # Initialize parameters
    params = {
        'learning_rate': learning_rate,
        'max_iter': max_iter,
        'tol': tol,
        'solver': solver,
        'regularization': regularization
    }

    # Choose solver function
    if solver == 'gradient_descent':
        result, metrics = _gradient_descent_solver(X_normalized, y, metric_func, distance_func, params)
    elif solver == 'newton':
        result, metrics = _newton_solver(X_normalized, y, metric_func, distance_func, params)
    elif solver == 'coordinate_descent':
        result, metrics = _coordinate_descent_solver(X_normalized, y, metric_func, distance_func, params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values.")

def _get_metric_function(metric: str, custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the input parameters."""
    if custom_metric is not None:
        return custom_metric
    if metric == 'mse':
        return _mean_squared_error
    elif metric == 'mae':
        return _mean_absolute_error
    elif metric == 'r2':
        return _r_squared
    elif metric == 'logloss':
        return _log_loss
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(distance: str, custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the input parameters."""
    if custom_distance is not None:
        return custom_distance
    if distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return _minkowski_distance
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    params: Dict[str, Any]
) -> tuple:
    """Gradient descent solver for learning rate optimization."""
    # Initialize parameters
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    learning_rate = params['learning_rate']
    max_iter = params['max_iter']
    tol = params['tol']

    # Gradient descent loop
    for _ in range(max_iter):
        gradients = np.zeros(n_features)
        for i in range(n_samples):
            prediction = np.dot(X[i], weights)
            error = y[i] - prediction
            gradients += error * X[i]

        # Update weights
        old_weights = np.copy(weights)
        weights += learning_rate * gradients / n_samples

        # Check convergence
        if distance_func(weights, old_weights) < tol:
            break

    # Compute metrics
    predictions = np.dot(X, weights)
    metrics = {
        'metric_value': metric_func(y, predictions),
        'final_weights': weights
    }

    return weights, metrics

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    params: Dict[str, Any]
) -> tuple:
    """Newton's method solver for learning rate optimization."""
    # Initialize parameters
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    learning_rate = params['learning_rate']
    max_iter = params['max_iter']
    tol = params['tol']

    # Newton's method loop
    for _ in range(max_iter):
        gradients = np.zeros(n_features)
        hessian = np.zeros((n_features, n_features))
        for i in range(n_samples):
            prediction = np.dot(X[i], weights)
            error = y[i] - prediction
            gradients += error * X[i]
            hessian -= np.outer(X[i], X[i])

        # Update weights
        old_weights = np.copy(weights)
        weights -= learning_rate * np.linalg.pinv(hessian) @ gradients

        # Check convergence
        if distance_func(weights, old_weights) < tol:
            break

    # Compute metrics
    predictions = np.dot(X, weights)
    metrics = {
        'metric_value': metric_func(y, predictions),
        'final_weights': weights
    }

    return weights, metrics

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    params: Dict[str, Any]
) -> tuple:
    """Coordinate descent solver for learning rate optimization."""
    # Initialize parameters
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    learning_rate = params['learning_rate']
    max_iter = params['max_iter']
    tol = params['tol']

    # Coordinate descent loop
    for _ in range(max_iter):
        old_weights = np.copy(weights)
        for j in range(n_features):
            # Compute residuals without the j-th feature
            residuals = y - np.dot(X[:, :j], weights[:j]) - np.dot(X[:, j+1:], weights[j+1:])

            # Compute the optimal weight for the j-th feature
            numerator = np.dot(X[:, j], residuals)
            denominator = np.dot(X[:, j], X[:, j])
            weights[j] += learning_rate * numerator / denominator

        # Check convergence
        if distance_func(weights, old_weights) < tol:
            break

    # Compute metrics
    predictions = np.dot(X, weights)
    metrics = {
        'metric_value': metric_func(y, predictions),
        'final_weights': weights
    }

    return weights, metrics

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3) -> float:
    """Compute the Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

################################################################################
# batch_size
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def batch_size_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver: str = 'gradient_descent',
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    batch_sizes: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Optimize batch size for a given dataset and metric.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    metric : str or callable
        Metric to optimize. Options: 'mse', 'mae', 'r2', or custom callable
    normalizer : callable, optional
        Function to normalize features. Options: None, standard, minmax, robust
    solver : str
        Optimization algorithm. Options: 'gradient_descent', 'newton'
    learning_rate : float
        Learning rate for gradient descent
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    batch_sizes : np.ndarray, optional
        Array of batch sizes to test. If None, uses default values.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = batch_size_fit(X, y, metric='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set default batch sizes if not provided
    if batch_sizes is None:
        batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64])

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize results dictionary
    results = {
        'result': None,
        'metrics': {},
        'params_used': {
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    # Get metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Test each batch size
    best_score = np.inf
    best_batch_size = None

    for bs in batch_sizes:
        try:
            # Split data into batches
            batches = _create_batches(X_normalized, bs)

            # Train model with current batch size
            if solver == 'gradient_descent':
                params = _gradient_descent(batches, y, learning_rate, max_iter, tol)
            else:
                params = _newton_optimization(X_normalized, y)

            # Calculate metric
            score = metric_func(y, _predict(X_normalized, params))

            # Update best batch size
            if score < best_score:
                best_score = score
                best_batch_size = bs

            # Store metrics for this batch size
            results['metrics'][bs] = {
                'score': score,
                'params': params
            }

        except Exception as e:
            results['warnings'].append(f"Batch size {bs} failed: {str(e)}")
            continue

    # Set best result
    results['result'] = {
        'optimal_batch_size': best_batch_size,
        'best_score': best_score
    }

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric_function(metric: str, custom_metric: Optional[Callable]) -> Callable:
    """Get metric function based on input."""
    if custom_metric is not None:
        return custom_metric

    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _create_batches(X: np.ndarray, batch_size: int) -> list:
    """Create batches from data."""
    n_samples = X.shape[0]
    return [X[i:i + batch_size] for i in range(0, n_samples, batch_size)]

def _gradient_descent(batches: list, y: np.ndarray, learning_rate: float,
                     max_iter: int, tol: float) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_features = batches[0].shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = np.zeros_like(params)
        n_batches = 0

        for batch in batches:
            if batch.shape[0] == 0:
                continue
            predictions = np.dot(batch, params)
            error = predictions - y[:batch.shape[0]]
            gradient += np.dot(batch.T, error)
            n_batches += 1

        if n_batches == 0:
            break

        gradient /= n_batches
        params_new = params - learning_rate * gradient

        if np.linalg.norm(params_new - params) < tol:
            break

        params = params_new

    return params

def _newton_optimization(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perform Newton optimization."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    # This is a simplified version - real implementation would need Hessian
    for _ in range(10):
        predictions = np.dot(X, params)
        error = predictions - y
        gradient = np.dot(X.T, error)

        # Approximate Hessian (in practice would calculate properly)
        hessian = np.dot(X.T, X) + 1e-6 * np.eye(n_features)

        params = params - np.linalg.solve(hessian, gradient)

    return params

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Make predictions using linear model."""
    return np.dot(X, params)

################################################################################
# epochs
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def epochs_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_epochs: int = 100,
    batch_size: Optional[int] = None,
    learning_rate: float = 0.01,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    regularization: str = 'none',
    reg_param: float = 0.1,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform epochs-based optimization for hyperparameter tuning.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    n_epochs : int, default=100
        Number of epochs to run
    batch_size : Optional[int], default=None
        Size of mini-batches. If None, uses full batch.
    learning_rate : float, default=0.01
        Learning rate for gradient-based solvers
    solver : str, default='gradient_descent'
        Optimization algorithm to use
    metric : Union[str, Callable], default='mse'
        Metric to optimize. Can be 'mse', 'mae', 'r2' or custom callable
    normalizer : Optional[Callable], default=None
        Normalization function to apply to features
    regularization : str, default='none'
        Regularization type: 'none', 'l1', 'l2' or 'elasticnet'
    reg_param : float, default=0.1
        Regularization parameter
    tol : float, default=1e-4
        Tolerance for early stopping
    random_state : Optional[int], default=None
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': Optimization result
        - 'metrics': Computed metrics
        - 'params_used': Parameters used during optimization
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = epochs_fit(X, y, n_epochs=50, solver='gradient_descent')
    """
    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if requested
    X_norm = _apply_normalization(X, normalizer)

    # Initialize parameters and metrics
    params = _initialize_parameters(X_norm.shape[1])
    best_params = None
    best_metric = float('inf')
    metrics_history = []
    warnings_list = []

    # Get solver function
    solver_func = _get_solver(solver)

    # Get metric function
    metric_func = _get_metric(metric)

    for epoch in range(n_epochs):
        # Get current batch
        X_batch, y_batch = _get_batch(X_norm, y, batch_size)

        # Update parameters
        params = solver_func(
            X_batch,
            y_batch,
            params,
            learning_rate=learning_rate,
            regularization=regularization,
            reg_param=reg_param
        )

        # Compute current metric
        current_metric = metric_func(y, X_norm @ params)

        metrics_history.append(current_metric)

        # Check for early stopping
        if best_params is None or current_metric < best_metric - tol:
            best_metric = current_metric
            best_params = params.copy()
        else:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Metric: {current_metric:.4f}")

    # Compute final metrics
    final_metrics = _compute_final_metrics(y, X_norm @ best_params)

    return {
        'result': best_params,
        'metrics': final_metrics,
        'params_used': {
            'n_epochs': epoch + 1,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'solver': solver,
            'metric': metric,
            'regularization': regularization,
            'reg_param': reg_param
        },
        'warnings': warnings_list
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
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

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to features."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize model parameters."""
    return np.zeros(n_features)

def _get_batch(X: np.ndarray, y: np.ndarray, batch_size: Optional[int]) -> tuple:
    """Get a random batch of data."""
    if batch_size is None or batch_size >= len(X):
        return X, y

    idx = np.random.choice(len(X), batch_size, replace=False)
    return X[idx], y[idx]

def _get_solver(solver: str) -> Callable:
    """Get the appropriate solver function."""
    solvers = {
        'gradient_descent': _gradient_descent_step,
        'newton': _newton_step,
        'coordinate_descent': _coordinate_descent_step
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver]

def _get_metric(metric: Union[str, Callable]) -> Callable:
    """Get the appropriate metric function."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if isinstance(metric, str):
        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}")
        return metrics[metric]
    elif callable(metric):
        return metric
    else:
        raise ValueError("Metric must be a string or callable")

def _gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    learning_rate: float = 0.01,
    regularization: str = 'none',
    reg_param: float = 0.1
) -> np.ndarray:
    """Perform a single gradient descent step."""
    residuals = y - X @ params
    gradient = -(X.T @ residuals) / len(X)

    if regularization == 'l1':
        gradient += reg_param * np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * reg_param * params
    elif regularization == 'elasticnet':
        gradient += reg_param * (np.sign(params) + 2 * params)

    return params - learning_rate * gradient

def _newton_step(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    learning_rate: float = 0.01,
    regularization: str = 'none',
    reg_param: float = 0.1
) -> np.ndarray:
    """Perform a single Newton step."""
    residuals = y - X @ params
    gradient = -(X.T @ residuals) / len(X)
    hessian = (X.T @ X) / len(X)

    if regularization == 'l2':
        hessian += 2 * reg_param * np.eye(len(params))

    try:
        params = params - learning_rate * np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        params = params - learning_rate * gradient

    return params

def _coordinate_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    learning_rate: float = 0.01,
    regularization: str = 'none',
    reg_param: float = 0.1
) -> np.ndarray:
    """Perform a single coordinate descent step."""
    params = params.copy()
    for i in range(len(params)):
        X_i = X[:, i]
        residuals = y - (X @ params - X_i * params[i])

        if regularization == 'l1':
            params[i] = _soft_threshold(residuals @ X_i, learning_rate * reg_param)
        elif regularization == 'l2':
            params[i] = (residuals @ X_i) / (X_i @ X_i + 2 * learning_rate * reg_param)
        else:
            params[i] = (residuals @ X_i) / (X_i @ X_i)

    return params

def _soft_threshold(x: float, threshold: float) -> float:
    """Soft-thresholding function for L1 regularization."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)

def _compute_final_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all final metrics."""
    return {
        'mse': _mean_squared_error(y_true, y_pred),
        'mae': _mean_absolute_error(y_true, y_pred),
        'r2': _r_squared(y_true, y_pred)
    }

################################################################################
# momentum
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    momentum_factor: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if learning_rate <= 0 or learning_rate >= 1:
        raise ValueError("learning_rate must be in (0, 1)")
    if momentum_factor < 0 or momentum_factor >= 1:
        raise ValueError("momentum_factor must be in [0, 1)")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize parameters with zeros."""
    return np.zeros(n_features)

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Callable = lambda y_true, y_pred: (y_pred - y_true).mean()
) -> np.ndarray:
    """Compute gradient based on the specified metric."""
    y_pred = X @ params
    return -2 * (X.T @ (y - y_pred)) / len(y)

def _update_parameters(
    params: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    velocity: Optional[np.ndarray] = None
) -> tuple:
    """Update parameters using momentum."""
    if velocity is None:
        velocity = np.zeros_like(params)
    velocity = momentum_factor * velocity + learning_rate * gradient
    params += velocity
    return params, velocity

def momentum_fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    momentum_factor: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: Callable = lambda y_true, y_pred: (y_pred - y_true).mean(),
    normalize: bool = True,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a model using momentum-based gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    learning_rate : float, optional
        Learning rate for gradient descent.
    momentum_factor : float, optional
        Momentum factor for acceleration.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    metric : Callable, optional
        Metric function to evaluate the model.
    normalize : bool, optional
        Whether to normalize the features.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, y, learning_rate, momentum_factor, max_iter, tol)

    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    n_features = X.shape[1]
    params = _initialize_parameters(n_features)
    velocity = np.zeros_like(params)

    for i in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric)
        params, velocity = _update_parameters(params, gradient, learning_rate, velocity)

        if np.linalg.norm(gradient) < tol:
            break

    y_pred = X @ params
    result = {
        "params": params,
        "y_pred": y_pred,
        "iterations": i + 1
    }

    metrics = {
        "metric_value": metric(y, y_pred)
    }

    params_used = {
        "learning_rate": learning_rate,
        "momentum_factor": momentum_factor,
        "max_iter": max_iter,
        "tol": tol
    }

    warnings = {}

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.rand(100)
# result = momentum_fit(X, y)

################################################################################
# weight_decay
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def weight_decay_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable] = "mse",
    solver: str = "closed_form",
    regularization: str = "none",
    lambda_: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute weight decay (L2 regularization) for linear regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalizer : Optional[Callable]
        Function to normalize features. Default is None.
    metric : Union[str, Callable]
        Metric to evaluate performance. Can be "mse", "mae", "r2", or a custom callable.
    solver : str
        Solver to use. Options: "closed_form", "gradient_descent".
    regularization : str
        Type of regularization. Options: "none", "l1", "l2".
    lambda_ : float
        Regularization strength.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function if needed.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = weight_decay_fit(X, y, solver="gradient_descent", regularization="l2")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    if normalizer is not None:
        X = normalizer(X)

    # Choose solver
    if solver == "closed_form":
        weights, metrics = _closed_form_solver(X, y, regularization, lambda_)
    elif solver == "gradient_descent":
        weights, metrics = _gradient_descent_solver(X, y, regularization, lambda_, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, X @ weights)

    return {
        "result": weights,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "lambda_": lambda_
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or Inf values")

def _closed_form_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: str,
    lambda_: float
) -> tuple[np.ndarray, Dict]:
    """Closed-form solution for weight decay."""
    n_samples, n_features = X.shape
    if regularization == "l2":
        identity = np.eye(n_features) * lambda_
        weights = np.linalg.inv(X.T @ X + identity) @ X.T @ y
    elif regularization == "none":
        weights = np.linalg.pinv(X) @ y
    else:
        raise ValueError(f"Unsupported regularization: {regularization}")

    metrics = {
        "mse": _compute_mse(y, X @ weights),
        "r2": _compute_r2(y, X @ weights)
    }
    return weights, metrics

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    regularization: str,
    lambda_: float,
    tol: float,
    max_iter: int
) -> tuple[np.ndarray, Dict]:
    """Gradient descent solver for weight decay."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, weights, regularization, lambda_)
        weights -= gradient

        current_loss = _compute_mse(y, X @ weights)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    metrics = {
        "mse": _compute_mse(y, X @ weights),
        "r2": _compute_r2(y, X @ weights)
    }
    return weights, metrics

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    regularization: str,
    lambda_: float
) -> np.ndarray:
    """Compute gradient for weight decay."""
    n_samples = X.shape[0]
    predictions = X @ weights
    error = predictions - y

    gradient = (X.T @ error) / n_samples

    if regularization == "l2":
        gradient += 2 * lambda_ * weights
    elif regularization == "l1":
        gradient += lambda_ * np.sign(weights)

    return gradient

def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _get_metric_function(metric: str) -> Callable:
    """Get metric function based on string input."""
    metrics = {
        "mse": _compute_mse,
        "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
        "r2": _compute_r2,
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

################################################################################
# dropout_rate
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def dropout_rate_fit(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    dropout_rates: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    n_splits: int = 5,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Fit a model with different dropout rates and evaluate performance.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    model : Callable
        A callable that implements fit and predict methods.
    dropout_rates : np.ndarray
        Array of dropout rates to test.
    metric : str or Callable, optional
        Metric to evaluate performance. Can be 'mse', 'mae', 'r2', or a custom callable.
    n_splits : int, optional
        Number of cross-validation splits.
    random_state : Optional[int], optional
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": Best dropout rate and corresponding metric value.
        - "metrics": Performance metrics for all dropout rates.
        - "params_used": Parameters used in the fitting process.
        - "warnings": Any warnings encountered during execution.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> model = lambda: None  # Replace with actual model
    >>> dropout_rates = np.linspace(0, 0.5, 6)
    >>> result = dropout_rate_fit(X, y, model, dropout_rates)
    """
    # Validate inputs
    _validate_inputs(X, y, dropout_rates)

    # Initialize results dictionary
    results = {
        "result": {},
        "metrics": {},
        "params_used": {
            "metric": metric,
            "n_splits": n_splits,
            "random_state": random_state
        },
        "warnings": []
    }

    # Define metric function
    metric_func = _get_metric_function(metric)

    # Evaluate each dropout rate
    metrics_values = []
    for rate in dropout_rates:
        try:
            # Fit model with current dropout rate
            model_instance = model()
            model_instance.dropout_rate = rate

            # Perform cross-validation
            cv_scores = _cross_validate(X, y, model_instance, n_splits, random_state)

            # Store results
            metrics_values.append(np.mean(cv_scores))
        except Exception as e:
            results["warnings"].append(f"Error with dropout rate {rate}: {str(e)}")
            metrics_values.append(np.inf)

    # Find best dropout rate
    best_idx = np.argmin(metrics_values)
    results["result"] = {
        "best_dropout_rate": dropout_rates[best_idx],
        "best_metric_value": metrics_values[best_idx]
    }

    # Store all metrics
    results["metrics"] = {
        "dropout_rates": dropout_rates,
        "metric_values": metrics_values
    }

    return results

def _validate_inputs(X: np.ndarray, y: np.ndarray, dropout_rates: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(dropout_rates < 0) or np.any(dropout_rates > 1):
        raise ValueError("Dropout rates must be between 0 and 1")

def _get_metric_function(metric: Union[str, Callable]) -> Callable:
    """Get metric function based on input."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }

    if isinstance(metric, str):
        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}")
        return metrics[metric]
    elif callable(metric):
        return metric
    else:
        raise ValueError("Metric must be either a string or callable")

def _cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model: Callable,
    n_splits: int,
    random_state: Optional[int]
) -> np.ndarray:
    """Perform cross-validation."""
    # Implement k-fold cross-validation
    cv_scores = []
    for i in range(n_splits):
        # Split data (simplified version)
        test_idx = np.arange(i * len(X) // n_splits, (i + 1) * len(X) // n_splits)
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        score = _mean_squared_error(y_test, y_pred)  # Default metric
        cv_scores.append(score)

    return np.array(cv_scores)

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

################################################################################
# hidden_units
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def hidden_units_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_layers: int = 1,
    layer_sizes: Optional[list] = None,
    activation: str = 'relu',
    solver: str = 'adam',
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    metric: str = 'mse',
    validation_split: float = 0.2,
    early_stopping: bool = True,
    patience: int = 5,
    normalize: str = 'standard',
    l1_ratio: float = 0.0,
    alpha: float = 0.0,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, dict]]:
    """
    Determine optimal number of hidden units for a neural network layer.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    n_layers : int, optional
        Number of hidden layers to consider, by default 1
    layer_sizes : list, optional
        List of layer sizes to test, by default None (auto-detect)
    activation : str, optional
        Activation function ('relu', 'sigmoid', 'tanh'), by default 'relu'
    solver : str, optional
        Optimization algorithm ('adam', 'sgd'), by default 'adam'
    learning_rate : float, optional
        Learning rate for optimization, by default 0.001
    batch_size : int, optional
        Batch size for training, by default 32
    epochs : int, optional
        Number of training epochs, by default 100
    tol : float, optional
        Tolerance for early stopping, by default 1e-4
    random_state : int, optional
        Random seed for reproducibility, by default None
    metric : str, optional
        Evaluation metric ('mse', 'mae', 'r2'), by default 'mse'
    validation_split : float, optional
        Fraction of data for validation, by default 0.2
    early_stopping : bool, optional
        Whether to use early stopping, by default True
    patience : int, optional
        Patience for early stopping, by default 5
    normalize : str, optional
        Normalization method ('standard', 'minmax'), by default 'standard'
    l1_ratio : float, optional
        L1 regularization ratio (0-1), by default 0.0
    alpha : float, optional
        Regularization strength, by default 0.0
    verbose : bool, optional
        Whether to print progress, by default False

    Returns
    -------
    Dict[str, Union[np.ndarray, float, dict]]
        Dictionary containing:
        - 'result': Optimal layer sizes
        - 'metrics': Performance metrics
        - 'params_used': Parameters used
        - 'warnings': Any warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = hidden_units_fit(X, y, n_layers=2, layer_sizes=[10, 20])
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, method=normalize)

    # Determine layer sizes to test if not provided
    if layer_sizes is None:
        layer_sizes = _determine_layer_sizes(X.shape[1], n_layers)

    # Initialize results storage
    results = []
    warnings_list = []

    for size in layer_sizes:
        try:
            # Train model with current layer size
            model = _train_model(
                X_norm, y_norm,
                layer_size=size,
                activation=activation,
                solver=solver,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                tol=tol,
                validation_split=validation_split,
                early_stopping=early_stopping,
                patience=patience,
                l1_ratio=l1_ratio,
                alpha=alpha
            )

            # Evaluate model
            metrics = _evaluate_model(model, X_norm, y_norm, metric=metric)

            results.append({
                'layer_size': size,
                'metrics': metrics
            })

        except Exception as e:
            warnings_list.append(f"Error with layer size {size}: {str(e)}")

    if not results:
        raise ValueError("No successful model evaluations. Check inputs and parameters.")

    # Select best configuration
    best_result = _select_best_configuration(results, metric)

    return {
        'result': best_result['layer_size'],
        'metrics': best_result['metrics'],
        'params_used': {
            'n_layers': n_layers,
            'activation': activation,
            'solver': solver,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'metric': metric,
            'normalize': normalize
        },
        'warnings': warnings_list if warnings_list else None
    }

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
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'standard'
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize input data."""
    if method == 'standard':
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)
    elif method == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    else:
        X_norm, y_norm = X.copy(), y.copy()
    return X_norm, y_norm

def _determine_layer_sizes(n_features: int, n_layers: int) -> list:
    """Determine reasonable layer sizes to test."""
    base_size = max(2, n_features // 2)
    return [base_size * (i+1) for i in range(n_layers)]

def _train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    layer_size: int,
    activation: str = 'relu',
    solver: str = 'adam',
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    tol: float = 1e-4,
    validation_split: float = 0.2,
    early_stopping: bool = True,
    patience: int = 5,
    l1_ratio: float = 0.0,
    alpha: float = 0.0
) -> dict:
    """Train neural network model with given parameters."""
    # This is a simplified placeholder for actual model training
    # In practice, this would use a proper neural network implementation

    class DummyModel:
        def __init__(self, layer_size):
            self.layer_size = layer_size
            self.weights = np.random.randn(X.shape[1], layer_size)
            self.bias = np.zeros(layer_size)

        def fit(self, X, y):
            # Simulate training
            for _ in range(epochs):
                pass

        def predict(self, X):
            return np.dot(X, self.weights) + self.bias

    model = DummyModel(layer_size)
    model.fit(X, y)
    return {'model': model}

def _evaluate_model(
    model: dict,
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'mse'
) -> dict:
    """Evaluate model performance."""
    y_pred = model['model'].predict(X)

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

def _select_best_configuration(
    results: list,
    metric: str = 'mse'
) -> dict:
    """Select best configuration based on evaluation metric."""
    if not results:
        raise ValueError("No results provided")

    # Determine which metric to optimize (minimize or maximize)
    if metric in ['mse', 'mae']:
        best_idx = np.argmin([r['metrics'][metric] for r in results])
    elif metric == 'r2':
        best_idx = np.argmax([r['metrics'][metric] for r in results])
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return results[best_idx]

################################################################################
# num_layers
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def num_layers_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_layers: int = 5,
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    learning_rate: float = 0.01,
    tol: float = 1e-4,
    max_iter: int = 1000,
    validation_split: float = 0.2,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Determine the optimal number of layers for a neural network based on given criteria.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    max_layers : int, optional
        Maximum number of layers to consider (default: 5)
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2') or custom callable (default: 'mse')
    solver : str, optional
        Solver to use ('gradient_descent', 'newton') (default: 'gradient_descent')
    learning_rate : float, optional
        Learning rate for gradient descent (default: 0.01)
    tol : float, optional
        Tolerance for convergence (default: 1e-4)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    validation_split : float, optional
        Fraction of data to use for validation (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    verbose : bool, optional
        Whether to print progress information (default: False)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': optimal number of layers
        - 'metrics': performance metrics for each layer count
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = num_layers_fit(X, y, max_layers=3, metric='r2')
    """
    # Validate inputs
    _validate_inputs(X, y)

    if random_state is not None:
        np.random.seed(random_state)

    # Split data
    X_train, y_train, X_val, y_val = _train_test_split(X, y, validation_split)

    # Initialize results storage
    metrics_history = []
    warnings_list = []

    best_layers = 1
    best_metric = -np.inf if metric != 'mse' else np.inf

    # Try different numbers of layers
    for n_layers in range(1, max_layers + 1):
        try:
            # Train model with current number of layers
            if solver == 'gradient_descent':
                params = _train_gradient_descent(
                    X_train, y_train,
                    n_layers=n_layers,
                    learning_rate=learning_rate,
                    tol=tol,
                    max_iter=max_iter
                )
            else:
                params = _train_newton(
                    X_train, y_train,
                    n_layers=n_layers
                )

            # Evaluate on validation set
            val_metric = _compute_metric(
                X_val, y_val,
                params['weights'],
                metric
            )

            metrics_history.append({
                'layers': n_layers,
                'metric_value': val_metric
            })

            # Update best configuration
            if (metric == 'mse' and val_metric < best_metric) or \
               (metric != 'mse' and val_metric > best_metric):
                best_layers = n_layers
                best_metric = val_metric

        except Exception as e:
            warnings_list.append(f"Failed with {n_layers} layers: {str(e)}")
            continue

    return {
        'result': best_layers,
        'metrics': metrics_history,
        'params_used': {
            'max_layers': max_layers,
            'metric': metric,
            'solver': solver,
            'learning_rate': learning_rate,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': warnings_list
    }

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
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float
) -> tuple:
    """Split data into training and validation sets."""
    n_samples = X.shape[0]
    n_val = int(n_samples * validation_split)

    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return (
        X[train_indices],
        y[train_indices],
        X[val_indices],
        y[val_indices]
    )

def _compute_metric(
    X: np.ndarray,
    y: np.ndarray,
    weights: list,
    metric: Union[str, Callable]
) -> float:
    """Compute the specified metric."""
    y_pred = _forward_pass(X, weights)

    if callable(metric):
        return metric(y, y_pred)
    elif metric == 'mse':
        return np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _forward_pass(X: np.ndarray, weights: list) -> np.ndarray:
    """Perform forward pass through the network."""
    a = X
    for w in weights:
        a = np.dot(a, w)
        # Simple ReLU activation
        a = np.maximum(0, a)
    return a

def _train_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    n_layers: int,
    learning_rate: float,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Train network using gradient descent."""
    n_features = X.shape[1]
    n_outputs = 1

    # Initialize weights
    weights = []
    for _ in range(n_layers):
        layer_size = np.random.randint(10, 50)  # Simple heuristic
        weights.append(np.random.randn(n_features if len(weights) == 0 else layer_size, layer_size))
    weights.append(np.random.randn(layer_size if n_layers > 0 else n_features, n_outputs))

    prev_loss = np.inf
    for _ in range(max_iter):
        # Forward pass
        y_pred = _forward_pass(X, weights)

        # Compute gradients (simplified)
        error = y_pred - y
        grads = []
        delta = error * (y_pred > 0)  # ReLU derivative

        for i in reversed(range(len(weights))):
            if i == len(weights) - 1:
                grad = np.dot(X.T, delta)
            else:
                delta = np.dot(delta, weights[i+1].T) * (y_pred > 0)
                grad = np.dot(X.T, delta)
            grads.append(grad)

        # Update weights
        for i in range(len(weights)):
            weights[i] -= learning_rate * grads[-(i+1)]

        # Check convergence
        current_loss = np.mean(error ** 2)
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return {'weights': weights}

def _train_newton(
    X: np.ndarray,
    y: np.ndarray,
    n_layers: int
) -> Dict[str, Any]:
    """Train network using Newton's method (simplified)."""
    # This is a placeholder for the actual implementation
    n_features = X.shape[1]
    weights = []

    # Simple initialization
    for _ in range(n_layers):
        layer_size = np.random.randint(10, 50)
        weights.append(np.random.randn(n_features if len(weights) == 0 else layer_size, layer_size))
    weights.append(np.random.randn(layer_size if n_layers > 0 else n_features, 1))

    # Simplified Newton step
    y_pred = _forward_pass(X, weights)
    error = y_pred - y

    # Compute Hessian approximation (simplified)
    hessian = np.dot(X.T, X)

    # Update weights
    grad = np.dot(X.T, error)
    weights[-1] -= np.linalg.inv(hessian) @ grad

    return {'weights': weights}

################################################################################
# activation_function
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def activation_function_fit(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Union[str, Callable] = 'relu',
    normalization: str = 'none',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_params: Optional[Dict] = None
) -> Dict:
    """
    Fit an activation function to the data with configurable hyperparameters.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    activation_func : Union[str, Callable], optional
        Activation function to use. Can be 'relu', 'sigmoid', 'tanh', or a custom callable.
    normalization : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    metric : Union[str, Callable], optional
        Metric to evaluate the fit. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str, optional
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization method. Can be 'none', 'l1', 'l2', or 'elasticnet'.
    learning_rate : float, optional
        Learning rate for gradient-based solvers.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    custom_params : Optional[Dict], optional
        Additional parameters for the solver or activation function.

    Returns
    -------
    Dict
        A dictionary containing:
        - "result": Fitted parameters.
        - "metrics": Computed metrics.
        - "params_used": Parameters used during fitting.
        - "warnings": Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = activation_function_fit(X, y, activation_func='relu', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalization)

    # Initialize parameters
    params = _initialize_params(X_normalized.shape[1], random_state)

    # Choose activation function
    if isinstance(activation_func, str):
        act_func = _get_activation_function(activation_func)
    else:
        act_func = activation_func

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _gradient_descent(
            X_normalized, y, act_func, learning_rate, max_iter, tol,
            regularization, custom_params
        )
    elif solver == 'newton':
        params = _newton_method(
            X_normalized, y, act_func, max_iter, tol,
            regularization, custom_params
        )
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(
            X_normalized, y, act_func, max_iter, tol,
            regularization, custom_params
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, params, act_func, metric)

    # Prepare output
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "activation_func": activation_func,
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "regularization": regularization,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or inf values")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_params(n_features: int, random_state: Optional[int] = None) -> np.ndarray:
    """Initialize parameters."""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randn(n_features)

def _get_activation_function(name: str) -> Callable:
    """Get activation function by name."""
    if name == 'relu':
        return lambda x: np.maximum(0, x)
    elif name == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))
    elif name == 'tanh':
        return lambda x: np.tanh(x)
    else:
        raise ValueError(f"Unknown activation function: {name}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed-form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Callable,
    learning_rate: float,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    custom_params: Optional[Dict]
) -> np.ndarray:
    """Gradient descent solver."""
    params = _initialize_params(X.shape[1])
    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, activation_func, regularization)
        params -= learning_rate * grad
    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    activation_func: Callable,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradient."""
    predictions = X @ params
    residuals = predictions - y
    grad = X.T @ (residuals * _activation_derivative(predictions, activation_func)) / X.shape[0]

    if regularization == 'l1':
        grad += np.sign(params)
    elif regularization == 'l2':
        grad += 2 * params
    return grad

def _activation_derivative(x: np.ndarray, activation_func: Callable) -> np.ndarray:
    """Compute derivative of activation function."""
    if activation_func == _get_activation_function('relu'):
        return (x > 0).astype(float)
    elif activation_func == _get_activation_function('sigmoid'):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    elif activation_func == _get_activation_function('tanh'):
        return 1 - np.tanh(x)**2
    else:
        raise ValueError("Unknown activation function for derivative")

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Callable,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    custom_params: Optional[Dict]
) -> np.ndarray:
    """Newton method solver."""
    params = _initialize_params(X.shape[1])
    for _ in range(max_iter):
        grad = _compute_gradient(X, y, params, activation_func, regularization)
        hessian = _compute_hessian(X, y, params, activation_func)
        params -= np.linalg.pinv(hessian) @ grad
    return params

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    activation_func: Callable
) -> np.ndarray:
    """Compute Hessian matrix."""
    predictions = X @ params
    der = _activation_derivative(predictions, activation_func)
    hessian = X.T @ np.diag(der) @ X / X.shape[0]
    return hessian

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    activation_func: Callable,
    max_iter: int,
    tol: float,
    regularization: Optional[str],
    custom_params: Optional[Dict]
) -> np.ndarray:
    """Coordinate descent solver."""
    params = _initialize_params(X.shape[1])
    for _ in range(max_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            params[i] = _solve_1d(X_i, y - X @ params + X_i * params[i], activation_func, regularization)
    return params

def _solve_1d(
    X_i: np.ndarray,
    y_residual: np.ndarray,
    activation_func: Callable,
    regularization: Optional[str]
) -> float:
    """Solve 1D problem in coordinate descent."""
    if regularization == 'l1':
        return np.sign(X_i.T @ y_residual) * np.maximum(
            0, np.abs(X_i.T @ y_residual) - 1
        ) / (X_i.T @ X_i)
    elif regularization == 'l2':
        return (X_i.T @ y_residual) / (X_i.T @ X_i + 1)
    else:
        return (X_i.T @ y_residual) / (X_i.T @ X_i)

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    activation_func: Callable,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics."""
    predictions = X @ params
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((predictions - y) ** 2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(predictions - y))}
        elif metric == 'r2':
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return {'r2': 1 - ss_res / ss_tot}
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return {'custom': metric(predictions, y)}

################################################################################
# optimizer
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def optimizer_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'gradient_descent',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a given model.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    metric : str or callable, optional
        Metric to optimize. Can be 'mse', 'mae', 'r2', or a custom callable.
    solver : str, optional
        Solver to use. Can be 'gradient_descent', 'newton', or 'coordinate_descent'.
    normalization : str, optional
        Normalization method. Can be 'standard', 'minmax', or 'robust'.
    regularization : str, optional
        Regularization method. Can be 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    learning_rate : float, optional
        Learning rate for gradient descent.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, used parameters, and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = optimizer_fit(X, y, metric='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    params = _initialize_parameters(X_normalized.shape[1])

    # Choose solver and optimize
    if solver == 'gradient_descent':
        optimized_params, metrics = _gradient_descent(
            X_normalized, y, params, metric, tol, max_iter,
            learning_rate, regularization, custom_metric
        )
    elif solver == 'newton':
        optimized_params, metrics = _newton_method(
            X_normalized, y, params, metric, tol, max_iter,
            regularization, custom_metric
        )
    elif solver == 'coordinate_descent':
        optimized_params, metrics = _coordinate_descent(
            X_normalized, y, params, metric, tol, max_iter,
            regularization, custom_metric
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare results
    result = {
        'result': optimized_params,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'solver': solver,
            'normalization': normalization,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate
        },
        'warnings': _check_warnings(metrics)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply specified normalization to input data."""
    if method is None:
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize parameters with zeros."""
    return np.zeros(n_features)

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    learning_rate: float,
    regularization: Optional[str],
    custom_metric: Optional[Callable]
) -> tuple:
    """Gradient descent optimization."""
    metrics = []
    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ params

        # Compute metric
        if isinstance(metric, str):
            current_metric = _compute_metric(y_pred, y, metric)
        else:
            current_metric = metric(y_pred, y)

        metrics.append(current_metric)

        # Check convergence
        if len(metrics) > 1 and abs(metrics[-2] - metrics[-1]) < tol:
            break

        # Compute gradient
        gradient = _compute_gradient(X, y, y_pred, params, regularization)

        # Update parameters
        params -= learning_rate * gradient

    return params, metrics

def _compute_metric(y_pred: np.ndarray, y_true: np.ndarray, metric: str) -> float:
    """Compute specified metric."""
    if metric == 'mse':
        return np.mean((y_pred - y_true) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_pred - y_true))
    elif metric == 'r2':
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    params: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute gradient with optional regularization."""
    gradient = -2 * X.T @ (y - y_pred) / len(y)

    if regularization == 'l1':
        gradient += np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * params
    elif regularization == 'elasticnet':
        gradient += np.sign(params) + 2 * params

    return gradient

def _check_warnings(metrics: list) -> list:
    """Check for warnings in optimization process."""
    warnings = []
    if len(metrics) >= 2 and abs(metrics[-1] - metrics[-2]) > 0.1:
        warnings.append("Slow convergence")
    if len(metrics) == max_iter:
        warnings.append("Maximum iterations reached")
    return warnings

# Additional solver implementations would follow the same pattern
def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    custom_metric: Optional[Callable]
) -> tuple:
    """Newton method optimization."""
    metrics = []
    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ params

        # Compute metric
        if isinstance(metric, str):
            current_metric = _compute_metric(y_pred, y, metric)
        else:
            current_metric = metric(y_pred, y)

        metrics.append(current_metric)

        # Check convergence
        if len(metrics) > 1 and abs(metrics[-2] - metrics[-1]) < tol:
            break

        # Compute gradient and hessian
        gradient = _compute_gradient(X, y, y_pred, params, regularization)
        hessian = _compute_hessian(X)

        # Update parameters
        params -= np.linalg.solve(hessian, gradient)

    return params, metrics

def _compute_hessian(X: np.ndarray) -> np.ndarray:
    """Compute Hessian matrix."""
    return 2 * X.T @ X / len(X)

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    tol: float,
    max_iter: int,
    regularization: Optional[str],
    custom_metric: Optional[Callable]
) -> tuple:
    """Coordinate descent optimization."""
    metrics = []
    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ params

        # Compute metric
        if isinstance(metric, str):
            current_metric = _compute_metric(y_pred, y, metric)
        else:
            current_metric = metric(y_pred, y)

        metrics.append(current_metric)

        # Check convergence
        if len(metrics) > 1 and abs(metrics[-2] - metrics[-1]) < tol:
            break

        # Update each parameter one at a time
        for i in range(X.shape[1]):
            X_i = X[:, i]
            residual = y - (y_pred - params[i] * X_i)

            if regularization == 'l1':
                params[i] = _soft_threshold(residual @ X_i, 1)
            elif regularization == 'l2':
                params[i] = (residual @ X_i) / (X_i @ X_i + 1)
            else:
                params[i] = (residual @ X_i) / (X_i @ X_i)

    return params, metrics

def _soft_threshold(x: float, lambda_: float) -> float:
    """Soft thresholding function."""
    if x > lambda_:
        return x - lambda_
    elif x < -lambda_:
        return x + lambda_
    else:
        return 0

################################################################################
# regularization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularization_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization_type: str = "none",
    alpha: float = 1.0,
    l1_ratio: Optional[float] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a regularized model to the data with configurable hyperparameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    normalization : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", "logloss", or custom callable.
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization_type : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    alpha : float, optional
        Regularization strength.
    l1_ratio : float, optional
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if metric is "custom".
    custom_distance : callable, optional
        Custom distance function if distance is "custom".

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": fitted model parameters
        - "metrics": computed metrics
        - "params_used": actual parameters used
        - "warnings": any warnings generated

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regularization_fit(X, y, normalization="standard", metric="mse")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)
    y_normalized = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Select metric
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        if custom_metric is None:
            raise ValueError("custom_metric must be provided when metric is a callable")
        metric_func = custom_metric

    # Select distance
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        if custom_distance is None:
            raise ValueError("custom_distance must be provided when distance is a callable")
        distance_func = custom_distance

    # Select solver
    if solver == "closed_form":
        params = _solve_closed_form(X_normalized, y_normalized,
                                  regularization_type, alpha, l1_ratio)
    elif solver == "gradient_descent":
        params = _solve_gradient_descent(X_normalized, y_normalized,
                                       metric_func, distance_func,
                                       regularization_type, alpha, l1_ratio,
                                       tol, max_iter)
    elif solver == "newton":
        params = _solve_newton(X_normalized, y_normalized,
                             metric_func, distance_func,
                             regularization_type, alpha, l1_ratio,
                             tol, max_iter)
    elif solver == "coordinate_descent":
        params = _solve_coordinate_descent(X_normalized, y_normalized,
                                         metric_func, distance_func,
                                         regularization_type, alpha, l1_ratio,
                                         tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y_normalized, params, metric_func)

    # Prepare output
    return {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization_type": regularization_type,
            "alpha": alpha,
            "l1_ratio": l1_ratio if l1_ratio is not None else 0.5,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == "none":
        return X
    elif method == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric_name: str) -> Callable:
    """Return the specified metric function."""
    metrics = {
        "mse": _mse,
        "mae": _mae,
        "r2": _r2,
        "logloss": _logloss
    }
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")
    return metrics[metric_name]

def _get_distance_function(distance_name: str) -> Callable:
    """Return the specified distance function."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": lambda x, y, p=2: np.sum(np.abs(x - y)**p, axis=1)**(1/p)
    }
    if distance_name not in distances:
        raise ValueError(f"Unknown distance: {distance_name}")
    return distances[distance_name]

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      reg_type: str, alpha: float, l1_ratio: Optional[float]) -> np.ndarray:
    """Solve using closed-form solution."""
    n_features = X.shape[1]
    if reg_type == "none":
        params = np.linalg.inv(X.T @ X) @ X.T @ y
    elif reg_type == "l2":
        params = np.linalg.inv(X.T @ X + alpha * np.eye(n_features)) @ X.T @ y
    elif reg_type == "l1":
        # For L1, we use coordinate descent as closed form doesn't exist
        return _solve_coordinate_descent(X, y, None, None,
                                       "l1", alpha, 1.0, 1e-4, 1000)
    elif reg_type == "elasticnet":
        if l1_ratio is None:
            l1_ratio = 0.5
        # ElasticNet requires coordinate descent or similar iterative method
        return _solve_coordinate_descent(X, y, None, None,
                                       "elasticnet", alpha, l1_ratio, 1e-4, 1000)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
    return params

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           metric_func: Callable, distance_func: Callable,
                           reg_type: str, alpha: float, l1_ratio: Optional[float],
                           tol: float, max_iter: int) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric_func, distance_func,
                                   reg_type, alpha, l1_ratio)
        params -= learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_newton(X: np.ndarray, y: np.ndarray,
                 metric_func: Callable, distance_func: Callable,
                 reg_type: str, alpha: float, l1_ratio: Optional[float],
                 tol: float, max_iter: int) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, metric_func, distance_func,
                                   reg_type, alpha, l1_ratio)
        hessian = _compute_hessian(X, y, params, metric_func, distance_func,
                                  reg_type, alpha)

        if np.linalg.cond(hessian) < 1e15:
            params -= np.linalg.solve(hessian, gradient)
        else:
            params -= 0.01 * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             metric_func: Callable, distance_func: Callable,
                             reg_type: str, alpha: float, l1_ratio: Optional[float],
                             tol: float, max_iter: int) -> np.ndarray:
    """Solve using coordinate descent."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, params) + params[j] * X_j

            if reg_type == "l1":
                if np.sum(X_j != 0) > 0:
                    params[j] = _soft_threshold(np.dot(X_j, residuals), alpha)
            elif reg_type == "l2":
                if np.sum(X_j != 0) > 0:
                    params[j] = np.dot(X_j, residuals) / (np.sum(X_j**2) + alpha)
            elif reg_type == "elasticnet":
                if l1_ratio is None:
                    l1_ratio = 0.5
                if np.sum(X_j != 0) > 0:
                    rho = l1_ratio * alpha
                    lambda_ = (1 - l1_ratio) * alpha
                    params[j] = _elasticnet_threshold(np.dot(X_j, residuals),
                                                    rho, lambda_, np.sum(X_j**2))
            else:
                if np.sum(X_j != 0) > 0:
                    params[j] = np.dot(X_j, residuals) / np.sum(X_j**2)

        if _check_convergence(params, tol):
            break

    return params

def _compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                     metric_func: Callable, distance_func: Callable,
                     reg_type: str, alpha: float, l1_ratio: Optional[float]) -> np.ndarray:
    """Compute gradient for optimization."""
    predictions = X @ params
    error = y - predictions

    if metric_func == _mse:
        gradient = -2 * X.T @ error / len(y)
    elif metric_func == _mae:
        gradient = -X.T @ np.sign(error) / len(y)
    else:
        # For other metrics, we need to compute the derivative
        gradient = _compute_metric_gradient(X, y, params, metric_func)

    # Add regularization terms
    if reg_type == "l1":
        gradient += alpha * np.sign(params)
    elif reg_type == "l2":
        gradient += 2 * alpha * params
    elif reg_type == "elasticnet":
        if l1_ratio is None:
            l1_ratio = 0.5
        gradient += alpha * (l1_ratio * np.sign(params) + (1 - l1_ratio) * 2 * params)

    return gradient

def _compute_hessian(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                    metric_func: Callable, distance_func: Callable,
                    reg_type: str, alpha: float) -> np.ndarray:
    """Compute Hessian matrix for optimization."""
    predictions = X @ params
    error = y - predictions

    if metric_func == _mse:
        hessian = 2 * X.T @ X / len(y)
    else:
        # For other metrics, we need to compute the second derivative
        hessian = _compute_metric_hessian(X, y, params, metric_func)

    # Add regularization terms
    if reg_type == "l2":
        hessian += 2 * alpha * np.eye(X.shape[1])

    return hessian

def _compute_metrics(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                    metric_func: Callable) -> Dict:
    """Compute evaluation metrics."""
    predictions = X @ params
    return {
        "metric_value": metric_func(y, predictions),
        "mse": _mse(y, predictions),
        "mae": _mae(y, predictions),
        "r2": _r2(y, predictions)
    }

def _check_convergence(params: np.ndarray, tol: float) -> bool:
    """Check if parameters have converged."""
    return np.linalg.norm(params) < tol

def _soft_threshold(value: float, threshold: float) -> float:
    """Soft-thresholding operator for L1 regularization."""
    if value > threshold:
        return value - threshold
    elif value < -threshold:
        return value + threshold
    else:
        return 0

def _elasticnet_threshold(value: float, rho: float,
                         lambda_: float, penalty: float) -> float:
    """Thresholding operator for ElasticNet regularization."""
    if value > 3 * (rho + lambda_):
        return value - rho
    elif value < -3 * (rho + lambda_):
        return value + rho
    else:
        return 0

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log Loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean distance."""
    return np.linalg.norm(x - y, axis=1)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manhattan distance."""
    return np.sum(np.abs(x - y), axis=1)

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _compute_metric_gradient(X: np.ndarray, y: np.ndarray,
                           params: np.ndarray, metric_func: Callable) -> np.ndarray:
    """Compute gradient for custom metrics."""
    # This is a placeholder implementation
    # Actual implementation depends on the specific metric
    predictions = X @ params
    error = y - predictions

    if callable(metric_func):
        # For custom metrics, we need to compute the derivative numerically
        h = 1e-5
        gradient = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += h
            metric_plus = metric_func(y, X @ params_plus)

            params_minus = params.copy()
            params_minus[i] -= h
            metric_minus = metric_func(y, X @ params_minus)

            gradient[i] = (metric_plus - metric_minus) / (2 * h)

        return gradient
    else:
        raise ValueError("Unknown metric for gradient computation")

def _compute_metric_hessian(X: np.ndarray, y: np.ndarray,
                           params: np.ndarray, metric_func: Callable) -> np.ndarray:
    """Compute a numeric Hessian approximation for a custom metric.

    Notes:
    - Assumes the metric can be evaluated as `metric_func(y, X @ params)`.
    - Uses central finite differences; falls back to identity if evaluation fails.
    """
    n_params = len(params)
    h = 1e-5
    hessian = np.zeros((n_params, n_params))

    def _eval(p: np.ndarray) -> float:
        return float(metric_func(y, X @ p))

    try:
        for i in range(n_params):
            for j in range(n_params):
                p_pp = params.copy(); p_pp[i] += h; p_pp[j] += h
                p_pm = params.copy(); p_pm[i] += h; p_pm[j] -= h
                p_mp = params.copy(); p_mp[i] -= h; p_mp[j] += h
                p_mm = params.copy(); p_mm[i] -= h; p_mm[j] -= h
                f_pp = _eval(p_pp)
                f_pm = _eval(p_pm)
                f_mp = _eval(p_mp)
                f_mm = _eval(p_mm)
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
    except Exception:
        hessian = np.eye(n_params)

    return hessian

################################################################################
# kernel_size
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def kernel_size_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute the optimal kernel size for a given dataset and parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    metric : str or callable, optional
        Metric to optimize. Options: 'mse', 'mae', 'r2'. Default is 'mse'.
    distance : str, optional
        Distance metric for kernel. Options: 'euclidean', 'manhattan', 'cosine'. Default is 'euclidean'.
    solver : str, optional
        Solver to use. Options: 'closed_form', 'gradient_descent'. Default is 'closed_form'.
    normalization : str, optional
        Normalization method. Options: 'standard', 'minmax', 'robust'. Default is None.
    regularization : str, optional
        Regularization method. Options: 'l1', 'l2'. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : callable, optional
        Custom metric function. Default is None.
    custom_distance : callable, optional
        Custom distance function. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalization) if normalization else X

    # Choose metric
    if callable(metric):
        metric_func = metric
    else:
        metric_func = _get_metric_function(metric)

    # Choose distance
    if callable(distance):
        distance_func = distance
    else:
        distance_func = _get_distance_function(distance)

    # Choose solver
    if solver == 'closed_form':
        result = _closed_form_solver(X_normalized, y, distance_func)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(X_normalized, y, distance_func, metric_func,
                                         regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(y, result['y_pred'], metric_func)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'normalization': normalization,
            'regularization': regularization
        },
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric: str) -> Callable:
    """Get metric function based on name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable:
    """Get distance function based on name."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Euclidean distance."""
    if Y is None:
        Y = X
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

def _manhattan_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Manhattan distance."""
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)

def _cosine_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute cosine distance."""
    if Y is None:
        Y = X
    dot_products = np.dot(X, Y.T)
    norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis] * np.sqrt(np.sum(Y ** 2, axis=1))
    return 1 - (dot_products / norms)

def _closed_form_solver(X: np.ndarray, y: np.ndarray, distance_func: Callable) -> Dict[str, Any]:
    """Solve using closed form solution."""
    # This is a placeholder for the actual implementation
    kernel_matrix = distance_func(X)
    beta = np.linalg.solve(kernel_matrix + 1e-6 * np.eye(X.shape[0]), y)
    y_pred = kernel_matrix @ beta
    return {'y_pred': y_pred, 'beta': beta}

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable,
    metric_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using gradient descent."""
    # This is a placeholder for the actual implementation
    n_samples = X.shape[0]
    beta = np.zeros(n_samples)
    kernel_matrix = distance_func(X)

    for _ in range(max_iter):
        y_pred = kernel_matrix @ beta
        gradient = -2 * (y - y_pred).T @ kernel_matrix

        if regularization == 'l1':
            gradient += np.sign(beta)
        elif regularization == 'l2':
            gradient += 2 * beta

        beta_new = beta - tol * gradient
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new

    y_pred = kernel_matrix @ beta
    return {'y_pred': y_pred, 'beta': beta}

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> Dict[str, float]:
    """Compute metrics."""
    return {'metric_value': metric_func(y_true, y_pred)}

################################################################################
# stride
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def stride_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a model using the stride method with configurable hyperparameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features.
    metric : str
        Metric to evaluate the model performance. Options: 'mse', 'mae', 'r2', 'logloss'.
    distance : str
        Distance metric for the solver. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    solver : str
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2', 'elasticnet'.
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_normalized = normalizer(X)

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Choose distance
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance(distance)

    # Choose solver
    if solver == 'closed_form':
        params = _closed_form_solver(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(X_normalized, y, distance_func, regularization, tol, max_iter)
    elif solver == 'newton':
        params = _newton_solver(X_normalized, y, distance_func, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent_solver(X_normalized, y, distance_func, regularization, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Calculate metrics
    y_pred = _predict(X_normalized, params)
    metrics = {
        'metric': metric_func(y, y_pred),
        'mse': _mean_squared_error(y, y_pred),
        'mae': _mean_absolute_error(y, y_pred),
        'r2': _r_squared(y, y_pred)
    }

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(y, y_pred)
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate the input data."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the specified metric."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Invalid metric: {metric}. Available options are: {list(metrics.keys())}.")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the specified distance."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }
    if distance not in distances:
        raise ValueError(f"Invalid distance: {distance}. Available options are: {list(distances.keys())}.")
    return distances[distance]

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, distance_func, regularization)
        params -= learning_rate * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _newton_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton method solver."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = _compute_gradient(X, y, params, distance_func, regularization)
        hessian = _compute_hessian(X, y, params, distance_func, regularization)
        params -= np.linalg.inv(hessian) @ gradient
        if np.linalg.norm(gradient) < tol:
            break
    return params

def _coordinate_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            params_j_old = params[j]
            params[j] = _update_parameter(X_j, y, params, j, distance_func, regularization)
            if np.abs(params[j] - params_j_old) < tol:
                break
    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str]
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    y_pred = _predict(X, params)
    error = y - y_pred
    gradient = -(X.T @ error) / len(y)
    if regularization == 'l1':
        gradient += np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * params
    return gradient

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str]
) -> np.ndarray:
    """Compute the Hessian matrix of the loss function."""
    hessian = (X.T @ X) / len(y)
    if regularization == 'l2':
        hessian += 2 * np.eye(X.shape[1])
    return hessian

def _update_parameter(
    X_j: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    j: int,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str]
) -> float:
    """Update a single parameter in coordinate descent."""
    params_j_old = params[j]
    params[j] = 0
    y_pred = _predict(X, params)
    error = y - y_pred
    numerator = X_j.T @ error
    denominator = X_j.T @ X_j
    if regularization == 'l1':
        params[j] = np.sign(numerator) * np.maximum(np.abs(numerator) - 1, 0) / denominator
    elif regularization == 'l2':
        params[j] = numerator / (denominator + 2)
    else:
        params[j] = numerator / denominator
    return params[j]

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict using the current parameters."""
    return X @ params

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine distance."""
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

def _check_warnings(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.any(np.isnan(y_pred)):
        warnings.append("Predictions contain NaN values.")
    if np.any(np.isinf(y_pred)):
        warnings.append("Predictions contain infinite values.")
    return warnings

################################################################################
# padding
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def padding_fit(
    X: np.ndarray,
    target_length: int,
    normalizer: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: str = "none",
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_normalizer: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit padding parameters to extend or truncate data arrays to a target length.

    Parameters:
    -----------
    X : np.ndarray
        Input data array of shape (n_samples, n_features).
    target_length : int
        Desired length for the output array.
    normalizer : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to evaluate padding quality: "mse", "mae", "r2", or custom callable.
    distance : str, optional
        Distance metric for padding: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", or "newton".
    regularization : str, optional
        Regularization type: "none", "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_normalizer : callable, optional
        Custom normalization function.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, target_length)

    # Normalize data if required
    normalized_X = _apply_normalization(X, normalizer, custom_normalizer)

    # Choose metric
    if isinstance(metric, str):
        metric_func = _get_metric(metric)
    else:
        metric_func = metric

    # Choose distance
    if isinstance(distance, str):
        distance_func = _get_distance(distance)
    else:
        distance_func = distance

    # Solve for padding parameters
    params = _solve_padding(
        normalized_X,
        target_length,
        solver,
        distance_func,
        regularization,
        tol,
        max_iter
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_X, params, metric_func)

    # Prepare output
    result = {
        "result": _apply_padding(X, params),
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": _check_warnings(normalized_X, params)
    }

    return result

def _validate_inputs(X: np.ndarray, target_length: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not isinstance(target_length, int) or target_length <= 0:
        raise ValueError("target_length must be a positive integer")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

def _apply_normalization(
    X: np.ndarray,
    method: str,
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Apply normalization to the input data."""
    if custom_func is not None:
        return custom_func(X)

    if method == "standard":
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == "robust":
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        return X

def _get_metric(metric: str) -> Callable:
    """Get the metric function based on the input string."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable:
    """Get the distance function based on the input string."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_padding(
    X: np.ndarray,
    target_length: int,
    solver: str,
    distance_func: Callable,
    regularization: str,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve for padding parameters using the specified solver."""
    if solver == "closed_form":
        return _solve_closed_form(X, target_length, distance_func)
    elif solver == "gradient_descent":
        return _solve_gradient_descent(X, target_length, distance_func, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _compute_metrics(
    X: np.ndarray,
    params: Dict[str, Any],
    metric_func: Callable
) -> Dict[str, float]:
    """Compute metrics for the padding results."""
    padded_X = _apply_padding(X, params)
    return {"metric": metric_func(padded_X)}

def _check_warnings(X: np.ndarray, params: Dict[str, Any]) -> List[str]:
    """Check for potential warnings in the padding process."""
    warnings = []
    if np.any(np.isnan(X)):
        warnings.append("Input data contains NaN values")
    if params.get("convergence", False):
        warnings.append("Solver did not converge within tolerance")
    return warnings

def _apply_padding(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply padding to the input data using the computed parameters."""
    # Implementation depends on the specific padding strategy
    pass

def _mean_squared_error(X: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean(np.square(X))

def _mean_absolute_error(X: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(X))

def _r_squared(X: np.ndarray) -> float:
    """Compute R-squared."""
    pass

def _euclidean_distance(X: np.ndarray) -> float:
    """Compute Euclidean distance."""
    pass

def _manhattan_distance(X: np.ndarray) -> float:
    """Compute Manhattan distance."""
    pass

def _cosine_distance(X: np.ndarray) -> float:
    """Compute cosine distance."""
    pass

def _solve_closed_form(X: np.ndarray, target_length: int, distance_func: Callable) -> Dict[str, Any]:
    """Solve padding parameters using closed-form solution."""
    pass

def _solve_gradient_descent(
    X: np.ndarray,
    target_length: int,
    distance_func: Callable,
    regularization: str,
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve padding parameters using gradient descent."""
    pass

################################################################################
# filter_count
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def filter_count_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a filter count model with configurable hyperparameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input features
    metric : str
        Metric to evaluate the model ('mse', 'mae', 'r2', 'logloss')
    distance : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_norm = normalizer(X)

    # Choose distance function
    if custom_distance:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_norm, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_norm, y, distance_func, tol, max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_norm, y, distance_func, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_norm, y, distance_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        params = _apply_regularization(params, X_norm, y, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y, params, metric, custom_metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance if not custom_distance else 'custom',
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(X_norm, y)
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

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the specified distance function."""
    if distance == 'euclidean':
        return lambda a, b: np.linalg.norm(a - b)
    elif distance == 'manhattan':
        return lambda a, b: np.sum(np.abs(a - b))
    elif distance == 'cosine':
        return lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif distance == 'minkowski':
        return lambda a, b: np.sum(np.abs(a - b)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using closed form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        params -= tol * gradient
    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using Newton's method."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        gradient = 2 * X.T @ (X @ params - y)
        hessian = 2 * X.T @ X
        params -= np.linalg.pinv(hessian) @ gradient
    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve using coordinate descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    for _ in range(max_iter):
        for i in range(n_features):
            X_i = X[:, i]
            params[i] = (y - X.dot(params) + params[i]*X_i).dot(X_i) / X_i.dot(X_i)
    return params

def _apply_regularization(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply specified regularization."""
    if regularization == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - 0.1, 0)
    elif regularization == 'l2':
        return params / (1 + 0.1 * np.linalg.norm(params))
    elif regularization == 'elasticnet':
        return (np.sign(params) * np.maximum(np.abs(params) - 0.1, 0) +
                params / (1 + 0.1 * np.linalg.norm(params))) / 2
    else:
        raise ValueError(f"Unknown regularization: {regularization}")

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate specified metrics."""
    y_pred = X @ params
    if custom_metric:
        return {'custom': custom_metric(y, y_pred)}
    elif metric == 'mse':
        return {'mse': np.mean((y - y_pred)**2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(y - y_pred))}
    elif metric == 'r2':
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return {'r2': 1 - ss_res / ss_tot}
    elif metric == 'logloss':
        return {'logloss': -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _check_warnings(X: np.ndarray, y: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if np.linalg.cond(X) > 1e6:
        warnings.append("High condition number detected - potential numerical instability")
    if np.any(np.isclose(y, 0)) and 'logloss' in _calculate_metrics(X, y, np.zeros(X.shape[1]), 'mse', None):
        warnings.append("Log loss may be undefined due to zero values in y")
    return warnings

################################################################################
# sequence_length
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def sequence_length_fit(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, dict]]:
    """
    Compute the optimal sequence length for time series data based on specified criteria.

    Parameters
    ----------
    X : np.ndarray
        Input time series data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    normalization : str, optional
        Normalization method (None, 'standard', 'minmax', 'robust').
    regularization : str, optional
        Regularization type (None, 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Optimal sequence length.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = sequence_length_fit(X, y, metric='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalization) if normalization else X

    # Initialize parameters
    params_used = {
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'normalization': normalization,
        'regularization': regularization
    }

    # Select solver and compute optimal sequence length
    if solver == 'closed_form':
        result = _closed_form_solver(X_normalized, y, metric, custom_metric)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(X_normalized, y, metric, custom_metric,
                                         distance, custom_distance, tol, max_iter)
    elif solver == 'newton':
        result = _newton_solver(X_normalized, y, metric, custom_metric,
                               distance, custom_distance, tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _coordinate_descent_solver(X_normalized, y, metric, custom_metric,
                                           distance, custom_distance, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, result, metric, custom_metric)

    # Return results
    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _closed_form_solver(X: np.ndarray, y: np.ndarray,
                       metric: str, custom_metric: Optional[Callable]) -> float:
    """Compute optimal sequence length using closed-form solution."""
    # Placeholder for actual implementation
    return 5.0

def _gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                            metric: str, custom_metric: Optional[Callable],
                            distance: str, custom_distance: Optional[Callable],
                            tol: float, max_iter: int) -> float:
    """Compute optimal sequence length using gradient descent."""
    # Placeholder for actual implementation
    return 5.0

def _newton_solver(X: np.ndarray, y: np.ndarray,
                   metric: str, custom_metric: Optional[Callable],
                   distance: str, custom_distance: Optional[Callable],
                   tol: float, max_iter: int) -> float:
    """Compute optimal sequence length using Newton's method."""
    # Placeholder for actual implementation
    return 5.0

def _coordinate_descent_solver(X: np.ndarray, y: np.ndarray,
                              metric: str, custom_metric: Optional[Callable],
                              distance: str, custom_distance: Optional[Callable],
                              tol: float, max_iter: int) -> float:
    """Compute optimal sequence length using coordinate descent."""
    # Placeholder for actual implementation
    return 5.0

def _compute_metrics(X: np.ndarray, y: np.ndarray,
                    result: float, metric: str, custom_metric: Optional[Callable]) -> dict:
    """Compute metrics based on the computed result."""
    if custom_metric is not None:
        return {'custom': custom_metric(X, y, result)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((y - result) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - result))
    elif metric == 'r2':
        ss_res = np.sum((y - result) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        metrics['logloss'] = -np.mean(y * np.log(result) + (1 - y) * np.log(1 - result))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# embedding_dimension
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def embedding_dimension_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    normalizer: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Estimate the optimal embedding dimension for a dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    normalizer : str
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    metric : Union[str, Callable]
        Metric to optimize: 'mse', 'mae', 'r2', or custom callable
    distance : Union[str, Callable]
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable
    solver : str
        Solver method: 'closed_form', 'gradient_descent'
    regularization : Optional[str]
        Regularization type: None, 'l1', 'l2', or 'elasticnet'
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function if needed
    custom_distance : Optional[Callable]
        Custom distance function if needed

    Returns:
    --------
    Dict containing:
        - 'result': Optimal embedding dimension
        - 'metrics': Dictionary of computed metrics
        - 'params_used': Parameters actually used
        - 'warnings': Any warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = embedding_dimension_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalizer)

    # Select metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Select distance function
    distance_func = _get_distance_function(distance, custom_distance)

    # Initialize parameters dictionary
    params_used = {
        'normalizer': normalizer,
        'metric': metric if isinstance(metric, str) else 'custom',
        'distance': distance if isinstance(distance, str) else 'custom',
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Determine optimal embedding dimension
    best_dim, metrics = _find_optimal_embedding(
        X_normalized,
        y,
        metric_func,
        distance_func,
        solver,
        regularization,
        tol,
        max_iter
    )

    # Prepare output dictionary
    result_dict = {
        'result': best_dim,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

    return result_dict

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        return (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Return the appropriate metric function."""
    if custom_metric is not None:
        return custom_metric
    if isinstance(metric, str):
        metric = metric.lower()
        if metric == 'mse':
            return _mean_squared_error
        elif metric == 'mae':
            return _mean_absolute_error
        elif metric == 'r2':
            return _r_squared
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metric

def _get_distance_function(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Return the appropriate distance function."""
    if custom_distance is not None:
        return custom_distance
    if isinstance(distance, str):
        distance = distance.lower()
        if distance == 'euclidean':
            return _euclidean_distance
        elif distance == 'manhattan':
            return _manhattan_distance
        elif distance == 'cosine':
            return _cosine_distance
        else:
            raise ValueError(f"Unknown distance: {distance}")
    return distance

def _find_optimal_embedding(
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric_func: Callable,
    distance_func: Callable,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple:
    """Find the optimal embedding dimension."""
    n_samples, n_features = X.shape
    max_dim = min(n_samples, n_features)

    best_score = -np.inf
    best_dim = 1
    metrics = {}

    for dim in range(1, max_dim + 1):
        # Project data to current dimension
        X_projected = _project_to_dimension(X, dim)

        # Fit model with current dimension
        if y is not None:
            params = _fit_model(X_projected, y, solver, regularization, tol, max_iter)
        else:
            params = _fit_unsupervised(X_projected, distance_func, solver, tol, max_iter)

        # Calculate metrics
        current_metrics = _calculate_metrics(X_projected, y, params, metric_func)

        # Update best dimension if needed
        current_score = current_metrics.get('score', 0)
        if current_score > best_score:
            best_score = current_score
            best_dim = dim

        metrics[dim] = current_metrics

    return best_dim, metrics

def _project_to_dimension(X: np.ndarray, dim: int) -> np.ndarray:
    """Project data to specified dimension using PCA."""
    if dim >= X.shape[1]:
        return X
    # Simple PCA implementation for demonstration
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    return X @ eigvecs[:, :dim]

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Fit a model to the projected data."""
    if solver == 'closed_form':
        return _fit_closed_form(X, y, regularization)
    elif solver == 'gradient_descent':
        return _fit_gradient_descent(X, y, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _fit_unsupervised(
    X: np.ndarray,
    distance_func: Callable,
    solver: str,
    tol: float,
    max_iter: int
) -> Dict:
    """Fit an unsupervised model to the projected data."""
    if solver == 'closed_form':
        return _fit_unsupervised_closed_form(X, distance_func)
    elif solver == 'gradient_descent':
        return _fit_unsupervised_gradient_descent(X, distance_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def _calculate_metrics(
    X: np.ndarray,
    y: Optional[np.ndarray],
    params: Dict,
    metric_func: Callable
) -> Dict:
    """Calculate metrics for the current model."""
    metrics = {}

    if y is not None:
        # Supervised case
        predictions = _predict(X, params)
        metrics['score'] = metric_func(y, predictions)

        # Additional common metrics
        if isinstance(metric_func, str) and metric_func.lower() != 'r2':
            metrics['r2'] = _r_squared(y, predictions)
    else:
        # Unsupervised case
        metrics['score'] = _unsupervised_score(X, params)

    return metrics

# Example metric functions
def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Example distance functions
def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example model fitting functions
def _fit_closed_form(X: np.ndarray, y: np.ndarray, regularization: Optional[str]) -> Dict:
    """Fit model using closed-form solution."""
    X_tx = np.dot(X.T, X)
    if regularization == 'l2':
        X_tx += 1e-4 * np.eye(X.shape[1])
    params = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return {'coefficients': params}

def _fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Fit model using gradient descent."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = np.dot(X.T, (np.dot(X, params) - y)) / X.shape[0]

        if regularization == 'l2':
            gradients += 1e-4 * params

        new_params = params - learning_rate * gradients
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return {'coefficients': params}

def _unsupervised_score(X: np.ndarray, params: Dict) -> float:
    """Calculate score for unsupervised case."""
    # Simple example: variance of projected data
    return np.var(X @ params['projection_matrix'])

def _predict(X: np.ndarray, params: Dict) -> np.ndarray:
    """Make predictions using fitted model."""
    return X @ params['coefficients']

################################################################################
# rnn_units
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def rnn_units_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_units: int = 100,
    activation: str = 'tanh',
    solver: str = 'adam',
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    validation_split: float = 0.2,
    metric: Union[str, Callable] = 'mse',
    normalize: str = 'standard',
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5,
    tol: float = 1e-4,
    early_stopping: bool = True,
    patience: int = 5,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit RNN units to the input data and return the results.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    n_units : int, optional
        Number of units in the RNN layer (default: 100).
    activation : str, optional
        Activation function ('tanh', 'relu', 'sigmoid') (default: 'tanh').
    solver : str, optional
        Solver ('adam', 'sgd', 'rmsprop') (default: 'adam').
    learning_rate : float, optional
        Learning rate for the solver (default: 0.001).
    batch_size : int, optional
        Batch size for training (default: 32).
    epochs : int, optional
        Number of epochs to train (default: 100).
    validation_split : float, optional
        Fraction of data to use for validation (default: 0.2).
    metric : Union[str, Callable], optional
        Metric to evaluate ('mse', 'mae', 'r2') or custom callable (default: 'mse').
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'standard').
    regularization : Optional[str], optional
        Regularization method ('none', 'l1', 'l2', 'elasticnet') (default: None).
    l1_ratio : float, optional
        Ratio of L1 regularization for elasticnet (default: 0.5).
    tol : float, optional
        Tolerance for early stopping (default: 1e-4).
    early_stopping : bool, optional
        Whether to use early stopping (default: True).
    patience : int, optional
        Number of epochs to wait before early stopping (default: 5).
    random_state : Optional[int], optional
        Random seed for reproducibility (default: None).
    custom_metric : Optional[Callable], optional
        Custom metric function (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalize)

    # Initialize RNN units
    rnn_units = _initialize_rnn_units(n_units, random_state)

    # Train RNN units
    history = _train_rnn_units(
        X_normalized, y, rnn_units,
        activation=activation,
        solver=solver,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        metric=metric,
        regularization=regularization,
        l1_ratio=l1_ratio,
        tol=tol,
        early_stopping=early_stopping,
        patience=patience
    )

    # Compute metrics
    metrics = _compute_metrics(y, history['y_pred'], metric, custom_metric)

    # Prepare results
    result = {
        'result': history,
        'metrics': metrics,
        'params_used': {
            'n_units': n_units,
            'activation': activation,
            'solver': solver,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'validation_split': validation_split,
            'metric': metric if isinstance(metric, str) else 'custom',
            'normalize': normalize,
            'regularization': regularization,
            'l1_ratio': l1_ratio,
            'tol': tol,
            'early_stopping': early_stopping,
            'patience': patience
        },
        'warnings': history.get('warnings', [])
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
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

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data."""
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

def _initialize_rnn_units(n_units: int, random_state: Optional[int] = None) -> Dict[str, Any]:
    """Initialize RNN units with random weights."""
    if random_state is not None:
        np.random.seed(random_state)
    weights = np.random.randn(n_units, n_units) * 0.1
    biases = np.zeros(n_units)
    return {'weights': weights, 'biases': biases}

def _train_rnn_units(
    X: np.ndarray,
    y: np.ndarray,
    rnn_units: Dict[str, Any],
    activation: str = 'tanh',
    solver: str = 'adam',
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    validation_split: float = 0.2,
    metric: Union[str, Callable] = 'mse',
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5,
    tol: float = 1e-4,
    early_stopping: bool = True,
    patience: int = 5
) -> Dict[str, Any]:
    """Train RNN units using the specified solver and parameters."""
    # Split data into training and validation sets
    split_idx = int(X.shape[0] * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Initialize history
    history = {
        'loss': [],
        'val_loss': [],
        'y_pred': None,
        'warnings': []
    }

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch training
        for i in range(0, X_shuffled.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Forward pass
            output, _ = _forward_pass(X_batch, rnn_units, activation)

            # Compute loss
            loss = _compute_loss(output, y_batch, metric, regularization, l1_ratio)

            # Backward pass and update weights
            _backward_pass(X_batch, y_batch, rnn_units, activation, solver, learning_rate)

        # Validation
        val_output, _ = _forward_pass(X_val, rnn_units, activation)
        val_loss = _compute_loss(val_output, y_val, metric, regularization, l1_ratio)

        # Store results
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)

        # Early stopping check
        if early_stopping and val_loss < best_val_loss - tol:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                history['warnings'].append(f"Early stopping triggered at epoch {epoch}.")
                break

    # Final prediction
    history['y_pred'], _ = _forward_pass(X, rnn_units, activation)

    return history

def _forward_pass(X: np.ndarray, rnn_units: Dict[str, Any], activation: str) -> tuple:
    """Perform forward pass through RNN units."""
    weights = rnn_units['weights']
    biases = rnn_units['biases']

    # Initialize hidden state
    h_prev = np.zeros((X.shape[0], weights.shape[0]))

    # Forward pass
    for t in range(X.shape[1]):
        h_prev = _activation_function(h_prev @ weights + X[:, t:t+1] + biases, activation)

    return h_prev, None

def _activation_function(x: np.ndarray, activation: str) -> np.ndarray:
    """Apply activation function."""
    if activation == 'tanh':
        return np.tanh(x)
    elif activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError(f"Unknown activation function: {activation}")

def _backward_pass(
    X: np.ndarray,
    y: np.ndarray,
    rnn_units: Dict[str, Any],
    activation: str,
    solver: str,
    learning_rate: float
) -> None:
    """Perform backward pass and update weights."""
    # Placeholder for actual backward pass implementation
    pass

def _compute_loss(
    output: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    regularization: Optional[str] = None,
    l1_ratio: float = 0.5
) -> float:
    """Compute loss with optional regularization."""
    if isinstance(metric, str):
        if metric == 'mse':
            loss = np.mean((output - y) ** 2)
        elif metric == 'mae':
            loss = np.mean(np.abs(output - y))
        elif metric == 'r2':
            ss_res = np.sum((output - y) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            loss = 1 - ss_res / (ss_tot + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        loss = metric(output, y)

    # Add regularization
    if regularization == 'l1':
        loss += np.sum(np.abs(rnn_units['weights'])) * l1_ratio
    elif regularization == 'l2':
        loss += np.sum(rnn_units['weights'] ** 2) * (1 - l1_ratio)
    elif regularization == 'elasticnet':
        loss += np.sum(np.abs(rnn_units['weights'])) * l1_ratio + np.sum(rnn_units['weights'] ** 2) * (1 - l1_ratio)

    return loss

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute metrics for the predictions."""
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics['mse'] = np.mean((y_pred - y_true) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(y_pred - y_true))
        elif metric == 'r2':
            ss_res = np.sum((y_pred - y_true) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - ss_res / (ss_tot + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metrics['custom'] = metric(y_pred, y_true)

    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(y_pred, y_true)

    return metrics

################################################################################
# temperature
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'none',
    distance_metric: Union[str, Callable] = 'euclidean'
) -> None:
    """Validate input data and parameters."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains infinite values")

    if isinstance(distance_metric, str):
        allowed_distances = ['euclidean', 'manhattan', 'cosine', 'minkowski']
        if distance_metric not in allowed_distances:
            raise ValueError(f"distance_metric must be one of {allowed_distances}")

def _normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'none'
) -> tuple:
    """Normalize input data based on specified method."""
    if normalize == 'none':
        return X, y
    elif normalize == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-8)
        return X_normalized, y_normalized
    elif normalize == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
        return X_normalized, y_normalized
    elif normalize == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_normalized = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        y_normalized = (y - y_median) / (y_iqr + 1e-8)
        return X_normalized, y_normalized
    else:
        raise ValueError("normalize must be one of 'none', 'standard', 'minmax', or 'robust'")

def _compute_distance(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: Union[str, Callable] = 'euclidean',
    p: float = 2.0
) -> np.ndarray:
    """Compute distance between samples based on specified metric."""
    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            return np.sqrt(np.sum((X - y)**2, axis=1))
        elif distance_metric == 'manhattan':
            return np.sum(np.abs(X - y), axis=1)
        elif distance_metric == 'cosine':
            return 1 - np.sum(X * y, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(y))
        elif distance_metric == 'minkowski':
            return np.sum(np.abs(X - y)**p, axis=1) ** (1/p)
    else:
        return distance_metric(X, y)

def _compute_temperature(
    distances: np.ndarray,
    solver: str = 'closed_form',
    regularization: str = 'none',
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Union[np.ndarray, float]]:
    """Compute temperature parameter based on specified solver."""
    if solver == 'closed_form':
        if regularization == 'none':
            temperature = np.mean(distances)
        elif regularization == 'l1':
            temperature = np.mean(np.abs(distances - alpha))
        elif regularization == 'l2':
            temperature = np.sqrt(np.mean((distances - alpha)**2))
        else:
            raise ValueError("regularization must be one of 'none', 'l1', or 'l2'")
    elif solver == 'gradient_descent':
        # Implement gradient descent logic here
        pass
    else:
        raise ValueError("solver must be one of 'closed_form' or 'gradient_descent'")

    return {'temperature': temperature}

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = 'mse'
) -> Dict[str, float]:
    """Compute performance metrics."""
    if isinstance(metric, str):
        if metric == 'mse':
            return {'mse': np.mean((y_true - y_pred)**2)}
        elif metric == 'mae':
            return {'mae': np.mean(np.abs(y_true - y_pred))}
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            return {'r2': 1 - (ss_res / (ss_tot + 1e-8))}
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return {'logloss': -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))}
    else:
        return {metric.__name__: metric(y_true, y_pred)}

def temperature_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = 'none',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: str = 'none',
    alpha: float = 1.0,
    metric: Union[str, Callable] = 'mse',
    p: float = 2.0,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute temperature parameter for hyperparameter tuning.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str or callable, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent')
    regularization : str, optional
        Regularization method ('none', 'l1', 'l2')
    alpha : float, optional
        Regularization strength
    metric : str or callable, optional
        Performance metric to compute ('mse', 'mae', 'r2', 'logloss')
    p : float, optional
        Power parameter for Minkowski distance (only used if distance_metric='minkowski')
    max_iter : int, optional
        Maximum number of iterations for iterative solvers
    tol : float, optional
        Tolerance for convergence

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Computed temperature parameter
        - 'metrics': Performance metrics
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated during computation

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = temperature_fit(X, y, normalize='standard', distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(X, y, normalize, distance_metric)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalize)

    # Compute distances
    distances = _compute_distance(X_norm, y_norm, distance_metric, p)

    # Compute temperature
    result = _compute_temperature(distances, solver, regularization, alpha, max_iter, tol)

    # Compute metrics
    metrics = _compute_metrics(y, y_norm if normalize != 'none' else y, metric)

    # Prepare output
    output = {
        'result': result['temperature'],
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'distance_metric': distance_metric if isinstance(distance_metric, str) else 'custom',
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'metric': metric if isinstance(metric, str) else 'custom',
            'p': p,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

    return output

################################################################################
# beam_width
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def beam_width_fit(
    X: np.ndarray,
    y: np.ndarray,
    beam_size: int = 5,
    max_iterations: int = 100,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    normalize: bool = True,
    tol: float = 1e-4,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit a beam search algorithm to find optimal hyperparameters.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    beam_size : int, optional
        Number of candidates to keep at each iteration (default: 5).
    max_iterations : int, optional
        Maximum number of iterations (default: 100).
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable (default: 'mse').
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable (default: 'euclidean').
    solver : str, optional
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent') (default: 'gradient_descent').
    normalize : bool, optional
        Whether to normalize the data (default: True).
    tol : float, optional
        Tolerance for early stopping (default: 1e-4).
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.
    **kwargs :
        Additional solver-specific parameters.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = beam_width_fit(X, y, beam_size=3, max_iterations=50)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_normalized = _normalize_data(X) if normalize else X

    # Initialize beam search
    best_params, best_metric = _beam_search(
        X_normalized,
        y,
        beam_size=beam_size,
        max_iterations=max_iterations,
        metric=metric,
        distance=distance,
        solver=solver,
        tol=tol,
        custom_metric=custom_metric,
        custom_distance=custom_distance,
        **kwargs
    )

    # Compute metrics on best parameters
    metrics = _compute_metrics(X_normalized, y, best_params, metric)

    return {
        "result": best_metric,
        "metrics": metrics,
        "params_used": best_params,
        "warnings": []
    }

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays.")
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

def _normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize data using standard scaling."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def _beam_search(
    X: np.ndarray,
    y: np.ndarray,
    beam_size: int,
    max_iterations: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    tol: float,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    **kwargs
) -> tuple:
    """Perform beam search to find optimal hyperparameters."""
    # Initialize beam with random candidates
    candidates = _initialize_candidates(X.shape[1], beam_size)

    for _ in range(max_iterations):
        new_candidates = []
        best_metric = None

        for candidate in candidates:
            # Fit model with current candidate
            params, current_metric = _fit_model(
                X,
                y,
                candidate,
                metric,
                distance,
                solver,
                custom_metric=custom_metric,
                custom_distance=custom_distance,
                **kwargs
            )

            # Update best metric if needed
            if best_metric is None or current_metric < best_metric:
                best_metric = current_metric

            # Generate new candidates
            new_candidates.extend(_generate_new_candidates(candidate, beam_size))

        # Select top candidates
        candidates = _select_top_candidates(new_candidates, X, y, metric, beam_size)

        # Early stopping if improvement is below tolerance
        if best_metric is not None and np.abs(best_metric - current_metric) < tol:
            break

    # Return best parameters and metric
    return candidates[0], best_metric

def _initialize_candidates(n_features: int, beam_size: int) -> list:
    """Initialize random candidates for hyperparameters."""
    return [np.random.rand(n_features) for _ in range(beam_size)]

def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    **kwargs
) -> tuple:
    """Fit model with given parameters and compute metric."""
    # Choose solver
    if solver == 'gradient_descent':
        optimized_params = _gradient_descent(X, y, params, **kwargs)
    elif solver == 'newton':
        optimized_params = _newton_method(X, y, params, **kwargs)
    elif solver == 'coordinate_descent':
        optimized_params = _coordinate_descent(X, y, params, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metric
    if custom_metric:
        current_metric = custom_metric(y, _predict(X, optimized_params))
    else:
        current_metric = _compute_metric(y, _predict(X, optimized_params), metric)

    return optimized_params, current_metric

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    learning_rate: float = 0.01,
    n_iter: int = 100
) -> np.ndarray:
    """Perform gradient descent optimization."""
    for _ in range(n_iter):
        gradient = 2 * X.T @ (X @ params - y)
        params -= learning_rate * gradient
    return params

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    n_iter: int = 10
) -> np.ndarray:
    """Perform Newton's method optimization."""
    for _ in range(n_iter):
        gradient = 2 * X.T @ (X @ params - y)
        hessian = 2 * X.T @ X
        params -= np.linalg.inv(hessian) @ gradient
    return params

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    n_iter: int = 100
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    for _ in range(n_iter):
        for i in range(X.shape[1]):
            X_i = X[:, i]
            params[i] = np.linalg.lstsq(X_i.reshape(-1, 1), y - X @ params + X_i * params[i], rcond=None)[0]
    return params

def _generate_new_candidates(
    current_params: np.ndarray,
    beam_size: int
) -> list:
    """Generate new candidates by perturbing current parameters."""
    perturbations = np.random.normal(0, 0.1, size=(beam_size, len(current_params)))
    return [current_params + perturbation for perturbation in perturbations]

def _select_top_candidates(
    candidates: list,
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable],
    beam_size: int
) -> list:
    """Select top candidates based on metric."""
    candidate_scores = []
    for params in candidates:
        prediction = _predict(X, params)
        if isinstance(metric, str):
            score = _compute_metric(y, prediction, metric)
        else:
            score = metric(y, prediction)
        candidate_scores.append((params, score))

    # Sort candidates by score and select top beam_size
    candidate_scores.sort(key=lambda x: x[1])
    return [params for params, _ in candidate_scores[:beam_size]]

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str
) -> float:
    """Compute specified metric."""
    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    elif metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute multiple metrics for the given parameters."""
    y_pred = _predict(X, params)
    metrics = {}

    if isinstance(metric, str):
        metrics[metric] = _compute_metric(y, y_pred, metric)
    else:
        metrics['custom'] = metric(y, y_pred)

    # Compute additional common metrics
    metrics['mse'] = np.mean((y - y_pred) ** 2)
    metrics['mae'] = np.mean(np.abs(y - y_pred))
    metrics['r2'] = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    return metrics

def _predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Make predictions using the given parameters."""
    return X @ params
