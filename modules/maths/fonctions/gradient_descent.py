"""
Quantix – Module gradient_descent
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# learning_rate
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def learning_rate_fit(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_func: Callable[[float, float], float] = lambda x, y: abs(x - y),
    solver: str = 'gradient_descent',
    **kwargs
) -> Dict[str, Any]:
    """
    Optimize the learning rate for gradient descent using a given objective function.

    Parameters:
    -----------
    objective_func : callable
        Function to minimize, takes parameters as input and returns a scalar.
    gradient_func : callable
        Gradient of the objective function, takes parameters as input and returns a vector.
    initial_params : np.ndarray
        Initial parameters for optimization.
    learning_rate : float, optional
        Initial learning rate (default: 0.01).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    normalizer : callable, optional
        Function to normalize parameters (default: None).
    metric_func : callable, optional
        Metric function to compare objective values (default: absolute difference).
    solver : str, optional
        Solver method ('gradient_descent', 'adam', etc.) (default: 'gradient_descent').

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_func, gradient_func, initial_params)

    # Initialize parameters and metrics
    params = initial_params.copy()
    if normalizer is not None:
        params = normalizer(params)

    # Solver selection
    if solver == 'gradient_descent':
        result, metrics = _gradient_descent_solver(
            objective_func, gradient_func, params,
            learning_rate, max_iter, tol, metric_func
        )
    elif solver == 'adam':
        result, metrics = _adam_solver(
            objective_func, gradient_func, params,
            learning_rate, max_iter, tol, metric_func
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": params.tolist(),
        "warnings": []
    }

    return output

def _validate_inputs(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray
) -> None:
    """
    Validate input functions and parameters.

    Parameters:
    -----------
    objective_func : callable
        Function to minimize.
    gradient_func : callable
        Gradient of the objective function.
    params : np.ndarray
        Initial parameters.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not callable(objective_func):
        raise ValueError("objective_func must be a callable")
    if not callable(gradient_func):
        raise ValueError("gradient_func must be a callable")
    if not isinstance(params, np.ndarray):
        raise ValueError("params must be a numpy array")
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        raise ValueError("params must not contain NaN or inf values")

def _gradient_descent_solver(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    learning_rate: float,
    max_iter: int,
    tol: float,
    metric_func: Callable[[float, float], float]
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform gradient descent optimization.

    Parameters:
    -----------
    objective_func : callable
        Function to minimize.
    gradient_func : callable
        Gradient of the objective function.
    params : np.ndarray
        Initial parameters.
    learning_rate : float
        Learning rate for gradient descent.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    metric_func : callable
        Metric function to compare objective values.

    Returns:
    --------
    tuple
        Optimized parameters and metrics dictionary.
    """
    current_params = params.copy()
    prev_value = objective_func(current_params)
    metrics = {"objective_values": [], "gradients": []}

    for i in range(max_iter):
        gradient = gradient_func(current_params)
        current_params -= learning_rate * gradient
        current_value = objective_func(current_params)

        metrics["objective_values"].append(current_value)
        metrics["gradients"].append(np.linalg.norm(gradient))

        if metric_func(current_value, prev_value) < tol:
            break
        prev_value = current_value

    return current_params, metrics

def _adam_solver(
    objective_func: Callable[[np.ndarray], float],
    gradient_func: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    learning_rate: float,
    max_iter: int,
    tol: float,
    metric_func: Callable[[float, float], float]
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform Adam optimization.

    Parameters:
    -----------
    objective_func : callable
        Function to minimize.
    gradient_func : callable
        Gradient of the objective function.
    params : np.ndarray
        Initial parameters.
    learning_rate : float
        Learning rate for Adam.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    metric_func : callable
        Metric function to compare objective values.

    Returns:
    --------
    tuple
        Optimized parameters and metrics dictionary.
    """
    current_params = params.copy()
    prev_value = objective_func(current_params)
    metrics = {"objective_values": [], "gradients": []}

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8

    for i in range(max_iter):
        gradient = gradient_func(current_params)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))

        current_params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        current_value = objective_func(current_params)

        metrics["objective_values"].append(current_value)
        metrics["gradients"].append(np.linalg.norm(gradient))

        if metric_func(current_value, prev_value) < tol:
            break
        prev_value = current_value

    return current_params, metrics

################################################################################
# batch_size
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def batch_size_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    batch_size: int,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    normalize: str = 'standard',
    metric_func: Callable[[np.ndarray, np.ndarray], float] = None,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform batch gradient descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    loss_func : Callable[[np.ndarray, np.ndarray], float]
        Loss function to minimize
    batch_size : int
        Size of the batches for mini-batch gradient descent
    learning_rate : float, default=0.01
        Learning rate for the gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for stopping criterion
    normalize : str, default='standard'
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric_func : Callable[[np.ndarray, np.ndarray], float], optional
        Metric function to evaluate performance
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Optimized parameters
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the optimization
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> def mse_loss(y_true, y_pred):
    ...     return np.mean((y_true - y_pred) ** 2)
    >>> result = batch_size_fit(X, y, loss_func=mse_loss, batch_size=32)
    """
    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(X, y)
    n_samples = X.shape[0]

    # Normalize data
    if normalize != 'none':
        X, y = _apply_normalization(X, y, method=normalize)

    # Initialize parameters
    n_features = X.shape[1]
    theta = np.zeros(n_features)

    # Prepare output dictionary
    output = {
        'result': None,
        'metrics': {},
        'params_used': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'tol': tol,
            'normalize': normalize
        },
        'warnings': []
    }

    # Check if metric function is provided
    if metric_func is None:
        output['warnings'].append("No metric function provided. Skipping metrics calculation.")

    # Gradient descent optimization
    for iteration in range(max_iter):
        # Shuffle data at each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process in batches
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Compute gradient
            gradient = _compute_gradient(X_batch, y_batch, theta, loss_func)

            # Update parameters
            theta -= learning_rate * gradient

        # Check convergence
        if iteration > 0:
            prev_loss = output['metrics'].get('loss', float('inf'))
            current_loss = loss_func(y, X @ theta)
            output['metrics']['loss'] = current_loss

            if abs(prev_loss - current_loss) < tol:
                break

    output['result'] = theta

    # Calculate metrics if provided
    if metric_func is not None:
        y_pred = X @ theta
        output['metrics']['metric'] = metric_func(y, y_pred)

    return output

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

def _apply_normalization(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str = 'standard'
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalization to input data."""
    if method == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_mean) / X_std

        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std != 0:
            y_normalized = (y - y_mean) / y_std
        else:
            y_normalized = y

    elif method == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)

        y_min = np.min(y)
        y_max = np.max(y)
        if y_max != y_min:
            y_normalized = (y - y_min) / (y_max - y_min)
        else:
            y_normalized = y

    elif method == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1
        X_normalized = (X - X_median) / X_iqr

        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr != 0:
            y_normalized = (y - y_median) / y_iqr
        else:
            y_normalized = y

    elif method == 'none':
        X_normalized, y_normalized = X, y
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized, y_normalized

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    y_pred = X @ theta
    error = y_pred - y

    # For MSE loss, this is the standard gradient
    gradient = (2 / X.shape[0]) * X.T @ error

    return gradient

################################################################################
# epochs
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

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

def normalize_data(X: np.ndarray, y: np.ndarray,
                  normalization: str = 'standard') -> tuple:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1
        y_normalized = (y - y_mean) / y_std
        return X_normalized, y_normalized
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_min) / X_range
        y_min = np.min(y)
        y_max = np.max(y)
        if y_max == y_min:
            y_normalized = np.zeros_like(y)
        else:
            y_normalized = (y - y_min) / (y_max - y_min)
        return X_normalized, y_normalized
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_iqr[X_iqr == 0] = 1
        X_normalized = (X - X_median) / X_iqr
        y_median = np.median(y)
        y_iqr = np.subtract(*np.percentile(y, [75, 25]))
        if y_iqr == 0:
            y_normalized = np.zeros_like(y)
        else:
            y_normalized = (y - y_median) / y_iqr
        return X_normalized, y_normalized
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
                  metric: Union[str, Callable]) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def gradient_descent_solver(X: np.ndarray, y: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iter: int = 1000,
                           tol: float = 1e-4) -> np.ndarray:
    """Perform gradient descent optimization."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Check for convergence
        if np.linalg.norm([dw, db]) < tol:
            break

    return np.concatenate([weights, [bias]])

def epochs_fit(X: np.ndarray, y: np.ndarray,
              normalization: str = 'standard',
              metric: Union[str, Callable] = 'mse',
              solver: str = 'gradient_descent',
              **solver_kwargs) -> Dict[str, Any]:
    """
    Perform epochs of optimization for gradient descent.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Evaluation metric (str or callable)
    - solver: Optimization method ('gradient_descent')
    - **solver_kwargs: Additional arguments for the solver

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = normalize_data(X, y, normalization)

    # Select solver
    if solver == 'gradient_descent':
        params = gradient_descent_solver(X_norm, y_norm, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute predictions and metrics
    y_pred = np.dot(X_norm, params[:-1]) + params[-1]
    main_metric = compute_metric(y_norm, y_pred, metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': {str(metric): main_metric},
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'solver': solver,
            **solver_kwargs
        },
        'warnings': []
    }

    return result

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = epochs_fit(
    X=X,
    y=y,
    normalization='standard',
    metric='mse',
    solver='gradient_descent',
    learning_rate=0.1,
    max_iter=500
)
"""

################################################################################
# gradient
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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

def normalize_data(X: np.ndarray, y: np.ndarray, normalization: str = 'none') -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-8)
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75 = np.percentile(X, 75, axis=0)
        X_q25 = np.percentile(X, 25, axis=0)
        X_normalized = (X - X_median) / (X_q75 - X_q25 + 1e-8)
        y_median = np.median(y)
        y_q75 = np.percentile(y, 75)
        y_q25 = np.percentile(y, 25)
        y_normalized = (y - y_median) / (y_q75 - y_q25 + 1e-8)
    else:
        X_normalized = X.copy()
        y_normalized = y.copy()

    return X_normalized, y_normalized

def compute_gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                     metric: str = 'mse', regularization: Optional[str] = None) -> np.ndarray:
    """Compute gradient of the loss function."""
    if metric == 'mse':
        residuals = X @ params - y
        gradient = (2 / len(y)) * X.T @ residuals
    elif metric == 'mae':
        residuals = np.abs(X @ params - y)
        gradient = (1 / len(y)) * X.T @ np.sign(X @ params - y)
    elif metric == 'r2':
        residuals = X @ params - y
        gradient = (2 / len(y)) * X.T @ residuals
    else:
        raise ValueError("Unsupported metric")

    if regularization == 'l1':
        gradient += np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * params

    return gradient

def gradient_descent(X: np.ndarray, y: np.ndarray,
                     learning_rate: float = 0.01, max_iter: int = 1000,
                     tol: float = 1e-4, verbose: bool = False) -> Dict[str, Any]:
    """Perform gradient descent optimization."""
    n_features = X.shape[1]
    params = np.zeros(n_features)
    prev_params = None

    for i in range(max_iter):
        gradient = compute_gradient(X, y, params)
        params -= learning_rate * gradient

        if prev_params is not None and np.linalg.norm(params - prev_params) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        prev_params = params.copy()

    return params

def gradient_fit(X: np.ndarray, y: np.ndarray,
                 normalization: str = 'none',
                 metric: str = 'mse',
                 regularization: Optional[str] = None,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 verbose: bool = False) -> Dict[str, Any]:
    """
    Perform gradient descent optimization with configurable parameters.

    Example:
        >>> X = np.random.rand(100, 5)
        >>> y = np.random.rand(100)
        >>> result = gradient_fit(X, y, normalization='standard', metric='mse')
    """
    validate_inputs(X, y)
    X_normalized, y_normalized = normalize_data(X, y, normalization)

    params = gradient_descent(
        X_normalized, y_normalized,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )

    # Compute metrics
    residuals = X_normalized @ params - y_normalized
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - mse / np.var(y_normalized)

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'regularization': regularization,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': []
    }

################################################################################
# cost_function
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def cost_function_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    solver: str = 'gradient_descent',
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute the cost function for gradient descent optimization.

    Parameters:
    -----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    reg_param : float, optional
        Regularization parameter
    solver : str, optional
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent')
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping criterion
    learning_rate : float, optional
        Learning rate for gradient descent
    custom_metric : callable, optional
        Custom metric function
    custom_distance : callable, optional
        Custom distance function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([5, 6])
    >>> result = cost_function_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data if required
    X_norm = _apply_normalization(X, normalization)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Initialize parameters
    params = _initialize_parameters(X_norm.shape[1])

    # Choose solver
    if solver == 'gradient_descent':
        params = _gradient_descent(
            X_norm, y_norm,
            metric=metric,
            custom_metric=custom_metric,
            regularization=regularization,
            reg_param=reg_param,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate
        )
    elif solver == 'newton':
        params = _newton_method(
            X_norm, y_norm,
            metric=metric,
            custom_metric=custom_metric
        )
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(
            X_norm, y_norm,
            metric=metric,
            custom_metric=custom_metric,
            max_iter=max_iter
        )

    # Compute metrics
    metrics = _compute_metrics(
        X_norm, y_norm,
        params,
        metric=metric,
        custom_metric=custom_metric
    )

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'normalization': normalization,
            'regularization': regularization,
            'reg_param': reg_param,
            'solver': solver
        },
        'warnings': []
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

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to input data."""
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

def _initialize_parameters(n_features: int) -> np.ndarray:
    """Initialize parameters for optimization."""
    return np.zeros(n_features)

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    regularization: Optional[str] = None,
    reg_param: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Perform gradient descent optimization."""
    params = _initialize_parameters(X.shape[1])
    prev_cost = float('inf')

    for _ in range(max_iter):
        gradient = _compute_gradient(
            X, y,
            params,
            metric=metric,
            custom_metric=custom_metric,
            regularization=regularization
        )

        params -= learning_rate * gradient

        current_cost = _compute_cost(
            X, y,
            params,
            metric=metric,
            custom_metric=custom_metric,
            regularization=regularization,
            reg_param=reg_param
        )

        if abs(prev_cost - current_cost) < tol:
            break

        prev_cost = current_cost

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    regularization: Optional[str] = None
) -> np.ndarray:
    """Compute gradient of the cost function."""
    if custom_metric is not None:
        return _custom_gradient(X, y, params, custom_metric)

    residuals = y - np.dot(X, params)
    if metric == 'mse':
        gradient = -2 * np.dot(X.T, residuals) / len(y)
    elif metric == 'mae':
        gradient = -np.dot(X.T, np.sign(residuals)) / len(y)
    elif metric == 'r2':
        gradient = -2 * np.dot(X.T, residuals) / len(y)
    elif metric == 'logloss':
        gradient = np.dot(X.T, (1 / (1 + np.exp(-y * np.dot(X, params))) - 1) * y)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if regularization == 'l1':
        gradient += reg_param * np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * reg_param * params

    return gradient

def _compute_cost(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    regularization: Optional[str] = None,
    reg_param: float = 1.0
) -> float:
    """Compute the cost function value."""
    if custom_metric is not None:
        return _custom_cost(X, y, params, custom_metric)

    residuals = y - np.dot(X, params)
    if metric == 'mse':
        cost = np.mean(residuals ** 2)
    elif metric == 'mae':
        cost = np.mean(np.abs(residuals))
    elif metric == 'r2':
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        cost = 1 - ss_res / ss_tot
    elif metric == 'logloss':
        cost = np.mean(np.log(1 + np.exp(-y * np.dot(X, params))))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if regularization == 'l1':
        cost += reg_param * np.sum(np.abs(params))
    elif regularization == 'l2':
        cost += reg_param * np.sum(params ** 2)

    return cost

def _compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict:
    """Compute various metrics for the model."""
    if custom_metric is not None:
        return {'custom': _custom_cost(X, y, params, custom_metric)}

    residuals = y - np.dot(X, params)
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean(residuals ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(residuals))
    elif metric == 'r2':
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        metrics['r2'] = 1 - ss_res / ss_tot
    elif metric == 'logloss':
        metrics['logloss'] = np.mean(np.log(1 + np.exp(-y * np.dot(X, params))))

    return metrics

def _custom_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> np.ndarray:
    """Compute gradient using custom metric function."""
    epsilon = 1e-8
    gradient = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        cost_plus = metric_func(y, np.dot(X, params_plus))

        params_minus = params.copy()
        params_minus[i] -= epsilon
        cost_minus = metric_func(y, np.dot(X, params_minus))

        gradient[i] = (cost_plus - cost_minus) / (2 * epsilon)

    return gradient

def _custom_cost(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> float:
    """Compute cost using custom metric function."""
    return metric_func(y, np.dot(X, params))

def _newton_method(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """Perform Newton method optimization."""
    params = _initialize_parameters(X.shape[1])
    prev_cost = float('inf')

    for _ in range(100):
        gradient = _compute_gradient(X, y, params, metric=metric, custom_metric=custom_metric)
        hessian = _compute_hessian(X, y, params)

        if np.linalg.det(hessian) == 0:
            break

        params -= np.linalg.solve(hessian, gradient)
        current_cost = _compute_cost(X, y, params, metric=metric, custom_metric=custom_metric)

        if abs(prev_cost - current_cost) < 1e-4:
            break

        prev_cost = current_cost

    return params

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """Compute Hessian matrix."""
    return 2 * np.dot(X.T, X) / len(y)

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None,
    max_iter: int = 1000
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    params = _initialize_parameters(X.shape[1])

    for _ in range(max_iter):
        for i in range(len(params)):
            X_i = X[:, i]
            mask = np.ones(X.shape[1], dtype=bool)
            mask[i] = False

            X_rest = X[:, mask]
            params_rest = params[mask]

            if np.linalg.norm(X_i) == 0:
                continue

            if custom_metric is not None:
                params[i] = _cd_custom(X, y, i, params_rest, custom_metric)
            else:
                params[i] = _cd_standard(X_i, y - np.dot(X_rest, params_rest), metric)

    return params

def _cd_standard(
    X_i: np.ndarray,
    residuals: np.ndarray,
    metric: str
) -> float:
    """Coordinate descent for standard metrics."""
    if metric == 'mse':
        numerator = np.dot(X_i, residuals)
        denominator = np.dot(X_i, X_i)
        return numerator / denominator
    elif metric == 'mae':
        return np.median(residuals) / X_i[0]
    else:
        raise ValueError(f"Unknown metric for coordinate descent: {metric}")

def _cd_custom(
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    params_rest: np.ndarray,
    metric_func: Callable
) -> float:
    """Coordinate descent with custom metric."""
    X_i = X[:, i]
    residuals = y - np.dot(X[:, params_rest != 0], params_rest)

    best_param = 0
    best_cost = float('inf')

    for param in np.linspace(-1, 1, 20):
        current_params = params_rest.copy()
        current_params = np.insert(current_params, i, param)
        cost = metric_func(y, np.dot(X, current_params))

        if cost < best_cost:
            best_cost = cost
            best_param = param

    return best_param

################################################################################
# local_minimum
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    gradient_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None:
    """Validate input data and functions."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be a 2D array and y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """Normalize the input data."""
    if method == "none":
        return X
    elif method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Compute the loss value."""
    return loss_func(y_true, y_pred)

def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    gradient_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    return gradient_func(X, y)

def local_minimum_fit(
    X: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], float],
    gradient_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    initial_params: Optional[np.ndarray] = None,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    normalization: str = "standard",
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Find the local minimum of a loss function using gradient descent.

    Parameters:
    - X: Input features (2D array)
    - y: Target values (1D array)
    - loss_func: Loss function to minimize
    - gradient_func: Gradient of the loss function
    - initial_params: Initial parameters (optional)
    - learning_rate: Learning rate for gradient descent
    - max_iter: Maximum number of iterations
    - tol: Tolerance for stopping criterion
    - normalization: Normalization method ("none", "standard", "minmax", "robust")
    - metric_funcs: Dictionary of metric functions to compute
    - verbose: Whether to print progress

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    validate_inputs(X, y, loss_func, gradient_func)

    # Normalize data
    X_norm = normalize_data(X, normalization)

    # Initialize parameters
    if initial_params is None:
        params = np.zeros(X_norm.shape[1])
    else:
        params = initial_params.copy()

    # Initialize metrics
    if metric_funcs is None:
        metric_funcs = {}

    # Gradient descent loop
    for i in range(max_iter):
        y_pred = X_norm @ params
        loss = compute_loss(y, y_pred, loss_func)

        if verbose:
            print(f"Iteration {i+1}, Loss: {loss}")

        gradient = compute_gradient(X_norm, y, params, gradient_func)
        params_new = params - learning_rate * gradient

        if np.linalg.norm(params_new - params) < tol:
            break

        params = params_new

    # Compute metrics
    y_pred_final = X_norm @ params
    metrics = {name: func(y, y_pred_final) for name, func in metric_funcs.items()}

    # Prepare output
    result = {
        "result": y_pred_final,
        "metrics": metrics,
        "params_used": params,
        "warnings": []
    }

    return result

################################################################################
# global_minimum
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def global_minimum_fit(
    objective_func: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    solver: str = 'gradient_descent',
    metric: Union[str, Callable[[np.ndarray], float]] = 'mse',
    normalization: str = 'none',
    distance: str = 'euclidean',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Find the global minimum of an objective function using various optimization methods.

    Parameters:
    -----------
    objective_func : callable
        The objective function to minimize. Must accept a numpy array and return a float.
    initial_params : np.ndarray
        Initial parameters for the optimization.
    solver : str, optional
        Optimization method to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    metric : str or callable, optional
        Metric to evaluate the optimization. Options: 'mse', 'mae', 'r2', custom callable.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    distance : str, optional
        Distance metric for some solvers. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski'.
    regularization : str, optional
        Regularization method. Options: 'none', 'l1', 'l2', 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    learning_rate : float, optional
        Learning rate for gradient descent.
    custom_params : dict, optional
        Additional parameters for the solver.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(objective_func, initial_params)

    # Normalize data if needed
    normalized_params = _apply_normalization(initial_params, normalization)

    # Choose solver
    if solver == 'gradient_descent':
        result = _gradient_descent(
            objective_func, normalized_params,
            tol=tol, max_iter=max_iter,
            learning_rate=learning_rate
        )
    elif solver == 'newton':
        result = _newton_method(
            objective_func, normalized_params,
            tol=tol, max_iter=max_iter
        )
    elif solver == 'coordinate_descent':
        result = _coordinate_descent(
            objective_func, normalized_params,
            tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result['params'], metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'solver': solver,
            'metric': metric,
            'normalization': normalization,
            'distance': distance,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter,
            'learning_rate': learning_rate
        },
        'warnings': []
    }

def _validate_inputs(objective_func: Callable, params: np.ndarray) -> None:
    """Validate the inputs for the optimization."""
    if not callable(objective_func):
        raise TypeError("objective_func must be a callable")
    if not isinstance(params, np.ndarray):
        raise TypeError("initial_params must be a numpy array")
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        raise ValueError("initial_params contains NaN or inf values")

def _apply_normalization(params: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the parameters."""
    if method == 'none':
        return params
    elif method == 'standard':
        mean = np.mean(params)
        std = np.std(params)
        return (params - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(params)
        max_val = np.max(params)
        return (params - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(params)
        iqr = np.percentile(params, 75) - np.percentile(params, 25)
        return (params - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _gradient_descent(
    objective_func: Callable,
    initial_params: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    learning_rate: float = 0.01
) -> Dict[str, Any]:
    """Perform gradient descent optimization."""
    params = initial_params.copy()
    prev_value = objective_func(params)
    for _ in range(max_iter):
        gradient = _compute_gradient(objective_func, params)
        params -= learning_rate * gradient
        current_value = objective_func(params)
        if abs(current_value - prev_value) < tol:
            break
        prev_value = current_value
    return {'params': params, 'value': current_value}

def _compute_gradient(
    objective_func: Callable,
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Compute the gradient of the objective function."""
    gradient = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        gradient[i] = (objective_func(params_plus) - objective_func(params_minus)) / (2 * epsilon)
    return gradient

def _newton_method(
    objective_func: Callable,
    initial_params: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Perform Newton's method optimization."""
    params = initial_params.copy()
    prev_value = objective_func(params)
    for _ in range(max_iter):
        gradient = _compute_gradient(objective_func, params)
        hessian = _compute_hessian(objective_func, params)
        try:
            params -= np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            params -= 0.1 * gradient
        current_value = objective_func(params)
        if abs(current_value - prev_value) < tol:
            break
        prev_value = current_value
    return {'params': params, 'value': current_value}

def _compute_hessian(
    objective_func: Callable,
    params: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Compute the Hessian matrix of the objective function."""
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        params_plus_i = params.copy()
        params_plus_i[i] += epsilon
        gradient_plus_i = _compute_gradient(objective_func, params_plus_i)
        for j in range(n):
            params_plus_j = params.copy()
            params_plus_j[j] += epsilon
            gradient_plus_j = _compute_gradient(objective_func, params_plus_j)
            hessian[i, j] = (gradient_plus_i[j] - gradient_plus_j[i]) / (2 * epsilon)
    return hessian

def _coordinate_descent(
    objective_func: Callable,
    initial_params: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Perform coordinate descent optimization."""
    params = initial_params.copy()
    prev_value = objective_func(params)
    for _ in range(max_iter):
        for i in range(len(params)):
            params_minus = params.copy()
            params_minus[i] = 0
            gradient_i = _compute_gradient(objective_func, params_minus)[i]
            params[i] -= gradient_i
        current_value = objective_func(params)
        if abs(current_value - prev_value) < tol:
            break
        prev_value = current_value
    return {'params': params, 'value': current_value}

def _calculate_metrics(
    params: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate the metrics for the optimization result."""
    if callable(metric):
        return {'custom_metric': metric(params)}
    elif metric == 'mse':
        return {'mse': np.mean(np.square(params))}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(params))}
    elif metric == 'r2':
        return {'r2': 1 - np.sum(np.square(params)) / len(params)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

################################################################################
# saddle_point
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def saddle_point_fit(
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    metric: str = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    random_state: Optional[int] = None
) -> Callable[[np.ndarray, np.ndarray], Dict]:
    """
    Configure and return a saddle point optimization function.

    Parameters:
    -----------
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.
    learning_rate : float
        Learning rate for gradient descent.
    metric : str or callable
        Metric to evaluate performance ('mse', 'mae', 'r2', etc.).
    normalizer : callable
        Function to normalize input data.
    distance : str or callable
        Distance metric ('euclidean', 'manhattan', etc.).
    solver : str
        Optimization solver ('gradient_descent', 'newton', etc.).
    regularization : str or None
        Regularization type ('l1', 'l2', 'elasticnet').
    alpha : float
        Regularization strength for L1 or ElasticNet.
    beta : float
        Regularization strength for L2 or ElasticNet.
    random_state : int or None
        Random seed for reproducibility.

    Returns:
    --------
    callable
        A function that performs saddle point optimization.
    """
    def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values.")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains infinite values.")

    def _normalize(X: np.ndarray) -> np.ndarray:
        if normalizer is None:
            return X
        return normalizer(X)

    def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if callable(metric):
            return metric(y_true, y_pred)
        elif metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _compute_distance(X: np.ndarray, y: np.ndarray) -> float:
        if callable(distance):
            return distance(X, y)
        elif distance == 'euclidean':
            return np.linalg.norm(X - y)
        elif distance == 'manhattan':
            return np.sum(np.abs(X - y))
        elif distance == 'cosine':
            return 1 - np.dot(X, y) / (np.linalg.norm(X) * np.linalg.norm(y))
        else:
            raise ValueError(f"Unknown distance: {distance}")

    def _saddle_point_optimization(X: np.ndarray, y: np.ndarray) -> Dict:
        X = _normalize(X)
        _validate_inputs(X, y)

        if solver == 'gradient_descent':
            theta = np.zeros(X.shape[1])
            for _ in range(max_iter):
                gradient = 2 * X.T @ (X @ theta - y) / len(y)
                if regularization == 'l1':
                    gradient += alpha * np.sign(theta)
                elif regularization == 'l2':
                    gradient += 2 * beta * theta
                elif regularization == 'elasticnet':
                    gradient += alpha * np.sign(theta) + 2 * beta * theta
                theta -= learning_rate * gradient
                if np.linalg.norm(gradient) < tol:
                    break
        else:
            raise ValueError(f"Unknown solver: {solver}")

        y_pred = X @ theta
        result = {
            'result': y_pred,
            'metrics': {'metric': _compute_metric(y, y_pred)},
            'params_used': {
                'max_iter': max_iter,
                'tol': tol,
                'learning_rate': learning_rate,
                'metric': metric,
                'normalizer': normalizer.__name__ if normalizer else None,
                'distance': distance,
                'solver': solver,
                'regularization': regularization,
                'alpha': alpha,
                'beta': beta
            },
            'warnings': []
        }
        return result

    return _saddle_point_optimization

################################################################################
# convergence_criteria
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def convergence_criteria_fit(
    loss_history: np.ndarray,
    metric_func: Callable[[np.ndarray], float] = lambda x: np.mean(x),
    tolerance: float = 1e-5,
    max_iterations: int = 1000,
    patience: Optional[int] = None,
    custom_criteria: Optional[Callable[[np.ndarray], bool]] = None
) -> Dict[str, Any]:
    """
    Compute convergence criteria for gradient descent optimization.

    Parameters:
    -----------
    loss_history : np.ndarray
        Array of loss values over iterations.
    metric_func : Callable[[np.ndarray], float]
        Function to compute convergence metric from loss history.
    tolerance : float
        Minimum change in metric to consider as converged.
    max_iterations : int
        Maximum number of iterations before forcing convergence.
    patience : Optional[int]
        Number of iterations to wait after last improvement before stopping.
    custom_criteria : Optional[Callable[[np.ndarray], bool]]
        Custom convergence criteria function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing convergence results and metrics.
    """
    # Validate inputs
    _validate_inputs(loss_history, metric_func, tolerance)

    # Compute metrics
    metrics = _compute_metrics(loss_history, metric_func)

    # Determine convergence status
    converged = _check_convergence(
        metrics['values'],
        tolerance,
        max_iterations,
        patience,
        custom_criteria
    )

    # Prepare results
    result = {
        'converged': converged,
        'final_metric_value': metrics['values'][-1],
        'convergence_iteration': converged['iteration'],
        'params_used': {
            'tolerance': tolerance,
            'max_iterations': max_iterations,
            'patience': patience
        },
        'warnings': converged['warnings']
    }

    return result

def _validate_inputs(
    loss_history: np.ndarray,
    metric_func: Callable[[np.ndarray], float],
    tolerance: float
) -> None:
    """Validate input parameters."""
    if not isinstance(loss_history, np.ndarray):
        raise TypeError("loss_history must be a numpy array")
    if loss_history.ndim != 1:
        raise ValueError("loss_history must be 1-dimensional")
    if not np.all(np.isfinite(loss_history)):
        raise ValueError("loss_history contains NaN or infinite values")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

def _compute_metrics(
    loss_history: np.ndarray,
    metric_func: Callable[[np.ndarray], float]
) -> Dict[str, Any]:
    """Compute convergence metrics from loss history."""
    values = np.array([metric_func(loss_history[:i+1]) for i in range(len(loss_history))])
    return {'values': values}

def _check_convergence(
    metric_values: np.ndarray,
    tolerance: float,
    max_iterations: int,
    patience: Optional[int],
    custom_criteria: Optional[Callable[[np.ndarray], bool]]
) -> Dict[str, Any]:
    """Check convergence based on various criteria."""
    warnings = []

    # Check max iterations
    if len(metric_values) >= max_iterations:
        warnings.append(f"Reached maximum iterations ({max_iterations})")

    # Check custom criteria if provided
    if custom_criteria is not None:
        if custom_criteria(metric_values):
            return {
                'iteration': len(metric_values) - 1,
                'warnings': warnings
            }

    # Check tolerance-based convergence
    if len(metric_values) > 1:
        delta = np.abs(metric_values[-1] - metric_values[-2])
        if delta < tolerance:
            return {
                'iteration': len(metric_values) - 1,
                'warnings': warnings
            }

    # Check patience-based convergence
    if patience is not None and len(metric_values) > patience:
        best_value = np.min(metric_values[-patience:])
        current_value = metric_values[-1]
        if np.abs(current_value - best_value) < tolerance:
            return {
                'iteration': len(metric_values) - 1,
                'warnings': warnings
            }

    # No convergence yet
    return {
        'iteration': None,
        'warnings': warnings
    }

# Example usage:
"""
loss_history = np.array([1.0, 0.9, 0.85, 0.82, 0.81, 0.805])
result = convergence_criteria_fit(
    loss_history,
    metric_func=np.min,
    tolerance=1e-3
)
print(result)
"""

################################################################################
# momentum
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    momentum_factor: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> None:
    """
    Validate the inputs for momentum gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    learning_rate : float, optional
        Learning rate for gradient descent.
    momentum_factor : float, optional
        Momentum factor (between 0 and 1).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if learning_rate <= 0 or learning_rate >= 1:
        raise ValueError("learning_rate must be in (0, 1).")
    if momentum_factor < 0 or momentum_factor > 1:
        raise ValueError("momentum_factor must be in [0, 1].")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")

def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute the gradient of the loss function.

    Parameters
    ----------
    X : np.ndarray
        Input features array.
    y : np.ndarray
        Target values array.
    weights : np.ndarray
        Current weights.

    Returns
    ------
    np.ndarray
        Gradient of the loss function.
    """
    residuals = y - X.dot(weights)
    gradient = -2 * X.T.dot(residuals) / len(y)
    return gradient

def momentum_fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    momentum_factor: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-4,
    metric: str = "mse",
    normalize: bool = True
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform momentum gradient descent to fit linear regression.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : np.ndarray
        Target values array of shape (n_samples,).
    learning_rate : float, optional
        Learning rate for gradient descent.
    momentum_factor : float, optional
        Momentum factor (between 0 and 1).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    metric : str, optional
        Metric to evaluate the model ("mse", "mae").
    normalize : bool, optional
        Whether to normalize the features.

    Returns
    ------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    validate_inputs(X, y, learning_rate, momentum_factor, max_iter, tol)

    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    n_features = X.shape[1]
    weights = np.zeros(n_features)
    velocity = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = compute_gradient(X, y, weights)
        velocity = momentum_factor * velocity + learning_rate * gradient
        weights -= velocity

        if np.linalg.norm(gradient) < tol:
            break

    result = weights
    metrics = {"mse": np.mean((y - X.dot(result)) ** 2)}
    params_used = {
        "learning_rate": learning_rate,
        "momentum_factor": momentum_factor,
        "max_iter": max_iter,
        "tol": tol,
        "metric": metric,
        "normalize": normalize
    }
    warnings = []

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
# adaptive_learning_rate
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def adaptive_learning_rate_fit(
    loss_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    learning_rate_func: Callable[[int, float], float],
    max_iter: int = 1000,
    tol: float = 1e-6,
    metrics_funcs: Optional[Dict[str, Callable[[np.ndarray], float]]] = None,
    normalize_grad: bool = False,
    line_search_func: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform gradient descent with adaptive learning rate.

    Parameters:
    - loss_func: Callable that computes the loss given parameters.
    - grad_func: Callable that computes the gradient of the loss given parameters.
    - initial_params: Initial parameters for optimization.
    - learning_rate_func: Callable that returns the learning rate given iteration and previous loss.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for stopping criterion.
    - metrics_funcs: Dictionary of metric functions to compute during optimization.
    - normalize_grad: Whether to normalize the gradient.
    - line_search_func: Optional callable for line search.
    - verbose: Whether to print progress.

    Returns:
    - Dictionary containing 'result', 'metrics', 'params_used', and 'warnings'.
    """
    params = initial_params.copy()
    prev_loss = loss_func(params)
    metrics_history = {}
    warnings_list = []

    if metrics_funcs is None:
        metrics_funcs = {}

    for i in range(max_iter):
        grad = grad_func(params)

        if normalize_grad:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm

        current_lr = learning_rate_func(i, prev_loss)

        if line_search_func is not None:
            current_lr = line_search_func(params, grad, current_lr)

        params_new = params - current_lr * grad
        new_loss = loss_func(params_new)

        if verbose:
            print(f"Iteration {i}, Loss: {new_loss}")

        # Check for convergence
        if abs(new_loss - prev_loss) < tol:
            warnings_list.append(f"Converged at iteration {i}")
            break

        # Update metrics
        for name, metric_func in metrics_funcs.items():
            if name not in metrics_history:
                metrics_history[name] = []
            metrics_history[name].append(metric_func(params_new))

        params, prev_loss = params_new, new_loss

    result_dict = {
        "result": params,
        "metrics": metrics_history,
        "params_used": {
            "max_iter": max_iter,
            "tol": tol,
            "normalize_grad": normalize_grad
        },
        "warnings": warnings_list if warnings_list else None
    }

    return result_dict

def standard_learning_rate(iteration: int, prev_loss: float) -> float:
    """
    Standard learning rate schedule.

    Parameters:
    - iteration: Current iteration.
    - prev_loss: Previous loss value.

    Returns:
    - Learning rate.
    """
    return 0.1 * (0.9 ** iteration)

def adaptive_learning_rate(iteration: int, prev_loss: float) -> float:
    """
    Adaptive learning rate schedule.

    Parameters:
    - iteration: Current iteration.
    - prev_loss: Previous loss value.

    Returns:
    - Learning rate.
    """
    return 0.1 / (1 + iteration)

def backtracking_line_search(
    params: np.ndarray,
    grad: np.ndarray,
    initial_lr: float,
    c1: float = 0.1,
    alpha: float = 0.5
) -> float:
    """
    Backtracking line search for adaptive learning rate.

    Parameters:
    - params: Current parameters.
    - grad: Gradient at current parameters.
    - initial_lr: Initial learning rate.
    - c1: Armijo condition constant.
    - alpha: Reduction factor.

    Returns:
    - Learning rate after line search.
    """
    lr = initial_lr
    while True:
        new_params = params - lr * grad
        new_loss = loss_func(new_params)
        old_loss = loss_func(params)
        if new_loss <= old_loss - c1 * lr * np.dot(grad, grad):
            break
        lr *= alpha
    return lr

# Example usage:
if __name__ == "__main__":
    # Define a simple quadratic loss function and its gradient
    def quadratic_loss(params: np.ndarray) -> float:
        return np.sum(params ** 2)

    def quadratic_grad(params: np.ndarray) -> np.ndarray:
        return 2 * params

    initial_params = np.array([10.0, 5.0])
    result = adaptive_learning_rate_fit(
        loss_func=quadratic_loss,
        grad_func=quadratic_grad,
        initial_params=initial_params,
        learning_rate_func=adaptive_learning_rate,
        max_iter=100,
        tol=1e-6,
        metrics_funcs={"norm": lambda p: np.linalg.norm(p)},
        normalize_grad=True,
        verbose=True
    )
    print(result)

################################################################################
# stochastic_gradient_descent
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def stochastic_gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    batch_size: Optional[int] = None,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    penalty: Optional[str] = None,
    alpha: float = 0.01,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Perform stochastic gradient descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    learning_rate : float, optional
        Learning rate for gradient descent.
    n_iterations : int, optional
        Maximum number of iterations.
    batch_size : Optional[int], optional
        Size of mini-batches. If None, uses full gradient descent.
    tol : float, optional
        Tolerance for stopping criterion.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    metric : Union[str, Callable], optional
        Metric to evaluate performance. Can be 'mse', 'mae', or a custom callable.
    normalizer : Optional[Callable], optional
        Function to normalize features. If None, no normalization.
    penalty : Optional[str], optional
        Type of regularization: 'l1', 'l2', or None.
    alpha : float, optional
        Regularization strength.
    verbose : bool, optional
        Whether to print progress.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Optimized parameters.
        - 'metrics': Performance metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = stochastic_gradient_descent_fit(X, y)
    """
    # Initialize random state
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features if specified
    X_normalized = normalizer(X) if normalizer is not None else X

    # Initialize parameters
    n_features = X_normalized.shape[1]
    theta = np.zeros(n_features)

    # Prepare metric function
    if isinstance(metric, str):
        if metric == 'mse':
            metric_func = _mean_squared_error
        elif metric == 'mae':
            metric_func = _mean_absolute_error
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metric_func = metric

    # Prepare penalty function
    if penalty is None:
        penalty_func = lambda t: 0.0
    elif penalty == 'l1':
        penalty_func = lambda t: alpha * np.sum(np.abs(t))
    elif penalty == 'l2':
        penalty_func = lambda t: alpha * np.sum(t**2)
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

    # Stochastic gradient descent
    for i in range(n_iterations):
        if batch_size is None:
            # Full gradient descent
            gradients = _compute_gradient(X_normalized, y, theta)
        else:
            # Mini-batch gradient descent
            indices = np.random.choice(X_normalized.shape[0], batch_size, replace=False)
            X_batch = X_normalized[indices]
            y_batch = y[indices]
            gradients = _compute_gradient(X_batch, y_batch, theta)

        # Add penalty gradient if applicable
        if penalty == 'l1':
            gradients += alpha * np.sign(theta)
        elif penalty == 'l2':
            gradients += 2 * alpha * theta

        # Update parameters
        theta -= learning_rate * gradients

        # Check convergence
        if i > 0 and np.linalg.norm(gradients) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

    # Compute final metrics
    predictions = X_normalized @ theta
    current_metric = metric_func(y, predictions)

    # Prepare output
    result_dict = {
        'result': theta,
        'metrics': {'final_metric': current_metric},
        'params_used': {
            'learning_rate': learning_rate,
            'n_iterations': i + 1,
            'batch_size': batch_size,
            'tol': tol,
            'metric': metric if isinstance(metric, str) else 'custom',
            'penalty': penalty,
            'alpha': alpha
        },
        'warnings': []
    }

    return result_dict

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

def _compute_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute gradient of the loss function."""
    predictions = X @ theta
    residuals = predictions - y
    gradient = (X.T @ residuals) / X.shape[0]
    return gradient

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

################################################################################
# mini_batch_gradient_descent
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def mini_batch_gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    loss_func: Callable[[np.ndarray, np.ndarray], float] = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    gradient_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = lambda X_batch, y_batch, weights: (2 / len(X_batch)) * X_batch.T @ (X_batch @ weights - y_batch),
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, object]]]:
    """
    Perform mini-batch gradient descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    batch_size : int, optional
        Size of mini-batches, by default 32
    learning_rate : float, optional
        Learning rate for gradient descent, by default 0.01
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    tol : float, optional
        Tolerance for stopping criterion, by default 1e-4
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None
    loss_func : Callable[[np.ndarray, np.ndarray], float], optional
        Loss function to minimize, by default MSE
    gradient_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], optional
        Gradient function, by default gradient of MSE
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function, by default None
    metric_funcs : Dict[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Dictionary of metric functions to compute, by default None
    verbose : bool, optional
        Whether to print progress, by default False

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, object]]]
        Dictionary containing:
        - "result": Optimized weights
        - "metrics": Computed metrics
        - "params_used": Parameters used in the optimization
        - "warnings": Any warnings encountered

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = mini_batch_gradient_descent_fit(X, y)
    """
    # Input validation
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if batch_size <= 0 or batch_size > len(X):
        raise ValueError("batch_size must be between 1 and n_samples")

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Add bias term if needed
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    n_features = X_with_bias.shape[1]

    # Initialize weights
    weights = np.zeros(n_features)

    # Normalize data if normalizer is provided
    X_normalized = normalizer(X_with_bias) if normalizer else X_with_bias

    # Initialize metrics dictionary
    metrics = {}
    if metric_funcs is not None:
        for name, func in metric_funcs.items():
            metrics[name] = []

    # Optimization loop
    for i in range(max_iter):
        # Shuffle data
        indices = rng.permutation(len(X_normalized))
        X_shuffled = X_normalized[indices]
        y_shuffled = y[indices]

        # Mini-batch iteration
        for j in range(0, len(X_shuffled), batch_size):
            X_batch = X_shuffled[j:j + batch_size]
            y_batch = y_shuffled[j:j + batch_size]

            # Compute gradient
            grad = gradient_func(X_batch, y_batch, weights)

            # Update weights
            weights -= learning_rate * grad

        # Compute and store metrics if provided
        y_pred = X_with_bias @ weights
        current_loss = loss_func(y, y_pred)

        if metric_funcs is not None:
            for name, func in metric_funcs.items():
                metrics[name].append(func(y, y_pred))

        # Check convergence
        if i > 0 and abs(prev_loss - current_loss) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        prev_loss = current_loss
        if verbose and i % 10 == 0:
            print(f"Iteration {i}, Loss: {current_loss:.4f}")

    # Prepare output
    result = {
        "result": weights,
        "metrics": metrics if metric_funcs is not None else {},
        "params_used": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "tol": tol
        },
        "warnings": []
    }

    return result

def _validate_input(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Input arrays must not contain NaN values")
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError("Input arrays must not contain infinite values")

def _standard_normalizer(X: np.ndarray) -> np.ndarray:
    """Standard normalizer (mean=0, std=1)."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def _minmax_normalizer(X: np.ndarray) -> np.ndarray:
    """Min-max normalizer (range [0, 1])."""
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def _robust_normalizer(X: np.ndarray) -> np.ndarray:
    """Robust normalizer (median and IQR)."""
    return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))

def mse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error metric."""
    return np.mean((y_true - y_pred) ** 2)

def mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error metric."""
    return np.mean(np.abs(y_true - y_pred))

def r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def logloss_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log loss metric."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

################################################################################
# learning_rate_scheduling
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def learning_rate_scheduling_fit(
    loss_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    initial_learning_rate: float,
    learning_rate_schedule: Callable[[int, float], float],
    max_iter: int,
    tol: float = 1e-4,
    metrics: Optional[Dict[str, Callable[[np.ndarray], float]]] = None,
    params_init: Optional[np.ndarray] = None,
    validation_data: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform gradient descent with learning rate scheduling.

    Parameters:
    - loss_function: Callable that computes the loss given parameters.
    - gradient_function: Callable that computes the gradient of the loss given parameters.
    - initial_learning_rate: Initial learning rate for the optimization.
    - learning_rate_schedule: Callable that updates the learning rate given iteration and current loss.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for stopping criterion (relative change in loss).
    - metrics: Dictionary of metric functions to compute during optimization.
    - params_init: Initial parameters for the optimization. If None, random initialization is used.
    - validation_data: Optional data to compute metrics on during optimization.

    Returns:
    - Dictionary containing 'result' (optimized parameters), 'metrics' (computed metrics),
      'params_used' (parameters used during optimization), and 'warnings'.
    """
    # Initialize parameters
    if params_init is None:
        params = np.random.randn(gradient_function(np.zeros(10)).shape[0])
    else:
        params = np.array(params_init, dtype=float)

    current_learning_rate = initial_learning_rate
    previous_loss = float('inf')
    loss_history = []
    params_history = [params.copy()]
    warnings_list = []

    # Initialize metrics if provided
    computed_metrics = {}
    if metrics is not None:
        for name, metric_func in metrics.items():
            computed_metrics[name] = []

    # Main optimization loop
    for iteration in range(max_iter):
        current_loss = loss_function(params)
        loss_history.append(current_loss)

        # Update learning rate
        current_learning_rate = learning_rate_schedule(iteration, current_loss)

        # Compute gradient and update parameters
        grad = gradient_function(params)
        params -= current_learning_rate * grad
        params_history.append(params.copy())

        # Check for convergence
        if abs(previous_loss - current_loss) < tol * max(1, abs(previous_loss)):
            warnings_list.append(f"Convergence reached at iteration {iteration}")
            break

        previous_loss = current_loss

        # Compute metrics if provided
        if metrics is not None:
            for name, metric_func in metrics.items():
                computed_metrics[name].append(metric_func(params))

    # Prepare output
    result = {
        'result': params,
        'metrics': computed_metrics if metrics is not None else {},
        'params_used': {
            'initial_learning_rate': initial_learning_rate,
            'final_learning_rate': current_learning_rate,
            'max_iter': max_iter,
            'tol': tol
        },
        'warnings': warnings_list if warnings_list else None
    }

    return result

def constant_learning_rate_schedule(iteration: int, loss: float) -> float:
    """
    Constant learning rate schedule.

    Parameters:
    - iteration: Current iteration number.
    - loss: Current loss value.

    Returns:
    - Constant learning rate.
    """
    return 0.1

def exponential_decay_schedule(iteration: int, loss: float) -> float:
    """
    Exponential decay learning rate schedule.

    Parameters:
    - iteration: Current iteration number.
    - loss: Current loss value.

    Returns:
    - Learning rate with exponential decay.
    """
    return 0.1 * np.exp(-iteration / 100)

def adaptive_learning_rate_schedule(iteration: int, loss: float) -> float:
    """
    Adaptive learning rate schedule based on loss.

    Parameters:
    - iteration: Current iteration number.
    - loss: Current loss value.

    Returns:
    - Adaptive learning rate based on current loss.
    """
    return 0.1 / (1 + iteration) * max(1e-4, loss)

def validate_input_gradient_function(
    gradient_function: Callable[[np.ndarray], np.ndarray],
    params_shape: tuple
) -> None:
    """
    Validate the gradient function input.

    Parameters:
    - gradient_function: Callable that computes the gradient.
    - params_shape: Expected shape of parameters.

    Raises:
    - ValueError if the gradient function output does not match expected shape.
    """
    test_params = np.random.randn(*params_shape)
    grad = gradient_function(test_params)
    if grad.shape != params_shape:
        raise ValueError(f"Gradient function output shape {grad.shape} does not match expected shape {params_shape}")

def validate_loss_function(
    loss_function: Callable[[np.ndarray], float],
    params_shape: tuple
) -> None:
    """
    Validate the loss function input.

    Parameters:
    - loss_function: Callable that computes the loss.
    - params_shape: Expected shape of parameters.

    Raises:
    - ValueError if the loss function output is not a scalar.
    """
    test_params = np.random.randn(*params_shape)
    loss = loss_function(test_params)
    if not isinstance(loss, (float, np.floating)):
        raise ValueError("Loss function must return a scalar value")

# Example usage
if __name__ == "__main__":
    # Define a simple quadratic loss function and its gradient
    def quadratic_loss(params: np.ndarray) -> float:
        return np.sum(params**2)

    def quadratic_gradient(params: np.ndarray) -> np.ndarray:
        return 2 * params

    # Define metrics
    def l1_norm(params: np.ndarray) -> float:
        return np.sum(np.abs(params))

    metrics = {
        'l1_norm': l1_norm
    }

    # Run learning rate scheduling
    result = learning_rate_scheduling_fit(
        loss_function=quadratic_loss,
        gradient_function=quadratic_gradient,
        initial_learning_rate=0.1,
        learning_rate_schedule=exponential_decay_schedule,
        max_iter=100,
        metrics=metrics
    )

    print(result)

################################################################################
# regularization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def regularization_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "gradient_descent",
    regularization: str = "none",
    alpha: float = 1.0,
    l1_ratio: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    learning_rate: float = 0.01,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform regularized gradient descent optimization.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize features
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate performance
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance metric for gradient computation
    solver : str
        Optimization algorithm to use
    regularization : str
        Type of regularization (none, l1, l2, elasticnet)
    alpha : float
        Regularization strength
    l1_ratio : Optional[float]
        ElasticNet mixing parameter (0 <= l1_ratio <= 1)
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criteria
    learning_rate : float
        Learning rate for gradient descent
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": Optimized parameters
        - "metrics": Performance metrics
        - "params_used": Parameters used in the optimization
        - "warnings": Any warnings generated during execution

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = regularization_fit(X, y, solver="gradient_descent", regularization="l2")
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize parameters
    n_features = X.shape[1]
    coef = np.zeros(n_features)
    intercept = 0.0

    # Normalize features
    X_norm = normalizer(X)

    # Set up metric function
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        metric_func = metric

    # Set up distance function
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        distance_func = distance

    # Set up regularization function
    reg_func, reg_deriv = _get_regularization_functions(regularization, alpha, l1_ratio)

    # Set up solver
    if solver == "gradient_descent":
        coef, intercept = _gradient_descent(
            X_norm, y,
            distance_func=distance_func,
            reg_func=reg_func,
            reg_deriv=reg_deriv,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate
        )
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Calculate metrics
    y_pred = _predict(X_norm, coef, intercept)
    metrics = {
        "metric": metric_func(y, y_pred),
        "mse": _mean_squared_error(y, y_pred),
        "mae": _mean_absolute_error(y, y_pred)
    }

    # Prepare output
    result = {
        "result": {"coef": coef, "intercept": intercept},
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, "__name__") else "custom",
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "tol": tol,
            "learning_rate": learning_rate
        },
        "warnings": []
    }

    return result

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

def _get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on name."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Metric {metric} not supported")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on name."""
    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Distance {distance} not supported")
    return distances[distance]

def _get_regularization_functions(
    regularization: str,
    alpha: float,
    l1_ratio: Optional[float]
) -> tuple:
    """Get regularization functions based on type."""
    if regularization == "none":
        return lambda x: 0, lambda x: np.zeros_like(x)
    elif regularization == "l1":
        if l1_ratio is not None:
            raise ValueError("l1_ratio should be None for L1 regularization")
        return lambda x: alpha * np.linalg.norm(x, 1), lambda x: alpha * np.sign(x)
    elif regularization == "l2":
        if l1_ratio is not None:
            raise ValueError("l1_ratio should be None for L2 regularization")
        return lambda x: alpha * np.linalg.norm(x, 2)**2, lambda x: 2 * alpha * x
    elif regularization == "elasticnet":
        if l1_ratio is None or not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1 for ElasticNet")
        l1 = alpha * l1_ratio
        l2 = alpha * (1 - l1_ratio)
        return (
            lambda x: l1 * np.linalg.norm(x, 1) + l2 * np.linalg.norm(x, 2)**2,
            lambda x: l1 * np.sign(x) + 2 * l2 * x
        )
    else:
        raise ValueError(f"Regularization {regularization} not supported")

def _gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    reg_func: Callable[[np.ndarray], float],
    reg_deriv: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
    tol: float,
    learning_rate: float
) -> tuple:
    """Perform gradient descent optimization."""
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)
    intercept = 0.0

    for _ in range(max_iter):
        # Compute predictions
        y_pred = X @ coef + intercept

        # Compute gradients
        error = y_pred - y
        grad_coef = (X.T @ error) / n_samples + reg_deriv(coef)
        grad_intercept = np.sum(error) / n_samples

        # Update parameters
        coef_new = coef - learning_rate * grad_coef
        intercept_new = intercept - learning_rate * grad_intercept

        # Check convergence
        if np.linalg.norm(coef_new - coef) < tol and abs(intercept_new - intercept) < tol:
            break

        coef, intercept = coef_new, intercept_new

    return coef, intercept

def _predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Make predictions using linear model."""
    return X @ coef + intercept

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
    return 1 - ss_res / ss_tot

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

################################################################################
# gradient_clipping
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def gradient_clipping_fit(
    loss_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    params_init: np.ndarray,
    clip_value: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float = 0.01,
    normalize_grad: bool = False,
    metric_func: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Perform gradient descent with gradient clipping.

    Parameters:
    -----------
    loss_func : callable
        Function to compute the loss given parameters.
    grad_func : callable
        Function to compute the gradient of the loss with respect to parameters.
    params_init : ndarray
        Initial parameters.
    clip_value : float, optional
        Maximum norm of the gradient before clipping (default: 1.0).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-6).
    learning_rate : float, optional
        Learning rate for gradient descent (default: 0.01).
    normalize_grad : bool, optional
        Whether to normalize the gradient (default: False).
    metric_func : callable, optional
        Function to compute a custom metric (default: None).

    Returns:
    --------
    dict
        Dictionary containing the optimized parameters, metrics, and warnings.
    """
    # Validate inputs
    _validate_inputs(loss_func, grad_func, params_init)

    params = params_init.copy()
    loss_history = []
    metric_history = []

    for i in range(max_iter):
        grad = grad_func(params)

        # Clip the gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > clip_value:
            grad = grad * (clip_value / grad_norm)

        # Normalize the gradient if requested
        if normalize_grad:
            grad = grad / (np.linalg.norm(grad) + 1e-8)

        # Update parameters
        params -= learning_rate * grad

        # Compute loss and metric
        current_loss = loss_func(params)
        loss_history.append(current_loss)

        if metric_func is not None:
            current_metric = metric_func(params)
            metric_history.append(current_metric)

        # Check for convergence
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break

    # Prepare the result dictionary
    result = {
        "result": params,
        "metrics": {
            "loss_history": loss_history,
            "metric_history": metric_history if metric_func is not None else None,
        },
        "params_used": {
            "clip_value": clip_value,
            "max_iter": max_iter,
            "tol": tol,
            "learning_rate": learning_rate,
            "normalize_grad": normalize_grad,
        },
        "warnings": [],
    }

    return result

def _validate_inputs(
    loss_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    params_init: np.ndarray,
) -> None:
    """
    Validate the inputs for gradient clipping.

    Parameters:
    -----------
    loss_func : callable
        Function to compute the loss given parameters.
    grad_func : callable
        Function to compute the gradient of the loss with respect to parameters.
    params_init : ndarray
        Initial parameters.

    Raises:
    -------
    ValueError
        If any of the inputs are invalid.
    """
    if not callable(loss_func):
        raise ValueError("loss_func must be a callable.")
    if not callable(grad_func):
        raise ValueError("grad_func must be a callable.")
    if not isinstance(params_init, np.ndarray):
        raise ValueError("params_init must be a numpy array.")
    if np.any(np.isnan(params_init)) or np.any(np.isinf(params_init)):
        raise ValueError("params_init must not contain NaN or inf values.")
