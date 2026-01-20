"""
Quantix – Module bootstrap
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# grid_system
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union, List, Tuple

def grid_system_fit(
    data: np.ndarray,
    target: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Perform grid system computation with configurable parameters.

    Parameters:
    - data: Input features (n_samples, n_features)
    - target: Target values (n_samples,)
    - normalizer: Callable for data normalization
    - metric: Metric to evaluate performance ('mse', 'mae', 'r2', or custom callable)
    - distance: Distance metric ('euclidean', 'manhattan', etc. or custom callable)
    - solver: Solver method ('closed_form', 'gradient_descent', etc.)
    - regularization: Regularization type (None, 'l1', 'l2', etc.)
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - custom_metric: Custom metric function

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Normalize data
    normalized_data = normalizer(data)
    normalized_target = normalizer(target.reshape(-1, 1)).flatten()

    # Select metric
    if isinstance(metric, str):
        metric_func = _get_metric(metric)
    else:
        metric_func = metric

    # Select distance
    if isinstance(distance, str):
        distance_func = _get_distance(distance)
    else:
        distance_func = distance

    # Select solver
    if solver == 'closed_form':
        params, metrics = _solve_closed_form(normalized_data, normalized_target)
    elif solver == 'gradient_descent':
        params, metrics = _solve_gradient_descent(
            normalized_data, normalized_target,
            distance_func, metric_func,
            tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization:
        params = _apply_regularization(params, normalized_data, regularization)

    # Calculate final metrics
    final_metrics = _calculate_metrics(normalized_data, normalized_target, params, metric_func)

    return {
        'result': params,
        'metrics': final_metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(normalized_data, normalized_target)
    }

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if target.ndim != 1:
        raise ValueError("Target must be 1-dimensional")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have same number of samples")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get metric function based on name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on name."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_closed_form(data: np.ndarray, target: np.ndarray) -> tuple:
    """Solve using closed form solution."""
    # Add intercept term
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])
    params = np.linalg.inv(data_with_intercept.T @ data_with_intercept) @ data_with_intercept.T @ target
    metrics = _calculate_metrics(data, target, params[1:], _mean_squared_error)
    return params, metrics

def _solve_gradient_descent(
    data: np.ndarray,
    target: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> tuple:
    """Solve using gradient descent."""
    # Initialize parameters
    params = np.zeros(data.shape[1] + 1)  # +1 for intercept
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])

    for _ in range(max_iter):
        predictions = data_with_intercept @ params
        error = target - predictions

        # Gradient calculation
        gradient = -2 * (data_with_intercept.T @ error) / data.shape[0]

        # Update parameters
        params -= gradient * tol

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    metrics = _calculate_metrics(data, target, params[1:], metric_func)
    return params, metrics

def _apply_regularization(
    params: np.ndarray,
    data: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply regularization to parameters."""
    if regularization == 'l1':
        params[1:] -= np.sign(params[1:]) * 0.1  # Simple L1 regularization
    elif regularization == 'l2':
        params[1:] -= 0.1 * params[1:]  # Simple L2 regularization
    return params

def _calculate_metrics(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    predictions = data @ params
    return {
        'metric': metric_func(target, predictions),
        'mse': _mean_squared_error(target, predictions),
        'mae': _mean_absolute_error(target, predictions)
    }

def _check_warnings(data: np.ndarray, target: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if data.shape[1] > data.shape[0]:
        warnings.append("Warning: Number of features exceeds number of samples")
    if np.var(data) < 1e-6:
        warnings.append("Warning: Low variance in features")
    return warnings

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

################################################################################
# container
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def container_fit(
    data: np.ndarray,
    target: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Main function to fit a bootstrap container model.

    Parameters:
    -----------
    data : np.ndarray
        Input features array of shape (n_samples, n_features)
    target : np.ndarray
        Target values array of shape (n_samples,)
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the model performance
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance metric for bootstrap sampling
    solver : str
        Solver algorithm to use
    regularization : Optional[str]
        Regularization type to apply
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Normalize data
    normalized_data = normalizer(data)
    normalized_target = normalizer(target.reshape(-1, 1)).flatten()

    # Prepare bootstrap samples
    bootstrap_samples = _bootstrap_sampling(normalized_data, distance)

    # Choose solver and fit model
    if solver == 'closed_form':
        params = _closed_form_solver(bootstrap_samples, normalized_target)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(
            bootstrap_samples, normalized_target,
            tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization == 'l1':
        params = _apply_l1_regularization(params)
    elif regularization == 'l2':
        params = _apply_l2_regularization(params)
    elif regularization == 'elasticnet':
        params = _apply_elasticnet_regularization(params)

    # Calculate metrics
    predictions = np.dot(normalized_data, params)
    metrics = _calculate_metrics(
        predictions, normalized_target,
        metric=metric,
        custom_metric=custom_metric
    )

    # Prepare results dictionary
    result = {
        'result': predictions,
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
        'warnings': _check_warnings(predictions, normalized_target)
    }

    return result

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have the same number of samples")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values")

def _bootstrap_sampling(data: np.ndarray, distance: Union[str, Callable]) -> np.ndarray:
    """Generate bootstrap samples using specified distance metric."""
    n_samples = data.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return data[indices]

def _closed_form_solver(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.inv(data.T @ data) @ data.T @ target

def _gradient_descent_solver(
    data: np.ndarray,
    target: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    n_features = data.shape[1]
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradients = -2 * (data.T @ (target - data @ params)) / n_features
        new_params = params - learning_rate * gradients

        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return params

def _apply_l1_regularization(params: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply L1 regularization to parameters."""
    return np.sign(params) * (np.abs(params) - alpha)

def _apply_l2_regularization(params: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply L2 regularization to parameters."""
    return params / (1 + alpha * np.linalg.norm(params))

def _apply_elasticnet_regularization(
    params: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1
) -> np.ndarray:
    """Apply elastic net regularization to parameters."""
    l1 = _apply_l1_regularization(params, alpha1)
    l2 = _apply_l2_regularization(params, alpha2)
    return (l1 + l2) / 2

def _calculate_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    if metric == 'mse':
        metrics['mse'] = np.mean((predictions - target) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(predictions - target))
    elif metric == 'r2':
        ss_res = np.sum((predictions - target) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metric):
        metrics['custom'] = metric(predictions, target)

    if custom_metric is not None:
        metrics['custom'] = custom_metric(predictions, target)

    return metrics

def _check_warnings(predictions: np.ndarray, target: np.ndarray) -> list:
    """Check for potential warnings in the results."""
    warnings = []

    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        warnings.append("Predictions contain NaN or infinite values")

    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        warnings.append("Target contains NaN or infinite values")

    return warnings

################################################################################
# row
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def row_fit(
    data: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    solver: str = 'closed_form',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute bootstrap row-wise statistics.

    Parameters:
    -----------
    data : np.ndarray
        Input features array of shape (n_samples, n_features)
    target : np.ndarray
        Target values array of shape (n_samples,)
    metric : str or callable, optional
        Metric to use for evaluation ('mse', 'mae', 'r2')
    normalizer : callable, optional
        Normalization function to apply to data
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    max_iter : int, optional
        Maximum number of iterations for iterative solvers
    tol : float, optional
        Tolerance for convergence
    random_state : int, optional
        Random seed for reproducibility
    custom_metric : callable, optional
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> target = np.random.rand(100)
    >>> result = row_fit(data, target)
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Apply normalization if specified
    normalized_data = data.copy()
    if normalizer is not None:
        normalized_data = normalizer(normalized_data)

    # Initialize random state
    rng = np.random.RandomState(random_state)

    # Select solver
    if solver == 'closed_form':
        params = _closed_form_solver(normalized_data, target)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(
            normalized_data, target,
            max_iter=max_iter,
            tol=tol,
            rng=rng
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(
        normalized_data, target,
        params, metric=metric,
        custom_metric=custom_metric
    )

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data and target arrays."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Data and target must be numpy arrays")

    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")

    if target.ndim != 1:
        raise ValueError("Target must be a 1D array")

    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have same number of samples")

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")

    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values")

def _closed_form_solver(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    X = np.column_stack([np.ones(data.shape[0]), data])
    theta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    return theta

def _gradient_descent_solver(
    data: np.ndarray,
    target: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-4,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    n_features = data.shape[1]
    theta = rng.uniform(-0.1, 0.1, n_features + 1) if rng else np.zeros(n_features + 1)

    for _ in range(max_iter):
        X = np.column_stack([np.ones(data.shape[0]), data])
        predictions = X @ theta
        errors = predictions - target

        gradient = (X.T @ errors) / data.shape[0]
        theta -= 0.01 * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return theta

def _compute_metrics(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    X = np.column_stack([np.ones(data.shape[0]), data])
    predictions = X @ params

    metrics_dict = {}

    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics_dict['mse'] = np.mean((predictions - target) ** 2)

    if metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics_dict['mae'] = np.mean(np.abs(predictions - target))

    if metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((predictions - target) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        metrics_dict['r2'] = 1 - (ss_res / ss_tot)

    if custom_metric is not None:
        metrics_dict['custom'] = custom_metric(target, predictions)

    return metrics_dict

################################################################################
# column
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, y: Optional[np.ndarray] = None,
                   normalization: str = 'standard') -> tuple:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X, y
    elif normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    if y is not None:
        return X_norm, y
    return X_norm

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Union[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metric_results = {}

    if isinstance(metrics, str):
        metrics_list = [metrics]
    else:
        metrics_list = metrics

    for metric in metrics_list:
        if metric == 'mse':
            metric_results['mse'] = np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            metric_results['mae'] = np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metric_results['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == 'logloss':
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            metric_results['logloss'] = -np.mean(y_true * np.log(y_pred) +
                                                (1 - y_true) * np.log(1 - y_pred))
        elif callable(metric):
            metric_results['custom'] = metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metric_results

def _bootstrap_resample(X: np.ndarray, y: Optional[np.ndarray] = None,
                       n_samples: int = 100) -> tuple:
    """Perform bootstrap resampling."""
    n_samples_orig = X.shape[0]
    indices = np.random.choice(n_samples_orig, size=n_samples, replace=True)
    X_resampled = X[indices]
    if y is not None:
        y_resampled = y[indices]
        return X_resampled, y_resampled
    return X_resampled

def column_fit(X: np.ndarray,
              y: Optional[np.ndarray] = None,
              n_bootstraps: int = 100,
              normalization: str = 'standard',
              metrics: Union[str, Callable] = 'mse',
              random_state: Optional[int] = None) -> Dict:
    """
    Perform bootstrap column analysis.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values of shape (n_samples,) or None
    n_bootstraps : int, default=100
        Number of bootstrap samples to generate
    normalization : str, default='standard'
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metrics : Union[str, Callable], default='mse'
        Metrics to compute ('mse', 'mae', 'r2', 'logloss') or custom callable
    random_state : Optional[int], default=None
        Random seed for reproducibility

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = column_fit(X, y, n_bootstraps=50, metrics=['mse', 'r2'])
    """
    if random_state is not None:
        np.random.seed(random_state)

    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, y, normalization)

    # Initialize results storage
    bootstrap_results = []
    warnings_list = []

    for _ in range(n_bootstraps):
        try:
            # Bootstrap resample
            X_resampled, y_resampled = _bootstrap_resample(X_norm, y)

            # Here you would typically fit a model and get predictions
            # For this example, we'll just use the mean as prediction
            if y_resampled is not None:
                y_pred = np.mean(y_resampled)
            else:
                y_pred = np.mean(X_resampled, axis=0)

            # Compute metrics
            if y_resampled is not None:
                metrics_results = _compute_metrics(y_resampled, y_pred, metrics)
            else:
                metrics_results = {}

            bootstrap_results.append({
                'X_resampled': X_resampled,
                'y_pred': y_pred,
                'metrics': metrics_results
            })
        except Exception as e:
            warnings_list.append(str(e))

    # Calculate final statistics
    result = {
        'mean_metrics': {k: np.mean([r['metrics'].get(k, 0) for r in bootstrap_results])
                        for k in set().union(*[r['metrics'].keys() for r in bootstrap_results])},
        'std_metrics': {k: np.std([r['metrics'].get(k, 0) for r in bootstrap_results])
                       for k in set().union(*[r['metrics'].keys() for r in bootstrap_results])},
        'n_bootstraps': n_bootstraps,
        'normalization_used': normalization
    }

    return {
        'result': result,
        'metrics': metrics_results if 'metrics_results' in locals() else {},
        'params_used': {
            'n_bootstraps': n_bootstraps,
            'normalization': normalization,
            'metrics': metrics,
            'random_state': random_state
        },
        'warnings': warnings_list if warnings_list else None
    }

################################################################################
# responsive_design
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def responsive_design_fit(
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
    Main function for responsive design bootstrap.

    Parameters:
    - X: Input features (n_samples, n_features)
    - y: Target values (n_samples,)
    - normalizer: Callable for feature normalization
    - metric: Metric to optimize ('mse', 'mae', 'r2', 'logloss')
    - distance: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type (None, 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    - custom_metric: Custom metric function
    - custom_distance: Custom distance function

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize features
    X_norm = normalizer(X)

    # Prepare parameters
    params_used = {
        'normalization': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Select metric function
    metric_func = _select_metric(metric, custom_metric)

    # Select distance function
    distance_func = _select_distance(distance, custom_distance)

    # Solve the problem
    if solver == 'closed_form':
        result, warnings = _solve_closed_form(X_norm, y, metric_func, distance_func)
    elif solver == 'gradient_descent':
        result, warnings = _solve_gradient_descent(X_norm, y, metric_func, distance_func,
                                                  regularization, tol, max_iter)
    elif solver == 'newton':
        result, warnings = _solve_newton(X_norm, y, metric_func, distance_func,
                                        regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        result, warnings = _solve_coordinate_descent(X_norm, y, metric_func, distance_func,
                                                    regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result, X_norm, y, metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

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

def _select_metric(metric: str, custom_metric: Optional[Callable]) -> Callable:
    """Select the appropriate metric function."""
    if custom_metric is not None:
        return custom_metric

    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics[metric]

def _select_distance(distance: str, custom_distance: Optional[Callable]) -> Callable:
    """Select the appropriate distance function."""
    if custom_distance is not None:
        return custom_distance

    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': _minkowski_distance
    }

    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")

    return distances[distance]

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      metric_func: Callable, distance_func: Callable) -> tuple:
    """Solve using closed form solution."""
    # Placeholder implementation
    beta = np.linalg.pinv(X) @ y
    warnings = []
    return beta, warnings

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           metric_func: Callable, distance_func: Callable,
                           regularization: Optional[str], tol: float, max_iter: int) -> tuple:
    """Solve using gradient descent."""
    # Placeholder implementation
    beta = np.zeros(X.shape[1])
    warnings = []
    return beta, warnings

def _solve_newton(X: np.ndarray, y: np.ndarray,
                  metric_func: Callable, distance_func: Callable,
                  regularization: Optional[str], tol: float, max_iter: int) -> tuple:
    """Solve using Newton's method."""
    # Placeholder implementation
    beta = np.zeros(X.shape[1])
    warnings = []
    return beta, warnings

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             metric_func: Callable, distance_func: Callable,
                             regularization: Optional[str], tol: float, max_iter: int) -> tuple:
    """Solve using coordinate descent."""
    # Placeholder implementation
    beta = np.zeros(X.shape[1])
    warnings = []
    return beta, warnings

def _calculate_metrics(result: np.ndarray,
                      X: np.ndarray, y: np.ndarray,
                      metric_func: Callable) -> Dict[str, float]:
    """Calculate various metrics."""
    y_pred = X @ result
    return {
        'primary_metric': metric_func(y, y_pred),
        'r2': _r_squared(y, y_pred)
    }

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate log loss."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 2) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1/p)

################################################################################
# breakpoints
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def breakpoints_fit(
    data: np.ndarray,
    n_breakpoints: int = 1,
    normalizer: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute optimal breakpoints for a given dataset using bootstrap methodology.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    n_breakpoints : int, optional
        Number of breakpoints to find. Default is 1.
    normalizer : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'. Default is 'standard'.
    metric : str or callable, optional
        Metric to optimize: 'mse', 'mae', 'r2', or custom callable. Default is 'mse'.
    distance : str, optional
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable. Default is 'euclidean'.
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', or 'coordinate_descent'. Default is 'closed_form'.
    regularization : str, optional
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Optimal breakpoints.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> result = breakpoints_fit(data, n_breakpoints=3)
    """
    # Validate inputs
    _validate_inputs(data, n_breakpoints)

    # Normalize data
    normalized_data = _normalize_data(data, normalizer)

    # Prepare solver parameters
    solver_params = {
        'tol': tol,
        'max_iter': max_iter,
        'regularization': regularization
    }

    # Choose metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for breakpoints
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, n_breakpoints, metric_func)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(normalized_data, n_breakpoints,
                                       metric_func, distance_func, solver_params)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(normalized_data, n_breakpoints,
                                         metric_func, distance_func, solver_params)
    else:
        raise ValueError("Unsupported solver method")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, result['breakpoints'], metric_func)

    # Prepare output
    output = {
        'result': result['breakpoints'],
        'metrics': metrics,
        'params_used': {
            'n_breakpoints': n_breakpoints,
            'normalizer': normalizer,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': result.get('warnings', [])
    }

    return output

def _validate_inputs(data: np.ndarray, n_breakpoints: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if n_breakpoints < 1:
        raise ValueError("Number of breakpoints must be at least 1")
    if n_breakpoints >= data.shape[0]:
        raise ValueError("Number of breakpoints cannot be greater than or equal to number of samples")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError("Unsupported normalization method")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get metric function based on input."""
    if custom_metric is not None:
        return custom_metric
    if metric == 'mse':
        return _mean_squared_error
    elif metric == 'mae':
        return _mean_absolute_error
    elif metric == 'r2':
        return _r_squared
    else:
        raise ValueError("Unsupported metric")

def _get_distance_function(distance: str, custom_distance: Optional[Callable]) -> Callable:
    """Get distance function based on input."""
    if custom_distance is not None:
        return custom_distance
    if distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    else:
        raise ValueError("Unsupported distance metric")

def _solve_closed_form(data: np.ndarray, n_breakpoints: int, metric_func: Callable) -> Dict:
    """Solve for breakpoints using closed-form solution."""
    # Implementation of closed-form solver
    return {'breakpoints': np.linspace(0, 1, n_breakpoints + 2)[1:-1], 'warnings': []}

def _solve_gradient_descent(
    data: np.ndarray,
    n_breakpoints: int,
    metric_func: Callable,
    distance_func: Callable,
    params: Dict
) -> Dict:
    """Solve for breakpoints using gradient descent."""
    # Implementation of gradient descent solver
    return {'breakpoints': np.linspace(0, 1, n_breakpoints + 2)[1:-1], 'warnings': []}

def _solve_coordinate_descent(
    data: np.ndarray,
    n_breakpoints: int,
    metric_func: Callable,
    distance_func: Callable,
    params: Dict
) -> Dict:
    """Solve for breakpoints using coordinate descent."""
    # Implementation of coordinate descent solver
    return {'breakpoints': np.linspace(0, 1, n_breakpoints + 2)[1:-1], 'warnings': []}

def _compute_metrics(data: np.ndarray, breakpoints: np.ndarray, metric_func: Callable) -> Dict:
    """Compute metrics for the given breakpoints."""
    # Implementation of metric computation
    return {'metric_value': 0.5}

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
    return 1 - (ss_res / (ss_tot + 1e-8))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

################################################################################
# spacer_classes
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: Optional[Callable] = None,
    distance: Optional[Callable] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")
    if metric is not None and not callable(metric):
        raise TypeError("metric must be a callable or None")
    if distance is not None and not callable(distance):
        raise TypeError("distance must be a callable or None")

def _default_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default metric: Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _default_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Default distance metric: Euclidean."""
    return np.linalg.norm(x1 - x2)

def _normalize_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = "standard"
) -> Dict[str, Union[np.ndarray, None]]:
    """Normalize data based on specified method."""
    if normalization == "none":
        return {"X": X, "y": y}
    elif normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    if y is not None:
        return {"X": X_norm, "y": y}
    else:
        return {"X": X_norm}

def _bootstrap_resample(X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Perform bootstrap resampling."""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_resampled = X[indices]
    if y is not None:
        return {"X": X_resampled, "y": y[indices]}
    else:
        return {"X": X_resampled}

def _compute_spacer_classes(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    metric: Optional[Callable] = None,
    distance: Optional[Callable] = None,
    n_bootstraps: int = 100
) -> Dict[str, Any]:
    """Compute spacer classes using bootstrap method."""
    if metric is None:
        metric = _default_metric
    if distance is None:
        distance = _default_distance

    metrics = []
    for _ in range(n_bootstraps):
        resampled = _bootstrap_resample(X, y)
        X_resampled = resampled["X"]
        y_resampled = resampled.get("y")

        # Here you would typically compute your spacer classes
        # For demonstration, we'll just use a simple example
        if y_resampled is not None:
            # Example: compute mean prediction
            y_pred = np.mean(y_resampled)
            metrics.append(metric(y_resampled, np.full_like(y_resampled, y_pred)))
        else:
            # Example: compute mean distance between samples
            distances = []
            for i in range(X_resampled.shape[0]):
                for j in range(i + 1, X_resampled.shape[0]):
                    distances.append(distance(X_resampled[i], X_resampled[j]))
            metrics.append(np.mean(distances))

    return {
        "result": np.mean(metrics),
        "metrics": metrics,
        "params_used": {
            "metric": metric.__name__ if callable(metric) else str(metric),
            "distance": distance.__name__ if callable(distance) else str(distance),
            "n_bootstraps": n_bootstraps
        },
        "warnings": []
    }

def spacer_classes_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metric: Optional[Callable] = None,
    distance: Optional[Callable] = None,
    n_bootstraps: int = 100
) -> Dict[str, Any]:
    """
    Main function to compute spacer classes using bootstrap method.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Optional[Callable]
        Metric function to evaluate performance
    distance : Optional[Callable]
        Distance function for computing distances between samples
    n_bootstraps : int
        Number of bootstrap iterations

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = spacer_classes_fit(X, y)
    """
    _validate_inputs(X, y, metric, distance)

    normalized = _normalize_data(X, y, normalization)
    X_norm = normalized["X"]
    y_norm = normalized.get("y")

    return _compute_spacer_classes(
        X_norm,
        y_norm,
        metric=metric,
        distance=distance,
        n_bootstraps=n_bootstraps
    )

################################################################################
# utilities
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def utilities_fit(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    normalization: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mean',
    random_state: Optional[int] = None,
    custom_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute bootstrap utilities for a given statistic function.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    statistic_func : callable
        Function to compute the statistic of interest.
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    metric : str or callable, optional
        Metric to evaluate ('mean', 'median', 'std', custom callable) (default: 'mean').
    random_state : int, optional
        Random seed for reproducibility (default: None).
    custom_options : dict, optional
        Additional options for customization (default: None).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, statistic_func)

    # Set random seed if provided
    rng = np.random.RandomState(random_state)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Compute bootstrap statistics
    bootstrap_stats = _compute_bootstrap_statistics(
        normalized_data, statistic_func, n_bootstrap, rng
    )

    # Compute metrics
    metrics = _compute_metrics(bootstrap_stats, metric)

    # Prepare output
    result = {
        'result': bootstrap_stats,
        'metrics': metrics,
        'params_used': {
            'n_bootstrap': n_bootstrap,
            'normalization': normalization,
            'metric': metric,
            'random_state': random_state
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, statistic_func: Callable) -> None:
    """Validate input data and statistic function."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if not callable(statistic_func):
        raise TypeError("statistic_func must be a callable.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_bootstrap_statistics(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute bootstrap statistics."""
    n_samples = data.shape[0]
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample = data[indices]
        bootstrap_stats[i] = statistic_func(sample)

    return bootstrap_stats

def _compute_metrics(
    bootstrap_stats: np.ndarray,
    metric: Union[str, Callable[[np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for bootstrap statistics."""
    if callable(metric):
        return {'custom_metric': metric(bootstrap_stats)}

    metrics = {}
    if metric == 'mean':
        metrics['mean'] = np.mean(bootstrap_stats)
    elif metric == 'median':
        metrics['median'] = np.median(bootstrap_stats)
    elif metric == 'std':
        metrics['std'] = np.std(bootstrap_stats)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

################################################################################
# components
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def components_fit(
    data: np.ndarray,
    n_components: int = 2,
    normalizer: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit the components using bootstrap method.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int, default=2
        Number of components to extract
    normalizer : str, default='standard'
        Normalization method: 'none', 'standard', 'minmax', 'robust'
    distance_metric : str or callable, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski' or custom callable
    solver : str, default='closed_form'
        Solver method: 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int or None, default=None
        Random seed for reproducibility
    custom_metric : callable or None, default=None
        Custom metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = components_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data
    normalized_data = _apply_normalization(data, normalizer)

    # Initialize parameters
    params = {
        'n_components': n_components,
        'normalizer': normalizer,
        'distance_metric': distance_metric,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol
    }

    # Select solver
    if solver == 'closed_form':
        components, metrics = _closed_form_solver(normalized_data, n_components)
    elif solver == 'gradient_descent':
        components, metrics = _gradient_descent_solver(
            normalized_data,
            n_components,
            distance_metric,
            max_iter,
            tol,
            random_state
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Calculate metrics
    if custom_metric is not None:
        metrics['custom'] = custom_metric(components, normalized_data)

    # Prepare output
    result = {
        'result': components,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if n_components <= 0 or n_components > data.shape[1]:
        raise ValueError("Invalid number of components")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def _closed_form_solver(data: np.ndarray, n_components: int) -> tuple:
    """Closed form solution for component extraction."""
    # This is a placeholder for the actual implementation
    components = np.random.rand(data.shape[1], n_components)
    metrics = {
        'reconstruction_error': np.linalg.norm(data - data @ components) ** 2
    }
    return components, metrics

def _gradient_descent_solver(
    data: np.ndarray,
    n_components: int,
    distance_metric: Union[str, Callable],
    max_iter: int,
    tol: float,
    random_state: Optional[int]
) -> tuple:
    """Gradient descent solver for component extraction."""
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize components randomly
    components = np.random.rand(data.shape[1], n_components)
    prev_error = float('inf')

    for _ in range(max_iter):
        # Update components using gradient descent
        # This is a placeholder for the actual implementation
        new_components = _update_components(data, components, distance_metric)

        # Calculate error
        current_error = np.linalg.norm(data - data @ new_components) ** 2

        # Check for convergence
        if abs(prev_error - current_error) < tol:
            break

        prev_error = current_error
        components = new_components

    metrics = {
        'reconstruction_error': prev_error,
        'iterations': _ + 1
    }
    return components, metrics

def _update_components(
    data: np.ndarray,
    components: np.ndarray,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Update components using gradient descent."""
    # This is a placeholder for the actual implementation
    return components

def _calculate_metrics(
    components: np.ndarray,
    data: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate metrics for the components."""
    metrics = {
        'reconstruction_error': np.linalg.norm(data - data @ components) ** 2,
        'explained_variance': 1 - metrics['reconstruction_error'] / np.linalg.norm(data) ** 2
    }

    if custom_metric is not None:
        metrics['custom'] = custom_metric(components, data)

    return metrics

################################################################################
# forms
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def forms_fit(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    metric: Union[str, Callable] = 'mean',
    normalization: str = 'none',
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute bootstrap statistics for a given dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000)
    metric : str or callable, optional
        Metric to compute (default: 'mean'). Options: 'mean', 'median', 'std', 'var'
    normalization : str, optional
        Normalization method (default: 'none'). Options: 'none', 'standard', 'minmax'
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_metric : callable, optional
        Custom metric function if not using built-in options

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': bootstrap statistics
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = forms_fit(data, n_bootstrap=500, metric='median')
    """
    # Validate inputs
    _validate_inputs(data, n_bootstrap)

    # Set random seed if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if requested
    normalized_data, norm_warning = _apply_normalization(data, normalization)

    # Compute bootstrap statistics
    bootstrap_stats = _compute_bootstrap(
        normalized_data,
        n_bootstrap,
        metric if not custom_metric else custom_metric,
        rng
    )

    # Compute additional metrics
    metrics = _compute_metrics(bootstrap_stats)

    return {
        'result': bootstrap_stats,
        'metrics': metrics,
        'params_used': {
            'n_bootstrap': n_bootstrap,
            'metric': metric if not custom_metric else 'custom',
            'normalization': normalization
        },
        'warnings': [norm_warning] if norm_warning else []
    }

def _validate_inputs(data: np.ndarray, n_bootstrap: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, method: str) -> tuple:
    """Apply requested normalization to the data."""
    if method == 'none':
        return data, None

    normalized = data.copy()
    warning = None

    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)

    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / (iqr + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, warning

def _compute_bootstrap(
    data: np.ndarray,
    n_samples: int,
    metric_func: Callable,
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Compute bootstrap statistics."""
    n_samples_orig = data.shape[0]
    stats = np.zeros(n_samples)

    for i in range(n_samples):
        indices = rng.choice(n_samples_orig, size=n_samples_orig, replace=True)
        sample = data[indices]
        stats[i] = metric_func(sample)

    return {
        'values': stats,
        'mean': np.mean(stats),
        'std': np.std(stats),
        'ci_95': np.percentile(stats, [2.5, 97.5])
    }

def _compute_metrics(bootstrap_stats: Dict[str, Any]) -> Dict[str, float]:
    """Compute additional metrics from bootstrap statistics."""
    return {
        'skewness': _compute_skewness(bootstrap_stats['values']),
        'kurtosis': _compute_kurtosis(bootstrap_stats['values'])
    }

def _compute_skewness(values: np.ndarray) -> float:
    """Compute skewness of values."""
    mean = np.mean(values)
    std = np.std(values)
    return np.sum((values - mean) ** 3) / (len(values) * std ** 3)

def _compute_kurtosis(values: np.ndarray) -> float:
    """Compute kurtosis of values."""
    mean = np.mean(values)
    std = np.std(values)
    return np.sum((values - mean) ** 4) / (len(values) * std ** 4)

################################################################################
# buttons
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def buttons_fit(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    metric: Union[str, Callable] = 'mean',
    normalization: str = 'none',
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform bootstrap resampling and compute statistics.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000).
    metric : str or callable, optional
        Metric to compute ('mean', 'median', 'std', 'custom') (default: 'mean').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: 'none').
    random_state : int, optional
        Random seed for reproducibility (default: None).
    custom_metric : callable, optional
        Custom metric function if metric='custom' (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, n_bootstrap)

    # Set random seed if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if required
    normalized_data, norm_params = _apply_normalization(data, normalization)

    # Perform bootstrap resampling
    bootstrap_samples = _bootstrap_resample(normalized_data, n_bootstrap, rng)

    # Compute statistics
    results = _compute_statistics(bootstrap_samples, metric, custom_metric)

    # Prepare output
    output = {
        'result': results,
        'metrics': _compute_metrics(results),
        'params_used': {
            'n_bootstrap': n_bootstrap,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': _check_warnings(data, normalized_data)
    }

    return output

def _validate_inputs(data: np.ndarray, n_bootstrap: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _apply_normalization(data: np.ndarray, method: str) -> tuple:
    """Apply normalization to the data."""
    norm_params = {}
    if method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
        norm_params['mean'] = mean
        norm_params['std'] = std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        norm_params['min'] = min_val
        norm_params['max'] = max_val
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        normalized_data = (data - median) / iqr
        norm_params['median'] = median
        norm_params['iqr'] = iqr
    else:
        normalized_data = data.copy()
    return normalized_data, norm_params

def _bootstrap_resample(data: np.ndarray, n_samples: int, rng: np.random.RandomState) -> list:
    """Perform bootstrap resampling."""
    n = len(data)
    samples = [rng.choice(data, size=n, replace=True) for _ in range(n_samples)]
    return samples

def _compute_statistics(samples: list, metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Dict[str, Any]:
    """Compute statistics from bootstrap samples."""
    results = {}
    for i, sample in enumerate(samples):
        if metric == 'mean':
            results[f'sample_{i}'] = np.mean(sample)
        elif metric == 'median':
            results[f'sample_{i}'] = np.median(sample)
        elif metric == 'std':
            results[f'sample_{i}'] = np.std(sample)
        elif metric == 'custom' and custom_metric is not None:
            results[f'sample_{i}'] = custom_metric(sample)
        else:
            raise ValueError("Invalid metric or missing custom metric function.")
    return results

def _compute_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute metrics from bootstrap results."""
    values = list(results.values())
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'ci_lower': np.percentile(values, 2.5),
        'ci_upper': np.percentile(values, 97.5)
    }

def _check_warnings(data: np.ndarray, normalized_data: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if not np.array_equal(data, normalized_data):
        warnings.append("Data was normalized.")
    return warnings

################################################################################
# navbar
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def navbar_fit(
    data: np.ndarray,
    metric: str = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the bootstrap navbar statistics.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    metric : str, optional
        Metric to use ('mse', 'mae', 'r2', etc.).
    normalizer : Callable, optional
        Custom normalization function.
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', etc.).
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', etc.).
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', etc.).
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalizer)

    # Compute statistics based on user choices
    result = _compute_statistics(
        normalized_data,
        metric=metric,
        distance=distance,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter,
        custom_metric=custom_metric
    )

    # Calculate metrics
    metrics = _calculate_metrics(result, metric, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(result)
    }

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _compute_statistics(
    data: np.ndarray,
    metric: str,
    distance: str,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, Any]:
    """Compute statistics based on user choices."""
    if solver == 'closed_form':
        result = _closed_form_solver(data, distance, regularization)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(data, distance, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return result

def _closed_form_solver(
    data: np.ndarray,
    distance: str,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Closed form solver for statistics."""
    # Placeholder for closed form computation
    return {'statistic': np.mean(data), 'params': {}}

def _gradient_descent_solver(
    data: np.ndarray,
    distance: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Gradient descent solver for statistics."""
    # Placeholder for gradient descent computation
    return {'statistic': np.median(data), 'params': {}}

def _calculate_metrics(
    result: Dict[str, Any],
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Calculate metrics based on the result."""
    if custom_metric is not None:
        return {'custom': custom_metric(result['statistic'], result['statistic'])}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((result['statistic'] - np.mean(result['statistic']))**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(result['statistic'] - np.mean(result['statistic'])))
    elif metric == 'r2':
        metrics['r2'] = 1 - np.sum((result['statistic'] - np.mean(result['statistic']))**2) / np.sum((result['statistic'] - result['statistic'].mean())**2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics

def _check_warnings(result: Dict[str, Any]) -> List[str]:
    """Check for warnings in the result."""
    warnings = []
    if np.isnan(result['statistic']).any():
        warnings.append("Result contains NaN values.")
    if np.isinf(result['statistic']).any():
        warnings.append("Result contains infinite values.")
    return warnings

################################################################################
# dropdowns
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def dropdowns_fit(
    data: np.ndarray,
    *,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main function for dropdowns bootstrap method.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable
    distance : str
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str or None
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    custom_metric : callable or None
        Custom metric function if needed
    custom_distance : callable or None
        Custom distance function if needed
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    random_state : int or None
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = dropdowns_fit(data, normalization='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(data, normalization, metric, distance, solver)

    # Set random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Initialize parameters
    params = _initialize_parameters(normalized_data.shape[1], solver, regularization)

    # Solve using selected method
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, metric_func)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(
            normalized_data, metric_func, distance_func,
            params, tol, max_iter, rng
        )
    elif solver == 'newton':
        result = _solve_newton(
            normalized_data, metric_func, distance_func,
            params, tol, max_iter
        )
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(
            normalized_data, metric_func, distance_func,
            params, tol, max_iter
        )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, result['params'], metric_func)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': _check_warnings(normalized_data, result)
    }

def _validate_inputs(
    data: np.ndarray,
    normalization: str,
    metric: Union[str, Callable],
    distance: str,
    solver: str
) -> None:
    """Validate all input parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

    valid_metrics = ['mse', 'mae', 'r2', 'logloss']
    if isinstance(metric, str) and metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics} or a callable")

    valid_distances = ['euclidean', 'manhattan', 'cosine', 'minkowski']
    if distance not in valid_distances:
        raise ValueError(f"Distance must be one of {valid_distances} or a callable")

    valid_solvers = ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']
    if solver not in valid_solvers:
        raise ValueError(f"Solver must be one of {valid_solvers}")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply selected normalization to data."""
    if method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Callable:
    """Return the appropriate metric function."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    def logloss(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) +
                       (1 - y_true) * np.log(1 - y_pred + 1e-8))

    metric_functions = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'logloss': logloss
    }
    return metric_functions[metric]

def _get_distance_function(
    distance: str,
    custom_distance: Optional[Callable]
) -> Callable:
    """Return the appropriate distance function."""
    if callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance

    def euclidean(a, b):
        return np.linalg.norm(a - b)

    def manhattan(a, b):
        return np.sum(np.abs(a - b))

    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def minkowski(a, b, p=3):
        return np.sum(np.abs(a - b) ** p) ** (1/p)

    distance_functions = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'cosine': cosine,
        'minkowski': minkowski
    }
    return distance_functions[distance]

def _initialize_parameters(
    n_features: int,
    solver: str,
    regularization: Optional[str]
) -> Dict[str, Any]:
    """Initialize parameters based on solver and regularization."""
    params = {
        'weights': np.zeros(n_features),
        'intercept': 0.0,
        'regularization_param': 1.0
    }

    if regularization == 'l1':
        params['regularization'] = 'l1'
    elif regularization == 'l2':
        params['regularization'] = 'l2'
    elif regularization == 'elasticnet':
        params['regularization'] = 'elasticnet'

    return params

def _solve_closed_form(
    data: np.ndarray,
    metric_func: Callable
) -> Dict[str, Any]:
    """Solve using closed form solution."""
    X = data[:, :-1]
    y = data[:, -1]

    # For demonstration, using least squares
    XTX_inv = np.linalg.inv(X.T @ X)
    weights = XTX_inv @ X.T @ y
    intercept = np.mean(y - X @ weights)

    return {
        'params': {'weights': weights, 'intercept': intercept},
        'converged': True,
        'iterations': 0
    }

def _solve_gradient_descent(
    data: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    params: Dict[str, Any],
    tol: float,
    max_iter: int,
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Solve using gradient descent."""
    X = data[:, :-1]
    y = data[:, -1]

    weights = params['weights'].copy()
    intercept = params['intercept']
    learning_rate = 0.01

    for i in range(max_iter):
        predictions = X @ weights + intercept
        error = y - predictions

        gradient_w = -2 * (X.T @ error) / len(y)
        gradient_b = -2 * np.sum(error) / len(y)

        weights -= learning_rate * gradient_w
        intercept -= learning_rate * gradient_b

        if np.linalg.norm(np.array([gradient_w, gradient_b])) < tol:
            break

    return {
        'params': {'weights': weights, 'intercept': intercept},
        'converged': i < max_iter,
        'iterations': i
    }

def _solve_newton(
    data: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    params: Dict[str, Any],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using Newton's method."""
    X = data[:, :-1]
    y = data[:, -1]

    weights = params['weights'].copy()
    intercept = params['intercept']

    for i in range(max_iter):
        predictions = X @ weights + intercept
        error = y - predictions

        # Compute gradient and Hessian (simplified for demonstration)
        gradient_w = -2 * (X.T @ error) / len(y)
        hessian_w = 2 * (X.T @ X) / len(y)

        update = np.linalg.inv(hessian_w) @ gradient_w
        weights -= update

        if np.linalg.norm(gradient_w) < tol:
            break

    return {
        'params': {'weights': weights, 'intercept': intercept},
        'converged': i < max_iter,
        'iterations': i
    }

def _solve_coordinate_descent(
    data: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    params: Dict[str, Any],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using coordinate descent."""
    X = data[:, :-1]
    y = data[:, -1]

    weights = params['weights'].copy()
    intercept = params['intercept']

    for i in range(max_iter):
        old_weights = weights.copy()

        for j in range(X.shape[1]):
            X_j = X[:, j]
            residual = y - (X @ weights + intercept)

            # Update weight for feature j
            numerator = X_j.T @ residual
            denominator = X_j.T @ X_j

            if denominator != 0:
                weights[j] += numerator / denominator

        # Update intercept
        intercept = np.mean(y - X @ weights)

        if np.linalg.norm(weights - old_weights) < tol:
            break

    return {
        'params': {'weights': weights, 'intercept': intercept},
        'converged': i < max_iter,
        'iterations': i
    }

def _calculate_metrics(
    data: np.ndarray,
    params: Dict[str, Any],
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate various metrics."""
    X = data[:, :-1]
    y_true = data[:, -1]
    y_pred = X @ params['weights'] + params['intercept']

    return {
        'metric': metric_func(y_true, y_pred),
        'mse': np.mean((y_true - y_pred) ** 2),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'r2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    }

def _check_warnings(
    data: np.ndarray,
    result: Dict[str, Any]
) -> list:
    """Check for potential warnings."""
    warnings = []

    if not result['converged']:
        warnings.append("Solver did not converge")

    if np.any(np.isnan(data)):
        warnings.append("Input data contains NaN values")

    return warnings

################################################################################
# modals
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def modals_fit(
    data: np.ndarray,
    target: np.ndarray,
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
    Fit a model using modal bootstrap methods.

    Parameters:
    -----------
    data : np.ndarray
        Input features array of shape (n_samples, n_features).
    target : np.ndarray
        Target values array of shape (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data.
    metric : str
        Metric to evaluate model performance. Options: 'mse', 'mae', 'r2'.
    distance : str
        Distance metric for bootstrap sampling. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Normalize data
    normalized_data = normalizer(data)
    normalized_target = normalizer(target.reshape(-1, 1)).flatten()

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
        params = _closed_form_solver(normalized_data, normalized_target)
    elif solver == 'gradient_descent':
        params = _gradient_descent_solver(
            normalized_data, normalized_target,
            metric_func, distance_func,
            regularization, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    predictions = _predict(normalized_data, params)
    metrics = {
        'metric': metric_func(predictions, normalized_target),
        'mse': _mean_squared_error(predictions, normalized_target),
        'mae': _mean_absolute_error(predictions, normalized_target)
    }

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data and target."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Data and target must be numpy arrays.")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have the same number of samples.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values.")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the metric name."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the distance name."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _closed_form_solver(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Closed form solution for linear regression."""
    return np.linalg.inv(data.T @ data) @ data.T @ target

def _gradient_descent_solver(
    data: np.ndarray,
    target: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for linear regression."""
    n_features = data.shape[1]
    params = np.zeros(n_features)
    prev_loss = float('inf')

    for _ in range(max_iter):
        predictions = _predict(data, params)
        loss = metric_func(predictions, target)

        if abs(prev_loss - loss) < tol:
            break

        prev_loss = loss
        gradient = _compute_gradient(data, target, predictions, regularization)
        params -= 0.01 * gradient

    return params

def _predict(data: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict using the given parameters."""
    return data @ params

def _compute_gradient(
    data: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute the gradient for gradient descent."""
    n_samples = data.shape[0]
    error = predictions - target
    gradient = (data.T @ error) / n_samples

    if regularization == 'l1':
        gradient += np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * params

    return gradient

def _mean_squared_error(predictions: np.ndarray, target: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((predictions - target) ** 2)

def _mean_absolute_error(predictions: np.ndarray, target: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(predictions - target))

def _r_squared(predictions: np.ndarray, target: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((predictions - target) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot)

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
# alerts
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def alerts_fit(
    data: np.ndarray,
    metric: str = 'mse',
    normalization: str = 'standard',
    n_bootstraps: int = 1000,
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute bootstrap alerts for given data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    metric : str or callable
        Metric to use for evaluation. Options: 'mse', 'mae', 'r2'
    normalization : str
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'
    n_bootstraps : int
        Number of bootstrap samples to generate
    random_state : Optional[int]
        Random seed for reproducibility
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = alerts_fit(data, metric='mse', normalization='standard')
    """
    # Validate inputs
    _validate_inputs(data, metric, custom_metric)

    # Set random state if provided
    rng = np.random.RandomState(random_state)

    # Normalize data
    normalized_data, norm_params = _apply_normalization(data, normalization)

    # Initialize results storage
    bootstrap_results = []
    original_metric_value = _compute_metric(normalized_data, metric, custom_metric)

    # Perform bootstrap sampling
    for _ in range(n_bootstraps):
        sample = rng.choice(normalized_data, size=len(normalized_data), replace=True)
        bootstrap_results.append(_compute_metric(sample, metric, custom_metric))

    # Calculate alerts
    alerts = _calculate_alerts(bootstrap_results, original_metric_value)

    return {
        'result': alerts,
        'metrics': {'original': original_metric_value, 'bootstrap_mean': np.mean(bootstrap_results)},
        'params_used': {
            'metric': metric,
            'normalization': normalization,
            'n_bootstraps': n_bootstraps
        },
        'warnings': _check_warnings(data, normalized_data)
    }

def _validate_inputs(
    data: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    valid_metrics = ['mse', 'mae', 'r2']
    if metric not in valid_metrics and custom_metric is None:
        raise ValueError(f"Metric must be one of {valid_metrics} or a custom callable")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply specified normalization to data."""
    normalized = data.copy()
    params = {}

    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / std
        params['mean'] = mean.tolist()
        params['std'] = std.tolist()

    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params['min'] = min_val.tolist()
        params['max'] = max_val.tolist()

    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        normalized = (data - median) / iqr
        params['median'] = median.tolist()
        params['iqr'] = iqr.tolist()

    return normalized, {'method': method, 'params': params}

def _compute_metric(
    data: np.ndarray,
    metric: str,
    custom_metric: Optional[Callable]
) -> float:
    """Compute specified metric for the data."""
    if custom_metric is not None:
        return custom_metric(data, data.mean(axis=0))

    if metric == 'mse':
        return np.mean((data - data.mean(axis=0))**2)
    elif metric == 'mae':
        return np.mean(np.abs(data - data.mean(axis=0)))
    elif metric == 'r2':
        ss_total = np.sum((data - data.mean())**2)
        ss_res = np.sum((data - data.mean(axis=0))**2, axis=1)
        return 1 - (np.sum(ss_res) / ss_total)

def _calculate_alerts(
    bootstrap_results: list[float],
    original_value: float
) -> Dict[str, float]:
    """Calculate bootstrap alerts based on results."""
    bootstrap_array = np.array(bootstrap_results)
    lower_bound = np.percentile(bootstrap_array, 2.5)
    upper_bound = np.percentile(bootstrap_array, 97.5)

    return {
        'value': original_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'is_alert': original_value < lower_bound or original_value > upper_bound
    }

def _check_warnings(
    data: np.ndarray,
    normalized_data: np.ndarray
) -> list[str]:
    """Check for potential warnings in the data."""
    warnings = []

    if np.any(np.isinf(normalized_data)):
        warnings.append("Normalization resulted in infinite values")

    if np.any(np.isnan(normalized_data)):
        warnings.append("Normalization resulted in NaN values")

    if data.shape[0] < 30:
        warnings.append("Small sample size may affect bootstrap reliability")

    return warnings

################################################################################
# badges
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None
) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if sample_weights is not None and not isinstance(sample_weights, np.ndarray):
        raise TypeError("sample_weights must be a numpy array or None")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")
    if sample_weights is not None:
        if len(sample_weights) != X.shape[0]:
            raise ValueError("sample_weights must have the same length as X")
        if np.any(sample_weights < 0):
            raise ValueError("sample_weights must be non-negative")

def _normalize_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = "standard",
    sample_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Normalize data according to specified method."""
    if normalization == "none":
        return {"X": X, "y": y}

    if sample_weights is None:
        sample_weights = np.ones(X.shape[0])

    if normalization == "standard":
        mean = np.average(X, axis=0, weights=sample_weights)
        std = np.sqrt(np.average((X - mean)**2, axis=0, weights=sample_weights))
        std[std == 0] = 1.0
        X_normalized = (X - mean) / std
    elif normalization == "minmax":
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        delta = max_vals - min_vals
        delta[delta == 0] = 1.0
        X_normalized = (X - min_vals) / delta
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr[iqr == 0] = 1.0
        X_normalized = (X - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    return {"X": X_normalized, "y": y}

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    sample_weights: Optional[np.ndarray] = None
) -> float:
    """Compute specified metric between true and predicted values."""
    if sample_weights is None:
        sample_weights = np.ones(len(y_true))

    residuals = y_true - y_pred

    if metric == "mse":
        return np.average(residuals**2, weights=sample_weights)
    elif metric == "mae":
        return np.average(np.abs(residuals), weights=sample_weights)
    elif metric == "r2":
        ss_res = np.sum(residuals**2 * sample_weights)
        ss_tot = np.sum(((y_true - np.average(y_true, weights=sample_weights))**2) * sample_weights)
        return 1 - (ss_res / ss_tot)
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.average(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), weights=sample_weights)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _bootstrap_resample(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_samples: int = 1000,
    sample_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Perform bootstrap resampling."""
    n_samples_orig = X.shape[0]
    indices = np.random.choice(n_samples_orig, size=(n_samples, n_samples_orig), replace=True)

    X_resampled = X[indices]
    y_resampled = y[indices] if y is not None else None

    return {"X": X_resampled, "y": y_resampled}

def badges_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_bootstraps: int = 1000,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    sample_weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Bootstrap Aggregating of Dependent Estimates (BADGES) method.

    Parameters:
    -----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values array of shape (n_samples,)
    n_bootstraps : int
        Number of bootstrap samples to generate
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust")
    metric : Union[str, Callable]
        Metric to compute ("mse", "mae", "r2", "logloss") or custom callable
    sample_weights : Optional[np.ndarray]
        Sample weights array of shape (n_samples,)
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, y, sample_weights)

    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data
    normalized_data = _normalize_data(X, y, normalization, sample_weights)
    X_norm = normalized_data["X"]
    y_norm = normalized_data["y"]

    # Initialize results storage
    bootstrap_metrics = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        # Bootstrap resample
        resampled_data = _bootstrap_resample(X_norm, y_norm, 1, sample_weights)
        X_boot = resampled_data["X"][0]
        y_boot = resampled_data["y"][0] if y_norm is not None else None

        # Here you would typically fit a model and get predictions
        # For this example, we'll just use the mean as a simple estimator
        y_pred = np.mean(X_boot, axis=0) if y_norm is None else np.mean(y_boot)

        # Compute metric
        if callable(metric):
            bootstrap_metrics[i] = metric(y_boot, y_pred)
        else:
            if y_norm is None:
                raise ValueError("Metric requires target values (y)")
            bootstrap_metrics[i] = _compute_metric(y_boot, y_pred, metric)

    # Calculate final statistics
    mean_metric = np.mean(bootstrap_metrics)
    std_metric = np.std(bootstrap_metrics)

    return {
        "result": {
            "mean_metric": mean_metric,
            "std_metric": std_metric,
            "bootstrap_metrics": bootstrap_metrics
        },
        "metrics": {
            "final_metric": mean_metric,
            "metric_std": std_metric
        },
        "params_used": {
            "n_bootstraps": n_bootstraps,
            "normalization": normalization,
            "metric": metric.__name__ if callable(metric) else metric
        },
        "warnings": []
    }

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = badges_fit(
    X=X,
    y=y,
    n_bootstraps=100,
    normalization="standard",
    metric="mse"
)
"""

################################################################################
# cards
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_size: int = 100,
    n_bootstraps: int = 1000
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if not isinstance(n_bootstraps, int) or n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be a positive integer")

def _normalize_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = "standard"
) -> tuple:
    """Normalize data according to specified method."""
    if normalization == "none":
        return X, y
    elif normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    if y is not None:
        return X_norm, y
    return X_norm

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)
    elif metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == "logloss":
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _bootstrap_samples(
    X: np.ndarray,
    sample_size: int
) -> np.ndarray:
    """Generate bootstrap samples from data."""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=(sample_size, n_bootstraps), replace=True)
    return X[indices]

def _fit_model(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    model_func: Callable = np.mean
) -> Any:
    """Fit model to data using specified function."""
    if y is not None:
        return model_func(X, y)
    return model_func(X)

def cards_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_size: int = 100,
    n_bootstraps: int = 1000,
    normalization: str = "standard",
    metric: Union[str, Callable] = "mse",
    model_func: Callable = np.mean,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Bootstrap method to estimate model performance with confidence intervals.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values of shape (n_samples,) or None
    sample_size : int
        Size of each bootstrap sample, default 100
    n_bootstraps : int
        Number of bootstrap iterations, default 1000
    normalization : str
        Normalization method ("none", "standard", "minmax", "robust"), default "standard"
    metric : Union[str, Callable]
        Performance metric to compute ("mse", "mae", "r2", "logloss"), default "mse"
    model_func : Callable
        Function to fit the model, default np.mean
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    if random_state is not None:
        np.random.seed(random_state)

    _validate_inputs(X, y, sample_size, n_bootstraps)
    X_norm, y_norm = _normalize_data(X, y, normalization)

    metrics = []
    for _ in range(n_bootstraps):
        X_sample = _bootstrap_samples(X_norm, sample_size)
        y_sample = y_norm[np.random.choice(len(y_norm), size=sample_size, replace=True)] if y is not None else None

        model = _fit_model(X_sample, y_sample, model_func)
        if y is not None:
            y_pred = model.predict(X_sample) if hasattr(model, 'predict') else model(X_sample)
            metrics.append(_compute_metric(y_sample, y_pred, metric))
        else:
            metrics.append(model)

    result = {
        "mean": np.mean(metrics),
        "std": np.std(metrics),
        "ci_lower": np.percentile(metrics, 2.5),
        "ci_upper": np.percentile(metrics, 97.5)
    }

    return {
        "result": result,
        "metrics": {"metric_used": metric},
        "params_used": {
            "sample_size": sample_size,
            "n_bootstraps": n_bootstraps,
            "normalization": normalization
        },
        "warnings": []
    }

# Example usage:
# result = cards_fit(X_train, y_train,
#                   sample_size=100,
#                   n_bootstraps=1000,
#                   normalization="standard",
#                   metric="mse")

################################################################################
# carousel
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def carousel_fit(
    data: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    solver: str = 'closed_form',
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for carousel bootstrap method.

    Parameters:
    -----------
    data : np.ndarray
        Input features array of shape (n_samples, n_features)
    target : np.ndarray
        Target values array of shape (n_samples,)
    metric : str or callable
        Metric to optimize. Can be 'mse', 'mae', 'r2' or custom callable
    normalizer : callable, optional
        Normalization function. Can be None, standard, minmax or custom
    solver : str
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton'
    max_iter : int
        Maximum number of iterations for iterative solvers
    tol : float
        Tolerance for convergence
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> target = np.random.rand(100)
    >>> result = carousel_fit(data, target, metric='mse', solver='gradient_descent')
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize data if normalizer provided
    normalized_data = _apply_normalization(data, normalizer)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(normalized_data, target)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            normalized_data, target,
            metric=metric,
            max_iter=max_iter,
            tol=tol
        )
    elif solver == 'newton':
        params = _solve_newton(
            normalized_data, target,
            metric=metric,
            max_iter=max_iter,
            tol=tol
        )
    else:
        raise ValueurError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(data, target, params, metric)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'solver': solver
        },
        'warnings': _check_warnings(data, target)
    }

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data and target."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have same number of samples")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values")

def _apply_normalization(data: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _solve_closed_form(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve using closed form solution."""
    # Add intercept term
    data_with_intercept = np.column_stack([np.ones(data.shape[0]), data])
    # Calculate parameters
    params = np.linalg.inv(data_with_intercept.T @ data_with_intercept) @ data_with_intercept.T @ target
    return params

def _solve_gradient_descent(
    data: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable],
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Solve using gradient descent."""
    # Initialize parameters
    n_features = data.shape[1]
    params = np.zeros(n_features + 1)  # +1 for intercept

    for _ in range(max_iter):
        # Calculate predictions
        predictions = data @ params[1:] + params[0]

        # Calculate gradient
        if isinstance(metric, str):
            if metric == 'mse':
                error = predictions - target
                gradient = np.zeros(n_features + 1)
                gradient[0] = 2 * np.mean(error)
                gradient[1:] = 2 * (data.T @ error) / data.shape[0]
            elif metric == 'mae':
                gradient = np.zeros(n_features + 1)
                gradient[0] = np.mean(np.sign(predictions - target))
                gradient[1:] = (data.T @ np.sign(predictions - target)) / data.shape[0]
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            # Custom metric
            gradient = metric.gradient(data, target, predictions)

        # Update parameters
        params -= tol * gradient

    return params

def _solve_newton(
    data: np.ndarray,
    target: np.ndarray,
    metric: Union[str, Callable],
    max_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """Solve using Newton's method."""
    # Initialize parameters
    n_features = data.shape[1]
    params = np.zeros(n_features + 1)  # +1 for intercept

    for _ in range(max_iter):
        # Calculate predictions
        predictions = data @ params[1:] + params[0]

        # Calculate gradient and hessian
        if isinstance(metric, str):
            if metric == 'mse':
                error = predictions - target
                gradient = np.zeros(n_features + 1)
                gradient[0] = 2 * np.mean(error)
                gradient[1:] = 2 * (data.T @ error) / data.shape[0]

                hessian = np.zeros((n_features + 1, n_features + 1))
                hessian[0, 0] = 2
                hessian[1:, 1:] = 2 * (data.T @ data) / data.shape[0]
            else:
                raise ValueError(f"Newton method not implemented for metric: {metric}")
        else:
            # Custom metric
            gradient, hessian = metric.gradient_hessian(data, target, predictions)

        # Update parameters
        params -= np.linalg.inv(hessian) @ gradient

    return params

def _calculate_metrics(
    data: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Calculate metrics for the model."""
    predictions = data @ params[1:] + params[0]
    metrics_dict = {}

    if isinstance(metric, str):
        if metric == 'mse':
            metrics_dict['mse'] = np.mean((predictions - target) ** 2)
        elif metric == 'mae':
            metrics_dict['mae'] = np.mean(np.abs(predictions - target))
        elif metric == 'r2':
            ss_res = np.sum((predictions - target) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            metrics_dict['r2'] = 1 - (ss_res / ss_tot)

    # Add custom metric if provided
    if isinstance(metric, Callable):
        metrics_dict['custom'] = metric(data, target, predictions)

    return metrics_dict

def _check_warnings(data: np.ndarray, target: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []

    if data.shape[1] > data.shape[0]:
        warnings.append("Warning: Number of features exceeds number of samples")

    if np.linalg.cond(data) > 1e6:
        warnings.append("Warning: Data matrix is ill-conditioned")

    return warnings

################################################################################
# tooltips
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_size: int = 1000,
    n_bootstraps: int = 1000
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")

def normalize_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    method: str = 'standard'
) -> Dict[str, np.ndarray]:
    """Normalize data using specified method."""
    normalized = {'X': X.copy()}
    if y is not None:
        normalized['y'] = y.copy()

    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        normalized['X'] = (X - mean) / std
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        normalized['X'] = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        normalized['X'] = (X - median) / (iqr + 1e-8)
    elif method != 'none':
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable]
) -> float:
    """Compute specified metric between true and predicted values."""
    if callable(metric):
        return metric(y_true, y_pred)

    if metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def bootstrap_samples(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_size: int = 1000,
    n_bootstraps: int = 1000
) -> Dict[str, np.ndarray]:
    """Generate bootstrap samples."""
    n_samples = X.shape[0]
    indices = np.random.randint(0, n_samples, size=(n_bootstraps, sample_size))

    samples = {'X': np.zeros((n_bootstraps, sample_size, X.shape[1]))}
    if y is not None:
        samples['y'] = np.zeros((n_bootstraps, sample_size))

    for i in range(n_bootstraps):
        samples['X'][i] = X[indices[i]]
        if y is not None:
            samples['y'][i] = y[indices[i]]

    return samples

def tooltips_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    sample_size: int = 1000,
    n_bootstraps: int = 1000,
    normalize_method: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form'
) -> Dict:
    """Main function to compute bootstrap tooltips."""
    # Validate inputs
    validate_inputs(X, y, sample_size, n_bootstraps)

    # Normalize data
    normalized = normalize_data(X, y, normalize_method)
    X_norm = normalized['X']
    y_norm = normalized.get('y', None)

    # Generate bootstrap samples
    samples = bootstrap_samples(X_norm, y_norm, sample_size, n_bootstraps)

    # Initialize results
    results = {'metrics': np.zeros(n_bootstraps)}
    params_used = {
        'sample_size': sample_size,
        'n_bootstraps': n_bootstraps,
        'normalize_method': normalize_method,
        'metric': metric if not callable(metric) else 'custom',
        'solver': solver
    }

    # Compute metrics for each bootstrap sample
    for i in range(n_bootstraps):
        X_sample = samples['X'][i]
        y_sample = samples['y'][i] if y_norm is not None else None

        # Here you would implement the actual model fitting based on solver
        # For now, we'll just compute a simple mean prediction as placeholder
        if y_sample is not None:
            y_pred = np.mean(y_sample)
            results['metrics'][i] = compute_metric(y_sample, y_pred, metric)

    # Prepare output
    output = {
        'result': results,
        'metrics': {
            'mean': np.mean(results['metrics']),
            'std': np.std(results['metrics']),
            'ci_95': np.percentile(results['metrics'], [2.5, 97.5])
        },
        'params_used': params_used,
        'warnings': []
    }

    return output

# Example usage:
"""
X = np.random.randn(100, 5)
y = np.random.randn(100)

result = tooltips_fit(
    X=X,
    y=y,
    sample_size=50,
    n_bootstraps=100,
    normalize_method='standard',
    metric='mse'
)
"""

################################################################################
# popovers
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def popovers_fit(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    metric: Union[str, Callable] = 'mean',
    normalization: str = 'none',
    random_state: Optional[int] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute bootstrap popovers for a given dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    n_bootstraps : int, optional
        Number of bootstrap samples to draw (default: 1000)
    metric : str or callable, optional
        Metric to compute for each bootstrap sample. Can be 'mean', 'median',
        or a custom callable function (default: 'mean')
    normalization : str, optional
        Normalization method. Can be 'none', 'standard', or 'minmax' (default: 'none')
    random_state : int, optional
        Random seed for reproducibility (default: None)
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': array of bootstrap estimates
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = popovers_fit(data, n_bootstraps=100, metric='median')
    """
    # Validate inputs
    _validate_inputs(data, n_bootstraps)

    # Set random state if provided
    rng = np.random.RandomState(random_state)

    # Normalize data if requested
    normalized_data, norm_params = _apply_normalization(data, normalization)

    # Compute bootstrap estimates
    bootstraps = _compute_bootstrap_samples(normalized_data, n_bootstraps, rng)

    # Compute metric for each bootstrap sample
    estimates = _compute_metrics(bootstraps, metric, custom_metric)

    # Calculate additional metrics
    metrics = _calculate_additional_metrics(estimates)

    return {
        'result': estimates,
        'metrics': metrics,
        'params_used': {
            'n_bootstraps': n_bootstraps,
            'metric': metric,
            'normalization': normalization
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, n_bootstraps: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply requested normalization to the data."""
    norm_params = {}
    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
        norm_params = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        norm_params = {'min': min_val, 'max': max_val}
    else:
        normalized_data = data.copy()
    return normalized_data, norm_params

def _compute_bootstrap_samples(
    data: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState
) -> list[np.ndarray]:
    """Compute bootstrap samples from the data."""
    n = data.shape[0]
    return [data[rng.choice(n, size=n, replace=True)] for _ in range(n_samples)]

def _compute_metrics(
    bootstraps: list[np.ndarray],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> np.ndarray:
    """Compute the requested metric for each bootstrap sample."""
    estimates = []
    for sample in bootstraps:
        if callable(metric):
            est = metric(sample)
        elif custom_metric is not None:
            est = custom_metric(sample)
        else:
            if metric == 'mean':
                est = np.mean(sample, axis=0)
            elif metric == 'median':
                est = np.median(sample, axis=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        estimates.append(est)
    return np.array(estimates)

def _calculate_additional_metrics(
    estimates: np.ndarray
) -> Dict[str, float]:
    """Calculate additional metrics from the bootstrap estimates."""
    return {
        'mean': np.mean(estimates, axis=0),
        'std': np.std(estimates, axis=0),
        'ci_lower': np.percentile(estimates, 2.5, axis=0),
        'ci_upper': np.percentile(estimates, 97.5, axis=0)
    }

################################################################################
# custom_styles
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: Optional[Union[str, Callable]] = None,
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    custom_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Validate inputs and parameters for custom_styles.

    Args:
        X: Input data array
        y: Target values (optional)
        normalize: Normalization method
        metric: Metric function or name
        distance: Distance function or name (optional)
        solver: Solver method
        regularization: Regularization type
        custom_func: Custom function (optional)

    Returns:
        Dictionary of validated parameters

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array or None")

    # Add more validation checks as needed

    return {
        "X": X,
        "y": y,
        "normalize": normalize,
        "metric": metric,
        "distance": distance,
        "solver": solver,
        "regularization": regularization,
        "custom_func": custom_func
    }

def _apply_normalization(
    data: np.ndarray,
    method: str = "standard"
) -> np.ndarray:
    """
    Apply normalization to data.

    Args:
        data: Input data
        method: Normalization method

    Returns:
        Normalized data
    """
    if method == "none":
        return data

    # Implement different normalization methods
    if method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)

    # Add other normalization methods

    raise ValueError(f"Unknown normalization method: {method}")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Union[str, Callable]
) -> float:
    """
    Compute specified metric between true and predicted values.

    Args:
        y_true: True values
        y_pred: Predicted values
        metric_func: Metric function or name

    Returns:
        Computed metric value
    """
    if callable(metric_func):
        return metric_func(y_true, y_pred)

    # Implement standard metrics
    if metric_func == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric_func == "mae":
        return np.mean(np.abs(y_true - y_pred))
    # Add other metrics

    raise ValueError(f"Unknown metric: {metric_func}")

def _solve_closed_form(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Solve using closed form solution.

    Args:
        X: Input data
        y: Target values

    Returns:
        Estimated parameters
    """
    # Implement closed form solution
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000
) -> np.ndarray:
    """
    Solve using gradient descent.

    Args:
        X: Input data
        y: Target values
        learning_rate: Learning rate
        max_iter: Maximum iterations

    Returns:
        Estimated parameters
    """
    # Implement gradient descent
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = 2/n_samples * X.T @ (X @ params - y)
        params -= learning_rate * gradient

    return params

def custom_styles_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: str = "standard",
    metric: Union[str, Callable] = "mse",
    distance: Optional[Union[str, Callable]] = None,
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    custom_func: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for custom styles bootstrap.

    Args:
        X: Input data array
        y: Target values (optional)
        normalize: Normalization method
        metric: Metric function or name
        distance: Distance function or name (optional)
        solver: Solver method
        regularization: Regularization type
        custom_func: Custom function (optional)
        **kwargs: Additional solver-specific parameters

    Returns:
        Dictionary containing results, metrics, and other information
    """
    # Validate inputs
    params = _validate_inputs(
        X, y, normalize, metric, distance, solver, regularization, custom_func
    )

    # Apply normalization
    X_norm = _apply_normalization(params["X"], params["normalize"])

    # Solve using specified method
    if solver == "closed_form":
        coeffs = _solve_closed_form(X_norm, params["y"])
    elif solver == "gradient_descent":
        coeffs = _solve_gradient_descent(X_norm, params["y"], **kwargs)
    # Add other solvers

    # Compute predictions
    y_pred = X_norm @ coeffs if params["y"] is not None else None

    # Compute metrics
    metrics = {}
    if params["y"] is not None:
        metrics["metric"] = _compute_metric(params["y"], y_pred, params["metric"])

    # Prepare results
    result = {
        "result": coeffs if params["y"] is not None else X_norm,
        "metrics": metrics,
        "params_used": {
            "normalize": params["normalize"],
            "metric": params["metric"],
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.random.rand(100, 5)
y = np.random.rand(100)

result = custom_styles_fit(
    X, y,
    normalize="standard",
    metric="mse",
    solver="gradient_descent",
    learning_rate=0.1,
    max_iter=500
)
"""

################################################################################
# javascript_plugins
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def javascript_plugins_fit(
    data: np.ndarray,
    target: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Main function for fitting bootstrap models with JavaScript plugins.

    Parameters:
    - data: Input features array of shape (n_samples, n_features)
    - target: Target values array of shape (n_samples,)
    - normalizer: Callable for data normalization
    - metric: Metric to optimize ('mse', 'mae', 'r2', 'logloss')
    - distance: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type (None, 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    - custom_metric: Custom metric function

    Returns:
    - Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(data, target)

    # Normalize data
    normalized_data = normalizer(data)
    normalized_target = normalizer(target.reshape(-1, 1)).flatten()

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = _get_metric(metric)

    # Choose distance
    distance_func = _get_distance(distance)

    # Solve model
    params, warnings = _solve_model(
        normalized_data,
        normalized_target,
        solver,
        distance_func,
        regularization,
        tol,
        max_iter
    )

    # Calculate metrics
    predictions = _predict(normalized_data, params)
    metrics = {
        'metric': metric_func(target, predictions),
        'mse': _mean_squared_error(target, predictions),
        'mae': _mean_absolute_error(target, predictions)
    }

    return {
        'result': predictions,
        'metrics': metrics,
        'params_used': params,
        'warnings': warnings
    }

def _validate_inputs(data: np.ndarray, target: np.ndarray) -> None:
    """Validate input data and target."""
    if not isinstance(data, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Data and target must have the same number of samples")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        raise ValueError("Target contains NaN or infinite values")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the selected metric function."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared,
        'logloss': _log_loss
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance(distance: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the selected distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.linalg.norm(x - y, ord=3)
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_model(
    data: np.ndarray,
    target: np.ndarray,
    solver: str,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple[Dict[str, Any], list]:
    """Solve the model using the selected solver."""
    warnings = []

    if solver == 'closed_form':
        params, _ = _solve_closed_form(data, target)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(data, target, distance_func, regularization, tol, max_iter)
    elif solver == 'newton':
        params = _solve_newton(data, target, distance_func, regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(data, target, distance_func, regularization, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return params, warnings

def _predict(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Make predictions using the model parameters."""
    return data @ params['coefficients'] + params['intercept']

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Log Loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def _solve_closed_form(data: np.ndarray, target: np.ndarray) -> tuple[Dict[str, Any], list]:
    """Solve using closed form solution."""
    X = np.c_[np.ones(data.shape[0]), data]
    theta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    return {'coefficients': theta[1:], 'intercept': theta[0]}, []

def _solve_gradient_descent(
    data: np.ndarray,
    target: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using gradient descent."""
    # Implementation of gradient descent
    pass

def _solve_newton(
    data: np.ndarray,
    target: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using Newton's method."""
    # Implementation of Newton's method
    pass

def _solve_coordinate_descent(
    data: np.ndarray,
    target: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Solve using coordinate descent."""
    # Implementation of coordinate descent
    pass
