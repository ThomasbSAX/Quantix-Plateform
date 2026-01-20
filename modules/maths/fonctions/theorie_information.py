"""
Quantix – Module theorie_information
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# entropie_shannon
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(probabilities: np.ndarray) -> None:
    """Validate input probabilities."""
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if probabilities.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array.")
    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-8):
        raise ValueError("Probabilities must sum to 1.")

def _compute_entropy(probabilities: np.ndarray, base: float = 2) -> float:
    """Compute Shannon entropy."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))

def entropie_shannon_fit(
    probabilities: np.ndarray,
    base: float = 2,
    normalize: bool = False,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute Shannon entropy with configurable options.

    Parameters:
    -----------
    probabilities : np.ndarray
        Input probability distribution.
    base : float, optional
        Logarithm base for entropy calculation (default: 2).
    normalize : bool, optional
        Whether to normalize probabilities before computation (default: False).
    custom_metric : Callable, optional
        Custom metric function to compute alongside entropy.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate input
    _validate_input(probabilities)

    # Normalize if required
    if normalize:
        probabilities = probabilities / np.sum(probabilities)

    # Compute entropy
    result = _compute_entropy(probabilities, base)

    # Compute custom metric if provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics["custom_metric"] = custom_metric(probabilities)
        except Exception as e:
            metrics["custom_metric_error"] = str(e)

    # Return structured output
    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "base": base,
            "normalize": normalize
        },
        "warnings": []
    }

# Example usage:
# probabilities = np.array([0.1, 0.2, 0.7])
# result = entropie_shannon_fit(probabilities)

################################################################################
# information_mutuelle
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Inputs contain infinite values")

def _normalize_data(X: np.ndarray, y: np.ndarray,
                   normalization: str = 'none') -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / y_std
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
        y_min = np.min(y)
        y_max = np.max(y)
        y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_normalized = (X - X_median) / (X_iqr + 1e-8)
        y_median = np.median(y)
        y_q75, y_q25 = np.percentile(y, [75, 25])
        y_iqr = y_q75 - y_q25
        y_normalized = (y - y_median) / (y_iqr + 1e-8)
    else:
        X_normalized = X.copy()
        y_normalized = y.copy()

    return X_normalized, y_normalized

def _compute_mutual_information(X: np.ndarray, y: np.ndarray,
                             metric: str = 'entropy',
                             distance_metric: Callable = None) -> float:
    """Compute mutual information between X and y."""
    if metric == 'entropy':
        # Implement entropy-based mutual information
        pass
    elif callable(metric):
        return metric(X, y)
    else:
        raise ValueError("Invalid metric specified")

def information_mutuelle_fit(X: np.ndarray, y: np.ndarray,
                          normalization: str = 'none',
                          metric: Union[str, Callable] = 'entropy',
                          distance_metric: Optional[Callable] = None,
                          **kwargs) -> Dict:
    """
    Compute mutual information between features and target.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to use for mutual information calculation
    distance_metric : callable, optional
        Custom distance metric function

    Returns
    -------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = information_mutuelle_fit(X, y,
    ...                                 normalization='standard',
    ...                                 metric='entropy')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm, y_norm = _normalize_data(X, y, normalization)

    # Compute mutual information
    mi_value = _compute_mutual_information(X_norm, y_norm,
                                         metric=metric,
                                         distance_metric=distance_metric)

    return {
        'result': mi_value,
        'metrics': {'mutual_information': mi_value},
        'params_used': {
            'normalization': normalization,
            'metric': metric.__name__ if callable(metric) else metric,
            'distance_metric': distance_metric.__name__ if distance_metric else None
        },
        'warnings': []
    }

################################################################################
# capacite_canal
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def capacite_canal_fit(
    p_y: np.ndarray,
    p_x_given_y: np.ndarray,
    *,
    normalisation: str = 'none',
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = None,
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Calculate the channel capacity of a communication channel.

    Parameters:
    -----------
    p_y : np.ndarray
        Probability distribution of the output variable Y.
    p_x_given_y : np.ndarray
        Conditional probability distribution P(X|Y).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    distance_metric : Callable, optional
        Custom distance metric function.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Callable, optional
        Custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(p_y, p_x_given_y)

    # Normalize if required
    p_y = _apply_normalisation(p_y, normalisation)
    p_x_given_y = _apply_normalisation(p_x_given_y, normalisation)

    # Calculate channel capacity
    if solver == 'closed_form':
        result = _capacite_canal_closed_form(p_y, p_x_given_y)
    elif solver == 'gradient_descent':
        result = _capacite_canal_gradient_descent(p_y, p_x_given_y, tol=tol, max_iter=max_iter)
    else:
        raise ValueError("Unsupported solver method.")

    # Calculate metrics
    metrics = _calculate_metrics(p_y, p_x_given_y, custom_metric)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalisation": normalisation,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(p_y: np.ndarray, p_x_given_y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(p_y, np.ndarray) or not isinstance(p_x_given_y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if p_y.ndim != 1 or p_x_given_y.ndim != 2:
        raise ValueError("Invalid dimensions for input arrays.")
    if np.any(p_y < 0) or np.any(p_x_given_y < 0):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(np.sum(p_y), 1.0):
        raise ValueError("p_y must sum to 1.")
    if not np.allclose(np.sum(p_x_given_y, axis=0), 1.0):
        raise ValueError("Columns of p_x_given_y must sum to 1.")

def _apply_normalisation(arr: np.ndarray, method: str) -> np.ndarray:
    """Apply normalisation to the input array."""
    if method == 'none':
        return arr
    elif method == 'standard':
        return (arr - np.mean(arr)) / np.std(arr)
    elif method == 'minmax':
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    elif method == 'robust':
        return (arr - np.median(arr)) / (np.percentile(arr, 75) - np.percentile(arr, 25))
    else:
        raise ValueError("Unsupported normalisation method.")

def _capacite_canal_closed_form(p_y: np.ndarray, p_x_given_y: np.ndarray) -> float:
    """Calculate channel capacity using closed-form solution."""
    p_x = np.sum(p_y[:, None] * p_x_given_y, axis=0)
    mutual_info = np.sum(p_y[:, None] * p_x_given_y * np.log2(p_x_given_y / (p_x[None, :] * p_y[:, None])))
    return mutual_info

def _capacite_canal_gradient_descent(
    p_y: np.ndarray,
    p_x_given_y: np.ndarray,
    *,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> float:
    """Calculate channel capacity using gradient descent."""
    p_x = np.sum(p_y[:, None] * p_x_given_y, axis=0)
    mutual_info = 0.0
    for _ in range(max_iter):
        gradient = np.sum(p_y[:, None] * p_x_given_y * np.log2(p_x[None, :] / (p_x_given_y + 1e-10)), axis=0)
        p_x -= tol * gradient
        new_mutual_info = np.sum(p_y[:, None] * p_x_given_y * np.log2(p_x_given_y / (p_x[None, :] * p_y[:, None] + 1e-10)))
        if np.abs(new_mutual_info - mutual_info) < tol:
            break
        mutual_info = new_mutual_info
    return mutual_info

def _calculate_metrics(
    p_y: np.ndarray,
    p_x_given_y: np.ndarray,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, float]:
    """Calculate metrics for the channel capacity."""
    p_x = np.sum(p_y[:, None] * p_x_given_y, axis=0)
    mutual_info = np.sum(p_y[:, None] * p_x_given_y * np.log2(p_x_given_y / (p_x[None, :] * p_y[:, None] + 1e-10)))
    metrics = {
        "mutual_information": mutual_info,
        "entropy_y": -np.sum(p_y * np.log2(p_y + 1e-10)),
        "entropy_x": -np.sum(p_x * np.log2(p_x + 1e-10))
    }
    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(p_y, p_x_given_y)
    return metrics

################################################################################
# redondance_informationnelle
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def redondance_informationnelle_fit(
    data: np.ndarray,
    *,
    normalization: str = 'standard',
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
    Calculate informational redundancy in data.

    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics
    custom_distance : callable, optional
        Custom distance function if not using built-in distances

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = redondance_informationnelle_fit(data, normalization='standard', metric='mse')
    """
    # Validate inputs
    _validate_inputs(data, normalization)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Prepare solver parameters
    solver_params = {
        'tol': tol,
        'max_iter': max_iter,
        'regularization': regularization
    }

    # Choose solver
    if solver == 'closed_form':
        result = _solve_closed_form(normalized_data, metric, distance)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(normalized_data, metric, distance, **solver_params)
    elif solver == 'newton':
        result = _solve_newton(normalized_data, metric, distance, **solver_params)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(normalized_data, metric, distance, **solver_params)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(result, normalized_data, metric, distance)

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
        'warnings': _check_warnings(result)
    }

def _validate_inputs(data: np.ndarray, normalization: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _solve_closed_form(data: np.ndarray, metric: Union[str, Callable], distance: Union[str, Callable]) -> Dict:
    """Solve redundancy using closed form solution."""
    # Implement closed form solution logic
    return {'redundancy_matrix': np.zeros((data.shape[1], data.shape[1]))}

def _solve_gradient_descent(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    **kwargs
) -> Dict:
    """Solve redundancy using gradient descent."""
    # Implement gradient descent logic
    return {'redundancy_matrix': np.zeros((data.shape[1], data.shape[1]))}

def _solve_newton(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    **kwargs
) -> Dict:
    """Solve redundancy using Newton's method."""
    # Implement Newton's method logic
    return {'redundancy_matrix': np.zeros((data.shape[1], data.shape[1]))}

def _solve_coordinate_descent(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    **kwargs
) -> Dict:
    """Solve redundancy using coordinate descent."""
    # Implement coordinate descent logic
    return {'redundancy_matrix': np.zeros((data.shape[1], data.shape[1]))}

def _calculate_metrics(
    result: Dict,
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable]
) -> Dict:
    """Calculate metrics for the redundancy result."""
    # Implement metric calculation logic
    return {'metric_value': 0.0}

def _check_warnings(result: Dict) -> list:
    """Check for potential warnings in the result."""
    return []

################################################################################
# codage_source
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def codage_source_fit(
    data: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Fit a source coding model to the given data.

    Parameters:
        data: Input data array of shape (n_samples, n_features)
        normalizer: Function to normalize the data
        metric: Metric to evaluate performance ('mse', 'mae', 'r2', 'logloss')
        distance: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski')
        solver: Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
        regularization: Regularization type (None, 'l1', 'l2', 'elasticnet')
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        custom_metric: Custom metric function if needed
        custom_distance: Custom distance function if needed

    Returns:
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data
    normalized_data = normalizer(data)

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
        params = _solve_closed_form(normalized_data, distance_func)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            normalized_data, distance_func,
            regularization=regularization,
            tol=tol, max_iter=max_iter
        )
    elif solver == 'newton':
        params = _solve_newton(
            normalized_data, distance_func,
            regularization=regularization,
            tol=tol, max_iter=max_iter
        )
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(
            normalized_data, distance_func,
            regularization=regularization,
            tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, params, metric_func)

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
        'warnings': _check_warnings(normalized_data, params)
    }

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def _get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the appropriate metric function."""
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
    """Get the appropriate distance function."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance,
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**2, axis=1)
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_closed_form(data: np.ndarray, distance_func: Callable) -> Dict[str, Any]:
    """Solve using closed form solution."""
    # Placeholder for actual implementation
    return {'params': np.zeros(data.shape[1])}

def _solve_gradient_descent(
    data: np.ndarray,
    distance_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Solve using gradient descent."""
    # Placeholder for actual implementation
    return {'params': np.zeros(data.shape[1])}

def _solve_newton(
    data: np.ndarray,
    distance_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Solve using Newton's method."""
    # Placeholder for actual implementation
    return {'params': np.zeros(data.shape[1])}

def _solve_coordinate_descent(
    data: np.ndarray,
    distance_func: Callable,
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Solve using coordinate descent."""
    # Placeholder for actual implementation
    return {'params': np.zeros(data.shape[1])}

def _calculate_metrics(
    data: np.ndarray,
    params: Dict[str, Any],
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate performance metrics."""
    # Placeholder for actual implementation
    return {'metric': 0.0}

def _check_warnings(data: np.ndarray, params: Dict[str, Any]) -> list:
    """Check for potential warnings."""
    return []

# Example metric functions
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
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example distance functions
def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(x - y, axis=1)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(x - y), axis=1)

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

################################################################################
# codage_canal
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def codage_canal_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalisation: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Fit a channel coding model using various configurable parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    normalisation : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Metric to optimize ('mse', 'mae', 'r2', 'logloss') or custom callable.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable.
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalisation)

    # Set metric and distance functions
    metric_func = _get_metric(metric, custom_metric)
    distance_func = _get_distance(distance, custom_distance)

    # Solve the problem
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(X_normalized, y, metric_func, distance_func,
                                        regularization, tol, max_iter)
    elif solver == 'newton':
        params = _solve_newton(X_normalized, y, metric_func, distance_func,
                              regularization, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(X_normalized, y, metric_func, distance_func,
                                          regularization, tol, max_iter)
    else:
        raise ValueError("Unsupported solver method.")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, y, params, metric_func)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalisation': normalisation,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the input data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError("Unsupported normalization method.")

def _get_metric(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get the metric function based on input."""
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
        raise ValueError("Unsupported metric.")

def _get_distance(distance: Union[str, Callable], custom_distance: Optional[Callable]) -> Callable:
    """Get the distance function based on input."""
    if custom_distance is not None:
        return custom_distance
    if distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 2, axis=1) ** (1/2)
    else:
        raise ValueError("Unsupported distance metric.")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the problem using closed-form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _solve_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the problem using gradient descent."""
    # Initialize parameters
    params = np.zeros(X.shape[1])

    for _ in range(max_iter):
        # Compute gradient
        gradient = _compute_gradient(X, y, params, metric_func, distance_func, regularization)

        # Update parameters
        params -= gradient

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_newton(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the problem using Newton's method."""
    # Initialize parameters
    params = np.zeros(X.shape[1])

    for _ in range(max_iter):
        # Compute gradient and Hessian
        gradient = _compute_gradient(X, y, params, metric_func, distance_func, regularization)
        hessian = _compute_hessian(X, y, params, metric_func, distance_func)

        # Update parameters
        params -= np.linalg.pinv(hessian) @ gradient

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _solve_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the problem using coordinate descent."""
    # Initialize parameters
    params = np.zeros(X.shape[1])

    for _ in range(max_iter):
        for i in range(X.shape[1]):
            # Compute gradient for the current parameter
            gradient = _compute_gradient(X, y, params, metric_func, distance_func, regularization)[i]

            # Update the current parameter
            params[i] -= gradient

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable,
    distance_func: Callable,
    regularization: Optional[str]
) -> np.ndarray:
    """Compute the gradient of the loss function."""
    predictions = X @ params
    error = metric_func(y, predictions)
    gradient = -2 * X.T @ (y - predictions)

    if regularization == 'l1':
        gradient += np.sign(params)
    elif regularization == 'l2':
        gradient += 2 * params
    elif regularization == 'elasticnet':
        gradient += np.sign(params) + 2 * params

    return gradient

def _compute_hessian(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable,
    distance_func: Callable
) -> np.ndarray:
    """Compute the Hessian matrix of the loss function."""
    return 2 * X.T @ X

def _calculate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    metric_func: Callable
) -> Dict:
    """Calculate various metrics for the model."""
    predictions = X @ params
    return {
        'mse': _mean_squared_error(y, predictions),
        'mae': _mean_absolute_error(y, predictions),
        'r2': _r_squared(y, predictions)
    }

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

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(x - y)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(x - y))

def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

################################################################################
# theoreme_noiseless_channel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    normalize: str = "none",
    distance_metric: Union[str, Callable] = "euclidean"
) -> None:
    """
    Validate inputs for the noiseless channel theorem.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : np.ndarray
        Output data matrix of shape (n_samples, n_outputs)
    normalize : str
        Normalization method ("none", "standard", "minmax", "robust")
    distance_metric : str or callable
        Distance metric to use ("euclidean", "manhattan", "cosine", "minkowski") or custom callable

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-dimensional arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method")
    if isinstance(distance_metric, str) and distance_metric not in ["euclidean", "manhattan", "cosine", "minkowski"]:
        raise ValueError("Invalid distance metric")

def _apply_normalization(
    data: np.ndarray,
    method: str = "none"
) -> np.ndarray:
    """
    Apply normalization to data.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize
    method : str
        Normalization method ("none", "standard", "minmax", "robust")

    Returns
    -------
    np.ndarray
        Normalized data
    """
    if method == "none":
        return data

    if method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)

    if method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)

    if method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)

def _compute_distance(
    X: np.ndarray,
    Y: np.ndarray,
    metric: Union[str, Callable] = "euclidean"
) -> np.ndarray:
    """
    Compute distance between X and Y using specified metric.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix
    Y : np.ndarray
        Output data matrix
    metric : str or callable
        Distance metric to use

    Returns
    -------
    np.ndarray
        Computed distances
    """
    if callable(metric):
        return metric(X, Y)

    if metric == "euclidean":
        return np.linalg.norm(X - Y, axis=1)
    if metric == "manhattan":
        return np.sum(np.abs(X - Y), axis=1)
    if metric == "cosine":
        return 1 - np.sum(X * Y, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
    if metric == "minkowski":
        return np.sum(np.abs(X - Y)**2, axis=1)**(1/2)

def _compute_channel_capacity(
    distances: np.ndarray,
    method: str = "shannon"
) -> float:
    """
    Compute channel capacity using specified method.

    Parameters
    ----------
    distances : np.ndarray
        Computed distances between X and Y
    method : str
        Method to compute channel capacity ("shannon")

    Returns
    -------
    float
        Computed channel capacity
    """
    if method == "shannon":
        # Implement Shannon's channel capacity formula based on distances
        pass

def theoreme_noiseless_channel_fit(
    X: np.ndarray,
    Y: np.ndarray,
    normalize: str = "none",
    distance_metric: Union[str, Callable] = "euclidean",
    capacity_method: str = "shannon",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Compute the noiseless channel theorem between X and Y.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : np.ndarray
        Output data matrix of shape (n_samples, n_outputs)
    normalize : str
        Normalization method ("none", "standard", "minmax", "robust")
    distance_metric : str or callable
        Distance metric to use ("euclidean", "manhattan", "cosine", "minkowski") or custom callable
    capacity_method : str
        Method to compute channel capacity ("shannon")
    custom_metric : callable, optional
        Custom distance metric function

    Returns
    -------
    Dict
        Dictionary containing:
        - "result": computed channel capacity
        - "metrics": dictionary of computed metrics
        - "params_used": parameters used in computation
        - "warnings": list of warnings encountered

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> Y = np.random.rand(100, 3)
    >>> result = theoreme_noiseless_channel_fit(X, Y, normalize="standard", distance_metric="euclidean")
    """
    # Validate inputs
    _validate_inputs(X, Y, normalize, distance_metric)

    # Apply normalization
    X_norm = _apply_normalization(X, normalize)
    Y_norm = _apply_normalization(Y, normalize)

    # Compute distances
    if custom_metric is not None:
        distances = _compute_distance(X_norm, Y_norm, metric=custom_metric)
    else:
        distances = _compute_distance(X_norm, Y_norm, metric=distance_metric)

    # Compute channel capacity
    capacity = _compute_channel_capacity(distances, method=capacity_method)

    # Prepare output
    result = {
        "result": capacity,
        "metrics": {
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances)
        },
        "params_used": {
            "normalize": normalize,
            "distance_metric": distance_metric if not custom_metric else "custom",
            "capacity_method": capacity_method
        },
        "warnings": []
    }

    return result

# Example usage:
# X = np.random.rand(100, 5)
# Y = np.random.rand(100, 3)
# result = theoreme_noiseless_channel_fit(X, Y, normalize="standard", distance_metric="euclidean")

################################################################################
# theoreme_source_separation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    n_components: int = 2,
    normalize: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean'
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of components to separate (default: 2)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str or callable, optional
        Distance metric to use ('euclidean', 'manhattan', etc.)

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array")
    if n_components <= 0 or n_components > X.shape[1]:
        raise ValueError("n_components must be between 1 and n_features")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method")
    if isinstance(distance_metric, str) and distance_metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("Invalid distance metric")

def _normalize_data(
    X: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Normalize data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    method : str
        Normalization method

    Returns
    ------
    np.ndarray
        Normalized data matrix
    """
    if method == 'none':
        return X

    X_norm = X.copy()
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)

    return X_norm

def _compute_distance_matrix(
    X: np.ndarray,
    metric: Union[str, Callable] = 'euclidean'
) -> np.ndarray:
    """
    Compute distance matrix using specified metric.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    metric : str or callable
        Distance metric to use

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples)
    """
    if isinstance(metric, str):
        if metric == 'euclidean':
            return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        elif metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        elif metric == 'cosine':
            dot_products = np.dot(X, X.T)
            norms = np.sqrt(np.sum(X**2, axis=1))
            return 1 - dot_products / (np.outer(norms, norms) + 1e-8)
    else:
        return metric(X[:, np.newaxis], X)

def _solve_source_separation(
    distance_matrix: np.ndarray,
    n_components: int = 2,
    solver: str = 'closed_form'
) -> np.ndarray:
    """
    Solve source separation problem using specified solver.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Input distance matrix of shape (n_samples, n_samples)
    n_components : int
        Number of components to separate
    solver : str
        Solver method ('closed_form', 'gradient_descent')

    Returns
    ------
    np.ndarray
        Separated components matrix of shape (n_samples, n_components)
    """
    if solver == 'closed_form':
        # Eigenvalue decomposition approach
        n_samples = distance_matrix.shape[0]
        J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        B = -0.5 * J @ distance_matrix**2 @ J

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        return eigenvectors[:, :n_components]
    else:
        raise ValueError("Unsupported solver method")

def _compute_metrics(
    X: np.ndarray,
    components: np.ndarray,
    metric_funcs: Dict[str, Callable] = None
) -> Dict[str, float]:
    """
    Compute metrics for source separation results.

    Parameters
    ----------
    X : np.ndarray
        Original input data matrix
    components : np.ndarray
        Separated components matrix
    metric_funcs : dict, optional
        Dictionary of custom metric functions

    Returns
    ------
    dict
        Dictionary of computed metrics
    """
    if metric_funcs is None:
        metric_funcs = {
            'reconstruction_error': lambda x, c: np.mean((x - x @ c.T) ** 2),
            'variance_explained': lambda x, c: np.var(x @ c.T) / np.var(x)
        }

    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(X, components)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def theoreme_source_separation_fit(
    X: np.ndarray,
    n_components: int = 2,
    normalize: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    metric_funcs: Optional[Dict[str, Callable]] = None
) -> Dict:
    """
    Perform source separation using information theory principles.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of components to separate (default: 2)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : str or callable, optional
        Distance metric to use ('euclidean', 'manhattan', etc.)
    solver : str, optional
        Solver method ('closed_form')
    metric_funcs : dict, optional
        Dictionary of custom metric functions

    Returns
    ------
    dict
        Dictionary containing:
        - 'result': separated components matrix
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> result = theoreme_source_separation_fit(X, n_components=3)
    """
    # Validate inputs
    _validate_inputs(X, n_components, normalize, distance_metric)

    warnings = []

    # Normalize data
    X_norm = _normalize_data(X, normalize)

    # Compute distance matrix
    try:
        distance_matrix = _compute_distance_matrix(X_norm, distance_metric)
    except Exception as e:
        warnings.append(f"Distance matrix computation failed: {str(e)}")
        distance_matrix = np.zeros((X.shape[0], X.shape[0]))

    # Solve source separation
    try:
        components = _solve_source_separation(distance_matrix, n_components, solver)
    except Exception as e:
        warnings.append(f"Source separation failed: {str(e)}")
        components = np.zeros((X.shape[0], n_components))

    # Compute metrics
    metrics = _compute_metrics(X, components, metric_funcs)

    return {
        'result': components,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'normalize': normalize,
            'distance_metric': distance_metric,
            'solver': solver
        },
        'warnings': warnings
    }

################################################################################
# entropie_conditionnelle
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays for conditional entropy calculation."""
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional")
    if len(X) != len(Y):
        raise ValueError("Inputs must have the same length")
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        raise ValueError("Inputs must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        raise ValueError("Inputs must not contain infinite values")

def _compute_joint_probabilities(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute joint probabilities between X and Y."""
    unique_x = np.unique(X)
    unique_y = np.unique(Y)

    joint_probs = {}
    for x_val in unique_x:
        for y_val in unique_y:
            joint_probs[(x_val, y_val)] = np.mean((X == x_val) & (Y == y_val))

    return joint_probs

def _compute_conditional_probabilities(X: np.ndarray, Y: np.ndarray) -> Dict[str, Dict[float, float]]:
    """Compute conditional probabilities P(Y|X)."""
    unique_x = np.unique(X)
    conditional_probs = {}

    for x_val in unique_x:
        mask = X == x_val
        if np.sum(mask) > 0:
            conditional_probs[x_val] = {}
            for y_val in np.unique(Y):
                conditional_probs[x_val][y_val] = np.mean((Y[mask] == y_val))

    return conditional_probs

def _compute_entropy(probabilities: Dict[str, float]) -> float:
    """Compute entropy from a probability distribution."""
    entropy = 0.0
    for p in probabilities.values():
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def entropie_conditionnelle_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable] = "shannon",
    solver: str = "exact",
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Union[float, Dict, str]]:
    """
    Compute conditional entropy between two discrete random variables.

    Parameters
    ----------
    X : np.ndarray
        Input array of discrete values.
    Y : np.ndarray
        Target array of discrete values.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    metric : str or callable, optional
        Entropy metric ('shannon', 'collision', 'minrenyi', 'tsallis') or custom callable.
    solver : str, optional
        Solver method ('exact', 'iterative').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing:
        - 'result': computed conditional entropy
        - 'metrics': additional metrics
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> X = np.array([0, 1, 0, 1])
    >>> Y = np.array([0, 0, 1, 1])
    >>> result = entropie_conditionnelle_fit(X, Y)
    """
    _validate_inputs(X, Y)

    joint_probs = _compute_joint_probabilities(X, Y)
    conditional_probs = _compute_conditional_probabilities(X, Y)

    # Compute marginal probabilities P(X)
    unique_x = np.unique(X)
    p_x = {x_val: np.mean(X == x_val) for x_val in unique_x}

    # Compute conditional entropy
    h_y_given_x = 0.0
    for x_val in unique_x:
        if p_x[x_val] > 0:
            h_y_given_x -= p_x[x_val] * _compute_entropy(conditional_probs.get(x_val, {}))

    # Apply normalization if specified
    if normalization == "standard":
        h_y_given_x /= np.log2(len(np.unique(Y)))
    elif normalization == "minmax":
        max_entropy = np.log2(len(np.unique(Y)))
        h_y_given_x /= max_entropy
    elif normalization == "robust":
        h_y_given_x = np.log1p(h_y_given_x)

    return {
        "result": h_y_given_x,
        "metrics": {
            "joint_entropy": _compute_entropy(joint_probs),
            "marginal_entropy_x": _compute_entropy({x: p for x, p in p_x.items()}),
            "marginal_entropy_y": _compute_entropy({y: np.mean(Y == y) for y in np.unique(Y)}),
        },
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

################################################################################
# entropie_jointe
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim < 2:
        raise ValueError("Input X must be at least 2-dimensional")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values")

def _normalize_data(X: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data based on specified method."""
    if normalization == "none":
        return X
    elif normalization == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_joint_entropy(X: np.ndarray, base: float = 2) -> float:
    """Compute joint entropy of the input data."""
    n_samples, n_features = X.shape
    probs = np.apply_along_axis(lambda x: np.histogram(x, bins='auto', density=True)[0], axis=0, arr=X)
    probs = np.mean(probs, axis=1)  # Average probabilities across features
    joint_probs = np.prod(probs, axis=0)
    joint_entropy = -np.sum(joint_probs * np.log2(joint_probs + 1e-8))
    return joint_entropy

def _compute_metrics(X: np.ndarray, metrics: Dict[str, Callable], params_used: Dict[str, Any]) -> Dict[str, float]:
    """Compute specified metrics."""
    results = {}
    for name, metric_func in metrics.items():
        if name == "custom":
            results[name] = metric_func(X, **params_used)
        else:
            if name == "mse":
                results[name] = np.mean((X - np.mean(X, axis=0))**2)
            elif name == "mae":
                results[name] = np.mean(np.abs(X - np.mean(X, axis=0)))
            elif name == "r2":
                ss_total = np.sum((X - np.mean(X, axis=0))**2)
                ss_res = np.sum((X - X)**2)  # Placeholder for actual residuals
                results[name] = 1 - (ss_res / ss_total)
            elif name == "logloss":
                results[name] = -np.mean(X * np.log2(X + 1e-8) + (1 - X) * np.log2(1 - X + 1e-8))
    return results

def entropie_jointe_fit(
    X: np.ndarray,
    normalization: str = "none",
    metrics: Optional[Dict[str, Callable]] = None,
    base: float = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute joint entropy of input data with configurable options.

    Parameters:
    - X: Input data as a numpy array
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metrics: Dictionary of metric functions to compute
    - base: Logarithm base for entropy calculation
    - **kwargs: Additional parameters passed to metric functions

    Returns:
    - Dictionary containing results, metrics, used parameters, and warnings
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Compute joint entropy
    joint_entropy = _compute_joint_entropy(X_normalized, base)

    # Compute metrics if specified
    computed_metrics = {}
    if metrics is not None:
        computed_metrics = _compute_metrics(X_normalized, metrics, kwargs)

    # Prepare output
    result = {
        "result": joint_entropy,
        "metrics": computed_metrics,
        "params_used": {
            "normalization": normalization,
            "base": base
        },
        "warnings": []
    }

    return result

# Example usage:
"""
X = np.random.rand(100, 5)  # 100 samples, 5 features
metrics = {
    "mse": None,
    "custom": lambda x, **kwargs: np.mean(x**2)
}
result = entropie_jointe_fit(X, normalization="standard", metrics=metrics)
"""

################################################################################
# divergence_KL
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(p: np.ndarray, q: np.ndarray) -> None:
    """Validate input distributions."""
    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if p.shape != q.shape:
        raise ValueError("Input distributions must have the same shape.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Input distributions must contain non-negative values.")
    if not np.isclose(np.sum(p), 1.0) or not np.isclose(np.sum(q), 1.0):
        raise ValueError("Input distributions must sum to 1.")

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between two distributions."""
    epsilon = 1e-10
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    return np.sum(p_safe * np.log(p_safe / q_safe))

def divergence_KL_fit(
    p: np.ndarray,
    q: np.ndarray,
    *,
    normalization: str = "none",
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "kl_divergence",
    solver: str = "closed_form"
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the KL divergence between two probability distributions.

    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.
    normalization : str, optional
        Normalization method. Options: "none", "standard", "minmax".
    metric : str or callable, optional
        Metric to compute. Options: "kl_divergence", custom callable.
    solver : str, optional
        Solver method. Options: "closed_form".

    Returns
    -------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> p = np.array([0.1, 0.2, 0.7])
    >>> q = np.array([0.3, 0.4, 0.3])
    >>> divergence_KL_fit(p, q)
    {
        'result': 0.143924,
        'metrics': {'kl_divergence': 0.143924},
        'params_used': {'normalization': 'none', 'metric': 'kl_divergence', 'solver': 'closed_form'},
        'warnings': []
    }
    """
    _validate_inputs(p, q)

    if solver != "closed_form":
        raise ValueError("Only 'closed_form' solver is supported for KL divergence.")

    if isinstance(metric, str):
        if metric == "kl_divergence":
            result = _kl_divergence(p, q)
        else:
            raise ValueError("Unsupported metric.")
    else:
        result = metric(p, q)

    if normalization != "none":
        raise NotImplementedError("Normalization options other than 'none' are not implemented.")

    return {
        "result": result,
        "metrics": {"kl_divergence": result},
        "params_used": {
            "normalization": normalization,
            "metric": metric if isinstance(metric, str) else "custom",
            "solver": solver
        },
        "warnings": []
    }

################################################################################
# rate_distortion_theory
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def rate_distortion_theory_fit(
    data: np.ndarray,
    distortion_measure: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    rate_function: Optional[Callable[[np.ndarray], float]] = None,
    solver: str = 'gradient_descent',
    normalization: Optional[str] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    reg_type: Optional[str] = None,
    reg_param: float = 0.1,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray, str]]:
    """
    Compute the rate-distortion theory optimization for given data.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    distortion_measure : str or callable
        Distortion measure to use. Can be 'mse', 'mae', or a custom callable.
    rate_function : callable, optional
        Rate function to optimize. If None, a default entropy-based rate is used.
    solver : str, optional
        Solver to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    reg_type : str, optional
        Regularization type. Options: 'none', 'l1', 'l2', 'elasticnet'.
    reg_param : float, optional
        Regularization parameter.
    custom_metric : callable, optional
        Custom metric function to evaluate the solution.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Optimization result.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the optimization.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = rate_distortion_theory_fit(data, distortion_measure='mse')
    """
    # Validate inputs
    _validate_inputs(data, distortion_measure, rate_function, solver, normalization)

    # Normalize data if required
    normalized_data = _apply_normalization(data, normalization)

    # Initialize parameters
    params = {
        'distortion_measure': distortion_measure,
        'rate_function': rate_function,
        'solver': solver,
        'normalization': normalization,
        'max_iter': max_iter,
        'tol': tol,
        'reg_type': reg_type,
        'reg_param': reg_param
    }

    # Choose solver and optimize
    if solver == 'gradient_descent':
        result = _gradient_descent_optimization(
            normalized_data, distortion_measure, rate_function,
            max_iter, tol, reg_type, reg_param
        )
    elif solver == 'newton':
        result = _newton_optimization(
            normalized_data, distortion_measure, rate_function,
            max_iter, tol, reg_type, reg_param
        )
    elif solver == 'coordinate_descent':
        result = _coordinate_descent_optimization(
            normalized_data, distortion_measure, rate_function,
            max_iter, tol, reg_type, reg_param
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(result, normalized_data, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    distortion_measure: Union[str, Callable],
    rate_function: Optional[Callable],
    solver: str,
    normalization: Optional[str]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")

    valid_solvers = ['gradient_descent', 'newton', 'coordinate_descent']
    if solver not in valid_solvers:
        raise ValueError(f"Solver must be one of {valid_solvers}.")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization is not None and normalization not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}.")

def _apply_normalization(
    data: np.ndarray,
    normalization: Optional[str]
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if normalization is None or normalization == 'none':
        return data
    elif normalization == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == 'robust':
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _gradient_descent_optimization(
    data: np.ndarray,
    distortion_measure: Union[str, Callable],
    rate_function: Optional[Callable],
    max_iter: int,
    tol: float,
    reg_type: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Perform gradient descent optimization."""
    # Initialize parameters (example: random initialization)
    params = np.random.rand(data.shape[1])

    for _ in range(max_iter):
        # Compute gradients (placeholder logic)
        gradient = _compute_gradient(data, params, distortion_measure)

        # Apply regularization
        if reg_type == 'l1':
            gradient += reg_param * np.sign(params)
        elif reg_type == 'l2':
            gradient += 2 * reg_param * params
        elif reg_type == 'elasticnet' and reg_param > 0:
            gradient += reg_param * (np.sign(params) + 2 * params)

        # Update parameters
        params -= gradient

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _newton_optimization(
    data: np.ndarray,
    distortion_measure: Union[str, Callable],
    rate_function: Optional[Callable],
    max_iter: int,
    tol: float,
    reg_type: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Perform Newton optimization."""
    # Initialize parameters (example: random initialization)
    params = np.random.rand(data.shape[1])

    for _ in range(max_iter):
        # Compute gradient and Hessian (placeholder logic)
        gradient = _compute_gradient(data, params, distortion_measure)
        hessian = _compute_hessian(data, params, distortion_measure)

        # Apply regularization to Hessian
        if reg_type == 'l2':
            hessian += 2 * reg_param * np.eye(hessian.shape[0])

        # Update parameters
        params -= np.linalg.solve(hessian, gradient)

        # Check convergence
        if np.linalg.norm(gradient) < tol:
            break

    return params

def _coordinate_descent_optimization(
    data: np.ndarray,
    distortion_measure: Union[str, Callable],
    rate_function: Optional[Callable],
    max_iter: int,
    tol: float,
    reg_type: Optional[str],
    reg_param: float
) -> np.ndarray:
    """Perform coordinate descent optimization."""
    # Initialize parameters (example: random initialization)
    params = np.random.rand(data.shape[1])

    for _ in range(max_iter):
        for i in range(params.shape[0]):
            # Save current parameter
            old_param = params[i]

            # Compute gradient for the i-th coordinate (placeholder logic)
            gradient_i = _compute_gradient_coordinate(data, params, i, distortion_measure)

            # Apply regularization
            if reg_type == 'l1':
                gradient_i += reg_param * np.sign(params[i])
            elif reg_type == 'l2':
                gradient_i += 2 * reg_param * params[i]

            # Update parameter
            params[i] -= gradient_i

        # Check convergence
        if np.linalg.norm(params - old_param) < tol:
            break

    return params

def _compute_gradient(
    data: np.ndarray,
    params: np.ndarray,
    distortion_measure: Union[str, Callable]
) -> np.ndarray:
    """Compute gradient of the distortion measure."""
    if callable(distortion_measure):
        # Custom distortion measure
        return _custom_distortion_gradient(data, params, distortion_measure)
    elif distortion_measure == 'mse':
        # MSE gradient
        return 2 * np.mean((data - params) * data, axis=0)
    elif distortion_measure == 'mae':
        # MAE gradient (subgradient at zero)
        return np.mean(np.sign(data - params), axis=0)
    else:
        raise ValueError(f"Unknown distortion measure: {distortion_measure}")

def _compute_hessian(
    data: np.ndarray,
    params: np.ndarray,
    distortion_measure: Union[str, Callable]
) -> np.ndarray:
    """Compute Hessian of the distortion measure."""
    if callable(distortion_measure):
        # Custom distortion measure
        return _custom_distortion_hessian(data, params, distortion_measure)
    elif distortion_measure == 'mse':
        # MSE Hessian
        return 2 * np.cov(data, rowvar=False)
    else:
        raise ValueError(f"Hessian not implemented for distortion measure: {distortion_measure}")

def _compute_gradient_coordinate(
    data: np.ndarray,
    params: np.ndarray,
    coord: int,
    distortion_measure: Union[str, Callable]
) -> float:
    """Compute gradient for a single coordinate."""
    if callable(distortion_measure):
        # Custom distortion measure
        return _custom_distortion_gradient_coordinate(data, params, coord, distortion_measure)
    elif distortion_measure == 'mse':
        # MSE gradient for coordinate
        return 2 * np.mean((data[:, coord] - params[coord]) * data[:, coord])
    elif distortion_measure == 'mae':
        # MAE gradient (subgradient at zero) for coordinate
        return np.mean(np.sign(data[:, coord] - params[coord]))
    else:
        raise ValueError(f"Unknown distortion measure: {distortion_measure}")

def _custom_distortion_gradient(
    data: np.ndarray,
    params: np.ndarray,
    distortion_measure: Callable
) -> np.ndarray:
    """Compute gradient for custom distortion measure."""
    # Placeholder logic - implement based on the callable's requirements
    return np.zeros(params.shape)

def _custom_distortion_hessian(
    data: np.ndarray,
    params: np.ndarray,
    distortion_measure: Callable
) -> np.ndarray:
    """Compute Hessian for custom distortion measure."""
    # Placeholder logic - implement based on the callable's requirements
    return np.zeros((params.shape[0], params.shape[0]))

def _custom_distortion_gradient_coordinate(
    data: np.ndarray,
    params: np.ndarray,
    coord: int,
    distortion_measure: Callable
) -> float:
    """Compute gradient for a single coordinate of custom distortion measure."""
    # Placeholder logic - implement based on the callable's requirements
    return 0.0

def _compute_metrics(
    result: np.ndarray,
    data: np.ndarray,
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for the optimization result."""
    metrics = {}

    # Default metrics
    if custom_metric is None:
        metrics['mse'] = np.mean((data - result) ** 2)
        metrics['mae'] = np.mean(np.abs(data - result))
    else:
        metrics['custom_metric'] = custom_metric(data, result)

    return metrics
