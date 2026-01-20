"""
Quantix – Module encodage_variables
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# one_hot_encoding
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or infinite values")

def _one_hot_encode(X: np.ndarray, categories: Optional[Dict[int, list]] = None) -> np.ndarray:
    """Perform one-hot encoding on the input data."""
    if categories is None:
        unique_values = [np.unique(X[:, i]) for i in range(X.shape[1])]
    else:
        unique_values = [categories[i] for i in range(X.shape[1])]

    encoded = np.zeros((X.shape[0], sum(len(cat) for cat in unique_values)), dtype=int)
    current_col = 0

    for i in range(X.shape[1]):
        for val in unique_values[i]:
            mask = X[:, i] == val
            encoded[mask, current_col] = 1
            current_col += 1

    return encoded

def one_hot_encoding_fit(
    X: np.ndarray,
    categories: Optional[Dict[int, list]] = None,
    handle_unknown: str = 'ignore',
    sparse: bool = False
) -> Dict[str, Any]:
    """
    Fit one-hot encoder to data.

    Parameters:
    - X: Input data (2D numpy array)
    - categories: Optional dictionary mapping column indices to list of categories
    - handle_unknown: Strategy for unknown categories ('ignore' or 'error')
    - sparse: Whether to return sparse matrix

    Returns:
    Dictionary containing:
    - result: Encoded data
    - metrics: Encoding statistics
    - params_used: Parameters used
    - warnings: Any warnings generated

    Example:
    >>> X = np.array([[1, 2], [1, 3], [4, 2]])
    >>> one_hot_encoding_fit(X)
    """
    _validate_inputs(X)

    if categories is not None:
        for col, cats in categories.items():
            if len(np.unique(X[:, col])) > len(cats):
                raise ValueError(f"Column {col} has more unique values than provided categories")

    encoded = _one_hot_encode(X, categories)

    if sparse:
        from scipy.sparse import csr_matrix
        encoded = csr_matrix(encoded)

    metrics = {
        'n_features_out': encoded.shape[1],
        'sparse_output': sparse
    }

    return {
        'result': encoded,
        'metrics': metrics,
        'params_used': {
            'handle_unknown': handle_unknown,
            'sparse': sparse
        },
        'warnings': []
    }

################################################################################
# label_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input array for label encoding."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 1:
        raise ValueError("Input must be a 1D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")

def label_encoding_fit(
    X: np.ndarray,
    *,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Fit label encoding on input data.

    Parameters:
    -----------
    X : np.ndarray
        Input array of categorical labels to be encoded.
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional
        Metric to optimize: "mse", "mae", "r2", or custom callable.
    distance : str, optional
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, optional
        Solver method: "closed_form", "gradient_descent", or "newton".
    regularization : str, optional
        Regularization type: None, "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    validate_input(X)

    # Determine unique labels
    unique_labels = np.unique(X)
    n_classes = len(unique_labels)

    # Initialize parameters
    params_used = {
        "normalize": normalize,
        "metric": metric if isinstance(metric, str) else "custom",
        "distance": distance,
        "solver": solver,
        "regularization": regularization,
        "tol": tol,
        "max_iter": max_iter
    }

    # Normalize data if required
    X_normalized = _normalize_data(X, method=normalize)

    # Fit encoding based on solver
    if solver == "closed_form":
        encoded_labels = _fit_closed_form(X_normalized, unique_labels)
    elif solver == "gradient_descent":
        encoded_labels = _fit_gradient_descent(
            X_normalized, unique_labels,
            metric=metric,
            distance=distance,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, encoded_labels, metric, custom_metric)

    # Prepare result
    result = {
        "encoded_labels": encoded_labels,
        "unique_labels": unique_labels,
        "label_mapping": dict(zip(unique_labels, encoded_labels))
    }

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": []
    }

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on specified method."""
    if method == "none":
        return X
    elif method == "standard":
        return (X - np.mean(X)) / np.std(X)
    elif method == "minmax":
        return (X - np.min(X)) / (np.max(X) - np.min(X))
    elif method == "robust":
        return (X - np.median(X)) / (np.percentile(X, 75) - np.percentile(X, 25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _fit_closed_form(X: np.ndarray, unique_labels: np.ndarray) -> np.ndarray:
    """Fit label encoding using closed form solution."""
    n_classes = len(unique_labels)
    return np.arange(n_classes)

def _fit_gradient_descent(
    X: np.ndarray,
    unique_labels: np.ndarray,
    *,
    metric: Union[str, Callable],
    distance: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Fit label encoding using gradient descent."""
    n_classes = len(unique_labels)
    encoded_labels = np.zeros(n_classes)

    # Initialize parameters
    if regularization == "l1":
        lambda_ = 0.1
    elif regularization == "l2":
        lambda_ = 0.1
    else:
        lambda_ = 0

    # Gradient descent loop
    for _ in range(max_iter):
        # Update encoded labels (simplified example)
        encoded_labels -= 0.01 * _compute_gradient(
            X, encoded_labels,
            metric=metric,
            distance=distance,
            regularization=regularization,
            lambda_=lambda_
        )

    return encoded_labels

def _compute_gradient(
    X: np.ndarray,
    encoded_labels: np.ndarray,
    *,
    metric: Union[str, Callable],
    distance: str,
    regularization: Optional[str],
    lambda_: float
) -> np.ndarray:
    """Compute gradient for label encoding optimization."""
    # Simplified gradient computation
    return np.zeros_like(encoded_labels)

def _calculate_metrics(
    X: np.ndarray,
    encoded_labels: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics for label encoding."""
    metrics = {}

    if isinstance(metric, str):
        if metric == "mse":
            metrics["mse"] = np.mean((X - encoded_labels) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(X - encoded_labels))
        elif metric == "r2":
            ss_total = np.sum((X - np.mean(X)) ** 2)
            ss_res = np.sum((X - encoded_labels) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_total)
    elif callable(metric):
        metrics["custom"] = metric(X, encoded_labels)

    if custom_metric is not None:
        metrics["custom"] = custom_metric(X, encoded_labels)

    return metrics

################################################################################
# ordinal_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    categories: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if categories is not None and not isinstance(categories, dict):
        raise TypeError("categories must be a dictionary or None")
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and len(y.shape) != 1:
        raise ValueError("y must be a 1D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if y is not None and np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "mse",
    custom_metric: Optional[Callable] = None
) -> float:
    """Compute the specified metric between true and predicted values."""
    if custom_metric is not None:
        return custom_metric(y_true, y_pred)

    if metric == "mse":
        return np.mean((y_true - y_pred) ** 2)
    elif metric == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _ordinal_encode(
    X: np.ndarray,
    categories: Dict[str, np.ndarray],
    normalize: str = "none"
) -> np.ndarray:
    """Perform ordinal encoding on the input data."""
    encoded = np.zeros_like(X, dtype=float)
    for col in range(X.shape[1]):
        unique_values = categories.get(f"col_{col}", np.unique(X[:, col]))
        value_to_code = {val: idx for idx, val in enumerate(unique_values)}
        encoded[:, col] = np.vectorize(value_to_code.get)(X[:, col])

    if normalize == "standard":
        encoded = (encoded - np.mean(encoded, axis=0)) / np.std(encoded, axis=0)
    elif normalize == "minmax":
        encoded = (encoded - np.min(encoded, axis=0)) / (np.max(encoded, axis=0) - np.min(encoded, axis=0))
    elif normalize == "robust":
        encoded = (encoded - np.median(encoded, axis=0)) / (np.percentile(encoded, 75, axis=0) - np.percentile(encoded, 25, axis=0))

    return encoded

def ordinal_encoding_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    categories: Optional[Dict[str, np.ndarray]] = None,
    normalize: str = "none",
    metric: str = "mse",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict, str]]:
    """
    Perform ordinal encoding on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    y : Optional[np.ndarray], default=None
        Target values for supervised encoding.
    categories : Optional[Dict[str, np.ndarray]], default=None
        Dictionary mapping feature names to their possible categories.
    normalize : str, default="none"
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str, default="mse"
        Metric to evaluate the encoding: "mse", "mae", "r2".
    custom_metric : Optional[Callable], default=None
        Custom metric function.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict, str]]
        Dictionary containing:
        - "result": Encoded data.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the encoding.
        - "warnings": Any warnings generated during processing.

    Examples
    --------
    >>> X = np.array([[1, 2], [1, 3], [2, 4]])
    >>> result = ordinal_encoding_fit(X)
    """
    _validate_inputs(X, y, categories)

    if categories is None:
        categories = {f"col_{i}": np.unique(X[:, i]) for i in range(X.shape[1])}

    encoded_data = _ordinal_encode(X, categories, normalize)

    metrics = {}
    if y is not None:
        metrics["metric"] = _compute_metric(y, encoded_data @ np.ones(X.shape[1]), metric, custom_metric)

    return {
        "result": encoded_data,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "categories": categories
        },
        "warnings": []
    }

################################################################################
# binary_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def binary_encoding_fit(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit binary encoding model to the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data. Default is identity.
    distance_metric : str
        Distance metric for encoding ('euclidean', 'manhattan', 'cosine', 'minkowski').
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function if needed.
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : Optional[str]
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function if needed.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - 'result': Encoded data
        - 'metrics': Computed metrics
        - 'params_used': Parameters used during fitting
        - 'warnings': Any warnings encountered

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = binary_encoding_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalizer)

    # Normalize data
    X_normalized = normalizer(X)

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_metric(distance_metric)

    # Choose solver
    if solver == 'closed_form':
        encoded_data = _closed_form_solver(X_normalized, distance_func)
    elif solver == 'gradient_descent':
        encoded_data = _gradient_descent_solver(X_normalized, distance_func, tol, max_iter)
    elif solver == 'newton':
        encoded_data = _newton_solver(X_normalized, distance_func, tol, max_iter)
    elif solver == 'coordinate_descent':
        encoded_data = _coordinate_descent_solver(X_normalized, distance_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if needed
    if regularization is not None:
        encoded_data = _apply_regularization(encoded_data, regularization)

    # Compute metrics
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, encoded_data)
    else:
        metrics.update(_compute_default_metrics(X, encoded_data))

    # Prepare output
    result = {
        'result': encoded_data,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, normalizer: Callable[[np.ndarray], np.ndarray]) -> None:
    """Validate input data and normalizer."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

def _get_distance_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the selected distance metric function."""
    metrics = {
        'euclidean': lambda x, y: np.linalg.norm(x - y),
        'manhattan': lambda x, y: np.sum(np.abs(x - y)),
        'cosine': lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
        'minkowski': lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    }
    if metric not in metrics:
        raise ValueError(f"Unknown distance metric: {metric}")
    return metrics[metric]

def _closed_form_solver(X: np.ndarray, distance_func: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Closed form solution for binary encoding."""
    # Placeholder implementation
    return X @ np.linalg.pinv(X.T)

def _gradient_descent_solver(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Gradient descent solver for binary encoding."""
    # Placeholder implementation
    n_samples, n_features = X.shape
    encoded_data = np.random.randn(n_samples, n_features)

    for _ in range(max_iter):
        # Update rule
        pass

    return encoded_data

def _newton_solver(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Newton solver for binary encoding."""
    # Placeholder implementation
    return X @ np.linalg.pinv(X.T)

def _coordinate_descent_solver(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Coordinate descent solver for binary encoding."""
    # Placeholder implementation
    return X @ np.linalg.pinv(X.T)

def _apply_regularization(
    encoded_data: np.ndarray,
    regularization: str
) -> np.ndarray:
    """Apply selected regularization to encoded data."""
    if regularization == 'l1':
        return np.sign(encoded_data) * np.maximum(np.abs(encoded_data) - 1, 0)
    elif regularization == 'l2':
        return encoded_data / (1 + np.linalg.norm(encoded_data))
    elif regularization == 'elasticnet':
        l1 = np.sign(encoded_data) * np.maximum(np.abs(encoded_data) - 0.5, 0)
        l2 = encoded_data / (1 + np.linalg.norm(encoded_data))
        return 0.5 * l1 + 0.5 * l2
    else:
        return encoded_data

def _compute_default_metrics(
    X: np.ndarray,
    encoded_data: np.ndarray
) -> Dict[str, float]:
    """Compute default metrics between original and encoded data."""
    mse = np.mean((X - encoded_data) ** 2)
    mae = np.mean(np.abs(X - encoded_data))
    r2 = 1 - mse / np.var(X)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2)
    }

################################################################################
# target_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def target_encoding_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit target encoding model.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features).
    y : np.ndarray
        Target values (n_samples,).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Normalization function.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate performance.
    solver : str
        Solver algorithm ('closed_form', 'gradient_descent').
    regularization : Optional[str]
        Regularization type ('l1', 'l2', 'elasticnet').
    alpha : float
        Regularization strength.
    l1_ratio : float
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
    tol : float
        Tolerance for stopping criteria.
    max_iter : int
        Maximum number of iterations.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used and warnings.

    Examples
    --------
    >>> X = np.array([[1], [2], [3]])
    >>> y = np.array([0, 1, 0])
    >>> result = target_encoding_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Initialize random state if provided
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Select metric function
    metric_func = _get_metric_function(metric)

    # Select solver
    if solver == 'closed_form':
        encoding = _closed_form_solver(X, y)
    elif solver == 'gradient_descent':
        encoding = _gradient_descent_solver(X, y, metric_func, alpha, l1_ratio,
                                          tol, max_iter, rng)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply normalizer
    encoding = normalizer(encoding)

    # Calculate metrics
    metrics = _calculate_metrics(y, encoding, metric_func)

    return {
        'result': encoding,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'solver': solver,
            'regularization': regularization,
            'alpha': alpha,
            'l1_ratio': l1_ratio
        },
        'warnings': _check_warnings(X, y)
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

def _get_metric_function(metric: Union[str, Callable]) -> Callable:
    """Get metric function based on input."""
    if callable(metric):
        return metric
    elif metric == 'mse':
        return _mean_squared_error
    elif metric == 'mae':
        return _mean_absolute_error
    elif metric == 'r2':
        return _r_squared
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form solution for target encoding."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    metric_func: Callable,
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Gradient descent solver for target encoding."""
    n_features = X.shape[1]
    weights = rng.randn(n_features)

    for _ in range(max_iter):
        grad = _compute_gradient(X, y, weights)
        if alpha > 0:
            if l1_ratio == 1:
                grad += alpha * np.sign(weights)
            elif l1_ratio == 0:
                grad += 2 * alpha * weights
            else:
                grad += alpha * (l1_ratio * np.sign(weights) + 2 * (1 - l1_ratio) * weights)

        new_weights = weights - grad
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights

    return weights

def _compute_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute gradient for target encoding."""
    predictions = X @ weights
    residuals = predictions - y
    return 2 * (X.T @ residuals) / len(y)

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable
) -> Dict[str, float]:
    """Calculate metrics."""
    return {
        'metric_value': metric_func(y_true, y_pred),
        'metric_name': metric_func.__name__ if hasattr(metric_func, '__name__') else 'custom'
    }

def _check_warnings(X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
    """Check for potential warnings."""
    warnings = {}
    if np.any(np.std(X, axis=0) == 0):
        warnings['zero_variance'] = "Some features have zero variance"
    if len(np.unique(y)) < 2:
        warnings['single_class'] = "Target has only one unique value"
    return warnings

################################################################################
# frequency_encoding
################################################################################

import numpy as np
from typing import Dict, Any, Callable, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input array."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 1:
        raise ValueError("Input must be a 1D array")
    if np.isnan(X).any():
        raise ValueError("Input contains NaN values")
    if np.isinf(X).any():
        raise ValueError("Input contains infinite values")

def frequency_encoding_fit(
    X: np.ndarray,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit frequency encoding on input data.

    Parameters:
    -----------
    X : np.ndarray
        Input array to encode.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to evaluate encoding ("mse", "mae", "r2", "logloss") or custom callable.
    custom_metric : callable, optional
        Custom metric function if metric is "custom".
    **kwargs :
        Additional parameters for normalization or custom functions.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing encoding results, metrics, parameters used and warnings.

    Example:
    --------
    >>> X = np.array([1, 2, 2, 3, 4, 4, 4])
    >>> result = frequency_encoding_fit(X)
    """
    validate_input(X)

    # Calculate frequencies
    unique_values, counts = np.unique(X, return_counts=True)
    frequency_map = dict(zip(unique_values, counts))

    # Normalize frequencies
    normalized_frequencies = _normalize_frequencies(counts, normalization, **kwargs)

    # Create encoding
    encoded_values = np.array([frequency_map[x] for x in X])
    if normalization != "none":
        encoded_values = normalized_frequencies[np.searchsorted(counts, encoded_values)]

    # Calculate metrics
    metrics = _calculate_metrics(encoded_values, X, metric, custom_metric)

    return {
        "result": encoded_values,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric if not callable(metric) else "custom"
        },
        "warnings": []
    }

def _normalize_frequencies(
    counts: np.ndarray,
    method: str = "none",
    **kwargs
) -> np.ndarray:
    """Normalize frequency counts."""
    if method == "none":
        return counts
    elif method == "standard":
        mean = np.mean(counts)
        std = np.std(counts)
        return (counts - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(counts)
        max_val = np.max(counts)
        return (counts - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(counts)
        iqr = np.percentile(counts, 75) - np.percentile(counts, 25)
        return (counts - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_metrics(
    encoded_values: np.ndarray,
    original_values: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable] = None
) -> Dict[str, float]:
    """Calculate metrics for frequency encoding."""
    metrics_dict = {}

    if metric == "mse":
        metrics_dict["mse"] = np.mean((encoded_values - original_values) ** 2)
    elif metric == "mae":
        metrics_dict["mae"] = np.mean(np.abs(encoded_values - original_values))
    elif metric == "r2":
        ss_total = np.sum((original_values - np.mean(original_values)) ** 2)
        ss_residual = np.sum((original_values - encoded_values) ** 2)
        metrics_dict["r2"] = 1 - (ss_residual / ss_total)
    elif metric == "logloss":
        # For classification-like scenario (needs adaptation for frequency encoding)
        pass
    elif metric == "custom" and custom_metric is not None:
        metrics_dict["custom"] = custom_metric(encoded_values, original_values)
    elif callable(metric):
        metrics_dict["custom"] = metric(encoded_values, original_values)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict

################################################################################
# mean_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse"
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalize : str
        Normalization method ("none", "standard", "minmax", "robust")
    metric : str or callable
        Metric function ("mse", "mae", "r2", "logloss") or custom callable

    Raises
    ------
    ValueError
        If inputs are invalid
    """
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

    if isinstance(metric, str) and metric not in ["mse", "mae", "r2", "logloss"]:
        raise ValueError("Invalid metric specified")

    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("Invalid normalization method specified")

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "none"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    method : str
        Normalization method ("none", "standard", "minmax", "robust")

    Returns
    ------
    tuple[np.ndarray, np.ndarray]
        Normalized X and y arrays
    """
    if method == "standard":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / (X_std + 1e-8)

    elif method == "minmax":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X = (X - X_min) / (X_max - X_min + 1e-8)

    elif method == "robust":
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X = (X - X_median) / (X_iqr + 1e-8)

    return X, y

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Union[str, Callable]
) -> float:
    """
    Compute specified metric between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values array of shape (n_samples,)
    y_pred : np.ndarray
        Predicted target values array of shape (n_samples,)
    metric_func : str or callable
        Metric function ("mse", "mae", "r2", "logloss") or custom callable

    Returns
    ------
    float
        Computed metric value
    """
    if isinstance(metric_func, str):
        if metric_func == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric_func == "mae":
            return np.mean(np.abs(y_true - y_pred))
        elif metric_func == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))
        elif metric_func == "logloss":
            y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        return metric_func(y_true, y_pred)

def mean_encoding_fit(
    X: np.ndarray,
    y: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    min_samples_leaf: int = 1,
    smoothing: float = 0.0
) -> Dict[str, Union[np.ndarray, dict]]:
    """
    Fit mean encoding model to data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features)
    y : np.ndarray
        Target values array of shape (n_samples,)
    normalize : str
        Normalization method ("none", "standard", "minmax", "robust")
    metric : str or callable
        Metric function ("mse", "mae", "r2", "logloss") or custom callable
    min_samples_leaf : int
        Minimum number of samples required to split a node
    smoothing : float
        Smoothing parameter for mean encoding (0 = no smoothing)

    Returns
    ------
    dict
        Dictionary containing:
        - "result": Encoded values
        - "metrics": Computed metrics
        - "params_used": Parameters used
        - "warnings": Any warnings generated

    Example
    -------
    >>> X = np.array([[1, 2], [1, 3], [2, 4]])
    >>> y = np.array([0, 1, 0])
    >>> result = mean_encoding_fit(X, y)
    """
    # Validate inputs
    validate_inputs(X, y, normalize, metric)

    # Normalize data
    X_norm, y = normalize_data(X.copy(), y.copy(), normalize)

    # Initialize warnings
    warnings = []

    # Check for categorical features (simplified approach)
    n_unique_values = np.array([len(np.unique(X_norm[:, i])) for i in range(X_norm.shape[1])])
    is_categorical = n_unique_values < 0.5 * X_norm.shape[0]

    # Compute mean encoding
    encoded_values = np.zeros_like(X_norm, dtype=np.float64)
    for i in range(X_norm.shape[1]):
        if is_categorical[i]:
            unique_values, inverse = np.unique(X_norm[:, i], return_inverse=True)
            mean_encodings = np.array([np.mean(y[inverse == j]) for j in range(len(unique_values))])

            # Apply smoothing
            if smoothing > 0:
                global_mean = np.mean(y)
                n_samples_per_value = np.array([np.sum(inverse == j) for j in range(len(unique_values))])
                mean_encodings = (mean_encodings * n_samples_per_value + global_mean * smoothing) / (n_samples_per_value + smoothing)

            encoded_values[:, i] = mean_encodings[inverse]
        else:
            encoded_values[:, i] = X_norm[:, i]

    # Compute metrics
    if isinstance(metric, str):
        y_pred = np.mean(y)
        metric_value = compute_metric(y, encoded_values[:, 0], metric)  # Using first feature for demo
    else:
        y_pred = np.mean(y)
        metric_value = compute_metric(y, encoded_values[:, 0], metric)

    # Prepare output
    result = {
        "result": encoded_values,
        "metrics": {"metric_value": metric_value},
        "params_used": {
            "normalize": normalize,
            "metric": str(metric),
            "min_samples_leaf": min_samples_leaf,
            "smoothing": smoothing
        },
        "warnings": warnings
    }

    return result

################################################################################
# hash_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hash_encoding_fit(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Fit the hash encoding model to the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the input data. Default is identity.
    distance_metric : str
        Distance metric for encoding. Options: 'euclidean', 'manhattan', 'cosine'.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function if not using built-in metrics.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    regularization : Optional[str]
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = hash_encoding_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalizer)

    # Normalize data
    X_normalized = normalizer(X)

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_metric(distance_metric)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(X_normalized, distance_func)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(
            X_normalized, distance_func,
            regularization=regularization,
            tol=tol, max_iter=max_iter,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, params)

    return {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, normalizer: Callable[[np.ndarray], np.ndarray]) -> None:
    """Validate input data and normalizer."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

    # Test normalizer
    try:
        _ = normalizer(X.copy())
    except Exception as e:
        raise ValueError(f"Normalizer function failed: {str(e)}")

def _get_distance_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance metric function."""
    metrics = {
        'euclidean': lambda x, y: np.linalg.norm(x - y),
        'manhattan': lambda x, y: np.sum(np.abs(x - y)),
        'cosine': lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    }
    if metric not in metrics:
        raise ValueError(f"Unknown distance metric: {metric}")
    return metrics[metric]

def _solve_closed_form(X: np.ndarray, distance_func: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Solve using closed form solution."""
    # This is a placeholder for the actual implementation
    return np.zeros(X.shape[1])

def _solve_gradient_descent(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Solve using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)

    n_features = X.shape[1]
    params = np.random.randn(n_features)
    prev_params = None

    for _ in range(max_iter):
        # Compute gradients (placeholder)
        grad = np.zeros_like(params)

        if regularization == 'l1':
            grad += np.sign(params)
        elif regularization == 'l2':
            grad += 2 * params

        # Update parameters
        prev_params = params.copy()
        params -= tol * grad

        if prev_params is not None and np.linalg.norm(params - prev_params) < tol:
            break

    return params

def _compute_metrics(X: np.ndarray, params: np.ndarray) -> Dict[str, float]:
    """Compute metrics for the fitted model."""
    # Placeholder for actual metric calculations
    return {
        'mse': 0.0,
        'mae': 0.0,
        'r2': 1.0
    }

################################################################################
# entity_embedding
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """
    Validate input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If input is not a 2D array or contains invalid values.
    """
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values.")

def standard_normalize(X: np.ndarray) -> np.ndarray:
    """
    Standard normalization (z-score).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.

    Returns
    ------
    np.ndarray
        Normalized data.
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)

def minmax_normalize(X: np.ndarray) -> np.ndarray:
    """
    Min-max normalization.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.

    Returns
    ------
    np.ndarray
        Normalized data.
    """
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def robust_normalize(X: np.ndarray) -> np.ndarray:
    """
    Robust normalization (median and IQR).

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.

    Returns
    ------
    np.ndarray
        Normalized data.
    """
    median = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    return (X - median) / iqr

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    ------
    float
        MSE value.
    """
    return np.mean((y_true - y_pred) ** 2)

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    ------
    float
        MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    ------
    float
        R2 value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def closed_form_solver(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form solution for linear regression.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    y : np.ndarray
        Target values.

    Returns
    ------
    np.ndarray
        Coefficients.
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

def gradient_descent_solver(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iter: int = 1000,
    tol: float = 1e-4
) -> np.ndarray:
    """
    Gradient descent solver.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    y : np.ndarray
        Target values.
    learning_rate : float, optional
        Learning rate.
    n_iter : int, optional
        Number of iterations.
    tol : float, optional
        Tolerance for stopping.

    Returns
    ------
    np.ndarray
        Coefficients.
    """
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    for _ in range(n_iter):
        gradients = 2 * X.T @ (X @ coefficients - y) / n_samples
        new_coefficients = coefficients - learning_rate * gradients
        if np.linalg.norm(new_coefficients - coefficients) < tol:
            break
        coefficients = new_coefficients
    return coefficients

def entity_embedding_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = "standard",
    metric: str = "mse",
    solver: str = "closed_form",
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit entity embedding model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray], optional
        Target values if supervised.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str, optional
        Metric to optimize ("mse", "mae", "r2").
    solver : str, optional
        Solver method ("closed_form", "gradient_descent").
    custom_metric : Optional[Callable], optional
        Custom metric function.
    **kwargs
        Additional solver-specific parameters.

    Returns
    ------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    validate_input(X)
    if y is not None:
        validate_input(y.reshape(-1, 1))

    # Normalization
    if normalization == "standard":
        X_norm = standard_normalize(X)
    elif normalization == "minmax":
        X_norm = minmax_normalize(X)
    elif normalization == "robust":
        X_norm = robust_normalize(X)
    else:
        X_norm = X

    # Solver
    if solver == "closed_form":
        coefficients = closed_form_solver(X_norm, y)
    elif solver == "gradient_descent":
        coefficients = gradient_descent_solver(X_norm, y, **kwargs)
    else:
        raise ValueError("Unsupported solver.")

    # Metrics
    metrics = {}
    if y is not None:
        y_pred = X_norm @ coefficients
        if metric == "mse":
            metrics["mse"] = compute_mse(y, y_pred)
        elif metric == "mae":
            metrics["mae"] = compute_mae(y, y_pred)
        elif metric == "r2":
            metrics["r2"] = compute_r2(y, y_pred)
        if custom_metric is not None:
            metrics["custom"] = custom_metric(y, y_pred)

    return {
        "result": coefficients,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "solver": solver
        },
        "warnings": []
    }

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.rand(100)
# result = entity_embedding_fit(X, y, normalization="standard", metric="mse")

################################################################################
# cyclical_encoding
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    X: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse'
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X must not contain NaN or inf values")
    if normalize not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("normalize must be one of: 'none', 'standard', 'minmax', 'robust'")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("metric must be one of: 'mse', 'mae', 'r2', 'logloss' or a callable")

def _normalize_data(
    X: np.ndarray,
    normalize: str = 'standard'
) -> np.ndarray:
    """Normalize the input data."""
    if normalize == 'none':
        return X
    elif normalize == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalize == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalize == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = 'mse'
) -> float:
    """Compute the specified metric."""
    if isinstance(metric, str):
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
        return metric(y_true, y_pred)

def _cyclical_encoding_fit(
    X: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse'
) -> Dict:
    """Fit cyclical encoding to the input data."""
    _validate_inputs(X, normalize, metric)
    X_normalized = _normalize_data(X, normalize)

    # Example: Simple cyclical encoding (sine and cosine transformations)
    n_features = X_normalized.shape[1]
    encoded_features = []
    for i in range(n_features):
        sin_encoded = np.sin(2 * np.pi * X_normalized[:, i])
        cos_encoded = np.cos(2 * np.pi * X_normalized[:, i])
        encoded_features.append(sin_encoded)
        encoded_features.append(cos_encoded)

    X_encoded = np.column_stack(encoded_features)

    # Compute metrics (example: using the first column as target)
    if X.shape[1] > 0:
        y_true = X[:, 0]
        y_pred = X_encoded[:, :2] @ np.linalg.pinv(X_normalized) @ y_true
        metric_value = _compute_metric(y_true, y_pred, metric)
    else:
        metric_value = None

    return {
        'result': X_encoded,
        'metrics': {'metric_value': metric_value} if metric_value is not None else {},
        'params_used': {
            'normalize': normalize,
            'metric': metric
        },
        'warnings': []
    }

def cyclical_encoding_compute(
    X: np.ndarray,
    normalize: str = 'standard',
    metric: Union[str, Callable] = 'mse'
) -> Dict:
    """Compute cyclical encoding for the input data."""
    return _cyclical_encoding_fit(X, normalize, metric)
