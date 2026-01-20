"""
Quantix – Module kernels
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# gaussian_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def _normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_distance_matrix(X: np.ndarray, distance_metric: Union[str, Callable]) -> np.ndarray:
    """Compute pairwise distance matrix."""
    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
        elif distance_metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        elif distance_metric == 'cosine':
            return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
        elif distance_metric == 'minkowski':
            return np.sum(np.abs(X[:, np.newaxis] - X) ** 3, axis=2) ** (1/3)
    else:
        return distance_metric(X[:, np.newaxis], X)

def _gaussian_kernel_matrix(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute Gaussian kernel matrix."""
    return np.exp(-distances ** 2 / (2 * bandwidth ** 2))

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_functions: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics."""
    metrics = {}
    for name, func in metric_functions.items():
        if name == 'mse':
            metrics[name] = np.mean((y_true - y_pred) ** 2)
        elif name == 'mae':
            metrics[name] = np.mean(np.abs(y_true - y_pred))
        elif name == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics[name] = 1 - (ss_res / ss_tot)
        else:
            metrics[name] = func(y_true, y_pred)
    return metrics

def gaussian_kernel_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    bandwidth: float = 1.0,
    distance_metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'none',
    metric_functions: Optional[Dict[str, Union[str, Callable]]] = None,
    custom_kernel: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Union[str, float]], str]]:
    """
    Compute Gaussian kernel and optionally fit a model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target values if fitting a model
    bandwidth : float, optional
        Bandwidth parameter for the Gaussian kernel
    distance_metric : Union[str, Callable], optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom function
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric_functions : Optional[Dict[str, Union[str, Callable]]], optional
        Dictionary of metrics to compute
    custom_kernel : Optional[Callable], optional
        Custom kernel function

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, Union[str, float]], str]]
        Dictionary containing results, metrics, parameters used, and warnings

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> Y = np.random.rand(100)
    >>> result = gaussian_kernel_fit(X, Y, bandwidth=0.5, distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Compute distance matrix
    distances = _compute_distance_matrix(X_normalized, distance_metric)

    # Compute kernel matrix
    if custom_kernel is not None:
        K = custom_kernel(distances)
    else:
        K = _gaussian_kernel_matrix(distances, bandwidth)

    # Compute metrics if Y is provided
    metrics = {}
    if Y is not None:
        if metric_functions is None:
            metric_functions = {'mse': 'mse', 'mae': 'mae', 'r2': 'r2'}
        metrics = _compute_metrics(Y, K @ np.linalg.pinv(K) @ Y, metric_functions)

    # Prepare output
    result = {
        'result': K,
        'metrics': metrics,
        'params_used': {
            'bandwidth': bandwidth,
            'distance_metric': distance_metric if isinstance(distance_metric, str) else 'custom',
            'normalization': normalization
        },
        'warnings': []
    }

    return result

################################################################################
# linear_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
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

def _normalize_data(X: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Normalize input data."""
    if normalization == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute the linear kernel matrix."""
    return X @ X.T

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_functions: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    for name, func in metric_functions.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def linear_kernel_fit(X: np.ndarray, y: Optional[np.ndarray] = None,
                      normalization: str = 'none',
                      metric_functions: Dict[str, Callable] = None,
                      custom_kernel: Optional[Callable] = None) -> Dict:
    """
    Compute the linear kernel between input samples.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : Optional[np.ndarray]
        Target values if available
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric_functions : Dict[str, Callable]
        Dictionary of metric functions to compute
    custom_kernel : Optional[Callable]
        Custom kernel function if not using linear kernel

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used and warnings

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = linear_kernel_fit(X, normalization='standard')
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Compute kernel matrix
    if custom_kernel is not None:
        K = custom_kernel(X_normalized)
    else:
        K = _compute_linear_kernel(X_normalized)

    # Prepare results
    result_dict = {
        'result': K,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'custom_kernel': custom_kernel is not None
        },
        'warnings': []
    }

    # Compute metrics if y is provided
    if y is not None:
        if metric_functions is None:
            metric_functions = {
                'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
                'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
            }
        result_dict['metrics'] = _compute_metrics(y, np.diag(K), metric_functions)

    return result_dict

################################################################################
# polynomial_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize input data."""
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

def _compute_polynomial_kernel(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 0.0
) -> np.ndarray:
    """Compute the polynomial kernel."""
    if Y is None:
        Y = X
    return (gamma * np.dot(X, Y.T) + coef0) ** degree

def polynomial_kernel_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 0.0,
    normalization: str = "none",
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the polynomial kernel between two sets of vectors.

    Parameters:
    -----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples, n_features). If None, Y = X.
    degree : int
        Degree of the polynomial kernel (default: 3)
    gamma : float
        Kernel coefficient for 'rbf', 'poly', 'sigmoid' (default: 1.0)
    coef0 : float
        Independent term in kernel function (default: 0.0)
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : Optional[Callable]
        Custom metric function. If None, no metric is computed.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the kernel matrix and other results.
    """
    _validate_inputs(X, Y)
    X_normalized = _normalize_data(X, normalization)
    if Y is not None:
        Y_normalized = _normalize_data(Y, normalization)
    else:
        Y_normalized = X_normalized

    kernel_matrix = _compute_polynomial_kernel(X_normalized, Y_normalized, degree, gamma, coef0)

    result = {
        "kernel_matrix": kernel_matrix,
        "params_used": {
            "degree": degree,
            "gamma": gamma,
            "coef0": coef0,
            "normalization": normalization
        },
        "warnings": []
    }

    if metric is not None:
        try:
            result["metrics"] = {"custom_metric": metric(X, Y)}
        except Exception as e:
            result["warnings"].append(f"Metric computation failed: {str(e)}")
    else:
        result["metrics"] = {}

    return result

# Example usage:
# kernel_result = polynomial_kernel_fit(X_train, Y_train, degree=2, gamma=0.5)

################################################################################
# laplacian_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data using specified method."""
    if method == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def compute_distance_matrix(X: np.ndarray, distance_metric: Union[str, Callable]) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if distance_metric == 'euclidean':
        for i in range(n_samples):
            dist_matrix[i, :] = np.sqrt(np.sum((X[i, :] - X) ** 2, axis=1))
    elif distance_metric == 'manhattan':
        for i in range(n_samples):
            dist_matrix[i, :] = np.sum(np.abs(X[i, :] - X), axis=1)
    elif distance_metric == 'cosine':
        for i in range(n_samples):
            dist_matrix[i, :] = 1 - np.dot(X[i, :], X.T) / (np.linalg.norm(X[i, :]) * np.linalg.norm(X, axis=1))
    elif callable(distance_metric):
        for i in range(n_samples):
            dist_matrix[i, :] = distance_metric(X[i, :], X)
    else:
        raise ValueError("Unsupported distance metric")

    return dist_matrix

def compute_laplacian_kernel(dist_matrix: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute Laplacian kernel from distance matrix."""
    return np.exp(-gamma * dist_matrix)

def laplacian_kernel_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize_method: str = 'none',
    distance_metric: Union[str, Callable] = 'euclidean',
    gamma: float = 1.0,
    solver: str = 'closed_form'
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Compute Laplacian kernel.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target values if available
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : Union[str, Callable]
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom callable
    gamma : float
        Kernel bandwidth parameter
    solver : str
        Solver method ('closed_form')

    Returns:
    --------
    Dict containing:
        - 'result': computed kernel matrix
        - 'metrics': dictionary of metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings
    """
    # Validate inputs
    validate_inputs(X, Y)

    # Normalize data
    X_normalized = normalize_data(X, normalize_method)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(X_normalized, distance_metric)

    # Compute Laplacian kernel
    kernel_matrix = compute_laplacian_kernel(dist_matrix, gamma)

    # Prepare output dictionary
    result_dict = {
        'result': kernel_matrix,
        'metrics': {},
        'params_used': {
            'normalize_method': normalize_method,
            'distance_metric': distance_metric if not callable(distance_metric) else 'custom',
            'gamma': gamma,
            'solver': solver
        },
        'warnings': []
    }

    return result_dict

# Example usage:
"""
X = np.random.rand(10, 5)
result = laplacian_kernel_fit(X, normalize_method='standard', distance_metric='euclidean')
"""

################################################################################
# sigmoid_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def _normalize_data(X: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Normalize data according to specified method."""
    if normalization == 'none':
        return X
    elif normalization == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_sigmoid_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                           alpha: float = 1.0) -> np.ndarray:
    """Compute the sigmoid kernel between X and Y."""
    if Y is None:
        Y = X
    XY = np.dot(X, Y.T)
    return np.tanh(alpha * XY)

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics between true and predicted values."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def sigmoid_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                      normalization: str = 'none',
                      alpha: float = 1.0,
                      metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """
    Compute the sigmoid kernel between X and Y.

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - Y: Optional target data matrix. If None, uses X.
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - alpha: Parameter for the sigmoid kernel
    - metric_funcs: Dictionary of metric functions to compute

    Returns:
    - Dictionary containing:
        * 'result': Computed kernel matrix
        * 'metrics': Computed metrics (if metric_funcs provided)
        * 'params_used': Parameters used in computation
        * 'warnings': Any warnings generated

    Example:
    >>> X = np.random.rand(10, 5)
    >>> result = sigmoid_kernel_fit(X, normalization='standard', alpha=0.5)
    """
    warnings = []

    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)
    if Y is not None:
        Y_norm = _normalize_data(Y, normalization)
    else:
        Y_norm = X_norm

    # Compute kernel
    kernel_matrix = _compute_sigmoid_kernel(X_norm, Y_norm, alpha)

    # Compute metrics if provided
    metrics = {}
    if metric_funcs is not None and Y is not None:
        try:
            metrics = _compute_metrics(Y, kernel_matrix.diagonal(), metric_funcs)
        except Exception as e:
            warnings.append(f"Error computing metrics: {str(e)}")

    return {
        'result': kernel_matrix,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'alpha': alpha
        },
        'warnings': warnings
    }

################################################################################
# cosine_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def _normalize_data(X: np.ndarray, normalization: str = 'none') -> np.ndarray:
    """Normalize input data."""
    if normalization == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalization == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalization == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _compute_cosine_kernel(X: np.ndarray) -> np.ndarray:
    """Compute cosine kernel matrix."""
    X_normalized = _normalize_data(X, 'standard')
    return np.dot(X_normalized, X_normalized.T)

def cosine_kernel_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalization: str = 'none',
    custom_kernel_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute cosine kernel and optionally fit a model.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target values if fitting a model
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_kernel_func : Optional[Callable]
        Custom kernel function to use instead of cosine

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    _validate_inputs(X, Y)

    # Compute kernel matrix
    if custom_kernel_func is not None:
        K = custom_kernel_func(X)
    else:
        K = _compute_cosine_kernel(X)

    # Prepare output
    result = {
        'kernel_matrix': K,
        'result': None,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'custom_kernel_func': custom_kernel_func is not None
        },
        'warnings': []
    }

    # Add warnings if needed
    if normalization != 'none' and custom_kernel_func is not None:
        result['warnings'].append("Normalization may affect custom kernel behavior")

    return result

# Example usage:
"""
X = np.random.rand(10, 5)
result = cosine_kernel_fit(X, normalization='standard')
print(result['kernel_matrix'].shape)  # Should print (10, 10)
"""

################################################################################
# rbf_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, Y: Optional[np.ndarray] = None,
                   normalization: str = 'none') -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        if Y is not None:
            return X_norm, (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
        return X_norm, None
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        if Y is not None:
            return X_norm, (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-8)
        return X_norm, None
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        if Y is not None:
            return X_norm, (Y - np.median(Y)) / (np.percentile(Y, 75) - np.percentile(Y, 25) + 1e-8)
        return X_norm, None
    else:
        if Y is not None:
            return X, Y
        return X, None

def _compute_rbf_kernel(X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute the RBF kernel matrix."""
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-gamma * sq_dists)

def _compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(Y_true, Y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def rbf_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                  gamma: float = 1.0,
                  normalization: str = 'none',
                  metric_funcs: Optional[Dict[str, Callable]] = None,
                  custom_kernel: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Compute the RBF kernel and evaluate metrics.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target values if available
    gamma : float, default=1.0
        Kernel coefficient for RBF kernel
    normalization : str, default='none'
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of metric functions to compute
    custom_kernel : Optional[Callable]
        Custom kernel function if not using RBF

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> Y = np.random.rand(10)
    >>> result = rbf_kernel_fit(X, Y, gamma=0.5, normalization='standard')
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Initialize warnings
    warnings = []

    # Normalize data
    X_norm, Y_norm = _normalize_data(X, Y, normalization)

    # Compute kernel
    if custom_kernel is not None:
        try:
            K = custom_kernel(X_norm, gamma)
        except Exception as e:
            warnings.append(f"Custom kernel failed: {str(e)}")
            K = _compute_rbf_kernel(X_norm, gamma)
    else:
        K = _compute_rbf_kernel(X_norm, gamma)

    # Prepare results
    result = {
        'kernel_matrix': K,
        'normalized_X': X_norm,
        'normalized_Y': Y_norm if Y is not None else None
    }

    # Compute metrics if targets are provided
    metrics = {}
    if Y is not None:
        if metric_funcs is None:
            # Default metrics
            from sklearn.metrics import mean_squared_error, r2_score

            def mae(y_true, y_pred):
                return np.mean(np.abs(y_true - y_pred))

            metric_funcs = {
                'mse': mean_squared_error,
                'mae': mae,
                'r2': r2_score
            }

        metrics = _compute_metrics(Y, Y_norm if Y is not None else np.zeros_like(X), metric_funcs)

    # Return structured output
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'gamma': gamma,
            'normalization': normalization
        },
        'warnings': warnings
    }

################################################################################
# exponential_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def exponential_kernel_fit(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: float = 1.0,
    metric: str = 'euclidean',
    normalization: str = 'none',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    reg_param: float = 1e-6,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the exponential kernel fit for given data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    Y : np.ndarray
        Target values array of shape (n_samples,) or (n_samples, n_outputs).
    bandwidth : float
        Bandwidth parameter for the exponential kernel.
    metric : str or callable
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine',
        'minkowski'. Or a custom callable.
    normalization : str
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent', 'newton',
        'coordinate_descent'.
    regularization : str or None
        Regularization type. Options: 'none', 'l1', 'l2', 'elasticnet'.
    reg_param : float
        Regularization parameter.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : callable or None
        Custom metric function if not using built-in metrics.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': fitted kernel parameters or predictions
        - 'metrics': computed metrics
        - 'params_used': parameters used in the fit
        - 'warnings': any warnings encountered

    Example
    -------
    >>> X = np.random.rand(10, 2)
    >>> Y = np.random.rand(10)
    >>> result = exponential_kernel_fit(X, Y)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if required
    X_normalized = _normalize_data(X, normalization)

    # Compute pairwise distances
    if custom_metric is not None:
        distances = _compute_custom_distance(X_normalized, metric=custom_metric)
    else:
        distances = _compute_distance_matrix(X_normalized, metric)

    # Compute kernel matrix
    K = _compute_exponential_kernel(distances, bandwidth)

    # Solve for kernel parameters
    if solver == 'closed_form':
        params = _solve_closed_form(K, Y)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(K, Y, reg_param, tol, max_iter)
    elif solver == 'newton':
        params = _solve_newton(K, Y, reg_param, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(K, Y, reg_param, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, regularization, reg_param)

    # Compute metrics
    metrics = _compute_metrics(X_normalized, Y, params, K)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'bandwidth': bandwidth,
            'metric': metric,
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization,
            'reg_param': reg_param
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("X and Y must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y.ndim not in (1, 2):
        raise ValueError("Y must be a 1D or 2D array")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data according to specified method."""
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

def _compute_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    """Compute pairwise distance matrix."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    elif metric == 'minkowski':
        return np.sum(np.abs(X[:, np.newaxis] - X) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_custom_distance(X: np.ndarray, metric: Callable) -> np.ndarray:
    """Compute pairwise distances using custom metric."""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = metric(X[i], X[j])
    return distances

def _compute_exponential_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute exponential kernel matrix."""
    return np.exp(-distances / (2 * bandwidth ** 2))

def _solve_closed_form(K: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Solve kernel parameters using closed form solution."""
    return np.linalg.solve(K + 1e-8 * np.eye(K.shape[0]), Y)

def _solve_gradient_descent(
    K: np.ndarray,
    Y: np.ndarray,
    reg_param: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve kernel parameters using gradient descent."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    for _ in range(max_iter):
        gradient = 2 * (np.dot(K, params) - Y)
        if np.linalg.norm(gradient) < tol:
            break
        params -= reg_param * gradient
    return params

def _solve_newton(
    K: np.ndarray,
    Y: np.ndarray,
    reg_param: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve kernel parameters using Newton's method."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    for _ in range(max_iter):
        gradient = 2 * (np.dot(K, params) - Y)
        hessian = 2 * K
        delta = np.linalg.solve(hessian + reg_param * np.eye(n_samples), gradient)
        params -= delta
        if np.linalg.norm(delta) < tol:
            break
    return params

def _solve_coordinate_descent(
    K: np.ndarray,
    Y: np.ndarray,
    reg_param: float,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve kernel parameters using coordinate descent."""
    n_samples = K.shape[0]
    params = np.zeros(n_samples)
    for _ in range(max_iter):
        old_params = params.copy()
        for i in range(n_samples):
            residual = Y - np.dot(K, params) + K[i] * params[i]
            params[i] = residual[i] / (K[i].dot(K[i]) + reg_param)
        if np.linalg.norm(params - old_params) < tol:
            break
    return params

def _apply_regularization(params: np.ndarray, method: str, reg_param: float) -> np.ndarray:
    """Apply regularization to parameters."""
    if method == 'l1':
        return np.sign(params) * np.maximum(np.abs(params) - reg_param, 0)
    elif method == 'l2':
        return params / (1 + reg_param * np.linalg.norm(params))
    elif method == 'elasticnet':
        l1 = np.sign(params) * np.maximum(np.abs(params) - reg_param, 0)
        l2 = params / (1 + reg_param * np.linalg.norm(params))
        return 0.5 * l1 + 0.5 * l2
    else:
        return params

def _compute_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    K: np.ndarray
) -> Dict[str, float]:
    """Compute various metrics for the kernel fit."""
    predictions = np.dot(K, params)
    mse = np.mean((predictions - Y) ** 2)
    mae = np.mean(np.abs(predictions - Y))
    r2 = 1 - mse / np.var(Y)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2)
    }

################################################################################
# anova_rbf_kernel
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

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

def _compute_anova_rbf_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                             sigma: float = 1.0, degree: int = 2) -> np.ndarray:
    """Compute the ANOVA RBF kernel."""
    if Y is None:
        Y = X
    n_samples = X.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            diff = X[i] - Y[j]
            rbf_part = np.exp(-np.linalg.norm(diff) ** 2 / (2 * sigma ** 2))
            anova_part = np.sum(diff ** degree)
            kernel_matrix[i, j] = rbf_part * anova_part
            kernel_matrix[j, i] = kernel_matrix[i, j]

    return kernel_matrix

def _compute_metrics(kernel_matrix: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for the kernel matrix."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            if name == 'r2':
                # Example R² calculation (adjust as needed)
                y_pred = np.mean(kernel_matrix, axis=1)
                y_true = np.mean(kernel_matrix)
                ss_res = np.sum((y_pred - y_true) ** 2)
                ss_tot = np.sum((kernel_matrix.mean(axis=1) - y_true) ** 2)
                metrics[name] = 1 - (ss_res / (ss_tot + 1e-8))
            else:
                metrics[name] = func(kernel_matrix)
        except Exception as e:
            metrics[f"{name}_error"] = str(e)

    return metrics

def anova_rbf_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                         sigma: float = 1.0, degree: int = 2,
                         normalize_method: str = 'standard',
                         metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict:
    """
    Fit the ANOVA RBF kernel.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target data matrix of shape (n_samples, n_features), defaults to X
    sigma : float
        Bandwidth parameter for the RBF kernel
    degree : int
        Degree for the ANOVA polynomial term
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of metric functions to compute

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = anova_rbf_kernel_fit(X, sigma=1.0, degree=2)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm = _normalize_data(X, normalize_method)
    Y_norm = None if Y is None else _normalize_data(Y, normalize_method)

    # Compute kernel
    kernel_matrix = _compute_anova_rbf_kernel(X_norm, Y_norm, sigma, degree)

    # Compute metrics
    default_metrics = {
        'mse': lambda x: np.mean((x - np.mean(x)) ** 2),
        'mae': lambda x: np.mean(np.abs(x - np.mean(x)))
    }
    if metric_funcs is None:
        metric_funcs = default_metrics
    else:
        metric_funcs.update(default_metrics)

    metrics = _compute_metrics(kernel_matrix, metric_funcs)

    # Prepare output
    result_dict = {
        'result': kernel_matrix,
        'metrics': metrics,
        'params_used': {
            'sigma': sigma,
            'degree': degree,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

    return result_dict

################################################################################
# chi2_kernel
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Input arrays must not contain infinite values")

def _normalize_data(X: np.ndarray, Y: Optional[np.ndarray] = None,
                   normalization: str = 'none') -> tuple:
    """Normalize input data."""
    if normalization == 'standard':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)
        if Y is not None:
            Y_mean = np.mean(Y, axis=0)
            Y_std = np.std(Y, axis=0)
            Y_norm = (Y - Y_mean) / (Y_std + 1e-8)
            return X_norm, Y_norm
        return X_norm, None
    elif normalization == 'minmax':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        if Y is not None:
            Y_min = np.min(Y, axis=0)
            Y_max = np.max(Y, axis=0)
            Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
            return X_norm, Y_norm
        return X_norm, None
    elif normalization == 'robust':
        X_median = np.median(X, axis=0)
        X_q75, X_q25 = np.percentile(X, [75, 25], axis=0)
        X_iqr = X_q75 - X_q25
        X_norm = (X - X_median) / (X_iqr + 1e-8)
        if Y is not None:
            Y_median = np.median(Y, axis=0)
            Y_q75, Y_q25 = np.percentile(Y, [75, 25], axis=0)
            Y_iqr = Y_q75 - Y_q25
            Y_norm = (Y - Y_median) / (Y_iqr + 1e-8)
            return X_norm, Y_norm
        return X_norm, None
    else:
        return X, Y

def _compute_chi2_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                         gamma: float = 1.0) -> np.ndarray:
    """Compute the Chi-squared kernel between X and Y."""
    if Y is None:
        Y = X

    # Compute the Chi-squared kernel
    XY_sum = np.dot(X, Y.T)
    X_norm = np.sum(X**2, axis=1, keepdims=True)
    Y_norm = np.sum(Y**2, axis=1, keepdims=True)

    kernel = XY_sum / (X_norm + Y_norm.T + 1e-8)
    return np.exp(-gamma * (1 - kernel))

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(y_true, y_pred)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def chi2_kernel_fit(X: np.ndarray, Y: Optional[np.ndarray] = None,
                   normalization: str = 'none',
                   gamma: float = 1.0,
                   metric_funcs: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """
    Compute the Chi-squared kernel between X and Y.

    Parameters:
    -----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples, n_outputs). If None, kernel is computed between X and itself.
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    gamma : float
        Kernel coefficient
    metric_funcs : Optional[Dict[str, Callable]]
        Dictionary of metric functions to compute

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the kernel matrix and metrics
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data
    X_norm, Y_norm = _normalize_data(X, Y, normalization)

    # Compute kernel
    kernel_matrix = _compute_chi2_kernel(X_norm, Y_norm, gamma)

    # Compute metrics if targets are provided
    metrics = {}
    if Y is not None and metric_funcs is not None:
        metrics = _compute_metrics(Y, kernel_matrix.diagonal(), metric_funcs)

    return {
        'result': kernel_matrix,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'gamma': gamma
        },
        'warnings': []
    }
