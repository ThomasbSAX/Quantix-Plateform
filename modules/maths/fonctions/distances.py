"""
Quantix – Module distances
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# euclidienne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for Euclidean distance computation.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples,) or None for unsupervised case

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if Y is not None and not isinstance(Y, np.ndarray):
        raise ValueError("Y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and Y.ndim != 1:
        raise ValueError("Y must be a 1D array or None")
    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if Y is not None and np.any(np.isnan(Y)):
        raise ValueError("Y contains NaN values")

def normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """
    Normalize data according to specified method.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')

    Returns
    ------
    np.ndarray
        Normalized array
    """
    if method == 'none':
        return X.copy()
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

def compute_euclidean_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Euclidean distance between samples.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples,) or None for unsupervised case

    Returns
    ------
    np.ndarray
        Array of Euclidean distances
    """
    if Y is None:
        # Unsupervised case - pairwise distances between samples
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))
    else:
        # Supervised case - distances to target
        diff = X - Y.reshape(-1, 1)
        return np.sqrt(np.sum(diff**2, axis=1))

def euclidienne_compute(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalization: str = 'none',
    distance_metric: Callable = compute_euclidean_distance,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute Euclidean distance with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples,) or None for unsupervised case
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    distance_metric : Callable
        Function to compute distances (default: Euclidean)
    **kwargs :
        Additional parameters for the distance metric function

    Returns
    ------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'result': computed distances
        - 'metrics': dictionary of metrics
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = euclidienne_compute(X)
    """
    # Validate inputs
    validate_inputs(X, Y)

    # Normalize data
    X_norm = normalize_data(X, normalization)
    if Y is not None:
        Y_norm = normalize_data(Y.reshape(-1, 1), normalization).flatten()
    else:
        Y_norm = None

    # Compute distances
    distances = distance_metric(X_norm, Y_norm)

    # Prepare output
    result_dict = {
        'result': distances,
        'metrics': {},
        'params_used': {
            'normalization': normalization,
            'distance_metric': distance_metric.__name__ if hasattr(distance_metric, '__name__') else 'custom'
        },
        'warnings': []
    }

    return result_dict

################################################################################
# manhattan
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    distance_func: Optional[Callable] = None
) -> None:
    """Validate input arrays and parameters."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-dimensional arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")
    if distance_func is None and metric not in ["euclidean", "manhattan", "cosine"]:
        raise ValueError("Invalid metric specified")

def _compute_manhattan_distance(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """Compute Manhattan distance between each pair of points in X and Y."""
    return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)

def _compute_distance(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    distance_func: Optional[Callable] = None
) -> np.ndarray:
    """Compute distance matrix between X and Y."""
    if distance_func is not None:
        return distance_func(X, Y)
    if metric == "euclidean":
        return np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2, axis=2))
    elif metric == "manhattan":
        return _compute_manhattan_distance(X, Y)
    elif metric == "cosine":
        dot_products = np.sum(X[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=2)
        norms_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
        norms_Y = np.linalg.norm(Y, axis=1)[np.newaxis, :]
        return 1 - dot_products / (norms_X * norms_Y)
    else:
        raise ValueError("Unsupported metric")

def manhattan_fit(
    X: np.ndarray,
    Y: np.ndarray,
    distance_metric: str = "manhattan",
    distance_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Compute Manhattan distance or other specified distance between two sets of points.

    Parameters:
    -----------
    X : np.ndarray
        First set of points, shape (n_samples_X, n_features)
    Y : np.ndarray
        Second set of points, shape (n_samples_Y, n_features)
    distance_metric : str
        Distance metric to use ("euclidean", "manhattan", "cosine")
    distance_func : callable, optional
        Custom distance function that takes two arrays and returns a distance matrix
    normalize : bool
        Whether to normalize the input data

    Returns:
    --------
    dict
        Dictionary containing:
        - "result": distance matrix between X and Y
        - "metrics": dictionary of computed metrics
        - "params_used": parameters used in the computation
        - "warnings": list of warnings encountered

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Y = np.array([[5, 6], [7, 8]])
    >>> result = manhattan_fit(X, Y)
    """
    warnings: list[str] = []

    # Validate inputs
    _validate_inputs(X, Y, distance_metric, distance_func)

    # Normalize if requested
    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
        warnings.append("Data was normalized")

    # Compute distance matrix
    distance_matrix = _compute_distance(X, Y, distance_metric, distance_func)

    # Compute some basic metrics
    metrics = {
        "mean_distance": np.mean(distance_matrix),
        "min_distance": np.min(distance_matrix),
        "max_distance": np.max(distance_matrix)
    }

    return {
        "result": distance_matrix,
        "metrics": metrics,
        "params_used": {
            "distance_metric": distance_metric,
            "normalize": normalize
        },
        "warnings": warnings
    }

################################################################################
# cosinus
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: str = "none",
) -> Dict[str, Any]:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y is not None and X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")
    if normalize not in ["none", "standard", "minmax", "robust"]:
        raise ValueError("normalize must be one of: none, standard, minmax, robust")
    return {"X": X, "y": y}

def _normalize_data(
    X: np.ndarray,
    normalize: str = "none",
) -> np.ndarray:
    """Normalize the input data."""
    if normalize == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X

def _cosine_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute the cosine distance between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1.0 - (dot_product / (norm_a * norm_b))

def cosinus_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: str = "none",
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = _cosine_distance,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Compute cosine similarity/distance between vectors.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray], default=None
        Target values. Not used in cosine distance computation.
    normalize : str, default="none"
        Normalization method: "none", "standard", "minmax", or "robust".
    distance_metric : Callable[[np.ndarray, np.ndarray], float], default=_cosine_distance
        Custom distance metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validated = _validate_inputs(X, y, normalize)
    X = validated["X"]
    y = validated.get("y")

    # Normalize data
    X_normalized = _normalize_data(X, normalize)

    # Compute cosine distances
    n_samples = X_normalized.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance_matrix[i, j] = distance_metric(X_normalized[i], X_normalized[j])

    # Prepare output
    result = {
        "distance_matrix": distance_matrix,
        "normalization_used": normalize,
        "metric_used": distance_metric.__name__ if hasattr(distance_metric, "__name__") else "custom",
    }

    return {
        "result": result,
        "metrics": {},
        "params_used": kwargs,
        "warnings": [],
    }

# Example usage:
# cosinus_fit(X=np.random.rand(10, 5), normalize="standard")

################################################################################
# hamming
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Hamming distance calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or inf values")

def hamming_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Hamming distance between two binary arrays."""
    return np.sum(x != y) / len(x)

def hamming_fit(
    x: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float] = hamming_distance,
    normalize: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute Hamming distance between two binary arrays with optional normalization.

    Parameters:
    -----------
    x : np.ndarray
        First input array (must be binary)
    y : np.ndarray
        Second input array (must be binary)
    distance_func : Callable, optional
        Custom distance function (default: hamming_distance)
    normalize : str, optional
        Normalization method ('none', 'standard', 'minmax') (default: None)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> x = np.array([1, 0, 1, 1])
    >>> y = np.array([1, 1, 0, 1])
    >>> result = hamming_fit(x, y)
    """
    # Validate inputs
    validate_inputs(x, y)

    # Check if arrays are binary
    if not np.all(np.isin(x, [0, 1])) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Input arrays must be binary (only 0 and 1 values)")

    # Calculate distance
    result = distance_func(x, y)

    # Apply normalization if specified
    if normalize == 'standard':
        result = (result - np.mean([0, 1])) / np.std([0, 1])
    elif normalize == 'minmax':
        result = (result - np.min([0, 1])) / (np.max([0, 1]) - np.min([0, 1]))

    # Prepare output
    output = {
        "result": result,
        "metrics": {"distance": result},
        "params_used": {
            "normalize": normalize,
            "distance_func": distance_func.__name__ if hasattr(distance_func, '__name__') else "custom"
        },
        "warnings": []
    }

    return output

################################################################################
# minkowski
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Inputs must have the same number of features.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Inputs must not contain NaN values.")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Inputs must not contain infinite values.")

def _normalize_data(x: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize input data based on the specified method."""
    if normalization == "none":
        return x, y
    elif normalization == "standard":
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
        x_normalized = (x - mean_x) / std_x
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)
        y_normalized = (y - mean_y) / std_y
        return x_normalized, y_normalized
    elif normalization == "minmax":
        min_x = np.min(x, axis=0)
        max_x = np.max(x, axis=0)
        x_normalized = (x - min_x) / (max_x - min_x)
        min_y = np.min(y, axis=0)
        max_y = np.max(y, axis=0)
        y_normalized = (y - min_y) / (max_y - min_y)
        return x_normalized, y_normalized
    elif normalization == "robust":
        median_x = np.median(x, axis=0)
        iqr_x = np.subtract(*np.percentile(x, [75, 25], axis=0))
        x_normalized = (x - median_x) / iqr_x
        median_y = np.median(y, axis=0)
        iqr_y = np.subtract(*np.percentile(y, [75, 25], axis=0))
        y_normalized = (y - median_y) / iqr_y
        return x_normalized, y_normalized
    else:
        raise ValueError("Invalid normalization method.")

def _compute_minkowski_distance(x: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """Compute Minkowski distance between two sets of points."""
    return np.sum(np.abs(x - y) ** p, axis=1) ** (1 / p)

def minkowski_fit(
    x: np.ndarray,
    y: np.ndarray,
    p: float = 2.0,
    normalization: str = "none",
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Compute Minkowski distance between two sets of points.

    Parameters:
    - x: Input array of shape (n_samples, n_features).
    - y: Input array of shape (n_samples, n_features).
    - p: Parameter for Minkowski distance.
    - normalization: Normalization method ("none", "standard", "minmax", "robust").
    - metric: Custom metric function (optional).

    Returns:
    - Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)
    x_norm, y_norm = _normalize_data(x, y, normalization)
    distances = _compute_minkowski_distance(x_norm, y_norm, p)

    metrics = {}
    if metric is not None:
        metrics["custom_metric"] = metric(x_norm, y_norm)

    result = {
        "result": distances,
        "metrics": metrics,
        "params_used": {
            "p": p,
            "normalization": normalization,
        },
        "warnings": [],
    }

    return result

# Example usage:
# x = np.array([[1, 2], [3, 4]])
# y = np.array([[5, 6], [7, 8]])
# result = minkowski_fit(x, y, p=2.0, normalization="standard")

################################################################################
# chebyshev
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _normalize_data(x: np.ndarray, y: np.ndarray, normalization: str) -> tuple:
    """Normalize input data based on the specified method."""
    if normalization == "none":
        return x, y
    elif normalization == "standard":
        x_mean = np.mean(x)
        x_std = np.std(x)
        y_mean = np.mean(y)
        y_std = np.std(y)
        x_normalized = (x - x_mean) / x_std
        y_normalized = (y - y_mean) / y_std
        return x_normalized, y_normalized
    elif normalization == "minmax":
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        x_normalized = (x - x_min) / (x_max - x_min)
        y_normalized = (y - y_min) / (y_max - y_min)
        return x_normalized, y_normalized
    elif normalization == "robust":
        x_median = np.median(x)
        x_q1 = np.percentile(x, 25)
        x_q3 = np.percentile(x, 75)
        y_median = np.median(y)
        y_q1 = np.percentile(y, 25)
        y_q3 = np.percentile(y, 75)
        x_normalized = (x - x_median) / (x_q3 - x_q1)
        y_normalized = (y - y_median) / (y_q3 - y_q1)
        return x_normalized, y_normalized
    else:
        raise ValueError("Invalid normalization method.")

def _compute_chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Chebyshev distance between two arrays."""
    return np.max(np.abs(x - y))

def chebyshev_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = "none",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Compute the Chebyshev distance between two arrays with optional normalization and custom metric.

    Parameters:
    - x: Input array 1.
    - y: Input array 2.
    - normalization: Normalization method ("none", "standard", "minmax", "robust").
    - custom_metric: Optional custom metric function.

    Returns:
    - A dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)
    x_norm, y_norm = _normalize_data(x, y, normalization)

    if custom_metric is not None:
        metric_value = custom_metric(x_norm, y_norm)
    else:
        metric_value = _compute_chebyshev_distance(x_norm, y_norm)

    result = {
        "result": metric_value,
        "metrics": {"chebyshev_distance": metric_value},
        "params_used": {
            "normalization": normalization,
            "custom_metric": custom_metric is not None,
        },
        "warnings": [],
    }

    return result

# Example usage:
# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6])
# result = chebyshev_fit(x, y, normalization="standard")

################################################################################
# jaccard
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> None:
    """Validate input arrays and metric function."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Input arrays must have the same number of samples")
    if metric is not None and not callable(metric):
        raise ValueError("Metric must be a callable function")

def _compute_jaccard(
    X: np.ndarray,
    Y: np.ndarray
) -> float:
    """Compute Jaccard distance between two binary arrays."""
    intersection = np.logical_and(X, Y).sum(axis=1)
    union = np.logical_or(X, Y).sum(axis=1)
    jaccard_distance = 1 - (intersection / union)
    return np.mean(jaccard_distance)

def jaccard_fit(
    X: np.ndarray,
    Y: np.ndarray,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute Jaccard distance between two binary arrays.

    Parameters:
    -----------
    X : np.ndarray
        First binary array of shape (n_samples, n_features)
    Y : np.ndarray
        Second binary array of shape (n_samples, n_features)
    metric : Optional[Callable]
        Custom metric function. If None, uses default Jaccard distance.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X, Y, metric)

    # Compute Jaccard distance
    if metric is None:
        result = _compute_jaccard(X, Y)
    else:
        result = metric(X, Y)

    # Prepare output
    output = {
        "result": result,
        "metrics": {"jaccard_distance": result},
        "params_used": {
            "metric": "default" if metric is None else "custom"
        },
        "warnings": []
    }

    return output

################################################################################
# canberra
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or Inf values")

def _canberra_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Canberra distance between two vectors."""
    return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))

def canberra_fit(
    x: np.ndarray,
    y: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float] = _canberra_distance,
    normalize: Optional[str] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute Canberra distance between two vectors with optional normalization.

    Parameters:
    -----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.
    distance_func : Callable[[np.ndarray, np.ndarray], float]
        Distance function to compute. Default is Canberra distance.
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    **kwargs
        Additional keyword arguments for normalization.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': computed distance
        - 'metrics': dictionary of metrics (empty for this function)
        - 'params_used': parameters used
        - 'warnings': any warnings generated

    Example:
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> canberra_fit(x, y)
    """
    warnings = []
    params_used = {
        'distance_func': distance_func.__name__,
        'normalize': normalize
    }

    _validate_inputs(x, y)

    if normalize == 'standard':
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalize == 'minmax':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalize == 'robust':
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
    elif normalize is not None:
        warnings.append(f"Unknown normalization method: {normalize}. Using none.")

    result = distance_func(x, y)

    return {
        'result': result,
        'metrics': {},
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# bhattacharyya
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """Validate input arrays and normalizer."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("Normalizer must be a callable or None.")

def _apply_normalization(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> tuple:
    """Apply normalization to input arrays."""
    if normalizer is not None:
        x = normalizer(x)
        y = normalizer(y)
    return x, y

def _compute_bhattacharyya_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute the Bhattacharyya distance between two distributions."""
    # Ensure inputs are probability distributions
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x /= np.sum(x)
    y /= np.sum(y)

    # Compute the Bhattacharyya coefficient
    bc = np.sum(np.sqrt(x * y))

    # Compute the Bhattacharyya distance
    bd = -np.log(bc)

    return bd

def bhattacharyya_compute(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the Bhattacharyya distance between two distributions.

    Parameters:
    -----------
    x : np.ndarray
        First input array representing a probability distribution.
    y : np.ndarray
        Second input array representing a probability distribution.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], default=None
        A callable that normalizes the input arrays. If None, no normalization is applied.

    Returns:
    --------
    dict
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(x, y, normalizer)

    # Apply normalization if specified
    x_norm, y_norm = _apply_normalization(x, y, normalizer)

    # Compute Bhattacharyya distance
    result = _compute_bhattacharyya_distance(x_norm, y_norm)

    # Prepare output dictionary
    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None
        },
        "warnings": []
    }

    return output

# Example usage:
# x = np.array([0.1, 0.2, 0.3, 0.4])
# y = np.array([0.4, 0.3, 0.2, 0.1])
# result = bhattacharyya_compute(x, y)

################################################################################
# wasserstein
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def wasserstein_fit(
    X: np.ndarray,
    Y: np.ndarray,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Wasserstein distance between two distributions.

    Parameters:
    -----------
    X : np.ndarray
        First distribution samples (n_samples, n_features)
    Y : np.ndarray
        Second distribution samples (m_samples, n_features)
    distance_metric : str or callable
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    normalization : str or None
        Normalization method ('none', 'standard', 'minmax', 'robust')
    regularization : str or None
        Regularization method ('none', 'l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_distance : callable or None
        Custom distance function if not using built-in metrics
    custom_metric : callable or None
        Custom metric function for evaluation

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.randn(100, 2)
    >>> Y = np.random.randn(100, 2) + 2
    >>> result = wasserstein_fit(X, Y)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if specified
    X_norm = _apply_normalization(X, normalization)
    Y_norm = _apply_normalization(Y, normalization)

    # Prepare distance matrix
    distance_matrix = _compute_distance_matrix(X_norm, Y_norm, distance_metric, custom_distance)

    # Solve for optimal transport plan
    if solver == 'closed_form':
        transport_plan = _solve_closed_form(distance_matrix, regularization)
    elif solver == 'gradient_descent':
        transport_plan = _solve_gradient_descent(distance_matrix, tol, max_iter, regularization)
    elif solver == 'newton':
        transport_plan = _solve_newton(distance_matrix, tol, max_iter, regularization)
    elif solver == 'coordinate_descent':
        transport_plan = _solve_coordinate_descent(distance_matrix, tol, max_iter, regularization)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute Wasserstein distance
    wasserstein_distance = _compute_wasserstein_distance(distance_matrix, transport_plan)

    # Compute metrics if custom metric is provided
    metrics = {}
    if custom_metric:
        metrics['custom'] = custom_metric(X, Y)

    # Prepare output
    result = {
        'result': wasserstein_distance,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric if not custom_distance else 'custom',
            'solver': solver,
            'normalization': normalization,
            'regularization': regularization
        },
        'warnings': _check_warnings(X, Y)
    }

    return result

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Number of features must match between X and Y")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

def _apply_normalization(X: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply specified normalization to data."""
    if method is None or method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_distance_matrix(X: np.ndarray, Y: np.ndarray,
                           metric: str, custom_metric: Optional[Callable]) -> np.ndarray:
    """Compute pairwise distance matrix between X and Y."""
    if custom_metric is not None:
        return np.array([[custom_metric(x, y) for y in Y] for x in X])

    n = X.shape[0]
    m = Y.shape[0]

    if metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2, axis=2))
    elif metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)
    elif metric == 'cosine':
        return 1 - np.sum(X[:, np.newaxis, :] * Y[np.newaxis, :, :], axis=2) / (
            np.linalg.norm(X[:, np.newaxis, :], axis=2) * np.linalg.norm(Y[np.newaxis, :, :], axis=2))
    elif metric == 'minkowski':
        return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :])**3, axis=2)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _solve_closed_form(distance_matrix: np.ndarray, regularization: Optional[str]) -> np.ndarray:
    """Solve for optimal transport plan using closed form solution."""
    n = distance_matrix.shape[0]
    m = distance_matrix.shape[1]

    # Simple uniform distribution for demonstration
    return np.ones((n, m)) / (n * m)

def _solve_gradient_descent(distance_matrix: np.ndarray,
                          tol: float, max_iter: int, regularization: Optional[str]) -> np.ndarray:
    """Solve for optimal transport plan using gradient descent."""
    n = distance_matrix.shape[0]
    m = distance_matrix.shape[1]

    # Initialize transport plan
    P = np.ones((n, m)) / (n * m)

    for _ in range(max_iter):
        # Compute gradient
        grad = distance_matrix - np.mean(distance_matrix, axis=1)[:, np.newaxis]

        # Update transport plan
        P -= 0.01 * grad

        # Project to feasible set (simple example)
        P = np.clip(P, 0, None)
        P /= np.sum(P)

    return P

def _solve_newton(distance_matrix: np.ndarray,
                 tol: float, max_iter: int, regularization: Optional[str]) -> np.ndarray:
    """Solve for optimal transport plan using Newton's method."""
    # Placeholder implementation
    return _solve_gradient_descent(distance_matrix, tol, max_iter, regularization)

def _solve_coordinate_descent(distance_matrix: np.ndarray,
                            tol: float, max_iter: int, regularization: Optional[str]) -> np.ndarray:
    """Solve for optimal transport plan using coordinate descent."""
    # Placeholder implementation
    return _solve_gradient_descent(distance_matrix, tol, max_iter, regularization)

def _compute_wasserstein_distance(distance_matrix: np.ndarray,
                                transport_plan: np.ndarray) -> float:
    """Compute Wasserstein distance from transport plan."""
    return np.sum(distance_matrix * transport_plan)

def _check_warnings(X: np.ndarray, Y: np.ndarray) -> list:
    """Check for potential issues and return warnings."""
    warnings = []

    if X.shape[0] < 10 or Y.shape[0] < 10:
        warnings.append("Small sample size may affect results")

    if np.any(np.std(X, axis=0) == 0) or np.any(np.std(Y, axis=0) == 0):
        warnings.append("Zero variance in some features")

    return warnings

################################################################################
# levenshtein
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def validate_inputs(str1: str, str2: str) -> None:
    """Validate input strings for Levenshtein distance calculation."""
    if not isinstance(str1, str) or not isinstance(str2, str):
        raise TypeError("Inputs must be strings")
    if len(str1) == 0 or len(str2) == 0:
        raise ValueError("Input strings cannot be empty")

def levenshtein_distance(str1: str, str2: str) -> int:
    """Compute the Levenshtein distance between two strings."""
    m = len(str1)
    n = len(str2)

    # Create a distance matrix
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    # Fill the distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # Deletion
                dp[i, j - 1] + 1,      # Insertion
                dp[i - 1, j - 1] + cost  # Substitution
            )

    return dp[m, n]

def levenshtein_fit(
    str1: str,
    str2: str,
    normalize: bool = False,
    metric: Optional[Callable[[int, int], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Levenshtein distance between two strings with optional normalization and custom metrics.

    Parameters:
    - str1: First input string
    - str2: Second input string
    - normalize: Whether to normalize the distance by the maximum string length (default: False)
    - metric: Optional custom metric function that takes (distance, max_length) and returns a float

    Returns:
    - Dictionary containing the result, metrics, parameters used, and any warnings
    """
    # Validate inputs
    validate_inputs(str1, str2)

    # Compute Levenshtein distance
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))

    # Apply normalization if requested
    normalized_distance = distance / max_len if normalize and max_len > 0 else distance

    # Compute custom metric if provided
    computed_metric = None
    if metric is not None:
        try:
            computed_metric = metric(distance, max_len)
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    # Prepare output
    result = {
        "result": normalized_distance if normalize else distance,
        "metrics": {"custom_metric": computed_metric} if metric is not None else {},
        "params_used": {
            "normalize": normalize,
            "metric_function": metric.__name__ if metric is not None else None
        },
        "warnings": []
    }

    return result

# Example usage:
"""
result = levenshtein_fit("kitten", "sitting")
print(result)
"""

################################################################################
# hausdorff
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays for Hausdorff distance computation."""
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Inputs must have the same number of features")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or infinite values")

def compute_distance_matrix(X: np.ndarray, Y: np.ndarray,
                           distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           normalize: Optional[str] = None) -> np.ndarray:
    """Compute distance matrix between two sets of points."""
    if normalize == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif normalize == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        Y = (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))
    elif normalize == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        Y = (Y - np.median(Y, axis=0)) / (np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0))

    n = X.shape[0]
    m = Y.shape[0]
    distance_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            distance_matrix[i, j] = distance_metric(X[i], Y[j])

    return distance_matrix

def hausdorff_distance(distance_matrix: np.ndarray) -> float:
    """Compute Hausdorff distance from a distance matrix."""
    h1 = np.max(np.min(distance_matrix, axis=1))
    h2 = np.max(np.min(distance_matrix, axis=0))
    return max(h1, h2)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(a - b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance between two points."""
    return np.sum(np.abs(a - b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cosine distance between two points."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hausdorff_fit(X: np.ndarray, Y: np.ndarray,
                  distance_metric: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
                  normalize: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute Hausdorff distance between two sets of points.

    Parameters:
    -----------
    X : np.ndarray
        First set of points (n_samples1, n_features)
    Y : np.ndarray
        Second set of points (n_samples2, n_features)
    distance_metric : Callable[[np.ndarray, np.ndarray], float]
        Distance metric function (default: euclidean_distance)
    normalize : Optional[str]
        Normalization method ('standard', 'minmax', 'robust') or None

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': Hausdorff distance value
        - 'metrics': Additional metrics if any
        - 'params_used': Parameters used in computation
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Y = np.array([[5, 6], [7, 8]])
    >>> result = hausdorff_fit(X, Y)
    """
    validate_inputs(X, Y)

    distance_matrix = compute_distance_matrix(X, Y, distance_metric, normalize)
    hd = hausdorff_distance(distance_matrix)

    return {
        'result': hd,
        'metrics': {},
        'params_used': {
            'distance_metric': distance_metric.__name__,
            'normalize': normalize
        },
        'warnings': []
    }

################################################################################
# mahalanobis
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Inputs must have the same number of features.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input y contains NaN or infinite values.")

def _standardize_data(x: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data based on the specified method."""
    if normalization == "standard":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(x, axis=0)
        max_val = np.max(x, axis=0)
        return (x - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(x, axis=0)
        iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
        return (x - median) / (iqr + 1e-8)
    else:
        return x

def _compute_covariance_matrix(x: np.ndarray) -> np.ndarray:
    """Compute the covariance matrix."""
    return np.cov(x, rowvar=False)

def _compute_mahalanobis_distance(
    x: np.ndarray,
    y: np.ndarray,
    inv_cov_matrix: np.ndarray
) -> np.ndarray:
    """Compute the Mahalanobis distance between two sets of points."""
    delta = x - y
    return np.sqrt(np.sum(delta @ inv_cov_matrix * delta, axis=1))

def mahalanobis_fit(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = "standard",
    distance_metric: str = "euclidean"
) -> Dict[str, Any]:
    """
    Compute the Mahalanobis distance between two sets of points.

    Parameters:
    -----------
    x : np.ndarray
        Input array of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target array of shape (m_samples, n_features). If None, y is set to x.
    normalization : str
        Normalization method: "none", "standard", "minmax", or "robust".
    distance_metric : str
        Distance metric: "euclidean" (default), "manhattan", or custom callable.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    if y is None:
        y = x

    _validate_inputs(x, y)

    # Normalize data
    x_norm = _standardize_data(x, normalization)
    y_norm = _standardize_data(y, normalization)

    # Compute covariance matrix and its inverse
    cov_matrix = _compute_covariance_matrix(x_norm)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distance
    distances = _compute_mahalanobis_distance(x_norm, y_norm, inv_cov_matrix)

    # Prepare output
    result = {
        "result": distances,
        "metrics": {"mean_distance": np.mean(distances)},
        "params_used": {
            "normalization": normalization,
            "distance_metric": distance_metric
        },
        "warnings": []
    }

    return result

# Example usage:
# mahalanobis_fit(np.random.rand(10, 5), normalization="standard")

################################################################################
# pearson
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def pearson_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    normalization: str = "standard",
    distance_metric: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute Pearson correlation-based distance between vectors.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    normalization : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    distance_metric : str or callable, optional
        Distance metric to use: 'euclidean', 'manhattan', 'cosine', or custom callable
    solver : str, optional
        Solver method: 'closed_form', 'gradient_descent', etc.
    regularization : str, optional
        Regularization type: None, 'l1', 'l2', or 'elasticnet'
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : callable, optional
        Custom metric function if needed
    weights : np.ndarray, optional
        Sample weights

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = pearson_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y, weights)

    # Normalize data if needed
    X_norm = _apply_normalization(X, normalization)
    y_norm = _apply_normalization(y.reshape(-1, 1), normalization).flatten()

    # Choose distance metric
    if callable(distance_metric):
        distance_func = distance_metric
    else:
        distance_func = _get_distance_function(distance_metric)

    # Solve for parameters
    if solver == "closed_form":
        params = _solve_closed_form(X_norm, y_norm)
    else:
        raise ValueError(f"Solver {solver} not implemented")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, X_norm, y_norm, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(X_norm, y_norm, params,
                                distance_func=distance_func,
                                custom_metric=custom_metric)

    # Prepare results
    result = {
        "result": params,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

def _validate_inputs(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if weights is not None:
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights must have same length as number of samples")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    """Apply specified normalization to data."""
    if method == "none":
        return data
    elif method == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == "robust":
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_distance_function(metric_name: str) -> Callable:
    """Return the specified distance function."""
    if metric_name == "euclidean":
        return lambda x, y: np.linalg.norm(x - y)
    elif metric_name == "manhattan":
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric_name == "cosine":
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown distance metric: {metric_name}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for parameters using closed-form solution."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def _apply_regularization(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                          method: str) -> np.ndarray:
    """Apply specified regularization to parameters."""
    if method == "l1":
        # Lasso regression
        pass  # Implementation would go here
    elif method == "l2":
        # Ridge regression
        pass  # Implementation would go here
    elif method == "elasticnet":
        # Elastic net regression
        pass  # Implementation would go here
    else:
        raise ValueError(f"Unknown regularization method: {method}")
    return params

def _calculate_metrics(X: np.ndarray, y: np.ndarray, params: np.ndarray,
                       distance_func: Callable, custom_metric: Optional[Callable]) -> Dict:
    """Calculate various metrics for the model."""
    predictions = X @ params
    residuals = y - predictions

    metrics = {
        "mse": np.mean(residuals**2),
        "mae": np.mean(np.abs(residuals)),
        "r2": 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
    }

    if custom_metric is not None:
        metrics["custom"] = custom_metric(y, predictions)

    return metrics

################################################################################
# spearman
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def spearman_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    normalization: str = 'none',
    handle_nan: str = 'raise',
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the Spearman distance between two arrays.

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    distance_metric : str, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    custom_distance : Callable, optional
        Custom distance function if not using built-in metrics.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax'.
    handle_nan : str, optional
        How to handle NaN values. Options: 'raise', 'omit'.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'result': computed Spearman distance
        - 'metrics': additional metrics if applicable
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> spearman_fit(x, y)
    """
    # Validate inputs
    _validate_inputs(x, y)

    # Handle NaN values
    x_clean, y_clean = _handle_nan(x, y, handle_nan)

    # Normalize data
    x_norm, y_norm = _normalize(x_clean, y_clean, normalization)

    # Compute ranks
    rank_x = _compute_ranks(x_norm)
    rank_y = _compute_ranks(y_norm)

    # Compute distance
    if custom_distance is not None:
        distance = custom_distance(rank_x, rank_y)
    else:
        distance = _compute_distance(rank_x, rank_y, distance_metric)

    # Prepare output
    result = {
        'result': distance,
        'metrics': {},
        'params_used': {
            'distance_metric': distance_metric,
            'normalization': normalization,
            'handle_nan': handle_nan
        },
        'warnings': []
    }

    return result

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Input arrays must not contain infinite values")

def _handle_nan(x: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Handle NaN values in input arrays."""
    if method == 'raise':
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError("Input arrays contain NaN values")
        return x, y
    elif method == 'omit':
        mask = ~(np.isnan(x) | np.isnan(y))
        return x[mask], y[mask]
    else:
        raise ValueError(f"Unknown NaN handling method: {method}")

def _normalize(x: np.ndarray, y: np.ndarray, method: str) -> tuple:
    """Normalize input arrays."""
    if method == 'none':
        return x, y
    elif method == 'standard':
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)
    elif method == 'minmax':
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return x_norm, y_norm

def _compute_ranks(x: np.ndarray) -> np.ndarray:
    """Compute ranks of input array."""
    sorted_indices = np.argsort(x)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(x))
    return ranks

def _compute_distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: str
) -> float:
    """Compute distance between two arrays."""
    if metric == 'euclidean':
        return np.sqrt(np.sum((x - y) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif metric == 'cosine':
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return 1 - (dot_product / (norm_x * norm_y))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
