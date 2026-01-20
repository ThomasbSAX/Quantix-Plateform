"""
Quantix – Module distance_distributions
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# euclidean_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def euclidean_distance_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    normalize: str = "none",
    metric: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute the Euclidean distance between two distributions with configurable options.

    Parameters
    ----------
    X : np.ndarray
        First input distribution.
    Y : np.ndarray
        Second input distribution.
    normalize : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to compute ("euclidean", "manhattan", "cosine", "minkowski").
    solver : str, optional
        Solver method ("closed_form", "gradient_descent", "newton").
    regularization : str, optional
        Regularization method ("none", "l1", "l2", "elasticnet").
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : callable, optional
        Custom metric function.
    custom_distance : callable, optional
        Custom distance function.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.array([1.0, 2.0, 3.0])
    >>> Y = np.array([4.0, 5.0, 6.0])
    >>> result = euclidean_distance_fit(X, Y)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if required
    X_norm, Y_norm = _apply_normalization(X, Y, normalize)

    # Choose distance metric
    if callable(metric):
        distance_func = metric
    elif custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_metric_function(metric)

    # Compute the Euclidean distance
    result = distance_func(X_norm, Y_norm)

    # Prepare output dictionary
    output = {
        "result": result,
        "metrics": {"distance": result},
        "params_used": {
            "normalize": normalize,
            "metric": metric if not callable(metric) else "custom",
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if X.shape != Y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        raise ValueError("Input arrays must not contain infinite values.")

def _apply_normalization(X: np.ndarray, Y: np.ndarray, method: str) -> tuple:
    """Apply normalization to input arrays."""
    if method == "standard":
        X = (X - np.mean(X)) / np.std(X)
        Y = (Y - np.mean(Y)) / np.std(Y)
    elif method == "minmax":
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    elif method == "robust":
        X = (X - np.median(X)) / (np.percentile(X, 75) - np.percentile(X, 25))
        Y = (Y - np.median(Y)) / (np.percentile(Y, 75) - np.percentile(Y, 25))
    return X, Y

def _get_metric_function(metric: str) -> Callable:
    """Get the appropriate metric function based on the input string."""
    metrics = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported metric: {metric}")
    return metrics[metric]

def _euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the Euclidean distance between two arrays."""
    return np.linalg.norm(X - Y)

def _manhattan_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the Manhattan distance between two arrays."""
    return np.sum(np.abs(X - Y))

def _cosine_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the cosine distance between two arrays."""
    return 1 - np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))

def _minkowski_distance(X: np.ndarray, Y: np.ndarray, p: float = 3) -> float:
    """Compute the Minkowski distance between two arrays."""
    return np.sum(np.abs(X - Y) ** p) ** (1 / p)

################################################################################
# manhattan_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    normalize_X: Optional[str] = None,
    normalize_Y: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate input arrays and normalization options.

    Parameters:
    -----------
    X : np.ndarray
        First input array.
    Y : np.ndarray
        Second input array.
    normalize_X : Optional[str]
        Normalization method for X ('none', 'standard', 'minmax', 'robust').
    normalize_Y : Optional[str]
        Normalization method for Y ('none', 'standard', 'minmax', 'robust').

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing validated inputs and normalization parameters.
    """
    if X.shape != Y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.isnan(X).any() or np.isnan(Y).any():
        raise ValueError("Input arrays must not contain NaN values.")
    if np.isinf(X).any() or np.isinf(Y).any():
        raise ValueError("Input arrays must not contain infinite values.")

    normalized_X, norm_params_X = _apply_normalization(X, normalize_X)
    normalized_Y, norm_params_Y = _apply_normalization(Y, normalize_Y)

    return {
        'X': normalized_X,
        'Y': normalized_Y,
        'norm_params_X': norm_params_X,
        'norm_params_Y': norm_params_Y
    }

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str] = None
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply normalization to the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data to normalize.
    method : Optional[str]
        Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns:
    --------
    tuple[np.ndarray, Dict[str, Any]]
        Normalized data and normalization parameters.
    """
    if method is None or method == 'none':
        return data, {}

    if method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        normalized_data = (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_data, {
        'method': method,
        'params': {'mean': mean, 'std': std} if method == 'standard' else
                  {'min': min_val, 'max': max_val} if method == 'minmax' else
                  {'median': median, 'iqr': iqr}
    }

def _compute_manhattan_distance(
    X: np.ndarray,
    Y: np.ndarray
) -> float:
    """
    Compute the Manhattan distance between two arrays.

    Parameters:
    -----------
    X : np.ndarray
        First input array.
    Y : np.ndarray
        Second input array.

    Returns:
    --------
    float
        Manhattan distance between X and Y.
    """
    return np.sum(np.abs(X - Y))

def manhattan_distance_fit(
    X: np.ndarray,
    Y: np.ndarray,
    normalize_X: Optional[str] = None,
    normalize_Y: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute the Manhattan distance between two arrays with optional normalization.

    Parameters:
    -----------
    X : np.ndarray
        First input array.
    Y : np.ndarray
        Second input array.
    normalize_X : Optional[str]
        Normalization method for X ('none', 'standard', 'minmax', 'robust').
    normalize_Y : Optional[str]
        Normalization method for Y ('none', 'standard', 'minmax', 'robust').

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> X = np.array([1, 2, 3])
    >>> Y = np.array([4, 5, 6])
    >>> manhattan_distance_fit(X, Y)
    {
        'result': 9.0,
        'metrics': {'distance': 9.0},
        'params_used': {
            'normalize_X': None,
            'normalize_Y': None
        },
        'warnings': []
    }
    """
    validated_inputs = _validate_inputs(X, Y, normalize_X, normalize_Y)

    distance = _compute_manhattan_distance(validated_inputs['X'], validated_inputs['Y'])

    return {
        'result': distance,
        'metrics': {'distance': distance},
        'params_used': {
            'normalize_X': normalize_X,
            'normalize_Y': normalize_Y
        },
        'warnings': []
    }

################################################################################
# cosine_similarity
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

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

def _normalize_none(x: np.ndarray) -> np.ndarray:
    """No normalization."""
    return x

def _normalize_standard(x: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    return (x - np.mean(x)) / np.std(x)

def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def _normalize_robust(x: np.ndarray) -> np.ndarray:
    """Robust normalization (using median and IQR)."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x - np.median(x)) / iqr

def cosine_similarity_compute(
    x: np.ndarray,
    y: np.ndarray,
    normalization: str = "none",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute cosine similarity between two vectors.

    Parameters:
    -----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    custom_normalization : Callable, optional
        Custom normalization function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result and metadata.
    """
    _validate_inputs(x, y)

    # Normalization
    if custom_normalization is not None:
        x_norm = custom_normalization(x)
        y_norm = custom_normalization(y)
    else:
        norm_funcs = {
            "none": _normalize_none,
            "standard": _normalize_standard,
            "minmax": _normalize_minmax,
            "robust": _normalize_robust,
        }
        norm_func = norm_funcs.get(normalization, _normalize_none)
        x_norm = norm_func(x)
        y_norm = norm_func(y)

    # Compute cosine similarity
    dot_product = np.dot(x_norm, y_norm)
    norm_x = np.linalg.norm(x_norm)
    norm_y = np.linalg.norm(y_norm)

    if norm_x == 0 or norm_y == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm_x * norm_y)

    return {
        "result": similarity,
        "metrics": {},
        "params_used": {"normalization": normalization},
        "warnings": [],
    }

# Example usage:
# cosine_similarity_compute(np.array([1, 2, 3]), np.array([4, 5, 6]))

################################################################################
# hamming_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def hamming_distance_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    normalize: str = "none",
    metric: Union[str, Callable] = "hamming",
    distance_metric: str = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute the Hamming distance between two distributions.

    Parameters:
    -----------
    X : np.ndarray
        First input distribution.
    Y : np.ndarray
        Second input distribution.
    normalize : str, optional (default="none")
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str or callable, optional (default="hamming")
        Metric to use: "hamming" or a custom callable.
    distance_metric : str, optional (default="euclidean")
        Distance metric: "euclidean", "manhattan", "cosine", or "minkowski".
    solver : str, optional (default="closed_form")
        Solver method: "closed_form", "gradient_descent", or "newton".
    regularization : str, optional (default=None)
        Regularization method: None, "l1", "l2", or "elasticnet".
    tol : float, optional (default=1e-4)
        Tolerance for convergence.
    max_iter : int, optional (default=1000)
        Maximum number of iterations.
    custom_metric : callable, optional (default=None)
        Custom metric function.
    custom_distance : callable, optional (default=None)
        Custom distance function.

    Returns:
    --------
    Dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if required
    X_norm, Y_norm = _normalize_data(X, Y, normalize)

    # Compute the Hamming distance
    result = _compute_hamming_distance(X_norm, Y_norm, metric, custom_metric)

    # Prepare the output dictionary
    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "normalize": normalize,
            "metric": metric,
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate the input arrays."""
    if X.shape != Y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values.")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Input Y contains NaN or infinite values.")

def _normalize_data(
    X: np.ndarray,
    Y: np.ndarray,
    normalize: str
) -> tuple:
    """Normalize the input data."""
    if normalize == "standard":
        X_norm = (X - np.mean(X)) / np.std(X)
        Y_norm = (Y - np.mean(Y)) / np.std(Y)
    elif normalize == "minmax":
        X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
        Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    elif normalize == "robust":
        X_norm = (X - np.median(X)) / (np.percentile(X, 75) - np.percentile(X, 25))
        Y_norm = (Y - np.median(Y)) / (np.percentile(Y, 75) - np.percentile(Y, 25))
    else:
        X_norm, Y_norm = X, Y
    return X_norm, Y_norm

def _compute_hamming_distance(
    X: np.ndarray,
    Y: np.ndarray,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> float:
    """Compute the Hamming distance between two distributions."""
    if custom_metric is not None:
        return custom_metric(X, Y)
    elif metric == "hamming":
        return np.mean(X != Y)
    else:
        raise ValueError("Unsupported metric.")

# Example usage
if __name__ == "__main__":
    X = np.array([1, 0, 1, 1])
    Y = np.array([1, 1, 0, 1])
    result = hamming_distance_fit(X, Y)
    print(result)

################################################################################
# jaccard_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "jaccard",
    normalize: bool = False
) -> None:
    """Validate input arrays and parameters."""
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs must be 2-dimensional arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Inputs must have the same number of features")
    if metric not in ["jaccard", "custom"]:
        raise ValueError("Invalid metric specified")
    if normalize and X.shape[0] != Y.shape[0]:
        raise ValueError("For normalization, inputs must have the same number of samples")

def _normalize_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "none"
) -> tuple:
    """Normalize input arrays."""
    if method == "standard":
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif method == "minmax":
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        Y = (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))
    elif method == "robust":
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        Y = (Y - np.median(Y, axis=0)) / (np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0))
    return X, Y

def _compute_jaccard_distance(
    X: np.ndarray,
    Y: np.ndarray
) -> float:
    """Compute Jaccard distance between two sets of binary vectors."""
    intersection = np.sum(X * Y, axis=1)
    union = np.sum(np.logical_or(X, Y), axis=1)
    jaccard_similarity = intersection / union
    return 1 - np.mean(jaccard_similarity)

def _compute_custom_distance(
    X: np.ndarray,
    Y: np.ndarray,
    distance_func: Callable
) -> float:
    """Compute custom distance between two sets of vectors."""
    return np.mean([distance_func(x, y) for x, y in zip(X, Y)])

def jaccard_distance_fit(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "jaccard",
    normalize: bool = False,
    normalization_method: str = "none",
    custom_distance_func: Optional[Callable] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Compute the Jaccard distance between two sets of vectors.

    Parameters
    ----------
    X : np.ndarray
        First set of binary vectors (n_samples, n_features)
    Y : np.ndarray
        Second set of binary vectors (n_samples, n_features)
    metric : str, optional
        Distance metric to use ("jaccard" or "custom"), by default "jaccard"
    normalize : bool, optional
        Whether to normalize the inputs before computing distance, by default False
    normalization_method : str, optional
        Normalization method ("none", "standard", "minmax", "robust"), by default "none"
    custom_distance_func : Callable, optional
        Custom distance function to use if metric="custom", by default None

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary containing:
        - "result": computed distance
        - "metrics": dictionary of metrics (currently empty)
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings (if any)

    Example
    -------
    >>> X = np.array([[1, 0, 1], [0, 1, 0]])
    >>> Y = np.array([[1, 1, 0], [0, 0, 1]])
    >>> jaccard_distance_fit(X, Y)
    {
        'result': 0.33333333333333337,
        'metrics': {},
        'params_used': {
            'metric': 'jaccard',
            'normalize': False,
            'normalization_method': 'none'
        },
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(X, Y, metric)

    # Normalize if requested
    X_norm, Y_norm = _normalize_arrays(X, Y, normalization_method) if normalize else (X, Y)

    # Compute distance
    if metric == "jaccard":
        result = _compute_jaccard_distance(X_norm, Y_norm)
    else:
        if custom_distance_func is None:
            raise ValueError("Custom distance function must be provided when metric='custom'")
        result = _compute_custom_distance(X_norm, Y_norm, custom_distance_func)

    # Prepare output
    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "metric": metric,
            "normalize": normalize,
            "normalization_method": normalization_method
        },
        "warnings": []
    }

    return output

################################################################################
# minkowski_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Minkowski distance calculation."""
    if x.ndim != y.ndim:
        raise ValueError("Input arrays must have the same number of dimensions")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input array x contains NaN or infinite values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input array y contains NaN or infinite values")

def normalize_data(x: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize input data based on specified method."""
    if method == 'none':
        return x
    elif method == 'standard':
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(x, axis=0)
        max_val = np.max(x, axis=0)
        return (x - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(x, axis=0)
        iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
        return (x - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def minkowski_distance_compute(
    x: np.ndarray,
    y: np.ndarray,
    p: float = 2.0,
    normalization: str = 'none',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute Minkowski distance between two arrays.

    Parameters:
    - x: First input array
    - y: Second input array
    - p: Power parameter for Minkowski distance (default 2.0 for Euclidean)
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - custom_metric: Optional custom metric function

    Returns:
    Dictionary containing:
    - result: Computed distance
    - metrics: Additional metrics if applicable
    - params_used: Parameters used in computation
    - warnings: Any warnings generated

    Example:
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[5, 6], [7, 8]])
    >>> minkowski_distance_compute(x, y, p=1.0)  # Manhattan distance
    """
    validate_inputs(x, y)

    warnings = []

    if normalization != 'none':
        x_norm = normalize_data(x, method=normalization)
        y_norm = normalize_data(y, method=normalization)
    else:
        x_norm = x
        y_norm = y

    if custom_metric is not None:
        result = custom_metric(x_norm, y_norm)
    else:
        if p <= 0:
            warnings.append("p value should be positive, using absolute difference instead")
            p = 1.0
        diff = np.abs(x_norm - y_norm)
        result = np.sum(diff ** p) ** (1/p)

    return {
        'result': float(result),
        'metrics': {},
        'params_used': {
            'p': p,
            'normalization': normalization
        },
        'warnings': warnings
    }

################################################################################
# chebyshev_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    normalize_X: Optional[Callable] = None,
    normalize_Y: Optional[Callable] = None
) -> None:
    """Validate input arrays and optional normalization functions."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Input arrays must have the same number of features")
    if normalize_X is not None and not callable(normalize_X):
        raise ValueError("normalize_X must be a callable or None")
    if normalize_Y is not None and not callable(normalize_Y):
        raise ValueError("normalize_Y must be a callable or None")

def _apply_normalization(
    data: np.ndarray,
    normalize_func: Optional[Callable] = None
) -> np.ndarray:
    """Apply normalization to data if a normalization function is provided."""
    if normalize_func is not None:
        return normalize_func(data)
    return data

def _compute_chebyshev_distance(
    X: np.ndarray,
    Y: np.ndarray
) -> float:
    """Compute the Chebyshev distance between two sets of points."""
    return np.max(np.abs(X[:, np.newaxis] - Y), axis=2)

def chebyshev_distance_fit(
    X: np.ndarray,
    Y: np.ndarray,
    normalize_X: Optional[Callable] = None,
    normalize_Y: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute the Chebyshev distance between two sets of points with optional normalization.

    Parameters:
    -----------
    X : np.ndarray
        First set of points, shape (n_samples_X, n_features)
    Y : np.ndarray
        Second set of points, shape (n_samples_Y, n_features)
    normalize_X : Optional[Callable], default=None
        Function to normalize X before computation. If None, no normalization is applied.
    normalize_Y : Optional[Callable], default=None
        Function to normalize Y before computation. If None, no normalization is applied.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": Computed Chebyshev distance matrix
        - "metrics": Dictionary of metrics (currently empty)
        - "params_used": Dictionary of parameters used
        - "warnings": List of warnings (if any)
    """
    _validate_inputs(X, Y, normalize_X, normalize_Y)

    X_normalized = _apply_normalization(X, normalize_X)
    Y_normalized = _apply_normalization(Y, normalize_Y)

    distance_matrix = _compute_chebyshev_distance(X_normalized, Y_normalized)

    return {
        "result": distance_matrix,
        "metrics": {},
        "params_used": {
            "normalize_X": normalize_X.__name__ if normalize_X else None,
            "normalize_Y": normalize_Y.__name__ if normalize_Y else None
        },
        "warnings": []
    }

# Example usage:
# X = np.array([[1, 2], [3, 4]])
# Y = np.array([[5, 6], [7, 8]])
# result = chebyshev_distance_fit(X, Y)

################################################################################
# canberra_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

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

def _canberra_distance_compute(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Canberra distance between two arrays."""
    return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))

def canberra_distance_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: Optional[str] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = _canberra_distance_compute,
    **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute the Canberra distance between two distributions with optional normalization.

    Parameters:
    - x: First input array.
    - y: Second input array.
    - normalization: Optional normalization method ('none', 'standard', 'minmax', 'robust').
    - distance_metric: Callable to compute the distance metric.
    - **kwargs: Additional keyword arguments for future extensions.

    Returns:
    - A dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    # Apply normalization if specified
    normalized_x = x.copy()
    normalized_y = y.copy()

    if normalization == 'standard':
        mean_x, std_x = np.mean(x), np.std(x)
        mean_y, std_y = np.mean(y), np.std(y)
        normalized_x = (x - mean_x) / std_x
        normalized_y = (y - mean_y) / std_y
    elif normalization == 'minmax':
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        normalized_x = (x - min_x) / (max_x - min_x)
        normalized_y = (y - min_y) / (max_y - min_y)
    elif normalization == 'robust':
        median_x, iqr_x = np.median(x), np.percentile(x, 75) - np.percentile(x, 25)
        median_y, iqr_y = np.median(y), np.percentile(y, 75) - np.percentile(y, 25)
        normalized_x = (x - median_x) / iqr_x
        normalized_y = (y - median_y) / iqr_y

    # Compute the distance
    result = distance_metric(normalized_x, normalized_y)

    return {
        "result": result,
        "metrics": {"distance": result},
        "params_used": {
            "normalization": normalization,
            "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom"
        },
        "warnings": []
    }

# Example usage:
# x = np.array([1.0, 2.0, 3.0])
# y = np.array([4.0, 5.0, 6.0])
# result = canberra_distance_fit(x, y, normalization='standard')

################################################################################
# bray_curtis_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Bray-Curtis distance calculation."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input array x contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Input array y contains NaN or infinite values.")

def _bray_curtis_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Bray-Curtis distance between two vectors."""
    abs_diff = np.abs(x - y)
    sum_abs_x = np.sum(np.abs(x))
    sum_abs_y = np.sum(np.abs(y))
    denominator = sum_abs_x + sum_abs_y
    if denominator == 0:
        return 0.0
    return np.sum(abs_diff) / denominator

def bray_curtis_distance_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalization: Optional[str] = None,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Bray-Curtis distance between two vectors with optional normalization and metric.

    Parameters:
    -----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : Callable, optional
        Custom metric function. Must take two numpy arrays and return a float.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(x, y)

    # Normalization
    if normalization == 'standard':
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    elif normalization == 'minmax':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    elif normalization == 'robust':
        x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))

    # Compute Bray-Curtis distance
    result = _bray_curtis_distance(x, y)

    # Compute additional metrics if provided
    metrics = {}
    if metric is not None:
        try:
            metrics['custom_metric'] = metric(x, y)
        except Exception as e:
            warnings.append(f"Custom metric computation failed: {str(e)}")

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'metric': metric.__name__ if metric else None
        },
        'warnings': []
    }

    return output

# Example usage:
# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6])
# result = bray_curtis_distance_fit(x, y)

################################################################################
# correlation_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def correlation_distance_fit(
    X: np.ndarray,
    y: np.ndarray,
    distance_metric: str = 'euclidean',
    normalization: str = 'standard',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute the correlation distance between two distributions.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    distance_metric : str or callable, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine',
        'minkowski'. Default is 'euclidean'.
    normalization : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax',
        'robust'. Default is 'standard'.
    solver : str, optional
        Solver to use. Options: 'closed_form', 'gradient_descent',
        'newton', 'coordinate_descent'. Default is 'closed_form'.
    regularization : str or None, optional
        Regularization method. Options: 'none', 'l1', 'l2',
        'elasticnet'. Default is None.
    custom_distance : callable or None, optional
        Custom distance function. If provided, overrides distance_metric.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Computed correlation distance.
        - 'metrics': Dictionary of metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': List of warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = correlation_distance_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Choose distance metric
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance_metric)

    # Solve for correlation distance
    if solver == 'closed_form':
        result = _solve_closed_form(X_normalized, y, distance_func)
    elif solver == 'gradient_descent':
        result = _solve_gradient_descent(X_normalized, y, distance_func,
                                        tol=tol, max_iter=max_iter,
                                        random_state=random_state)
    elif solver == 'newton':
        result = _solve_newton(X_normalized, y, distance_func,
                              tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        result = _solve_coordinate_descent(X_normalized, y, distance_func,
                                          tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, result)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'distance_metric': distance_metric if custom_distance is None else 'custom',
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

    return output

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
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

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on metric name."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _solve_closed_form(X: np.ndarray, y: np.ndarray,
                      distance_func: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """Solve for correlation distance using closed-form solution."""
    # Placeholder for actual implementation
    return np.mean([distance_func(x, y) for x in X])

def _solve_gradient_descent(X: np.ndarray, y: np.ndarray,
                           distance_func: Callable[[np.ndarray, np.ndarray], float],
                           tol: float = 1e-6,
                           max_iter: int = 1000,
                           random_state: Optional[int] = None) -> float:
    """Solve for correlation distance using gradient descent."""
    # Placeholder for actual implementation
    return np.mean([distance_func(x, y) for x in X])

def _solve_newton(X: np.ndarray, y: np.ndarray,
                 distance_func: Callable[[np.ndarray, np.ndarray], float],
                 tol: float = 1e-6,
                 max_iter: int = 1000) -> float:
    """Solve for correlation distance using Newton's method."""
    # Placeholder for actual implementation
    return np.mean([distance_func(x, y) for x in X])

def _solve_coordinate_descent(X: np.ndarray, y: np.ndarray,
                             distance_func: Callable[[np.ndarray, np.ndarray], float],
                             tol: float = 1e-6,
                             max_iter: int = 1000) -> float:
    """Solve for correlation distance using coordinate descent."""
    # Placeholder for actual implementation
    return np.mean([distance_func(x, y) for x in X])

def _compute_metrics(X: np.ndarray, y: np.ndarray,
                    result: float) -> Dict[str, float]:
    """Compute metrics for the correlation distance."""
    # Placeholder for actual implementation
    return {
        'mse': np.mean((X @ result - y) ** 2),
        'mae': np.mean(np.abs(X @ result - y)),
        'r2': 1 - np.sum((y - X @ result) ** 2) / np.sum((y - np.mean(y)) ** 2)
    }
