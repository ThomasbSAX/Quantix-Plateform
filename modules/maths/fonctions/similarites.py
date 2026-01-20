"""
Quantix – Module similarites
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# cosine_similarity
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize_X: bool = True,
    normalize_Y: bool = False
) -> Dict[str, np.ndarray]:
    """
    Validate input arrays and apply normalization if required.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples, n_features), defaults to X if None
    normalize_X : bool
        Whether to normalize X
    normalize_Y : bool
        Whether to normalize Y

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing validated and normalized arrays with keys:
        'X', 'Y', 'X_normalized', 'Y_normalized'
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if Y is not None and Y.ndim != 2:
        raise ValueError("Y must be a 2D array or None")

    if Y is None:
        Y = X.copy()

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    X_normalized = _normalize_array(X) if normalize_X else X
    Y_normalized = _normalize_array(Y) if normalize_Y else Y

    return {
        'X': X,
        'Y': Y,
        'X_normalized': X_normalized,
        'Y_normalized': Y_normalized
    }

def _normalize_array(
    arr: np.ndarray,
    method: str = 'l2'
) -> np.ndarray:
    """
    Normalize array using specified method.

    Parameters
    ----------
    arr : np.ndarray
        Input array to normalize
    method : str
        Normalization method ('l1', 'l2', or 'max')

    Returns
    -------
    np.ndarray
        Normalized array
    """
    if method == 'l1':
        norms = np.linalg.norm(arr, ord=1, axis=1, keepdims=True)
    elif method == 'l2':
        norms = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
    elif method == 'max':
        norms = np.max(np.abs(arr), axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Avoid division by zero
    norms[norms == 0] = 1.0

    return arr / norms

def _compute_cosine_similarity(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between arrays.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : np.ndarray
        Second input array of shape (n_samples, n_features)

    Returns
    -------
    np.ndarray
        Cosine similarity matrix of shape (n_samples, n_samples)
    """
    dot_products = np.dot(X, Y.T)
    norms_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
    norms_Y = np.linalg.norm(Y, axis=1)[np.newaxis, :]

    # Avoid division by zero
    norms_X[norms_X == 0] = 1.0
    norms_Y[norms_Y == 0] = 1.0

    return dot_products / (norms_X * norms_Y.T)

def cosine_similarity_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize_X: bool = True,
    normalize_Y: bool = False,
    normalization_method: str = 'l2',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compute cosine similarity between two arrays with configurable options.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Target array of shape (n_samples, n_features), defaults to X if None
    normalize_X : bool
        Whether to normalize X before computation
    normalize_Y : bool
        Whether to normalize Y before computation
    normalization_method : str
        Normalization method ('l1', 'l2', or 'max')
    custom_metric : Optional[Callable]
        Custom metric function to use instead of cosine similarity

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'result': similarity matrix
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used in computation
        - 'warnings': list of warnings

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = cosine_similarity_fit(X)
    """
    # Validate inputs and normalize if required
    validated = _validate_inputs(
        X, Y, normalize_X, normalize_Y
    )

    # Compute similarity
    if custom_metric is not None:
        result = custom_metric(validated['X_normalized'], validated['Y_normalized'])
    else:
        result = _compute_cosine_similarity(
            validated['X_normalized'], validated['Y_normalized']
        )

    # Compute metrics
    metrics = {
        'mean_similarity': np.mean(result),
        'min_similarity': np.min(result),
        'max_similarity': np.max(result)
    }

    # Prepare output
    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalize_X': normalize_X,
            'normalize_Y': normalize_Y,
            'normalization_method': normalization_method
        },
        'warnings': []
    }

################################################################################
# euclidean_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray, Y: Optional[np.ndarray] = None) -> None:
    """
    Validate input arrays for Euclidean distance calculation.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Second input array of shape (n_samples, n_features). If None, computes pairwise distances.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong shape, NaN/inf values)
    """
    if not isinstance(X, np.ndarray) or (Y is not None and not isinstance(Y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if Y is not None and Y.ndim != 2:
        raise ValueError("Y must be a 2D array or None")

    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))) or \
       np.any(np.isinf(X)) or (Y is not None and np.any(np.isinf(Y))):
        raise ValueError("Inputs contain NaN or infinite values")

def normalize_data(X: np.ndarray, method: str = 'none') -> np.ndarray:
    """
    Normalize input data using specified method.

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
        return X

    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)

    if method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / ((max_val - min_val + 1e-8))

    if method == 'robust':
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return (X - median) / (iqr + 1e-8)

    raise ValueError(f"Unknown normalization method: {method}")

def compute_euclidean_distance(X_norm: np.ndarray, Y_norm: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Euclidean distance between normalized arrays.

    Parameters
    ----------
    X_norm : np.ndarray
        First normalized array of shape (n_samples, n_features)
    Y_norm : Optional[np.ndarray]
        Second normalized array of shape (n_samples, n_features). If None, computes pairwise distances.

    Returns
    ------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples) or (n_samples,) if Y is None
    """
    if Y_norm is None:
        # Pairwise distances
        diff = X_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))
    else:
        # Between two arrays
        diff = X_norm - Y_norm
        return np.sqrt(np.sum(diff**2, axis=-1))

def euclidean_distance_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize_method: str = 'none',
    pairwise: bool = False
) -> Dict[str, Any]:
    """
    Compute Euclidean distance between arrays with configurable options.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Second input array of shape (n_samples, n_features). If None and pairwise=True,
        computes pairwise distances on X.
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    pairwise : bool
        If True and Y is None, computes pairwise distances on X

    Returns
    ------
    Dict[str, Any]
        Dictionary containing:
        - 'result': computed distances
        - 'metrics': dictionary of metrics (currently empty)
        - 'params_used': parameters used
        - 'warnings': list of warnings

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = euclidean_distance_fit(X)
    """
    warnings: list[str] = []

    # Validate inputs
    validate_inputs(X, Y)

    # Normalize data
    X_norm = normalize_data(X, normalize_method)
    if Y is not None:
        Y_norm = normalize_data(Y, normalize_method)

    # Compute distances
    if pairwise and Y is None:
        distances = compute_euclidean_distance(X_norm)
    else:
        if Y is None:
            warnings.append("Y is None and pairwise=False - computing distances between X and itself")
        distances = compute_euclidean_distance(X_norm, Y_norm if Y is not None else X_norm)

    return {
        'result': distances,
        'metrics': {},
        'params_used': {
            'normalize_method': normalize_method,
            'pairwise': pairwise
        },
        'warnings': warnings
    }

################################################################################
# manhattan_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    normalize: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Validate input arrays and apply normalization if specified.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : np.ndarray
        Second input array of shape (n_samples, n_features) or (1, n_features)
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing validated and normalized arrays with keys 'X' and 'Y'
    """
    if X.ndim != 2 or Y.ndim not in (2, 1):
        raise ValueError("X must be 2D and Y must be 1D or 2D")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if normalize == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif normalize == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        Y = (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))
    elif normalize == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        Y = (Y - np.median(Y, axis=0)) / (np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0))

    return {'X': X, 'Y': Y}

def _compute_manhattan_distance(
    X: np.ndarray,
    Y: np.ndarray
) -> float:
    """
    Compute Manhattan distance between two arrays.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : np.ndarray
        Second input array of shape (n_samples, n_features) or (1, n_features)

    Returns
    -------
    float
        Computed Manhattan distance
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    return np.sum(np.abs(X - Y), axis=1).mean()

def manhattan_distance_fit(
    X: np.ndarray,
    Y: np.ndarray,
    normalize: Optional[str] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute Manhattan distance between two arrays with optional normalization.

    Parameters
    ----------
    X : np.ndarray
        First input array of shape (n_samples, n_features)
    Y : np.ndarray
        Second input array of shape (n_samples, n_features) or (1, n_features)
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - 'result': computed Manhattan distance
        - 'metrics': empty dictionary (can be extended for additional metrics)
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Y = np.array([0, 1])
    >>> manhattan_distance_fit(X, Y)
    {
        'result': 2.5,
        'metrics': {},
        'params_used': {'normalize': None},
        'warnings': []
    }
    """
    warnings = []

    # Validate inputs and apply normalization
    validated_data = _validate_inputs(X, Y, normalize)
    X_validated = validated_data['X']
    Y_validated = validated_data['Y']

    # Compute Manhattan distance
    result = _compute_manhattan_distance(X_validated, Y_validated)

    return {
        'result': result,
        'metrics': {},
        'params_used': {'normalize': normalize},
        'warnings': warnings
    }

################################################################################
# jaccard_similarity
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    set1: np.ndarray,
    set2: np.ndarray,
    check_nan: bool = True
) -> None:
    """
    Validate input sets for Jaccard similarity.

    Parameters
    ----------
    set1 : np.ndarray
        First input set as a numpy array.
    set2 : np.ndarray
        Second input set as a numpy array.
    check_nan : bool, optional
        Whether to check for NaN values, by default True.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(set1, np.ndarray) or not isinstance(set2, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if set1.ndim != 1 or set2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")
    if check_nan and (np.isnan(set1).any() or np.isnan(set2).any()):
        raise ValueError("Inputs contain NaN values")

def _jaccard_similarity_compute(
    set1: np.ndarray,
    set2: np.ndarray
) -> float:
    """
    Compute Jaccard similarity between two sets.

    Parameters
    ----------
    set1 : np.ndarray
        First input set as a numpy array.
    set2 : np.ndarray
        Second input set as a numpy array.

    Returns
    ------
    float
        Jaccard similarity coefficient.
    """
    intersection = np.intersect1d(set1, set2).size
    union = np.union1d(set1, set2).size
    return intersection / union if union != 0 else 0.0

def jaccard_similarity_fit(
    set1: np.ndarray,
    set2: np.ndarray,
    normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = "jaccard",
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    check_nan: bool = True
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute Jaccard similarity between two sets with configurable options.

    Parameters
    ----------
    set1 : np.ndarray
        First input set as a numpy array.
    set2 : np.ndarray
        Second input set as a numpy array.
    normalize : Optional[Callable], optional
        Normalization function to apply to inputs, by default None.
    metric : str, optional
        Metric to compute ("jaccard"), by default "jaccard".
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.
    check_nan : bool, optional
        Whether to check for NaN values, by default True.

    Returns
    ------
    Dict[str, Union[float, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": computed similarity
        - "metrics": dictionary of metrics
        - "params_used": dictionary of parameters used
        - "warnings": dictionary of warnings

    Examples
    --------
    >>> set1 = np.array([1, 2, 3])
    >>> set2 = np.array([2, 3, 4])
    >>> jaccard_similarity_fit(set1, set2)
    {
        'result': 0.5,
        'metrics': {'jaccard': 0.5},
        'params_used': {
            'normalize': None,
            'metric': 'jaccard',
            'custom_metric': None
        },
        'warnings': {}
    }
    """
    # Validate inputs
    _validate_inputs(set1, set2, check_nan)

    # Apply normalization if specified
    if normalize is not None:
        set1 = normalize(set1)
        set2 = normalize(set2)

    # Compute metrics
    metrics = {}

    if metric == "jaccard" or custom_metric is None:
        jaccard = _jaccard_similarity_compute(set1, set2)
        metrics["jaccard"] = jaccard

    if custom_metric is not None:
        try:
            custom_result = custom_metric(set1, set2)
            metrics["custom"] = custom_result
        except Exception as e:
            warnings = {"custom_metric": f"Error computing custom metric: {str(e)}"}
        else:
            warnings = {}

    # Determine result based on requested metric
    if custom_metric is not None and "custom" in metrics:
        result = metrics["custom"]
    else:
        result = metrics.get("jaccard", 0.0)

    # Prepare output
    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize.__name__ if normalize is not None else None,
            "metric": metric,
            "custom_metric": custom_metric.__name__ if custom_metric is not None else None
        },
        "warnings": warnings if "warnings" in locals() else {}
    }

################################################################################
# pearson_correlation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def pearson_correlation_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: str = 'standard',
    metric: str = 'r2',
    solver: str = 'closed_form',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute Pearson correlation between features and target.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Target vector of shape (n_samples,) or None for pairwise correlation.
    normalize : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : str
        Metric to evaluate: 'mse', 'mae', 'r2', or custom callable.
    solver : str
        Solver method: 'closed_form' or gradient-based methods.
    custom_metric : Optional[Callable]
        Custom metric function if needed.
    custom_distance : Optional[Callable]
        Custom distance function if needed.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, method=normalize)
    y_norm = _normalize_data(y.reshape(-1, 1) if y is not None else np.array([]), method=normalize).flatten() if y is not None else None

    # Choose solver
    if solver == 'closed_form':
        correlation = _pearson_closed_form(X_norm, y_norm)
    else:
        raise ValueError(f"Solver {solver} not implemented for Pearson correlation.")

    # Compute metrics
    metrics = _compute_metrics(correlation, X_norm, y_norm, metric=metric, custom_metric=custom_metric)

    # Prepare output
    result = {
        'result': correlation,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'metric': metric,
            'solver': solver
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None.")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X)) or (y is not None and np.any(np.isinf(y))):
        raise ValueError("Input data contains infinite values.")

def _normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize data using specified method."""
    if data.size == 0:
        return data
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
        raise ValueError(f"Normalization method {method} not recognized.")

def _pearson_closed_form(X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
    """Compute Pearson correlation using closed-form solution."""
    if y is None:
        # Pairwise correlation
        cov = np.cov(X, rowvar=False)
        var = np.diag(cov)
        return cov / np.sqrt(np.outer(var, var))
    else:
        # Correlation between X and y
        cov = np.cov(X, y, rowvar=False)
        var_x = np.var(X, axis=0)
        var_y = np.var(y)
        return cov[:-1, -1] / np.sqrt(var_x * var_y)

def _compute_metrics(
    correlation: np.ndarray,
    X: np.ndarray,
    y: Optional[np.ndarray],
    metric: str = 'r2',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    metrics = {}
    if y is None:
        return metrics

    if metric == 'r2':
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - np.dot(X, correlation))**2)
        metrics['r2'] = 1 - (ss_residual / ss_total)
    elif metric == 'mse':
        metrics['mse'] = np.mean((y - np.dot(X, correlation))**2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(y - np.dot(X, correlation)))
    elif custom_metric is not None:
        metrics['custom'] = custom_metric(y, np.dot(X, correlation))
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    return metrics

################################################################################
# spearman_rank_correlation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Spearman rank correlation.

    Args:
        x: First input array.
        y: Second input array.

    Raises:
        ValueError: If inputs are invalid.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or inf values.")

def rank_data(data: np.ndarray) -> np.ndarray:
    """Compute ranks for input data, handling ties appropriately.

    Args:
        data: Input array to rank.

    Returns:
        Array of ranks.
    """
    sorted_data = np.sort(data)
    ranks = np.zeros_like(data, dtype=float)

    for i, value in enumerate(sorted_data):
        mask = data == value
        ranks[mask] = np.mean([i + 1, i + len(data) - np.sum(mask)])

    return ranks

def spearman_rank_correlation_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    normalize: bool = False,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Compute Spearman rank correlation between two arrays.

    Args:
        x: First input array.
        y: Second input array.
        normalize: Whether to normalize ranks (default: False).
        custom_metric: Optional custom metric function.

    Returns:
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(x, y)

    # Compute ranks
    rank_x = rank_data(x)
    rank_y = rank_data(y)

    # Normalize ranks if requested
    if normalize:
        rank_x = (rank_x - np.mean(rank_x)) / np.std(rank_x)
        rank_y = (rank_y - np.mean(rank_y)) / np.std(rank_y)

    # Compute covariance and variances
    cov_xy = np.cov(rank_x, rank_y)[0, 1]
    var_x = np.var(rank_x)
    var_y = np.var(rank_y)

    # Compute correlation
    if var_x == 0 or var_y == 0:
        spearman_corr = np.nan
    else:
        spearman_corr = cov_xy / np.sqrt(var_x * var_y)

    # Compute custom metric if provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom'] = custom_metric(rank_x, rank_y)
        except Exception as e:
            metrics['custom_error'] = str(e)

    # Prepare output
    result = {
        'spearman_correlation': spearman_corr,
        'metrics': metrics,
        'params_used': {
            'normalize': normalize,
            'custom_metric_provided': custom_metric is not None
        },
        'warnings': []
    }

    if np.isnan(spearman_corr):
        result['warnings'].append("Correlation could not be computed due to zero variance in ranks.")

    return result

# Example usage:
# result = spearman_rank_correlation_fit(
#     np.array([1, 2, 3, 4, 5]),
#     np.array([5, 4, 3, 2, 1])
# )

################################################################################
# dice_coefficient
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input arrays must not contain NaN or inf values.")

def _dice_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Dice coefficient between two binary arrays."""
    intersection = np.sum(x * y)
    sum_squares = np.sum(x) + np.sum(y)
    if sum_squares == 0:
        return 1.0
    return (2. * intersection) / sum_squares

def dice_coefficient_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalize: Optional[str] = None,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Dice coefficient between two binary arrays.

    Parameters:
    -----------
    x : np.ndarray
        First binary array.
    y : np.ndarray
        Second binary array.
    normalize : str, optional
        Normalization method. Not used for Dice coefficient but kept for consistency.
    metric : Callable, optional
        Custom metric function. If provided, will override the default Dice coefficient.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> x = np.array([1, 0, 1, 1])
    >>> y = np.array([1, 1, 0, 1])
    >>> result = dice_coefficient_fit(x, y)
    """
    _validate_inputs(x, y)

    if metric is not None:
        result = metric(x, y)
    else:
        result = _dice_coefficient(x, y)

    return {
        "result": result,
        "metrics": {"dice_coefficient": result},
        "params_used": {
            "normalize": normalize,
            "metric": "default" if metric is None else "custom"
        },
        "warnings": []
    }

################################################################################
# hamming_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input arrays for Hamming distance calculation."""
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if not (np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer)):
        raise ValueError("Input arrays must contain integer values.")
    if np.any((x < 0) | (y < 0)):
        raise ValueError("Input arrays must contain non-negative integers.")

def _compute_hamming_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Hamming distance between two arrays."""
    return np.sum(x != y)

def hamming_distance_fit(
    x: np.ndarray,
    y: np.ndarray,
    normalize: Optional[str] = None,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the Hamming distance between two arrays with optional normalization and custom metric.

    Parameters:
    -----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array.
    normalize : str, optional
        Normalization method. Options: 'none', 'standard', 'minmax', 'robust'.
    metric : Callable, optional
        Custom metric function. Must accept two numpy arrays and return a float.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([1, 0, 3])
    >>> hamming_distance_fit(x, y)
    {
        'result': 1.0,
        'metrics': {'hamming_distance': 1.0},
        'params_used': {'normalize': None, 'metric': None},
        'warnings': []
    }
    """
    _validate_inputs(x, y)

    params_used = {
        'normalize': normalize,
        'metric': metric
    }

    warnings = []

    if normalize is not None:
        if normalize == 'standard':
            x = (x - np.mean(x)) / np.std(x)
            y = (y - np.mean(y)) / np.std(y)
        elif normalize == 'minmax':
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        elif normalize == 'robust':
            x = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
            y = (y - np.median(y)) / (np.percentile(y, 75) - np.percentile(y, 25))
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")

    hamming_dist = _compute_hamming_distance(x, y)

    metrics = {
        'hamming_distance': hamming_dist
    }

    if metric is not None:
        custom_metric = metric(x, y)
        metrics['custom_metric'] = custom_metric

    return {
        'result': hamming_dist,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# levenshtein_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    s1: str,
    s2: str
) -> None:
    """
    Validate input strings for Levenshtein distance calculation.

    Parameters
    ----------
    s1 : str
        First input string.
    s2 : str
        Second input string.

    Raises
    ------
    ValueError
        If inputs are not strings or if they contain invalid characters.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise ValueError("Inputs must be strings.")
    if not s1.isascii() or not s2.isascii():
        raise ValueError("Strings must contain only ASCII characters.")

def _levenshtein_distance_compute(
    s1: str,
    s2: str
) -> int:
    """
    Compute the Levenshtein distance between two strings.

    Parameters
    ----------
    s1 : str
        First input string.
    s2 : str
        Second input string.

    Returns
    ------
    int
        The Levenshtein distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance_compute(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein_distance_fit(
    s1: str,
    s2: str,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Compute the Levenshtein distance between two strings with optional normalization.

    Parameters
    ----------
    s1 : str
        First input string.
    s2 : str
        Second input string.
    normalize : bool, optional
        Whether to normalize the distance by the maximum string length. Default is False.

    Returns
    ------
    Dict[str, Any]
        A dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_inputs(s1, s2)

    distance = _levenshtein_distance_compute(s1, s2)
    max_len = max(len(s1), len(s2))

    result = {
        "result": distance if not normalize else distance / max_len,
        "metrics": {
            "normalized_distance": distance / max_len if normalize else None
        },
        "params_used": {
            "normalize": normalize
        },
        "warnings": []
    }

    return result

# Example usage:
# levenshtein_distance_fit("kitten", "sitting")

################################################################################
# tanimoto_similarity
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def tanimoto_similarity_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize: str = 'none',
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Compute Tanimoto similarity between two sets of vectors.

    Parameters:
    -----------
    X : np.ndarray
        First set of vectors, shape (n_samples, n_features)
    Y : Optional[np.ndarray]
        Second set of vectors, shape (n_samples, n_features). If None, Y = X.
    normalize : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'
    metric : Union[str, Callable]
        Distance metric: 'euclidean', 'manhattan', 'cosine', or custom callable
    solver : str
        Solver method: 'closed_form' (default)
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    custom_metric : Optional[Callable]
        Custom metric function if needed

    Returns:
    --------
    Dict containing:
        - 'result': similarity matrix
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': potential warnings

    Example:
    --------
    >>> X = np.array([[1, 0], [0, 1]])
    >>> result = tanimoto_similarity_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, Y)

    # Normalize data if needed
    X_norm = _apply_normalization(X, normalize)
    Y_norm = _apply_normalization(Y if Y is not None else X, normalize)

    # Compute similarity
    similarity = _compute_tanimoto_similarity(X_norm, Y_norm, metric, custom_metric)

    # Prepare output
    return {
        'result': similarity,
        'metrics': {'tanimoto_similarity': np.mean(similarity)},
        'params_used': {
            'normalize': normalize,
            'metric': metric if isinstance(metric, str) else 'custom',
            'solver': solver
        },
        'warnings': _check_warnings(X_norm, Y_norm)
    }

def _validate_inputs(X: np.ndarray, Y: Optional[np.ndarray]) -> None:
    """Validate input arrays."""
    if not isinstance(X, np.ndarray) or (Y is not None and not isinstance(Y, np.ndarray)):
        raise ValueError("Inputs must be numpy arrays")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if Y is not None and X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    if np.any(np.isnan(X)) or (Y is not None and np.any(np.isnan(Y))):
        raise ValueError("Input arrays must not contain NaN values")

def _apply_normalization(
    X: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Apply normalization to input array."""
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    return X.copy()

def _compute_tanimoto_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: Union[str, Callable] = 'euclidean',
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """Compute Tanimoto similarity between vectors."""
    Y = X if Y is None else Y

    # Compute dot products
    dot_XY = np.dot(X, Y.T)
    norm_X = np.sum(X**2, axis=1)[:, np.newaxis]
    norm_Y = np.sum(Y**2, axis=1)[np.newaxis, :]

    # Tanimoto similarity formula
    return dot_XY / (norm_X + norm_Y - dot_XY)

def _check_warnings(X: np.ndarray, Y: Optional[np.ndarray] = None) -> Dict[str, str]:
    """Check for potential warnings."""
    Y = X if Y is None else Y
    warnings = {}

    if np.any(norm_X == 0) or np.any(norm_Y == 0):
        warnings['zero_norm'] = "Some vectors have zero norm, similarity will be undefined for these pairs"

    return warnings
