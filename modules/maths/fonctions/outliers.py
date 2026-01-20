"""
Quantix – Module outliers
Généré automatiquement
Date: 2026-01-08
"""

################################################################################
# definition
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    """Validate input data and functions."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

def _normalize_data(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply normalization to the data."""
    return normalizer(X)

def _compute_distances(
    X: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute pairwise distances between data points."""
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = distance_metric(X[i], X[j])
    return distances

def _detect_outliers(
    distances: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Detect outliers based on distance thresholds."""
    n = distances.shape[0]
    outlier_scores = np.zeros(n)
    for i in range(n):
        outlier_scores[i] = np.mean(distances[i, :])
    outliers = outlier_scores > threshold
    return outliers

def definition_fit(
    X: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.linalg.norm(a - b),
    threshold: float = 1.5,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Detect outliers in a dataset using the definition method.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data.
    distance_metric : Callable[[np.ndarray, np.ndarray], float]
        Function to compute the distance between two data points.
    threshold : float
        Threshold for outlier detection.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    _validate_inputs(X, normalizer, distance_metric)

    # Normalize data
    X_normalized = _normalize_data(X, normalizer)

    # Compute distances
    distances = _compute_distances(X_normalized, distance_metric)

    # Detect outliers
    outliers = _detect_outliers(distances, threshold)

    # Calculate metrics
    n_outliers = np.sum(outliers)
    outlier_percentage = (n_outliers / X.shape[0]) * 100

    # Prepare output
    result = {
        "outliers": outliers,
        "distances": distances
    }

    metrics = {
        "n_outliers": n_outliers,
        "outlier_percentage": outlier_percentage
    }

    params_used = {
        "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
        "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom",
        "threshold": threshold
    }

    warnings = []
    if n_outliers == 0:
        warnings.append("No outliers detected")
    elif n_outliers == X.shape[0]:
        warnings.append("All points are considered outliers")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

# Example usage:
# from sklearn.preprocessing import StandardScaler
# X = np.random.rand(100, 5)
# result = definition_fit(X, normalizer=StandardScaler().fit_transform)

################################################################################
# detection_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def detection_methods_fit(
    data: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0,
    normalization: Optional[str] = None,
    distance_metric: str = 'euclidean',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect outliers in data using various statistical methods.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    method : str, optional
        Outlier detection method ('zscore', 'iqr', 'dbscan', 'isolation_forest')
    threshold : float, optional
        Threshold for outlier detection (method specific)
    normalization : str or None, optional
        Normalization method ('standard', 'minmax', 'robust')
    distance_metric : str, optional
        Distance metric for spatial methods ('euclidean', 'manhattan', 'cosine')
    custom_metric : callable or None, optional
        Custom distance metric function

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': array of outlier flags (0=inlier, 1=outlier)
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters actually used
        - 'warnings': list of warnings

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = detection_methods_fit(data, method='zscore', threshold=3.0)
    """
    # Validate inputs
    _validate_inputs(data, method, normalization)

    # Normalize data if requested
    normalized_data = _apply_normalization(data, normalization)

    # Get method-specific parameters with defaults
    method_params = _get_method_parameters(method, **kwargs)

    # Detect outliers using selected method
    outlier_flags = _detect_outliers(
        normalized_data,
        method=method,
        threshold=threshold,
        distance_metric=distance_metric,
        custom_metric=custom_metric,
        **method_params
    )

    # Calculate metrics
    metrics = _calculate_metrics(data, outlier_flags)

    return {
        'result': outlier_flags,
        'metrics': metrics,
        'params_used': {**method_params, 'threshold': threshold},
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    method: str,
    normalization: Optional[str]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    valid_methods = ['zscore', 'iqr', 'dbscan', 'isolation_forest']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    valid_normalizations = [None, 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply requested normalization to data."""
    if method is None:
        return data

    if method == 'standard':
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

    raise ValueError("Unknown normalization method")

def _get_method_parameters(
    method: str,
    **kwargs
) -> Dict[str, Any]:
    """Get default parameters for each detection method."""
    params = {}

    if method == 'zscore':
        pass  # threshold is the only parameter

    elif method == 'iqr':
        params['lower_quantile'] = kwargs.get('lower_quantile', 0.25)
        params['upper_quantile'] = kwargs.get('upper_quantile', 0.75)
        params['factor'] = kwargs.get('factor', 1.5)

    elif method == 'dbscan':
        params['eps'] = kwargs.get('eps', 0.5)
        params['min_samples'] = kwargs.get('min_samples', 5)

    elif method == 'isolation_forest':
        params['n_estimators'] = kwargs.get('n_estimators', 100)
        params['contamination'] = kwargs.get('contamination', 'auto')

    return params

def _detect_outliers(
    data: np.ndarray,
    method: str,
    threshold: float,
    distance_metric: str,
    custom_metric: Optional[Callable],
    **kwargs
) -> np.ndarray:
    """Detect outliers using the selected method."""
    if method == 'zscore':
        return _detect_zscore_outliers(data, threshold)

    elif method == 'iqr':
        return _detect_iqr_outliers(data, **kwargs)

    elif method == 'dbscan':
        return _detect_dbscan_outliers(data, distance_metric, custom_metric, **kwargs)

    elif method == 'isolation_forest':
        return _detect_isolation_forest_outliers(data, **kwargs)

    raise ValueError("Unknown detection method")

def _detect_zscore_outliers(
    data: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8))
    return (z_scores > threshold).any(axis=1).astype(int)

def _detect_iqr_outliers(
    data: np.ndarray,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    factor: float = 1.5
) -> np.ndarray:
    """Detect outliers using IQR method."""
    q_low = np.percentile(data, lower_quantile * 100, axis=0)
    q_high = np.percentile(data, upper_quantile * 100, axis=0)
    iqr = q_high - q_low
    lower_bound = q_low - factor * iqr
    upper_bound = q_high + factor * iqr

    return ((data < lower_bound) | (data > upper_bound)).any(axis=1).astype(int)

def _detect_dbscan_outliers(
    data: np.ndarray,
    distance_metric: str,
    custom_metric: Optional[Callable],
    eps: float = 0.5,
    min_samples: int = 5
) -> np.ndarray:
    """Detect outliers using DBSCAN method."""
    # This is a simplified version - in practice you would use sklearn
    if custom_metric:
        distance_matrix = _compute_distance_matrix(data, custom_metric)
    else:
        distance_matrix = _compute_distance_matrix(data, distance_metric)

    # Simplified DBSCAN implementation
    labels = np.zeros(len(data))
    cluster_id = 1

    for i in range(len(data)):
        if labels[i] != -1:
            continue

        neighbors = np.where(distance_matrix[i] <= eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
            cluster_id += 1

    return (labels == -1).astype(int)

def _detect_isolation_forest_outliers(
    data: np.ndarray,
    n_estimators: int = 100,
    contamination: Union[str, float] = 'auto'
) -> np.ndarray:
    """Detect outliers using Isolation Forest method."""
    # This is a simplified version - in practice you would use sklearn
    scores = np.random.rand(len(data))  # Placeholder for actual isolation forest scores

    if contamination == 'auto':
        threshold = np.percentile(scores, 95)
    else:
        threshold = np.quantile(scores, contamination)

    return (scores > threshold).astype(int)

def _compute_distance_matrix(
    data: np.ndarray,
    metric: Union[str, Callable]
) -> np.ndarray:
    """Compute pairwise distance matrix."""
    if callable(metric):
        return _compute_custom_distance_matrix(data, metric)

    n = data.shape[0]
    distance_matrix = np.zeros((n, n))

    if metric == 'euclidean':
        for i in range(n):
            distance_matrix[i] = np.linalg.norm(data - data[i], axis=1)

    elif metric == 'manhattan':
        for i in range(n):
            distance_matrix[i] = np.sum(np.abs(data - data[i]), axis=1)

    elif metric == 'cosine':
        for i in range(n):
            dot_products = np.dot(data, data[i])
            norms = np.linalg.norm(data, axis=1) * np.linalg.norm(data[i])
            distance_matrix[i] = 1 - dot_products / (norms + 1e-8)

    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return distance_matrix

def _compute_custom_distance_matrix(
    data: np.ndarray,
    metric_func: Callable
) -> np.ndarray:
    """Compute distance matrix using custom metric function."""
    n = data.shape[0]
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            distance_matrix[i,j] = metric_func(data[i], data[j])
            distance_matrix[j,i] = distance_matrix[i,j]

    return distance_matrix

def _calculate_metrics(
    data: np.ndarray,
    outlier_flags: np.ndarray
) -> Dict[str, float]:
    """Calculate various metrics about the outliers."""
    n_outliers = np.sum(outlier_flags)
    outlier_ratio = n_outliers / len(data)

    # Calculate mean distance of outliers to their nearest neighbor
    if n_outliers > 0:
        outlier_indices = np.where(outlier_flags == 1)[0]
        distances = []
        for idx in outlier_indices:
            dists = np.linalg.norm(data - data[idx], axis=1)
            nearest_dist = np.min(dists[dists > 0]) if len(dists[dists > 0]) > 0 else np.inf
            distances.append(nearest_dist)
        mean_outlier_distance = np.mean(distances)
    else:
        mean_outlier_distance = 0.0

    return {
        'n_outliers': n_outliers,
        'outlier_ratio': outlier_ratio,
        'mean_outlier_distance': mean_outlier_distance
    }

################################################################################
# z_score
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for z-score calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(
    data: np.ndarray,
    normalization: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data based on specified method."""
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == "none":
        return data
    elif normalization == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif normalization == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _calculate_z_scores(
    data: np.ndarray,
    threshold: float = 3.0
) -> Dict[str, Any]:
    """Calculate z-scores and identify outliers."""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std

    outliers_mask = np.abs(z_scores) > threshold
    outliers_indices = np.where(outliers_mask)[0]
    outliers_values = data[outliers_mask]

    return {
        "z_scores": z_scores,
        "outliers_indices": outliers_indices,
        "outliers_values": outliers_values,
        "mean": mean,
        "std": std
    }

def z_score_fit(
    data: np.ndarray,
    normalization: str = "standard",
    threshold: float = 3.0,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Calculate z-scores and identify outliers in the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    threshold : float, optional
        Threshold for outlier detection (default: 3.0).
    custom_normalization : callable, optional
        Custom normalization function.

    Returns:
    --------
    dict
        Dictionary containing z-scores, outliers information, and other metrics.

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    >>> result = z_score_fit(data)
    """
    _validate_inputs(data)

    normalized_data = _normalize_data(
        data,
        normalization=normalization,
        custom_normalization=custom_normalization
    )

    result = _calculate_z_scores(normalized_data, threshold)

    return {
        "result": result,
        "metrics": {
            "mean": np.mean(data),
            "std": np.std(data),
            "outliers_count": len(result["outliers_indices"])
        },
        "params_used": {
            "normalization": normalization,
            "threshold": threshold
        },
        "warnings": []
    }

################################################################################
# iqr
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for IQR computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_iqr(
    data: np.ndarray,
    q1_func: Callable[[np.ndarray], float],
    q3_func: Callable[[np.ndarray], float]
) -> Dict[str, Any]:
    """Compute IQR and related statistics."""
    q1 = q1_func(data)
    q3 = q3_func(data)
    iqr_value = q3 - q1
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value

    return {
        "iqr": iqr_value,
        "q1": q1,
        "q3": q3,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

def _default_q1_func(data: np.ndarray) -> float:
    """Default Q1 (25th percentile) computation."""
    return np.percentile(data, 25)

def _default_q3_func(data: np.ndarray) -> float:
    """Default Q3 (75th percentile) computation."""
    return np.percentile(data, 75)

def iqr_fit(
    data: np.ndarray,
    q1_func: Optional[Callable[[np.ndarray], float]] = None,
    q3_func: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute Interquartile Range (IQR) and related statistics.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1-dimensional)
    q1_func : Callable[[np.ndarray], float], optional
        Custom function to compute Q1 (25th percentile)
    q3_func : Callable[[np.ndarray], float], optional
        Custom function to compute Q3 (75th percentile)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing IQR statistics and bounds

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> result = iqr_fit(data)
    """
    _validate_input(data)

    q1_func = q1_func if q1_func is not None else _default_q1_func
    q3_func = q3_func if q3_func is not None else _default_q3_func

    result = _compute_iqr(data, q1_func, q3_func)

    return {
        "result": result,
        "metrics": {},
        "params_used": {
            "q1_func": q1_func.__name__ if hasattr(q1_func, "__name__") else "custom",
            "q3_func": q3_func.__name__ if hasattr(q3_func, "__name__") else "custom"
        },
        "warnings": []
    }

################################################################################
# mahalanobis_distance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_normalizer is not None:
        return custom_normalizer(X)

    X_norm = X.copy()
    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / ((max_val - min_val) + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm

def compute_covariance_matrix(
    X: np.ndarray,
    regularization: str = "none",
    alpha: float = 1.0
) -> np.ndarray:
    """Compute covariance matrix with optional regularization."""
    cov = np.cov(X, rowvar=False)

    if regularization == "l1":
        cov += alpha * np.eye(cov.shape[0])
    elif regularization == "l2":
        cov += alpha * np.eye(cov.shape[0])
    elif regularization == "elasticnet":
        cov += alpha * (np.eye(cov.shape[0]) + np.ones_like(cov))
    elif regularization != "none":
        raise ValueError(f"Unknown regularization method: {regularization}")

    return cov

def mahalanobis_distance_fit(
    X: np.ndarray,
    normalize_method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    regularization: str = "none",
    alpha: float = 1.0
) -> Dict[str, Any]:
    """
    Compute Mahalanobis distances for outlier detection.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalizer : callable, optional
        Custom normalization function
    regularization : str
        Regularization method ('none', 'l1', 'l2', 'elasticnet')
    alpha : float
        Regularization strength

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = mahalanobis_distance_fit(X)
    """
    # Validate inputs
    validate_inputs(X)

    # Normalize data
    X_norm = normalize_data(X, method=normalize_method, custom_normalizer=custom_normalizer)

    # Compute covariance matrix
    cov = compute_covariance_matrix(X_norm, regularization=regularization, alpha=alpha)

    # Compute inverse covariance matrix
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular. Try different regularization.")

    # Compute Mahalanobis distances
    mean = np.mean(X_norm, axis=0)
    diff = X_norm - mean
    distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

    # Prepare results
    result = {
        "result": distances,
        "metrics": {
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances)
        },
        "params_used": {
            "normalize_method": normalize_method,
            "regularization": regularization,
            "alpha": alpha
        },
        "warnings": []
    }

    return result

def mahalanobis_distance_compute(
    X: np.ndarray,
    mean: np.ndarray,
    inv_covariance: np.ndarray
) -> np.ndarray:
    """
    Compute Mahalanobis distances given mean and inverse covariance matrix.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    mean : np.ndarray
        Mean vector of the distribution
    inv_covariance : np.ndarray
        Inverse covariance matrix

    Returns:
    --------
    np.ndarray
        Array of Mahalanobis distances
    """
    validate_inputs(X)
    if mean.shape[0] != X.shape[1]:
        raise ValueError("Mean vector dimension mismatch")
    if inv_covariance.shape != (X.shape[1], X.shape[1]):
        raise ValueError("Inverse covariance matrix dimension mismatch")

    diff = X - mean
    distances = np.sqrt(np.sum(diff @ inv_covariance * diff, axis=1))
    return distances

################################################################################
# dbscan
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def dbscan_fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    normalize: Optional[str] = None,
    algorithm: str = 'auto'
) -> Dict[str, Any]:
    """
    DBSCAN clustering algorithm for outlier detection.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    normalize : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    algorithm : str, optional
        The algorithm to use for neighbor search. Can be 'auto', 'ball_tree', or 'kd_tree'.

    Returns:
    --------
    dict
        A dictionary containing the clustering results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(X, eps, min_samples)

    # Normalize data if required
    X_normalized = apply_normalization(X, normalize)

    # Choose distance metric
    distance_func = get_distance_function(metric)

    # Perform DBSCAN clustering
    labels = _dbscan_core(X_normalized, eps, min_samples, distance_func, algorithm)

    # Calculate metrics
    metrics = calculate_metrics(X_normalized, labels, distance_func)

    # Prepare output
    result = {
        'result': {
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'outliers': np.sum(labels == -1)
        },
        'metrics': metrics,
        'params_used': {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'normalize': normalize,
            'algorithm': algorithm
        },
        'warnings': []
    }

    return result

def validate_inputs(
    X: np.ndarray,
    eps: float,
    min_samples: int
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

def apply_normalization(
    X: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply normalization to the input data."""
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

def get_distance_function(
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the metric parameter."""
    if callable(metric):
        return metric
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _dbscan_core(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    algorithm: str
) -> np.ndarray:
    """Core DBSCAN clustering algorithm."""
    labels = np.full(X.shape[0], -1, dtype=int)
    cluster_id = 0

    for i in range(X.shape[0]):
        if labels[i] != -1:
            continue

        neighbors = _region_query(X, i, eps, distance_func, algorithm)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            _expand_cluster(X, i, neighbors, cluster_id, labels, eps, min_samples, distance_func)
            cluster_id += 1

    return labels

def _region_query(
    X: np.ndarray,
    query_index: int,
    eps: float,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    algorithm: str
) -> np.ndarray:
    """Find all points within eps distance of the query point."""
    if algorithm == 'auto':
        # Use a simple implementation for demonstration
        distances = np.array([distance_func(X[query_index], x) for x in X])
        return np.where(distances <= eps)[0]
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented")

def _expand_cluster(
    X: np.ndarray,
    start_index: int,
    neighbors: np.ndarray,
    cluster_id: int,
    labels: np.ndarray,
    eps: float,
    min_samples: int,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> None:
    """Expand the cluster from the start point."""
    labels[start_index] = cluster_id
    i = 0

    while i < len(neighbors):
        point_idx = neighbors[i]

        if labels[point_idx] == -1:
            labels[point_idx] = cluster_id

        if labels[point_idx] == -1 or labels[point_idx] != cluster_id:
            labels[point_idx] = cluster_id

        if labels[point_idx] == cluster_id:
            new_neighbors = _region_query(X, point_idx, eps, distance_func, 'auto')
            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, new_neighbors))
        i += 1

def calculate_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Any]:
    """Calculate clustering metrics."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)

    metrics = {
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'silhouette_score': _calculate_silhouette(X, labels, distance_func)
    }

    return metrics

def _calculate_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Calculate silhouette score for clustering."""
    # Simplified implementation
    n_samples = X.shape[0]
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        if labels[i] == -1:
            continue

        a = np.mean([distance_func(X[i], X[j]) for j in range(n_samples) if labels[j] == labels[i] and i != j])
        b = np.min([np.mean([distance_func(X[i], X[j]) for j in range(n_samples) if labels[j] == l])
                    for l in set(labels) if l != -1 and l != labels[i]])

        silhouette_scores[i] = (b - a) / max(a, b)

    return np.mean(silhouette_scores)

################################################################################
# isolation_forest
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def isolation_forest_fit(
    X: np.ndarray,
    n_estimators: int = 100,
    max_samples: Union[int, float] = 'auto',
    contamination: Optional[float] = None,
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0
) -> Dict:
    """
    Fit the Isolation Forest model.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : int or float, default='auto'
        The number of samples to draw from X to train each base estimator.
    contamination : float, default=None
        The expected proportion of outliers in the data.
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
    bootstrap : bool, default=False
        Whether bootstrap samples are used when building trees.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel.
    random_state : int or None, default=None
        Controls the randomness of the estimator.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Returns:
    --------
    dict
        A dictionary containing the fitted model, metrics, parameters used,
        and any warnings.
    """
    # Validate inputs
    _validate_inputs(X, n_estimators, max_samples, contamination,
                     max_features, bootstrap)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Initialize results dictionary
    results = {
        "result": None,
        "metrics": {},
        "params_used": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "n_jobs": n_jobs,
            "random_state": random_state
        },
        "warnings": []
    }

    # Fit the isolation forest
    results["result"] = _fit_isolation_forest(
        X, n_estimators, max_samples, contamination,
        max_features, bootstrap, rng
    )

    # Compute metrics
    results["metrics"] = _compute_metrics(results["result"], X)

    return results

def _validate_inputs(
    X: np.ndarray,
    n_estimators: int,
    max_samples: Union[int, float],
    contamination: Optional[float],
    max_features: Union[int, float],
    bootstrap: bool
) -> None:
    """Validate the inputs for isolation forest."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer.")
    if contamination is not None and (contamination <= 0 or contamination >= 1):
        raise ValueError("contamination must be in (0, 1).")
    if max_features <= 0 or max_features > 1:
        raise ValueError("max_features must be in (0, 1].")

def _fit_isolation_forest(
    X: np.ndarray,
    n_estimators: int,
    max_samples: Union[int, float],
    contamination: Optional[float],
    max_features: Union[int, float],
    bootstrap: bool,
    rng: np.random.RandomState
) -> Dict:
    """Fit the isolation forest model."""
    n_samples, n_features = X.shape

    # Set max_samples if 'auto'
    if max_samples == 'auto':
        max_samples = min(256, n_samples)

    # Initialize trees
    trees = []
    for _ in range(n_estimators):
        # Sample data
        if bootstrap:
            sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        else:
            sample_idx = rng.choice(n_samples, size=min(max_samples, n_samples), replace=False)

        X_sample = X[sample_idx]

        # Sample features
        if max_features < 1.0:
            n_max_features = int(max_features * n_features)
        else:
            n_max_features = int(max_features)

        feature_idx = rng.choice(n_features, size=n_max_features, replace=False)
        X_sample = X_sample[:, feature_idx]

        # Build tree
        tree = _build_isolation_tree(X_sample, rng)
        trees.append(tree)

    return {"trees": trees}

def _build_isolation_tree(
    X: np.ndarray,
    rng: np.random.RandomState
) -> Dict:
    """Build a single isolation tree."""
    n_samples, n_features = X.shape

    # Base case: if all samples are the same or only one sample remains
    if n_samples == 0:
        return {"is_leaf": True, "path_length": float('inf')}
    if n_features == 0 or np.all(X[1:] == X[0]):
        return {"is_leaf": True, "path_length": 0}

    # Select a random feature
    feature_idx = rng.choice(n_features)
    feature_values = X[:, feature_idx]

    # Select a random split value
    min_val, max_val = np.min(feature_values), np.max(feature_values)
    split_value = rng.uniform(min_val, max_val)

    # Split the data
    left_mask = feature_values < split_value
    right_mask = ~left_mask

    # Recursively build left and right subtrees
    left_subtree = _build_isolation_tree(X[left_mask], rng)
    right_subtree = _build_isolation_tree(X[right_mask], rng)

    # Calculate path length
    left_path_length = left_subtree["path_length"] + 1
    right_path_length = right_subtree["path_length"] + 1

    return {
        "is_leaf": False,
        "feature_idx": feature_idx,
        "split_value": split_value,
        "left_subtree": left_subtree,
        "right_subtree": right_subtree,
        "path_length": min(left_path_length, right_path_length)
    }

def _compute_metrics(
    model: Dict,
    X: np.ndarray
) -> Dict:
    """Compute metrics for the isolation forest model."""
    anomaly_scores = _compute_anomaly_scores(model, X)
    return {
        "mean_anomaly_score": np.mean(anomaly_scores),
        "std_anomaly_score": np.std(anomaly_scores)
    }

def _compute_anomaly_scores(
    model: Dict,
    X: np.ndarray
) -> np.ndarray:
    """Compute anomaly scores for the input data."""
    n_samples = X.shape[0]
    anomaly_scores = np.zeros(n_samples)

    for i in range(n_samples):
        x = X[i]
        scores = []
        for tree in model["trees"]:
            path_length = _compute_path_length(tree, x)
            scores.append(path_length)

        # Average path length across all trees
        avg_path_length = np.mean(scores)
        anomaly_scores[i] = 2 ** (-avg_path_length / _c(model["trees"][0]["path_length"]))

    return anomaly_scores

def _compute_path_length(
    tree: Dict,
    x: np.ndarray
) -> float:
    """Compute the path length for a single sample in a tree."""
    if tree["is_leaf"]:
        return tree["path_length"]

    feature_idx = tree["feature_idx"]
    split_value = tree["split_value"]

    if x[feature_idx] < split_value:
        return _compute_path_length(tree["left_subtree"], x)
    else:
        return _compute_path_length(tree["right_subtree"], x)

def _c(path_length: float) -> float:
    """Compute the normalization factor c."""
    return 2 * (np.log(2) - 1)

################################################################################
# local_outlier_factor
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

def default_distance_metric(x: np.ndarray, y: np.ndarray) -> float:
    """Default Euclidean distance metric."""
    return np.linalg.norm(x - y)

def local_outlier_factor_fit(
    X: np.ndarray,
    n_neighbors: int = 20,
    metric: Union[str, Callable] = 'euclidean',
    algorithm: str = 'auto',
    leaf_size: int = 30,
    p: float = 2.0,
    metric_params: Optional[Dict] = None,
    normalizer: Callable = None
) -> Dict:
    """
    Compute the Local Outlier Factor (LOF) for each sample in X.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_neighbors : int, optional
        Number of neighbors to consider (default: 20).
    metric : str or callable, optional
        Distance metric to use (default: 'euclidean').
    algorithm : str, optional
        Algorithm to compute nearest neighbors (default: 'auto').
    leaf_size : int, optional
        Leaf size for BallTree or KDTree (default: 30).
    p : float, optional
        Parameter for Minkowski metric (default: 2.0).
    metric_params : dict, optional
        Additional parameters for the distance metric (default: None).
    normalizer : callable, optional
        Function to normalize the data before computation (default: None).

    Returns:
    --------
    dict
        Dictionary containing the LOF scores, metrics, parameters used, and warnings.
    """
    # Validate input
    validate_input(X)

    if normalizer is not None:
        X = normalizer(X.copy())

    # Set default metric parameters if not provided
    if metric_params is None:
        metric_params = {}

    # Define distance metric function
    if callable(metric):
        distance_metric = metric
    elif isinstance(metric, str):
        if metric == 'euclidean':
            distance_metric = lambda x, y: np.linalg.norm(x - y)
        elif metric == 'manhattan':
            distance_metric = lambda x, y: np.sum(np.abs(x - y))
        elif metric == 'cosine':
            distance_metric = lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        elif metric == 'minkowski':
            distance_metric = lambda x, y: np.sum(np.abs(x - y) ** p) ** (1 / p)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        raise TypeError("Metric must be a string or callable")

    # Compute pairwise distances
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = distance_metric(X[i], X[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Compute k-distance for each sample
    k_distances = np.zeros(n_samples)
    for i in range(n_samples):
        distances = distance_matrix[i]
        k_distances[i] = np.sort(distances)[n_neighbors]

    # Compute reachability distance
    reachability_distance = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if distance_matrix[i, j] <= k_distances[j]:
                reachability_distance[i, j] = distance_matrix[i, j]
            else:
                reachability_distance[i, j] = k_distances[j]

    # Compute local reachability density
    lrd = np.zeros(n_samples)
    for i in range(n_samples):
        neighbors = np.argsort(distance_matrix[i])[1:n_neighbors + 1]
        lrd[i] = n_neighbors / np.sum(reachability_distance[neighbors, i])

    # Compute LOF scores
    lof_scores = np.zeros(n_samples)
    for i in range(n_samples):
        neighbors = np.argsort(distance_matrix[i])[1:n_neighbors + 1]
        lof_scores[i] = np.mean(lrd[neighbors]) / lrd[i]

    return {
        "result": lof_scores,
        "metrics": {"n_neighbors": n_neighbors, "algorithm": algorithm},
        "params_used": {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "algorithm": algorithm,
            "leaf_size": leaf_size,
            "p": p
        },
        "warnings": []
    }

# Example usage:
# lof_result = local_outlier_factor_fit(X, n_neighbors=20)

################################################################################
# one_class_svm
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input contains NaN or infinite values")

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

def compute_kernel(X: np.ndarray, kernel: str = 'rbf', gamma: float = 1.0) -> np.ndarray:
    """Compute kernel matrix."""
    if kernel == 'linear':
        return X @ X.T
    elif kernel == 'rbf':
        pairwise_sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
        return np.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'poly':
        return (X @ X.T + 1) ** gamma
    elif kernel == 'sigmoid':
        return np.tanh(gamma * X @ X.T + 1)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

def one_class_svm_fit(
    X: np.ndarray,
    nu: float = 0.1,
    kernel: str = 'rbf',
    gamma: float = 1.0,
    normalize_method: str = 'standard',
    tol: float = 1e-3,
    max_iter: int = 1000,
    custom_kernel_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit One-Class SVM model.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    nu : float
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    kernel : str
        Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    custom_kernel_func : callable, optional
        Custom kernel function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': fitted model parameters
        - 'metrics': performance metrics
        - 'params_used': parameters used for fitting
        - 'warnings': any warnings encountered

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> result = one_class_svm_fit(X, nu=0.1)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_normalized = normalize_data(X, method=normalize_method)

    # Compute kernel matrix
    if custom_kernel_func is not None:
        K = custom_kernel_func(X_normalized)
    else:
        K = compute_kernel(X_normalized, kernel=kernel, gamma=gamma)

    # Initialize variables
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    errors = np.ones(n_samples)

    # SVM parameters
    C = 1.0 / (nu * n_samples)
    b = 0.0

    # Training loop
    for _ in range(max_iter):
        alpha_prev = np.copy(alpha)

        # Compute errors
        K_alpha = (K * alpha[:, None]).sum(axis=0)
        errors = 1 - K_alpha + b

        # Update alphas
        for i in range(n_samples):
            if (0 < alpha[i] < C) and (errors[i] > tol or errors[i] < -tol):
                # Compute gradient
                grad = K[i, i] * alpha[i] - np.dot(K[i], alpha) + b

                # Update alpha
                alpha_new = alpha[i] - errors[i] / (K[i, i] + 1e-8)
                alpha_new = np.clip(alpha_new, 0, C)

                # Update bias
                b_new = b - errors[i] + alpha[i] * K[i, i] - alpha_new * K[i, i]

                # Update variables
                alpha[i] = alpha_new
                b = b_new

        # Check convergence
        diff = np.linalg.norm(alpha - alpha_prev)
        if diff < tol:
            break

    # Compute support vectors
    sv_indices = alpha > 1e-5
    support_vectors = X_normalized[sv_indices]
    sv_alphas = alpha[sv_indices]

    # Compute decision function
    def decision_function(x: np.ndarray) -> float:
        x_normalized = normalize_data(x.reshape(1, -1), method=normalize_method)
        if custom_kernel_func is not None:
            k = custom_kernel_func(x_normalized)
        else:
            k = compute_kernel(x_normalized, kernel=kernel, gamma=gamma)
        return np.dot(k[0], sv_alphas) - b

    # Compute metrics
    def compute_metrics():
        distances = np.abs(decision_function(X_normalized))
        return {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances)
        }

    metrics = compute_metrics()

    # Return results
    return {
        'result': {
            'decision_function': decision_function,
            'support_vectors': support_vectors,
            'alphas': sv_alphas
        },
        'metrics': metrics,
        'params_used': {
            'nu': nu,
            'kernel': kernel if custom_kernel_func is None else 'custom',
            'gamma': gamma,
            'normalize_method': normalize_method
        },
        'warnings': []
    }

################################################################################
# visualization_techniques
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for visualization techniques."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(
    data: np.ndarray,
    normalization: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == "none":
        return data
    elif normalization == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif normalization == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _calculate_metrics(
    data: np.ndarray,
    outliers: np.ndarray,
    metrics: Union[str, List[str], Callable[[np.ndarray, np.ndarray], Dict[str, float]]],
    custom_metrics: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None
) -> Dict[str, float]:
    """Calculate specified metrics for outlier detection."""
    if custom_metrics is not None:
        return custom_metrics(data, outliers)

    result = {}
    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == "mse":
            result["mse"] = np.mean((data - outliers) ** 2)
        elif metric == "mae":
            result["mae"] = np.mean(np.abs(data - outliers))
        elif metric == "r2":
            ss_res = np.sum((data - outliers) ** 2)
            ss_tot = np.sum((data - np.mean(data, axis=0)) ** 2)
            result["r2"] = 1 - (ss_res / (ss_tot + 1e-8))
        elif metric == "logloss":
            result["logloss"] = -np.mean(data * np.log(outliers + 1e-8) + (1 - data) * np.log(1 - outliers + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result

def _detect_outliers(
    data: np.ndarray,
    method: str = "zscore",
    threshold: float = 3.0,
    custom_outlier_detection: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Detect outliers using specified method or custom function."""
    if custom_outlier_detection is not None:
        return custom_outlier_detection(data)

    if method == "zscore":
        z_scores = np.abs((data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8))
        return np.where(z_scores > threshold, data, np.nan)
    elif method == "iqr":
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return np.where((data < lower_bound) | (data > upper_bound), data, np.nan)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN().fit(data)
        return np.where(clustering.labels_ == -1, data, np.nan)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def visualization_techniques_fit(
    data: np.ndarray,
    normalization: str = "standard",
    outlier_method: str = "zscore",
    threshold: float = 3.0,
    metrics: Union[str, List[str]] = "mse",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_outlier_detection: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metrics: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Main function for outlier visualization techniques.

    Parameters:
    - data: Input data as 2D numpy array
    - normalization: Normalization method ("none", "standard", "minmax", "robust")
    - outlier_method: Method for outlier detection ("zscore", "iqr", "dbscan")
    - threshold: Threshold for outlier detection
    - metrics: Metrics to calculate ("mse", "mae", "r2", "logloss")
    - custom_normalization: Custom normalization function
    - custom_outlier_detection: Custom outlier detection function
    - custom_metrics: Custom metrics calculation function

    Returns:
    Dictionary containing results, metrics, parameters used, and warnings
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        normalization=normalization,
        custom_normalization=custom_normalization
    )

    # Detect outliers
    outliers = _detect_outliers(
        normalized_data,
        method=outlier_method,
        threshold=threshold,
        custom_outlier_detection=custom_outlier_detection
    )

    # Calculate metrics
    calculated_metrics = _calculate_metrics(
        normalized_data,
        outliers,
        metrics=metrics,
        custom_metrics=custom_metrics
    )

    # Prepare results
    result = {
        "result": outliers,
        "metrics": calculated_metrics,
        "params_used": {
            "normalization": normalization,
            "outlier_method": outlier_method,
            "threshold": threshold,
            "metrics": metrics
        },
        "warnings": []
    }

    # Check for warnings
    if np.all(np.isnan(outliers)):
        result["warnings"].append("No outliers detected with current parameters")

    return result

################################################################################
# handling_strategies
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def handling_strategies_fit(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]:
    """
    Main function to handle outlier detection strategies.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to use. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric. Can be "euclidean", "manhattan", "cosine", or a custom callable.
    solver : str, optional
        Solver method. Can be "closed_form", "gradient_descent", etc.
    regularization : Optional[str], optional
        Regularization type. Can be "l1", "l2", or None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str], list]]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve the problem based on the chosen solver
    result = _solve_problem(
        normalized_data,
        metric_func,
        distance_func,
        solver,
        regularization,
        tol,
        max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, result, metric_func)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

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

def _get_metric_function(
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the input."""
    if callable(metric):
        return metric
    if custom_metric is not None:
        return custom_metric

    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared
    }

    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the input."""
    if callable(distance):
        return distance
    if custom_distance is not None:
        return custom_distance

    distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance
    }

    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _solve_problem(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the outlier detection problem based on the chosen solver."""
    solvers = {
        "closed_form": _solve_closed_form,
        "gradient_descent": _solve_gradient_descent
    }

    if solver not in solvers:
        raise ValueError(f"Unknown solver: {solver}")
    return solvers[solver](data, metric_func, distance_func, regularization, tol, max_iter)

def _calculate_metrics(
    data: np.ndarray,
    result: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the result."""
    return {
        "metric_value": metric_func(data, result)
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

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _solve_closed_form(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the problem using closed-form solution."""
    # Placeholder for actual implementation
    return np.zeros(data.shape[1])

def _solve_gradient_descent(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Solve the problem using gradient descent."""
    # Placeholder for actual implementation
    return np.zeros(data.shape[1])

################################################################################
# domain_specific_outliers
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def domain_specific_outliers_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Fit a domain-specific outlier detection model.

    Parameters:
    - X: Input data matrix of shape (n_samples, n_features)
    - y: Optional target vector of shape (n_samples,) for supervised methods
    - normalization: Normalization method ('none', 'standard', 'minmax', 'robust')
    - metric: Metric to use ('mse', 'mae', 'r2', 'logloss') or custom callable
    - distance: Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski') or custom callable
    - solver: Solver method ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent')
    - regularization: Regularization type ('none', 'l1', 'l2', 'elasticnet')
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    - custom_metric: Custom metric function if needed
    - custom_distance: Custom distance function if needed

    Returns:
    A dictionary containing:
    - 'result': Fitted model parameters
    - 'metrics': Computed metrics
    - 'params_used': Parameters used during fitting
    - 'warnings': Any warnings generated

    Example:
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> result = domain_specific_outliers_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_normalized = _apply_normalization(X, normalization)

    # Initialize parameters
    params = {
        'normalization': normalization,
        'metric': metric,
        'distance': distance,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Fit model based on solver choice
    if solver == 'closed_form':
        result = _fit_closed_form(X_normalized, y)
    elif solver == 'gradient_descent':
        result = _fit_gradient_descent(X_normalized, y, tol, max_iter)
    elif solver == 'newton':
        result = _fit_newton(X_normalized, y, tol, max_iter)
    elif solver == 'coordinate_descent':
        result = _fit_coordinate_descent(X_normalized, y, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, y, result, metric, custom_metric)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None")
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

def _fit_closed_form(X: np.ndarray, y: Optional[np.ndarray]) -> Dict[str, Any]:
    """Fit model using closed-form solution."""
    if y is None:
        raise ValueError("Closed form solver requires target y")
    X_tx = np.dot(X.T, X)
    if np.linalg.det(X_tx) == 0:
        raise ValueError("Matrix is singular")
    params = np.linalg.inv(X_tx).dot(X.T).dot(y)
    return {'params': params}

def _fit_gradient_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit model using gradient descent."""
    if y is None:
        raise ValueError("Gradient descent requires target y")
    n_samples, n_features = X.shape
    params = np.zeros(n_features)
    learning_rate = 0.01

    for _ in range(max_iter):
        gradient = 2/n_samples * X.T.dot(X.dot(params) - y)
        new_params = params - learning_rate * gradient
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return {'params': params}

def _fit_newton(
    X: np.ndarray,
    y: Optional[np.ndarray],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit model using Newton's method."""
    if y is None:
        raise ValueError("Newton's method requires target y")
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        gradient = 2/n_samples * X.T.dot(X.dot(params) - y)
        hessian = 2/n_samples * X.T.dot(X)
        new_params = params - np.linalg.inv(hessian).dot(gradient)
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    return {'params': params}

def _fit_coordinate_descent(
    X: np.ndarray,
    y: Optional[np.ndarray],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit model using coordinate descent."""
    if y is None:
        raise ValueError("Coordinate descent requires target y")
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            X_j = X[:, j]
            residuals = y - np.dot(X, params) + params[j] * X_j
            params[j] = np.sum(X_j * residuals) / np.sum(X_j**2)
        if np.linalg.norm(params - params) < tol:
            break

    return {'params': params}

def _compute_metrics(
    X: np.ndarray,
    y: Optional[np.ndarray],
    result: Dict[str, Any],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute specified metrics."""
    if y is None:
        raise ValueError("Metrics computation requires target y")

    predictions = X.dot(result['params'])
    metrics_dict = {}

    if metric == 'mse' or (custom_metric is None and metric == 'mse'):
        metrics_dict['mse'] = np.mean((predictions - y)**2)
    elif metric == 'mae' or (custom_metric is None and metric == 'mae'):
        metrics_dict['mae'] = np.mean(np.abs(predictions - y))
    elif metric == 'r2' or (custom_metric is None and metric == 'r2'):
        ss_res = np.sum((y - predictions)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        metrics_dict['r2'] = 1 - ss_res / ss_tot
    elif metric == 'logloss' or (custom_metric is None and metric == 'logloss'):
        metrics_dict['logloss'] = -np.mean(y * np.log(predictions) + (1-y) * np.log(1-predictions))
    elif callable(metric) or custom_metric:
        metric_func = custom_metric if custom_metric else metric
        metrics_dict['custom'] = metric_func(y, predictions)

    return metrics_dict

def _compute_distance(
    X: np.ndarray,
    distance_metric: Union[str, Callable],
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """Compute distance matrix using specified metric."""
    if callable(distance_metric) or custom_distance:
        dist_func = custom_distance if custom_distance else distance_metric
        return dist_func(X)
    elif distance_metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif distance_metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif distance_metric == 'cosine':
        dot_products = np.dot(X, X.T)
        norms = np.linalg.norm(X, axis=1)[:, np.newaxis]
        return 1 - dot_products / (norms * norms.T)
    elif distance_metric == 'minkowski':
        return np.sum(np.abs(X[:, np.newaxis] - X) ** 3, axis=2) ** (1/3)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
