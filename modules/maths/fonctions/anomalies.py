"""
Quantix – Module anomalies
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# outlier_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def outlier_detection_fit(
    X: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0,
    normalize: bool = True,
    distance_metric: Union[str, Callable] = 'euclidean',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Detect outliers in a dataset using various statistical methods.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    method : str, optional
        Method for outlier detection. Options: 'zscore', 'iqr', 'mahalanobis', 'dbscan'.
    threshold : float, optional
        Threshold for outlier detection.
    normalize : bool, optional
        Whether to normalize the data before detection.
    distance_metric : str or callable, optional
        Distance metric for methods like Mahalanobis. Options: 'euclidean', 'manhattan', 'cosine'.
    custom_metric : callable, optional
        Custom distance metric function.

    Returns:
    --------
    Dict containing:
        - 'result': bool array indicating outliers
        - 'metrics': dict of computed metrics
        - 'params_used': dict of parameters used
        - 'warnings': list of warnings

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = outlier_detection_fit(X, method='zscore', threshold=3.0)
    """
    # Validate inputs
    validate_inputs(X)

    # Normalize data if required
    if normalize:
        X = apply_normalization(X, method='standard')

    # Initialize result dictionary
    result_dict: Dict = {
        'result': np.zeros(X.shape[0], dtype=bool),
        'metrics': {},
        'params_used': {
            'method': method,
            'threshold': threshold,
            'normalize': normalize,
            'distance_metric': distance_metric
        },
        'warnings': []
    }

    # Dispatch to appropriate method
    if method == 'zscore':
        result_dict['result'] = _detect_zscore_outliers(X, threshold)
    elif method == 'iqr':
        result_dict['result'] = _detect_iqr_outliers(X, threshold)
    elif method == 'mahalanobis':
        result_dict['result'] = _detect_mahalanobis_outliers(X, threshold, distance_metric)
    elif method == 'dbscan':
        result_dict['result'] = _detect_dbscan_outliers(X, threshold, distance_metric)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics
    result_dict['metrics'] = compute_metrics(X, result_dict['result'])

    return result_dict

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

def apply_normalization(
    X: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """Apply normalization to data."""
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

def _detect_zscore_outliers(
    X: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Detect outliers using z-score method."""
    z_scores = np.abs((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8))
    return np.any(z_scores > threshold, axis=1)

def _detect_iqr_outliers(
    X: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Detect outliers using IQR method."""
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return np.any((X < lower_bound) | (X > upper_bound), axis=1)

def _detect_mahalanobis_outliers(
    X: np.ndarray,
    threshold: float,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Detect outliers using Mahalanobis distance."""
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(X.shape[1]))
    mean = np.mean(X, axis=0)

    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            def mahalanobis(x):
                return np.sqrt(np.dot(np.dot((x - mean), inv_cov), (x - mean).T))
        elif distance_metric == 'manhattan':
            def mahalanobis(x):
                return np.sum(np.abs((x - mean) @ inv_cov))
        elif distance_metric == 'cosine':
            def mahalanobis(x):
                return 1 - np.dot((x - mean), inv_cov @ (x - mean).T) / (
                    np.linalg.norm(x - mean) * np.linalg.norm(inv_cov @ (x - mean))
                )
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    else:
        mahalanobis = distance_metric

    distances = np.array([mahalanobis(x) for x in X])
    return distances > threshold

def _detect_dbscan_outliers(
    X: np.ndarray,
    threshold: float,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Detect outliers using DBSCAN algorithm."""
    # This is a simplified version of DBSCAN
    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            def compute_distance(x1, x2):
                return np.linalg.norm(x1 - x2)
        elif distance_metric == 'manhattan':
            def compute_distance(x1, x2):
                return np.sum(np.abs(x1 - x2))
        elif distance_metric == 'cosine':
            def compute_distance(x1, x2):
                return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    else:
        compute_distance = distance_metric

    labels = np.zeros(X.shape[0])
    is_outlier = np.zeros(X.shape[0], dtype=bool)

    for i in range(X.shape[0]):
        if labels[i] != 0:
            continue

        neighbors = []
        for j in range(X.shape[0]):
            if i == j:
                continue
            dist = compute_distance(X[i], X[j])
            if dist <= threshold:
                neighbors.append(j)

        if len(neighbors) < 2:  # Simplified condition for outlier
            labels[i] = -1
            is_outlier[i] = True
        else:
            for j in neighbors:
                labels[j] = len(np.unique(labels[labels != 0])) + 1

    return is_outlier

def compute_metrics(
    X: np.ndarray,
    outliers: np.ndarray
) -> Dict:
    """Compute metrics for outlier detection."""
    return {
        'outlier_ratio': np.mean(outliers),
        'mean_distance': np.mean(np.linalg.norm(X[outliers], axis=1)),
        'std_distance': np.std(np.linalg.norm(X[outliers], axis=1))
    }

################################################################################
# statistical_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    X: np.ndarray,
    threshold_func: Callable[[np.ndarray], float],
    normalization: str = "standard",
    distance_metric: str = "euclidean"
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")

    normalization_options = ["none", "standard", "minmax", "robust"]
    if normalization not in normalization_options:
        raise ValueError(f"normalization must be one of {normalization_options}")

    distance_metrics = ["euclidean", "manhattan", "cosine", "minkowski"]
    if distance_metric not in distance_metrics:
        raise ValueError(f"distance_metric must be one of {distance_metrics}")

def _normalize_data(
    X: np.ndarray,
    normalization: str = "standard"
) -> np.ndarray:
    """Normalize the input data based on specified method."""
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

def _compute_distance(
    X: np.ndarray,
    distance_metric: str = "euclidean",
    p: float = 2.0
) -> np.ndarray:
    """Compute pairwise distances between data points."""
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))

    if distance_metric == "euclidean":
        dist_matrix = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif distance_metric == "manhattan":
        dist_matrix = np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif distance_metric == "cosine":
        dot_products = np.dot(X, X.T)
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        dist_matrix = 1 - dot_products / (np.outer(norms, norms) + 1e-8)
    elif distance_metric == "minkowski":
        dist_matrix = np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)

    return dist_matrix

def _detect_anomalies(
    X: np.ndarray,
    threshold_func: Callable[[np.ndarray], float],
    distance_metric: str = "euclidean",
    p: float = 2.0
) -> np.ndarray:
    """Detect anomalies based on distance metrics and threshold function."""
    dist_matrix = _compute_distance(X, distance_metric, p)
    mean_distances = np.mean(dist_matrix, axis=1)
    threshold = threshold_func(mean_distances)
    anomalies = mean_distances > threshold
    return anomalies

def statistical_anomaly_fit(
    X: np.ndarray,
    threshold_func: Callable[[np.ndarray], float],
    normalization: str = "standard",
    distance_metric: str = "euclidean",
    p: float = 2.0
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Detect statistical anomalies in the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    threshold_func : Callable[[np.ndarray], float]
        Function to compute the anomaly threshold from mean distances.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    distance_metric : str, optional
        Distance metric ("euclidean", "manhattan", "cosine", "minkowski").
    p : float, optional
        Power parameter for Minkowski distance.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - "result": boolean array of anomaly flags
        - "metrics": dictionary of computed metrics
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> def threshold_func(distances):
    ...     return np.percentile(distances, 95)
    >>> result = statistical_anomaly_fit(X, threshold_func)
    """
    _validate_inputs(X, threshold_func, normalization, distance_metric)

    warnings = []
    params_used = {
        "normalization": normalization,
        "distance_metric": distance_metric,
        "p": p
    }

    X_normalized = _normalize_data(X, normalization)
    anomalies = _detect_anomalies(X_normalized, threshold_func, distance_metric, p)

    metrics = {
        "n_anomalies": np.sum(anomalies),
        "anomaly_rate": np.mean(anomalies)
    }

    return {
        "result": anomalies,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# machine_learning_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def machine_learning_anomaly_fit(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect anomalies in data using machine learning approach.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalizer : Optional[Callable]
        Function to normalize the data. If None, no normalization.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    solver : str
        Solver to use ('gradient_descent', 'newton', 'coordinate_descent')
    regularization : Optional[str]
        Regularization type ('l1', 'l2', 'elasticnet')
    custom_metric : Optional[Callable]
        Custom metric function if not using built-in metrics
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize parameters
    params = {
        'distance_metric': distance_metric,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }

    # Get distance function
    distance_func = _get_distance_function(distance_metric)

    # Fit model based on solver choice
    if solver == 'gradient_descent':
        result, metrics = _gradient_descent_solver(
            X_normalized,
            distance_func,
            regularization,
            tol,
            max_iter,
            random_state
        )
    elif solver == 'newton':
        result, metrics = _newton_solver(
            X_normalized,
            distance_func,
            regularization,
            tol,
            max_iter
        )
    elif solver == 'coordinate_descent':
        result, metrics = _coordinate_descent_solver(
            X_normalized,
            distance_func,
            regularization,
            tol,
            max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, result)

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input X must be 2-dimensional")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input X contains infinite values")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to data."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get distance function based on metric name."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**3)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _gradient_descent_solver(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> tuple[np.ndarray, Dict[str, float]]:
    """Gradient descent solver for anomaly detection."""
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    # Initialize parameters
    if regularization == 'l1':
        params = np.random.randn(n_features)
    else:
        params = np.zeros(n_features)

    # Main loop
    for _ in range(max_iter):
        grad = np.zeros(n_features)
        for i in range(n_samples):
            # Compute gradient
            grad += 2 * (distance_func(X[i], params) / distance_func(X[i], np.zeros_like(params))) * (X[i] - params)

        # Apply regularization
        if regularization == 'l1':
            grad += np.sign(params)
        elif regularization == 'l2':
            grad += 2 * params

        # Update parameters
        params -= tol * grad

        # Check convergence (simplified)
        if np.linalg.norm(grad) < tol:
            break

    # Calculate metrics
    residuals = np.array([distance_func(X[i], params) for i in range(n_samples)])
    metrics = {
        'mse': np.mean(residuals**2),
        'mae': np.mean(np.abs(residuals)),
        'r2': 1 - np.sum(residuals**2) / np.sum((X - np.mean(X, axis=0))**2)
    }

    return params, metrics

def _newton_solver(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple[np.ndarray, Dict[str, float]]:
    """Newton solver for anomaly detection."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        # Compute gradient and hessian (simplified)
        grad = np.zeros(n_features)
        hess = np.eye(n_features)

        for i in range(n_samples):
            grad += 2 * (X[i] - params)
            hess += 2 * np.eye(n_features)

        # Apply regularization
        if regularization == 'l1':
            grad += np.sign(params)
        elif regularization == 'l2':
            hess += 2 * np.eye(n_features)

        # Update parameters
        params -= np.linalg.inv(hess) @ grad

        # Check convergence (simplified)
        if np.linalg.norm(grad) < tol:
            break

    # Calculate metrics
    residuals = np.array([distance_func(X[i], params) for i in range(n_samples)])
    metrics = {
        'mse': np.mean(residuals**2),
        'mae': np.mean(np.abs(residuals)),
        'r2': 1 - np.sum(residuals**2) / np.sum((X - np.mean(X, axis=0))**2)
    }

    return params, metrics

def _coordinate_descent_solver(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple[np.ndarray, Dict[str, float]]:
    """Coordinate descent solver for anomaly detection."""
    n_samples, n_features = X.shape
    params = np.zeros(n_features)

    for _ in range(max_iter):
        for j in range(n_features):
            # Compute gradient for feature j
            grad_j = 0
            for i in range(n_samples):
                grad_j += 2 * (X[i, j] - params[j])

            # Apply regularization
            if regularization == 'l1':
                grad_j += np.sign(params[j])
            elif regularization == 'l2':
                grad_j += 2 * params[j]

            # Update parameter j
            params[j] -= tol * grad_j

        # Check convergence (simplified)
        if np.linalg.norm(grad_j) < tol:
            break

    # Calculate metrics
    residuals = np.array([distance_func(X[i], params) for i in range(n_samples)])
    metrics = {
        'mse': np.mean(residuals**2),
        'mae': np.mean(np.abs(residuals)),
        'r2': 1 - np.sum(residuals**2) / np.sum((X - np.mean(X, axis=0))**2)
    }

    return params, metrics

################################################################################
# time_series_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def time_series_anomaly_fit(
    data: np.ndarray,
    window_size: int = 10,
    threshold_method: str = 'std',
    normalization: Optional[str] = None,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = np.linalg.norm,
    anomaly_threshold: Optional[float] = None,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, float, Dict[str, float], str]]:
    """
    Detect anomalies in a time series using configurable parameters.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    window_size : int, optional
        Size of the sliding window for anomaly detection, by default 10.
    threshold_method : str, optional
        Method to determine the anomaly threshold ('std', 'iqr'), by default 'std'.
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust'), by default None.
    distance_metric : Callable, optional
        Distance metric function, by default np.linalg.norm.
    anomaly_threshold : float, optional
        Custom threshold for anomalies, by default None.
    custom_normalization : Callable, optional
        Custom normalization function, by default None.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict[str, float], str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> result = time_series_anomaly_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, window_size)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization, custom_normalization)

    # Compute rolling statistics
    rolling_stats = _compute_rolling_statistics(normalized_data, window_size)

    # Detect anomalies
    anomalies = _detect_anomalies(
        normalized_data,
        rolling_stats,
        threshold_method,
        distance_metric,
        anomaly_threshold
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, anomalies)

    return {
        'result': {
            'anomalies': anomalies,
            'normalized_data': normalized_data
        },
        'metrics': metrics,
        'params_used': {
            'window_size': window_size,
            'threshold_method': threshold_method,
            'normalization': normalization if normalization is not None else 'none',
            'distance_metric': distance_metric.__name__ if hasattr(distance_metric, '__name__') else 'custom'
        },
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, window_size: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

def _apply_normalization(
    data: np.ndarray,
    normalization: Optional[str],
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if custom_normalization is not None:
        return custom_normalization(data)
    if normalization == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif normalization == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        return data.copy()

def _compute_rolling_statistics(data: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
    """Compute rolling statistics for the data."""
    rolling_mean = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    rolling_std = np.sqrt(np.convolve((data - np.mean(data))**2, np.ones(window_size)/window_size, mode='valid'))
    return {
        'mean': rolling_mean,
        'std': rolling_std
    }


def _detect_anomalies(
    data: np.ndarray,
    rolling_stats: Dict[str, np.ndarray],
    threshold_method: str,
    distance_metric: Callable[[np.ndarray, np.ndarray], float],
    anomaly_threshold: Optional[float]
) -> np.ndarray:
    """Detect anomalies based on rolling statistics and threshold method."""
    mean = rolling_stats['mean']
    std = rolling_stats['std']

    if threshold_method == 'std':
        threshold = np.mean(std) + 2 * np.std(std) if anomaly_threshold is None else anomaly_threshold
    elif threshold_method == 'iqr':
        q1 = np.percentile(std, 25)
        q3 = np.percentile(std, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr if anomaly_threshold is None else anomaly_threshold
    else:
        raise ValueError("Invalid threshold method.")

    anomalies = np.zeros_like(data, dtype=bool)
    for i in range(len(mean)):
        window = data[i:i+window_size]
        current_mean = mean[i]
        current_std = std[i]
        if distance_metric(window, np.full(window_size, current_mean)) > threshold:
            anomalies[i:i+window_size] = True
    return anomalies

def _compute_metrics(data: np.ndarray, anomalies: np.ndarray) -> Dict[str, float]:
    """Compute metrics for anomaly detection."""
    total_points = len(data)
    anomaly_count = np.sum(anomalies)
    return {
        'total_points': total_points,
        'anomaly_count': anomaly_count,
        'anomaly_ratio': anomaly_count / total_points if total_points > 0 else 0.0
    }

################################################################################
# spatial_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

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
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        X_norm = (X - median) / (iqr + 1e-8)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm

def compute_distance_matrix(
    X: np.ndarray,
    distance_metric: str = "euclidean",
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute distance matrix using specified metric or custom function."""
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    if custom_distance is not None:
        for i in range(n_samples):
            for j in range(n_samples):
                dist_matrix[i, j] = custom_distance(X[i], X[j])
        return dist_matrix

    if distance_metric == "euclidean":
        for i in range(n_samples):
            dist_matrix[i] = np.linalg.norm(X - X[i], axis=1)
    elif distance_metric == "manhattan":
        for i in range(n_samples):
            dist_matrix[i] = np.sum(np.abs(X - X[i]), axis=1)
    elif distance_metric == "cosine":
        for i in range(n_samples):
            dot_products = np.dot(X, X[i])
            norms = np.linalg.norm(X, axis=1) * np.linalg.norm(X[i])
            dist_matrix[i] = 1 - (dot_products / (norms + 1e-8))
    elif distance_metric == "minkowski":
        p = 3
        for i in range(n_samples):
            dist_matrix[i] = np.sum(np.abs(X - X[i])**p, axis=1)**(1/p)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    return dist_matrix

def detect_anomalies(
    dist_matrix: np.ndarray,
    method: str = "kNN",
    k: int = 5,
    threshold: float = 1.5
) -> np.ndarray:
    """Detect anomalies based on distance matrix."""
    n_samples = dist_matrix.shape[0]
    anomaly_scores = np.zeros(n_samples)

    if method == "kNN":
        for i in range(n_samples):
            # Exclude self-distance
            distances = dist_matrix[i][np.arange(n_samples) != i]
            if len(distances) < k:
                raise ValueError(f"Not enough neighbors for sample {i}")
            anomaly_scores[i] = np.mean(np.sort(distances)[:k])
    elif method == "median":
        for i in range(n_samples):
            # Exclude self-distance
            distances = dist_matrix[i][np.arange(n_samples) != i]
            if len(distances) == 0:
                raise ValueError(f"No neighbors for sample {i}")
            anomaly_scores[i] = np.median(distances)
    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")

    # Normalize scores
    if np.max(anomaly_scores) - np.min(anomaly_scores) > 0:
        anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (
            np.max(anomaly_scores) - np.min(anomaly_scores)
        )

    # Apply threshold
    anomalies = anomaly_scores > threshold

    return anomalies, anomaly_scores

def compute_metrics(
    X: np.ndarray,
    anomalies: np.ndarray,
    anomaly_scores: np.ndarray
) -> Dict[str, float]:
    """Compute various metrics for anomaly detection."""
    n_anomalies = np.sum(anomalies)
    total_samples = X.shape[0]

    metrics = {
        "anomaly_ratio": n_anomalies / total_samples,
        "mean_anomaly_score": np.mean(anomaly_scores[anomalies]),
        "max_anomaly_score": np.max(anomaly_scores[anomalies]),
        "min_anomaly_score": np.min(anomaly_scores[anomalies])
    }

    return metrics

def spatial_anomaly_fit(
    X: np.ndarray,
    normalization_method: str = "standard",
    distance_metric: str = "euclidean",
    detection_method: str = "kNN",
    k: int = 5,
    threshold: float = 1.5,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, str]]]:
    """
    Detect spatial anomalies in a dataset.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    normalization_method : str, optional
        Normalization method to apply (default: "standard")
    distance_metric : str, optional
        Distance metric to use (default: "euclidean")
    detection_method : str, optional
        Anomaly detection method (default: "kNN")
    k : int, optional
        Number of neighbors for kNN method (default: 5)
    threshold : float, optional
        Threshold for anomaly detection (default: 1.5)
    custom_normalizer : callable, optional
        Custom normalization function
    custom_distance : callable, optional
        Custom distance function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = spatial_anomaly_fit(X)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_norm = normalize_data(
        X,
        method=normalization_method,
        custom_normalizer=custom_normalizer
    )

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(
        X_norm,
        distance_metric=distance_metric,
        custom_distance=custom_distance
    )

    # Detect anomalies
    anomalies, anomaly_scores = detect_anomalies(
        dist_matrix,
        method=detection_method,
        k=k,
        threshold=threshold
    )

    # Compute metrics
    metrics = compute_metrics(X, anomalies, anomaly_scores)

    # Prepare output
    result = {
        "result": {
            "anomalies": anomalies,
            "anomaly_scores": anomaly_scores
        },
        "metrics": metrics,
        "params_used": {
            "normalization_method": normalization_method,
            "distance_metric": distance_metric,
            "detection_method": detection_method,
            "k": k,
            "threshold": threshold
        },
        "warnings": []
    }

    return result

################################################################################
# contextual_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def contextual_anomaly_fit(
    X: np.ndarray,
    context: Optional[np.ndarray] = None,
    normalizer: str = 'standard',
    distance_metric: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit a contextual anomaly detection model.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    context : Optional[np.ndarray]
        Contextual data matrix of shape (n_samples, n_context_features).
    normalizer : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    distance_metric : Union[str, Callable]
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable.
    solver : str
        Solver method: 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : Optional[str]
        Regularization type: 'none', 'l1', 'l2', or 'elasticnet'.
    custom_metric : Optional[Callable]
        Custom metric function.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, context)

    # Normalize data
    X_norm = _apply_normalization(X, normalizer)
    context_norm = _apply_normalization(context, normalizer) if context is not None else None

    # Initialize parameters
    params = _initialize_parameters(X_norm.shape[1], context_norm.shape[1] if context_norm is not None else 0)

    # Fit model based on solver choice
    if solver == 'closed_form':
        params = _fit_closed_form(X_norm, context_norm)
    elif solver == 'gradient_descent':
        params = _fit_gradient_descent(X_norm, context_norm, tol, max_iter, random_state)
    elif solver == 'newton':
        params = _fit_newton(X_norm, context_norm, tol, max_iter)
    elif solver == 'coordinate_descent':
        params = _fit_coordinate_descent(X_norm, context_norm, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Apply regularization if needed
    if regularization is not None:
        params = _apply_regularization(params, regularization)

    # Compute metrics
    metrics = _compute_metrics(X_norm, context_norm, params, custom_metric)

    # Return results
    return {
        'result': {'params': params},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer,
            'distance_metric': distance_metric,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray, context: Optional[np.ndarray]) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    if context is not None and (not isinstance(context, np.ndarray) or context.ndim != 2):
        raise ValueError("Context must be a 2D numpy array or None.")
    if context is not None and X.shape[0] != context.shape[0]:
        raise ValueError("X and context must have the same number of samples.")

def _apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to the data."""
    if method == 'none':
        return X
    elif method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError("Invalid normalization method specified.")

def _initialize_parameters(n_features: int, n_context_features: int) -> Dict[str, Any]:
    """Initialize model parameters."""
    return {
        'weights': np.zeros(n_features),
        'context_weights': np.zeros(n_context_features) if n_context_features > 0 else None,
        'intercept': 0.0
    }

def _fit_closed_form(X: np.ndarray, context: Optional[np.ndarray]) -> Dict[str, Any]:
    """Fit model using closed-form solution."""
    if context is None:
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        params = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ X
    else:
        X_aug = np.hstack([X, context, np.ones((X.shape[0], 1))])
        params = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ X
    return {
        'weights': params[:-1],
        'context_weights': params[-X.shape[1]-1:-1] if context is not None else None,
        'intercept': params[-1]
    }

def _fit_gradient_descent(
    X: np.ndarray,
    context: Optional[np.ndarray],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> Dict[str, Any]:
    """Fit model using gradient descent."""
    np.random.seed(random_state)
    params = _initialize_parameters(X.shape[1], context.shape[1] if context is not None else 0)
    for _ in range(max_iter):
        # Update parameters using gradient descent
        pass  # Implement gradient descent logic
    return params

def _fit_newton(
    X: np.ndarray,
    context: Optional[np.ndarray],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit model using Newton's method."""
    params = _initialize_parameters(X.shape[1], context.shape[1] if context is not None else 0)
    for _ in range(max_iter):
        # Update parameters using Newton's method
        pass  # Implement Newton's method logic
    return params

def _fit_coordinate_descent(
    X: np.ndarray,
    context: Optional[np.ndarray],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Fit model using coordinate descent."""
    params = _initialize_parameters(X.shape[1], context.shape[1] if context is not None else 0)
    for _ in range(max_iter):
        # Update parameters using coordinate descent
        pass  # Implement coordinate descent logic
    return params

def _apply_regularization(params: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Apply regularization to the parameters."""
    if method == 'l1':
        params['weights'] = np.sign(params['weights']) * np.maximum(np.abs(params['weights']) - 1, 0)
    elif method == 'l2':
        pass  # Implement L2 regularization
    elif method == 'elasticnet':
        pass  # Implement elastic net regularization
    return params

def _compute_metrics(
    X: np.ndarray,
    context: Optional[np.ndarray],
    params: Dict[str, Any],
    custom_metric: Optional[Callable]
) -> Dict[str, float]:
    """Compute metrics for the model."""
    predictions = _predict(X, context, params)
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(X, predictions)
    return metrics

def _predict(
    X: np.ndarray,
    context: Optional[np.ndarray],
    params: Dict[str, Any]
) -> np.ndarray:
    """Make predictions using the fitted model."""
    if context is None:
        return X @ params['weights'] + params['intercept']
    else:
        return (X @ params['weights'] + context @ params['context_weights']) + params['intercept']

################################################################################
# point_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def point_anomaly_fit(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    threshold_func: Optional[Callable[[np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Detect point anomalies in a dataset using configurable parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data, by default None
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine'), by default 'euclidean'
    threshold_func : Optional[Callable[[np.ndarray], float]], optional
        Function to compute the anomaly threshold, by default None
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function, by default None

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - 'result': array of anomaly scores
        - 'metrics': dictionary of computed metrics
        - 'params_used': parameters used in the computation
        - 'warnings': any warnings generated

    Example
    -------
    >>> data = np.random.randn(100, 5)
    >>> result = point_anomaly_fit(data, distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Compute distance matrix
    distance_matrix = _compute_distance_matrix(
        normalized_data,
        distance_metric=distance_metric,
        custom_distance=custom_distance
    )

    # Compute anomaly scores
    anomaly_scores = _compute_anomaly_scores(distance_matrix)

    # Compute threshold if threshold_func is provided
    threshold = None
    if threshold_func is not None:
        threshold = threshold_func(anomaly_scores)

    # Compute metrics
    metrics = _compute_metrics(anomaly_scores, threshold)

    # Prepare output
    result_dict = {
        'result': anomaly_scores,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'threshold_func': threshold_func.__name__ if threshold_func else None
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    try:
        normalized_data = normalizer(data)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")
    return normalized_data

def _compute_distance_matrix(
    data: np.ndarray,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute the distance matrix."""
    if custom_distance is not None:
        return _compute_custom_distance_matrix(data, custom_distance)

    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if distance_metric == 'euclidean':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.linalg.norm(data[i] - data[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif distance_metric == 'manhattan':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = np.sum(np.abs(data[i] - data[j]))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    elif distance_metric == 'cosine':
        for i in range(n_samples):
            for j in range(i, n_samples):
                distance = 1 - np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    return distance_matrix

def _compute_custom_distance_matrix(
    data: np.ndarray,
    custom_distance: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute distance matrix using custom distance function."""
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            distance = custom_distance(data[i], data[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def _compute_anomaly_scores(distance_matrix: np.ndarray) -> np.ndarray:
    """Compute anomaly scores from distance matrix."""
    # This is a simple implementation using the average distance to other points
    anomaly_scores = np.mean(distance_matrix, axis=1)
    return anomaly_scores

def _compute_metrics(
    anomaly_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """Compute metrics for anomaly detection."""
    metrics = {
        'mean_score': np.mean(anomaly_scores),
        'std_score': np.std(anomaly_scores),
        'min_score': np.min(anomaly_scores),
        'max_score': np.max(anomaly_scores)
    }

    if threshold is not None:
        anomalies = anomaly_scores > threshold
        metrics['n_anomalies'] = np.sum(anomalies)
        metrics['anomaly_rate'] = np.mean(anomalies)

    return metrics

################################################################################
# collective_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def collective_anomaly_fit(
    data: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Detect collective anomalies in a dataset using various configurable parameters.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data. Default is identity function.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski').
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
    **kwargs : dict
        Additional solver-specific parameters.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.rand(100, 5)
    >>> result = collective_anomaly_fit(data, normalizer=np.std, distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data
    normalized_data = normalizer(data)

    # Choose distance metric
    distance_func = _get_distance_function(distance_metric)

    # Compute anomaly scores
    anomaly_scores = _compute_anomaly_scores(normalized_data, distance_func)

    # Solve for parameters
    params = _solve_anomaly_model(anomaly_scores, solver, regularization, tol, max_iter, **kwargs)

    # Compute metrics
    metrics = _compute_metrics(anomaly_scores, params, custom_metric)

    # Prepare output
    result = {
        "result": anomaly_scores,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if hasattr(normalizer, '__name__') else "custom",
            "distance_metric": distance_metric,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return the distance function based on the metric name."""
    metrics = {
        'euclidean': lambda x, y: np.linalg.norm(x - y),
        'manhattan': lambda x, y: np.sum(np.abs(x - y)),
        'cosine': lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
        'minkowski': lambda x, y, p=3: np.sum(np.abs(x - y) ** p) ** (1 / p)
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return metrics[metric]

def _compute_anomaly_scores(data: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute anomaly scores for each data point."""
    n_samples = data.shape[0]
    anomaly_scores = np.zeros(n_samples)
    for i in range(n_samples):
        distances = [distance_func(data[i], data[j]) for j in range(n_samples) if i != j]
        anomaly_scores[i] = np.mean(distances)
    return anomaly_scores

def _solve_anomaly_model(
    scores: np.ndarray,
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    **kwargs
) -> Dict[str, Union[np.ndarray, float]]:
    """Solve for model parameters using the specified solver."""
    if solver == 'closed_form':
        params = _closed_form_solution(scores, regularization)
    elif solver == 'gradient_descent':
        params = _gradient_descent(scores, tol, max_iter, **kwargs)
    elif solver == 'newton':
        params = _newton_method(scores, tol, max_iter, **kwargs)
    elif solver == 'coordinate_descent':
        params = _coordinate_descent(scores, tol, max_iter, **kwargs)
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    return params

def _closed_form_solution(scores: np.ndarray, regularization: Optional[str]) -> Dict[str, Union[np.ndarray, float]]:
    """Closed form solution for anomaly detection."""
    if regularization == 'l1':
        # Lasso regression
        pass  # Implement L1 regularized solution
    elif regularization == 'l2':
        # Ridge regression
        pass  # Implement L2 regularized solution
    elif regularization == 'elasticnet':
        # Elastic net regression
        pass  # Implement elastic net solution
    else:
        # No regularization
        pass  # Implement ordinary least squares
    return {"coefficients": np.zeros(scores.shape[0]), "intercept": 0.0}

def _gradient_descent(
    scores: np.ndarray,
    tol: float,
    max_iter: int,
    learning_rate: float = 0.01,
    **kwargs
) -> Dict[str, Union[np.ndarray, float]]:
    """Gradient descent solver."""
    # Implement gradient descent
    return {"coefficients": np.zeros(scores.shape[0]), "intercept": 0.0}

def _newton_method(
    scores: np.ndarray,
    tol: float,
    max_iter: int,
    **kwargs
) -> Dict[str, Union[np.ndarray, float]]:
    """Newton's method solver."""
    # Implement Newton's method
    return {"coefficients": np.zeros(scores.shape[0]), "intercept": 0.0}

def _coordinate_descent(
    scores: np.ndarray,
    tol: float,
    max_iter: int,
    **kwargs
) -> Dict[str, Union[np.ndarray, float]]:
    """Coordinate descent solver."""
    # Implement coordinate descent
    return {"coefficients": np.zeros(scores.shape[0]), "intercept": 0.0}

def _compute_metrics(
    scores: np.ndarray,
    params: Dict[str, Union[np.ndarray, float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for anomaly detection."""
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(scores, params["coefficients"])
    else:
        # Default metrics
        pass  # Implement default metrics
    return metrics

################################################################################
# conditional_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def conditional_anomaly_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    condition: Optional[np.ndarray] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    distance_metric: str = 'euclidean',
    anomaly_threshold: float = 3.0,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Fit a conditional anomaly detection model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    y : Optional[np.ndarray], default=None
        Target values if available. If None, unsupervised mode.
    condition : Optional[np.ndarray], default=None
        Condition vector of shape (n_samples,) to filter data.
    normalizer : Callable[[np.ndarray], np.ndarray], default=lambda x: x
        Function to normalize the input data.
    distance_metric : str, default='euclidean'
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    anomaly_threshold : float, default=3.0
        Threshold for anomaly detection.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], default=None
        Custom distance function if needed.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict, str]]
        Dictionary containing:
        - 'result': Anomaly scores or labels.
        - 'metrics': Performance metrics.
        - 'params_used': Parameters used in the fitting process.
        - 'warnings': Any warnings encountered.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = conditional_anomaly_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, y, condition)

    # Normalize data
    X_normalized = normalizer(X)

    # Filter data based on condition if provided
    if condition is not None:
        X_normalized = X_normalized[condition]

    # Compute distance matrix
    distance_matrix = _compute_distance_matrix(X_normalized, distance_metric, custom_distance)

    # Compute anomaly scores
    anomaly_scores = _compute_anomaly_scores(distance_matrix, anomaly_threshold)

    # Compute metrics
    metrics = _compute_metrics(anomaly_scores, y)

    return {
        'result': anomaly_scores,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'distance_metric': distance_metric,
            'anomaly_threshold': anomaly_threshold
        },
        'warnings': []
    }

def _validate_inputs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    condition: Optional[np.ndarray] = None
) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array or None.")
    if condition is not None and not isinstance(condition, np.ndarray):
        raise TypeError("condition must be a numpy array or None.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if condition is not None and X.shape[0] != condition.shape[0]:
        raise ValueError("X and condition must have the same number of samples.")

def _compute_distance_matrix(
    X: np.ndarray,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute the distance matrix."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    if custom_distance is not None:
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = custom_distance(X[i], X[j])
    else:
        if distance_metric == 'euclidean':
            for i in range(n_samples):
                for j in range(n_samples):
                    distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
        elif distance_metric == 'manhattan':
            for i in range(n_samples):
                for j in range(n_samples):
                    distance_matrix[i, j] = np.sum(np.abs(X[i] - X[j]))
        elif distance_metric == 'cosine':
            for i in range(n_samples):
                for j in range(n_samples):
                    distance_matrix[i, j] = 1 - np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    return distance_matrix

def _compute_anomaly_scores(
    distance_matrix: np.ndarray,
    anomaly_threshold: float = 3.0
) -> np.ndarray:
    """Compute anomaly scores based on distance matrix."""
    n_samples = distance_matrix.shape[0]
    anomaly_scores = np.zeros(n_samples)

    for i in range(n_samples):
        distances = distance_matrix[i, :]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        anomaly_scores[i] = (distances[i] - mean_distance) / std_distance if std_distance != 0 else 0

    return anomaly_scores > anomaly_threshold

def _compute_metrics(
    anomaly_scores: np.ndarray,
    y: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute performance metrics."""
    metrics = {}

    if y is not None:
        # Add supervised metrics here
        pass

    return metrics

################################################################################
# z_score
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(
    data: np.ndarray,
    threshold: float = 3.0,
    axis: int = 0
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if threshold <= 0:
        raise ValueError("Threshold must be positive")

def _compute_z_scores(
    data: np.ndarray,
    mean: float,
    std: float
) -> np.ndarray:
    """Compute z-scores for the given data."""
    return (data - mean) / std

def _get_normalization(
    data: np.ndarray,
    method: str = "standard"
) -> Dict[str, float]:
    """Apply normalization to the data."""
    if method == "none":
        return {"mean": 0.0, "std": 1.0}
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return {"mean": mean, "std": std}
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return {"min": min_val, "max": max_val}
    elif method == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return {"median": median, "iqr": iqr}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def z_score_fit(
    data: np.ndarray,
    threshold: float = 3.0,
    normalization: str = "standard",
    axis: int = 0
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Compute z-scores for anomaly detection.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    threshold : float, optional
        Threshold for anomaly detection, by default 3.0.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust"), by default "standard".
    axis : int, optional
        Axis along which to compute statistics, by default 0.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = z_score_fit(data, threshold=3.0, normalization="standard")
    """
    _validate_inputs(data, threshold, axis)

    # Get normalization parameters
    norm_params = _get_normalization(data, normalization)
    if normalization in ["standard", "robust"]:
        mean = norm_params["mean"]
        std = norm_params.get("std", norm_params.get("iqr", 1.0))
    else:
        mean = 0.0
        std = 1.0

    # Compute z-scores
    z_scores = _compute_z_scores(data, mean, std)

    # Identify anomalies
    anomalies = np.abs(z_scores) > threshold

    return {
        "result": {
            "z_scores": z_scores,
            "anomalies": anomalies
        },
        "metrics": {
            "mean_z_score": np.mean(np.abs(z_scores)),
            "anomaly_rate": np.mean(anomalies)
        },
        "params_used": {
            "threshold": threshold,
            "normalization": normalization,
            "axis": axis
        },
        "warnings": []
    }

################################################################################
# iqr_method
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")

def _normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_func is not None:
        return custom_func(data)

    data = data.copy()
    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    elif method == "robust":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        return (data - q1) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_iqr(data: np.ndarray) -> float:
    """Calculate the interquartile range (IQR)."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return q3 - q1

def _detect_anomalies(
    data: np.ndarray,
    threshold: float = 1.5
) -> np.ndarray:
    """Detect anomalies using IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return np.where((data < lower_bound) | (data > upper_bound), 1, 0)

def iqr_method_fit(
    data: np.ndarray,
    normalization: str = "standard",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    threshold: float = 1.5
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Detect anomalies in data using the IQR method.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    custom_normalization : Callable, optional
        Custom normalization function.
    threshold : float, optional
        Threshold for anomaly detection.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> result = iqr_method_fit(data)
    """
    _validate_inputs(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        method=normalization,
        custom_func=custom_normalization
    )

    # Calculate IQR and detect anomalies
    iqr = _calculate_iqr(normalized_data)
    anomalies = _detect_anomalies(normalized_data, threshold=threshold)

    # Calculate metrics
    anomaly_count = np.sum(anomalies)
    anomaly_percentage = (anomaly_count / len(data)) * 100

    return {
        "result": anomalies,
        "metrics": {
            "iqr": iqr,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_percentage
        },
        "params_used": {
            "normalization": normalization,
            "custom_normalization": custom_normalization is not None,
            "threshold": threshold
        },
        "warnings": []
    }

################################################################################
# dbscan
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def dbscan_fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[str, Callable] = 'euclidean',
    normalize: Optional[str] = None,
    algorithm: str = 'auto',
    leaf_size: int = 30
) -> Dict[str, Any]:
    """
    Perform DBSCAN clustering on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str or callable, optional
        The distance metric to use. Can be 'euclidean', 'manhattan', 'cosine', or a custom callable.
    normalize : str, optional
        Normalization method. Can be 'none', 'standard', 'minmax', or 'robust'.
    algorithm : str, optional
        Algorithm used by the NearestNeighbors module to compute pointwise distances.
    leaf_size : int, optional
        Leaf size passed to NearestNeighbors.

    Returns:
    --------
    dict
        A dictionary containing the clustering results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, eps, min_samples)

    # Normalize data if specified
    X_normalized = _normalize_data(X, normalize) if normalize else X

    # Get distance function
    distance_func = _get_distance_function(metric)

    # Perform DBSCAN clustering
    labels = _dbscan_clustering(X_normalized, eps, min_samples, distance_func, algorithm, leaf_size)

    # Calculate metrics
    metrics = _calculate_metrics(X_normalized, labels)

    # Prepare output
    result = {
        'result': {'labels': labels},
        'metrics': metrics,
        'params_used': {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'normalize': normalize,
            'algorithm': algorithm,
            'leaf_size': leaf_size
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray, eps: float, min_samples: int) -> None:
    """Validate the input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")
    if eps <= 0:
        raise ValueError("eps must be a positive number.")
    if min_samples <= 0:
        raise ValueError("min_samples must be a positive integer.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize the input data based on the specified method."""
    if method == 'standard':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'minmax':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif method == 'robust':
        return (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_distance_function(metric: Union[str, Callable]) -> Callable:
    """Get the distance function based on the specified metric."""
    if callable(metric):
        return metric
    elif metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y) ** 3) ** (1/3)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _dbscan_clustering(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    distance_func: Callable,
    algorithm: str,
    leaf_size: int
) -> np.ndarray:
    """Perform DBSCAN clustering on the input data."""
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm=algorithm, leaf_size=leaf_size, metric=distance_func)
    nbrs.fit(X)

    distances, indices = nbrs.kneighbors(X)
    labels = np.zeros(X.shape[0], dtype=int)

    cluster_id = 1
    for i in range(X.shape[0]):
        if labels[i] != 0:
            continue
        neighbors = indices[i][distances[i] <= eps]
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
        else:
            _expand_cluster(X, i, neighbors, eps, min_samples, distance_func, labels, cluster_id)
            cluster_id += 1
    return labels

def _expand_cluster(
    X: np.ndarray,
    point_idx: int,
    neighbors: np.ndarray,
    eps: float,
    min_samples: int,
    distance_func: Callable,
    labels: np.ndarray,
    cluster_id: int
) -> None:
    """Expand the cluster from the given point."""
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id
            distances, new_neighbors = _get_neighbors(X, neighbor_idx, min_samples, distance_func)
            if len(new_neighbors[distances <= eps]) >= min_samples:
                neighbors = np.concatenate((neighbors, new_neighbors))
        i += 1

def _get_neighbors(
    X: np.ndarray,
    point_idx: int,
    min_samples: int,
    distance_func: Callable
) -> tuple:
    """Get the neighbors of a given point."""
    distances = np.array([distance_func(X[point_idx], x) for x in X])
    indices = np.argsort(distances)[1:min_samples + 1]
    return distances[indices], X[indices]

def _calculate_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for the clustering results."""
    from sklearn.metrics import silhouette_score

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {'silhouette_score': np.nan}

    metrics = {
        'silhouette_score': silhouette_score(X, labels)
    }
    return metrics

################################################################################
# isolation_forest
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def isolation_forest_fit(
    X: np.ndarray,
    n_estimators: int = 100,
    max_samples: Union[int, float] = "auto",
    contamination: float = 0.1,
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0
) -> Dict:
    """
    Fit the Isolation Forest model to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : int or float, default="auto"
        The number of samples to draw to train each base estimator.
    contamination : float, default=0.1
        The expected proportion of outliers in the data.
    max_features : int or float, default=1.0
        The number of features to draw for each split.
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
    Dict containing:
        - "result": Fitted model
        - "metrics": Dictionary of metrics
        - "params_used": Parameters used for fitting
        - "warnings": List of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = isolation_forest_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, n_estimators, max_samples, contamination,
                     max_features, bootstrap)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Initialize warnings list
    warnings_list = []

    # Set default max_samples if auto
    if max_samples == "auto":
        max_samples = min(256, X.shape[0])

    # Initialize ensemble
    trees = []

    for _ in range(n_estimators):
        # Sample data
        if bootstrap:
            sample_idx = rng.choice(X.shape[0], size=max_samples, replace=True)
        else:
            sample_idx = rng.choice(X.shape[0], size=max_samples, replace=False)

        X_sample = X[sample_idx]

        # Build tree
        tree = _build_isolation_tree(X_sample, max_features, rng)

        # Append to ensemble
        trees.append(tree)

    # Calculate anomaly scores
    scores = _calculate_anomaly_scores(X, trees, contamination)

    # Calculate metrics
    metrics = _calculate_metrics(scores, contamination)

    return {
        "result": {"trees": trees},
        "metrics": metrics,
        "params_used": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "max_features": max_features,
            "bootstrap": bootstrap
        },
        "warnings": warnings_list
    }

def _validate_inputs(
    X: np.ndarray,
    n_estimators: int,
    max_samples: Union[int, float],
    contamination: float,
    max_features: Union[int, float],
    bootstrap: bool
) -> None:
    """Validate input parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if contamination < 0 or contamination > 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    if max_features <= 0 or max_features > X.shape[1]:
        raise ValueError("max_features must be between 0 and n_features")

def _build_isolation_tree(
    X: np.ndarray,
    max_features: Union[int, float],
    rng: np.random.RandomState
) -> Dict:
    """Build a single isolation tree."""
    n_samples, n_features = X.shape

    # Determine number of features to use
    if isinstance(max_features, float):
        n_features_to_use = int(max_features * n_features)
    else:
        n_features_to_use = max_features

    # Select random features
    feature_idx = rng.choice(n_features, size=n_features_to_use, replace=False)

    # Build tree recursively
    node = _build_node(X[:, feature_idx], rng)

    return {"node": node, "features_used": feature_idx}

def _build_node(
    X: np.ndarray,
    rng: np.random.RandomState
) -> Dict:
    """Build a single node in the isolation tree."""
    if X.shape[0] <= 1:
        return {"is_leaf": True, "samples": X}

    # Select random split point
    feature_idx = rng.choice(X.shape[1])
    split_value = rng.uniform(X[:, feature_idx].min(), X[:, feature_idx].max())

    # Split data
    left_mask = X[:, feature_idx] < split_value
    right_mask = ~left_mask

    # Recursively build children
    left_node = _build_node(X[left_mask], rng)
    right_node = _build_node(X[right_mask], rng)

    return {
        "is_leaf": False,
        "feature_idx": feature_idx,
        "split_value": split_value,
        "left_node": left_node,
        "right_node": right_node
    }

def _calculate_anomaly_scores(
    X: np.ndarray,
    trees: list,
    contamination: float
) -> np.ndarray:
    """Calculate anomaly scores for each sample."""
    n_samples = X.shape[0]
    scores = np.zeros(n_samples)

    for tree in trees:
        # Get path lengths
        path_lengths = _get_path_lengths(X, tree["node"], tree["features_used"])

        # Calculate scores
        scores += _calculate_scores(path_lengths, contamination)

    return scores / len(trees)

def _get_path_lengths(
    X: np.ndarray,
    node: Dict,
    features_used: np.ndarray
) -> np.ndarray:
    """Get path lengths for each sample in the tree."""
    n_samples = X.shape[0]
    path_lengths = np.zeros(n_samples)

    for i in range(n_samples):
        current_node = node
        depth = 0

        while not current_node["is_leaf"]:
            feature_idx = features_used[current_node["feature_idx"]]
            if X[i, feature_idx] < current_node["split_value"]:
                current_node = current_node["left_node"]
            else:
                current_node = current_node["right_node"]
            depth += 1

        path_lengths[i] = depth

    return path_lengths

def _calculate_scores(
    path_lengths: np.ndarray,
    contamination: float
) -> np.ndarray:
    """Calculate scores from path lengths."""
    n_samples = path_lengths.shape[0]
    max_depth = np.max(path_lengths)

    # Calculate scores
    scores = 2 ** (-path_lengths / max_depth)
    scores[scores > contamination] = contamination

    return 1 - scores / np.max(scores)

def _calculate_metrics(
    scores: np.ndarray,
    contamination: float
) -> Dict:
    """Calculate metrics for the model."""
    # Calculate thresholds based on contamination
    threshold = np.quantile(scores, 1 - contamination)

    # Calculate binary predictions
    predictions = scores >= threshold

    # Calculate metrics (example: precision, recall)
    tp = np.sum((predictions == 1) & (scores >= threshold))
    fp = np.sum((predictions == 1) & (scores < threshold))
    tn = np.sum((predictions == 0) & (scores < threshold))
    fn = np.sum((predictions == 0) & (scores >= threshold))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "anomaly_score_mean": np.mean(scores),
        "anomaly_score_std": np.std(scores),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

################################################################################
# local_outlier_factor
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def local_outlier_factor_fit(
    X: np.ndarray,
    n_neighbors: int = 20,
    algorithm: str = 'auto',
    metric: Union[str, Callable] = 'euclidean',
    p: float = 2,
    metric_params: Optional[Dict] = None,
    leaf_size: int = 30,
    n_jobs: Optional[int] = None
) -> Dict:
    """
    Compute the Local Outlier Factor (LOF) for each sample in X.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_neighbors : int, optional
        Number of neighbors to consider (default=20).
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute nearest neighbors (default='auto').
    metric : str or callable, optional
        Distance metric to use (default='euclidean').
    p : float, optional
        Parameter for Minkowski metric (default=2).
    metric_params : dict, optional
        Additional keyword arguments for the metric function.
    leaf_size : int, optional
        Leaf size passed to BallTree or KDTree (default=30).
    n_jobs : int, optional
        Number of parallel jobs to run (default=None).

    Returns:
    --------
    dict
        Dictionary containing:
        - 'result': array of LOF scores for each sample
        - 'metrics': dictionary of computed metrics
        - 'params_used': dictionary of parameters used
        - 'warnings': list of warnings encountered

    Example:
    --------
    >>> X = np.random.rand(10, 5)
    >>> result = local_outlier_factor_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, n_neighbors)

    # Compute nearest neighbors distances and reachability distances
    kdist = _compute_k_distances(X, n_neighbors, algorithm, metric, p, metric_params, leaf_size, n_jobs)
    reach_dist = _compute_reachability_distance(X, kdist, n_neighbors, algorithm, metric, p, metric_params, leaf_size, n_jobs)

    # Compute local reachability density
    lrd = _compute_local_reachability_density(reach_dist, n_neighbors)

    # Compute LOF scores
    lof_scores = _compute_lof_scores(lrd, reach_dist)

    # Prepare output
    metrics = {
        'k_distance': kdist,
        'reachability_distance': reach_dist,
        'local_reachability_density': lrd
    }

    params_used = {
        'n_neighbors': n_neighbors,
        'algorithm': algorithm,
        'metric': metric,
        'p': p,
        'leaf_size': leaf_size
    }

    warnings = []

    return {
        'result': lof_scores,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

def _validate_inputs(X: np.ndarray, n_neighbors: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors > X.shape[0]:
        raise ValueError("n_neighbors must be less than or equal to the number of samples")

def _compute_k_distances(
    X: np.ndarray,
    n_neighbors: int,
    algorithm: str,
    metric: Union[str, Callable],
    p: float,
    metric_params: Optional[Dict],
    leaf_size: int,
    n_jobs: Optional[int]
) -> np.ndarray:
    """Compute the k-distance for each sample."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        algorithm=algorithm,
        metric=metric,
        p=p,
        metric_params=metric_params,
        leaf_size=leaf_size
    )
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # The k-distance is the distance to the n_neighbors-th neighbor
    return distances[:, -1]

def _compute_reachability_distance(
    X: np.ndarray,
    kdist: np.ndarray,
    n_neighbors: int,
    algorithm: str,
    metric: Union[str, Callable],
    p: float,
    metric_params: Optional[Dict],
    leaf_size: int,
    n_jobs: Optional[int]
) -> np.ndarray:
    """Compute the reachability distance for each sample pair."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric,
        p=p,
        metric_params=metric_params,
        leaf_size=leaf_size
    )
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    reach_dist = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j, idx in enumerate(indices[i]):
            if distances[i][j] <= kdist[idx]:
                reach_dist[i, idx] = distances[i][j]
            else:
                reach_dist[i, idx] = kdist[idx]

    return reach_dist

def _compute_local_reachability_density(
    reach_dist: np.ndarray,
    n_neighbors: int
) -> np.ndarray:
    """Compute the local reachability density for each sample."""
    lrd = 1.0 / (np.mean(reach_dist[:, :n_neighbors], axis=1) + 1e-10)
    return lrd

def _compute_lof_scores(
    lrd: np.ndarray,
    reach_dist: np.ndarray
) -> np.ndarray:
    """Compute the LOF scores for each sample."""
    lof_scores = np.zeros(lrd.shape[0])

    for i in range(len(lof_scores)):
        neighbor_lrd = lrd[i] / (np.mean(lrd[reach_dist[i, :len(reach_dist[i])] > 0], axis=0) + 1e-10)
        lof_scores[i] = np.mean(neighbor_lrd)

    return lof_scores

################################################################################
# one_class_svm
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Input contains infinite values")

def normalize_data(
    X: np.ndarray,
    method: str = "standard",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data using specified method or custom function."""
    if custom_normalizer is not None:
        return custom_normalizer(X)

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

def compute_kernel(
    X: np.ndarray,
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Compute kernel matrix."""
    if custom_kernel is not None:
        return custom_kernel(X, X)

    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    if kernel_type == "linear":
        K = X @ X.T
    elif kernel_type == "rbf":
        pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + \
                           np.sum(X**2, axis=1)[np.newaxis, :] - \
                           2 * X @ X.T
        K = np.exp(-gamma * pairwise_sq_dists)
    elif kernel_type == "poly":
        K = (X @ X.T + 1) ** gamma
    elif kernel_type == "sigmoid":
        K = np.tanh(gamma * X @ X.T + 1)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return K

def one_class_svm_fit(
    X: np.ndarray,
    nu: float = 0.1,
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    normalize_method: str = "standard",
    tol: float = 1e-3,
    max_iter: int = 1000,
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[np.ndarray, float, Dict[str, float], str]]:
    """
    Fit One-Class SVM model.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    nu : float
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    kernel_type : str
        Type of kernel to use ('linear', 'rbf', 'poly', 'sigmoid')
    gamma : float
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    normalize_method : str
        Normalization method ('none', 'standard', 'minmax', 'robust')
    tol : float
        Tolerance for stopping criteria
    max_iter : int
        Maximum number of iterations
    custom_kernel : callable, optional
        Custom kernel function
    custom_normalizer : callable, optional
        Custom normalization function

    Returns
    -------
    dict
        Dictionary containing:
        - result: fitted model parameters
        - metrics: performance metrics
        - params_used: parameters used for fitting
        - warnings: any warnings generated during fitting

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> result = one_class_svm_fit(X, nu=0.1)
    """
    # Validate input
    validate_input(X)

    # Normalize data
    X_normalized = normalize_data(
        X,
        method=normalize_method,
        custom_normalizer=custom_normalizer
    )

    # Compute kernel matrix
    K = compute_kernel(
        X_normalized,
        kernel_type=kernel_type,
        gamma=gamma,
        custom_kernel=custom_kernel
    )

    # Initialize variables for optimization
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    errors = np.ones(n_samples)

    # Optimization loop
    for _ in range(max_iter):
        # Compute support vectors
        sv_indices = np.where(alpha > 1e-5)[0]
        n_sv = len(sv_indices)

        if n_sv / n_samples > nu + tol:
            alpha[sv_indices] -= (n_sv - n_samples * nu) / n_sv
        elif n_sv / n_samples < nu - tol:
            alpha[sv_indices] += (n_samples * nu - n_sv) / n_sv

        # Update errors
        for i in range(n_samples):
            K_i = K[i, sv_indices]
            alpha_sv = alpha[sv_indices]

            # Compute error
            errors[i] = K[i, i] - np.sum(alpha_sv * K_i)

            # Update alpha
            if errors[i] < 0:
                alpha[i] += errors[i]
            elif alpha[i] > 0:
                alpha[i] -= errors[i]

        # Check convergence
        if np.max(np.abs(errors)) < tol:
            break

    # Compute offset
    rho = np.mean(alpha[sv_indices] * errors[sv_indices])

    # Prepare output
    result = {
        "alpha": alpha,
        "support_vectors": X[sv_indices],
        "rho": rho
    }

    metrics = {
        "n_support_vectors": n_sv,
        "nu_used": n_sv / n_samples
    }

    params_used = {
        "kernel_type": kernel_type,
        "gamma": gamma,
        "normalize_method": normalize_method,
        "tol": tol,
        "max_iter": max_iter
    }

    warnings = []
    if _ == max_iter - 1:
        warnings.append("Maximum iterations reached without convergence")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }

def predict_one_class_svm(
    X: np.ndarray,
    model: Dict[str, Union[np.ndarray, float]],
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    custom_kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Predict anomalies using fitted One-Class SVM model.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    model : dict
        Fitted One-Class SVM model from one_class_svm_fit
    kernel_type : str
        Type of kernel used in fitting ('linear', 'rbf', 'poly', 'sigmoid')
    gamma : float
        Kernel coefficient used in fitting
    custom_kernel : callable, optional
        Custom kernel function used in fitting

    Returns
    -------
    np.ndarray
        Array of decision values (1 for inliers, -1 for outliers)
    """
    validate_input(X)

    alpha = model["alpha"]
    support_vectors = model["support_vectors"]
    rho = model["rho"]

    if custom_kernel is not None:
        K = custom_kernel(X, support_vectors)
    else:
        if kernel_type == "linear":
            K = X @ support_vectors.T
        elif kernel_type == "rbf":
            pairwise_sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + \
                               np.sum(support_vectors**2, axis=1)[np.newaxis, :] - \
                               2 * X @ support_vectors.T
            K = np.exp(-gamma * pairwise_sq_dists)
        elif kernel_type == "poly":
            K = (X @ support_vectors.T + 1) ** gamma
        elif kernel_type == "sigmoid":
            K = np.tanh(gamma * X @ support_vectors.T + 1)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    decision_values = np.sum(alpha[:, np.newaxis] * K, axis=0) - rho
    return np.where(decision_values >= 0, 1, -1)

################################################################################
# autoencoder_anomaly
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def autoencoder_anomaly_fit(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'gradient_descent',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit an autoencoder model to detect anomalies in the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. If None, no normalization is applied.
    metric : str, optional
        Metric to evaluate the reconstruction error. Options: 'mse', 'mae', 'r2'.
    distance : str, optional
        Distance metric for anomaly scoring. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str, optional
        Solver to optimize the autoencoder. Options: 'gradient_descent', 'newton'.
    regularization : Optional[str], optional
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float, optional
        Tolerance for stopping criteria.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = autoencoder_anomaly_fit(X, normalizer=np.std, metric='mse')
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Initialize autoencoder parameters
    params = _initialize_autoencoder_params(X_normalized.shape[1])

    # Choose solver and optimize
    if solver == 'gradient_descent':
        optimized_params = _gradient_descent(
            X_normalized, params, metric, distance,
            regularization, tol, max_iter,
            custom_metric, custom_distance
        )
    elif solver == 'newton':
        optimized_params = _newton_method(
            X_normalized, params, metric, distance,
            regularization, tol, max_iter,
            custom_metric, custom_distance
        )
    else:
        raise ValueError("Unsupported solver. Choose from 'gradient_descent' or 'newton'.")

    # Compute reconstruction error and anomaly scores
    reconstruction_error = _compute_reconstruction_error(
        X_normalized, optimized_params, metric, custom_metric
    )
    anomaly_scores = _compute_anomaly_scores(
        reconstruction_error, distance, custom_distance
    )

    # Compute metrics
    metrics = _compute_metrics(reconstruction_error, metric)

    return {
        'result': {
            'reconstruction_error': reconstruction_error,
            'anomaly_scores': anomaly_scores
        },
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization
        },
        'warnings': []
    }

def _validate_inputs(X: np.ndarray) -> None:
    """Validate the input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X contains NaN or infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the input data."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _initialize_autoencoder_params(n_features: int) -> Dict[str, np.ndarray]:
    """Initialize autoencoder parameters."""
    return {
        'weights': np.random.randn(n_features, n_features),
        'bias': np.zeros(n_features)
    }

def _gradient_descent(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    metric: str,
    distance: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, np.ndarray]:
    """Optimize autoencoder parameters using gradient descent."""
    # Implementation of gradient descent
    pass

def _newton_method(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    metric: str,
    distance: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, np.ndarray]:
    """Optimize autoencoder parameters using Newton's method."""
    # Implementation of Newton's method
    pass

def _compute_reconstruction_error(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> np.ndarray:
    """Compute the reconstruction error."""
    # Implementation of reconstruction error computation
    pass

def _compute_anomaly_scores(
    reconstruction_error: np.ndarray,
    distance: str,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> np.ndarray:
    """Compute anomaly scores based on reconstruction error."""
    # Implementation of anomaly score computation
    pass

def _compute_metrics(
    reconstruction_error: np.ndarray,
    metric: str
) -> Dict[str, float]:
    """Compute metrics based on the reconstruction error."""
    # Implementation of metric computation
    pass

################################################################################
# mahalanobis_distance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(X: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> None:
    """Validate input dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(mu, np.ndarray) or not isinstance(inv_cov, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array")
    if inv_cov.ndim != 2:
        raise ValueError("inv_cov must be a 2D array")
    if X.shape[1] != mu.shape[0] or X.shape[1] != inv_cov.shape[0] or inv_cov.shape[0] != inv_cov.shape[1]:
        raise ValueError("Dimension mismatch between X, mu and inv_cov")
    if np.any(np.isnan(X)) or np.any(np.isnan(mu)) or np.any(np.isnan(inv_cov)):
        raise ValueError("Inputs contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(mu)) or np.any(np.isinf(inv_cov)):
        raise ValueError("Inputs contain infinite values")

def _compute_mahalanobis_distance(X: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distances."""
    delta = X - mu
    return np.sqrt(np.sum(delta @ inv_cov * delta, axis=1))

def mahalanobis_distance_fit(
    X: np.ndarray,
    mu: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    normalize: str = "standard",
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, dict]]:
    """
    Compute Mahalanobis distances for anomaly detection.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    mu : np.ndarray, optional
        Mean vector. If None, computed from X.
    cov : np.ndarray, optional
        Covariance matrix. If None, computed from X.
    normalize : str, optional
        Normalization method: "none", "standard", "minmax", or "robust"
    distance_metric : callable, optional
        Custom distance metric function. If None, uses Mahalanobis distance.
    **kwargs : dict
        Additional parameters for normalization and distance computation.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.

    Example:
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = mahalanobis_distance_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, mu if mu is not None else np.mean(X, axis=0),
                    cov if cov is not None else np.linalg.inv(np.cov(X.T)))

    # Normalize data if required
    if normalize == "standard":
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif normalize == "minmax":
        X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    elif normalize == "robust":
        X_norm = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
    else:
        X_norm = X.copy()

    # Compute mean and inverse covariance if not provided
    mu_used = np.mean(X_norm, axis=0) if mu is None else mu
    cov_used = np.cov(X_norm.T) if cov is None else cov
    inv_cov_used = np.linalg.inv(cov_used)

    # Compute distances
    if distance_metric is None:
        distances = _compute_mahalanobis_distance(X_norm, mu_used, inv_cov_used)
    else:
        distances = np.array([distance_metric(x, mu_used) for x in X_norm])

    # Prepare output
    result = {
        "result": distances,
        "metrics": {
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances)
        },
        "params_used": {
            "normalization": normalize,
            "mean_vector": mu_used,
            "covariance_matrix": cov_used
        },
        "warnings": []
    }

    return result

################################################################################
# grubbs_test
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for Grubbs' test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def _calculate_grubbs_statistic(data: np.ndarray, suspected_outlier_index: int) -> float:
    """Calculate Grubbs' test statistic for a suspected outlier."""
    data_without_suspect = np.delete(data, suspected_outlier_index)
    mean = np.mean(data_without_suspect)
    std = np.std(data_without_suspect, ddof=1)
    suspect_value = data[suspected_outlier_index]
    return np.abs(suspect_value - mean) / std

def _calculate_p_value(grubbs_statistic: float, n: int) -> float:
    """Calculate p-value for Grubbs' test statistic."""
    from scipy.stats import t
    df = n - 2
    if df <= 0:
        raise ValueError("Not enough data points to perform Grubbs' test")
    t_critical = t.ppf(1 - 0.5 / n, df)
    p_value = 2 * t.cdf(-np.abs(grubbs_statistic) / np.sqrt(n - 1), df)
    return p_value

def grubbs_test_fit(
    data: np.ndarray,
    alpha: float = 0.05,
    normalize: bool = False,
    normalization_method: str = 'standard',
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    suspected_outlier_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Grubbs' test for outliers in a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1D)
    alpha : float, optional
        Significance level (default: 0.05)
    normalize : bool, optional
        Whether to normalize data (default: False)
    normalization_method : str, optional
        Normalization method ('standard', 'minmax', 'robust') (default: 'standard')
    custom_normalization : callable, optional
        Custom normalization function
    suspected_outlier_index : int, optional
        Index of suspected outlier (if None, test all points)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 10.0])
    >>> result = grubbs_test_fit(data)
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if requested
    if normalize:
        if custom_normalization is not None:
            data = custom_normalization(data)
        else:
            if normalization_method == 'standard':
                data = (data - np.mean(data)) / np.std(data)
            elif normalization_method == 'minmax':
                data = (data - np.min(data)) / (np.max(data) - np.min(data))
            elif normalization_method == 'robust':
                data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
            else:
                raise ValueError("Invalid normalization method")

    n = len(data)
    results = []
    warnings = []

    if suspected_outlier_index is not None:
        indices_to_test = [suspected_outlier_index]
    else:
        indices_to_test = range(n)

    for i in indices_to_test:
        try:
            statistic = _calculate_grubbs_statistic(data, i)
            p_value = _calculate_p_value(statistic, n)

            results.append({
                'index': i,
                'statistic': statistic,
                'p_value': p_value,
                'is_outlier': p_value < alpha
            })
        except Exception as e:
            warnings.append(f"Error processing index {i}: {str(e)}")

    return {
        'result': results,
        'metrics': {
            'alpha': alpha,
            'n_points_tested': len(indices_to_test)
        },
        'params_used': {
            'normalization': normalize,
            'normalization_method': normalization_method if not normalize else None,
            'custom_normalization': custom_normalization is not None
        },
        'warnings': warnings if warnings else None
    }

################################################################################
# esd_test
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for ESD test."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_statistic(data: np.ndarray, alpha: float) -> float:
    """Compute the ESD test statistic."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std == 0:
        raise ValueError("Standard deviation is zero, cannot compute ESD test")

    sorted_data = np.sort(data)
    statistic_values = []

    for i in range(n):
        x_i = sorted_data[i]
        mean_rest = np.mean(sorted_data[i+1:])
        std_rest = np.std(sorted_data[i+1:], ddof=1)

        if std_rest == 0:
            continue

        statistic = abs(x_i - mean) / std
        statistic_values.append(statistic)

    if not statistic_values:
        raise ValueError("Could not compute any valid ESD statistics")

    return max(statistic_values)

def _compute_p_value(statistic: float, n: int) -> float:
    """Compute the p-value for the ESD test."""
    # This is a simplified version - in practice you would use more accurate methods
    p_value = 1 - statistic / (n - 1)
    return max(0, min(1, p_value))

def esd_test_fit(
    data: np.ndarray,
    alpha: float = 0.05,
    normalize: Optional[str] = None,
    metric: str = "mse",
    custom_metric: Optional[Callable] = None,
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Perform the ESD (Extreme Studentized Deviate) test for detecting outliers.

    Parameters:
    -----------
    data : np.ndarray
        Input data to test for anomalies.
    alpha : float, optional
        Significance level (default: 0.05).
    normalize : str or None, optional
        Normalization method (default: None). Options: 'standard', 'minmax'.
    metric : str or callable, optional
        Metric to use (default: 'mse'). Options: 'mse', 'mae'.
    custom_metric : callable or None, optional
        Custom metric function (default: None).
    solver : str, optional
        Solver method (default: 'closed_form'). Options: 'closed_form'.
    regularization : str or None, optional
        Regularization method (default: None). Options: 'l1', 'l2'.
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if specified
    if normalize == "standard":
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            raise ValueError("Standard deviation is zero, cannot standardize")
        data = (data - mean) / std
    elif normalize == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            raise ValueError("Min and max values are equal, cannot min-max normalize")
        data = (data - min_val) / (max_val - min_val)

    # Compute ESD statistic
    try:
        statistic = _compute_statistic(data, alpha)
    except ValueError as e:
        return {
            "result": None,
            "metrics": {},
            "params_used": locals(),
            "warnings": [str(e)]
        }

    # Compute p-value
    n = len(data)
    p_value = _compute_p_value(statistic, n)

    # Determine if anomaly is detected
    anomaly_detected = p_value < alpha

    # Prepare metrics
    metrics = {
        "statistic": statistic,
        "p_value": p_value,
        "anomaly_detected": anomaly_detected
    }

    # Prepare result dictionary
    result = {
        "result": anomaly_detected,
        "metrics": metrics,
        "params_used": locals(),
        "warnings": []
    }

    return result

# Example usage:
# result = esd_test_fit(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]))

################################################################################
# mad_based_outlier
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(
    data: np.ndarray,
    method: str = "standard",
    custom_func: Optional[Callable] = None
) -> np.ndarray:
    """Normalize data using specified method."""
    if custom_func is not None:
        return custom_func(data)

    if method == "none":
        return data
    elif method == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == "robust":
        median = np.median(data, axis=0)
        mad = _compute_mad(data, axis=0)
        return (data - median) / (mad + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_mad(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute Median Absolute Deviation."""
    median = np.median(data, axis=axis)
    diff = np.abs(data - median)
    return np.median(diff, axis=axis)

def _compute_threshold(
    mad_values: np.ndarray,
    threshold_factor: float = 3.0
) -> float:
    """Compute outlier threshold based on MAD."""
    return threshold_factor * mad_values

def _detect_outliers(
    data: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Detect outliers based on MAD threshold."""
    return np.abs(data) > threshold

def mad_based_outlier_fit(
    data: np.ndarray,
    normalization_method: str = "robust",
    custom_normalization: Optional[Callable] = None,
    threshold_factor: float = 3.0
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Detect outliers using MAD-based method.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    normalization_method : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_normalization : callable, optional
        Custom normalization function
    threshold_factor : float, optional
        Factor for outlier threshold (default: 3.0)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, and parameters used

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = mad_based_outlier_fit(data)
    """
    _validate_inputs(data)

    # Normalize data
    normalized_data = _normalize_data(
        data,
        method=normalization_method,
        custom_func=custom_normalization
    )

    # Compute MAD and threshold
    mad_values = _compute_mad(normalized_data)
    threshold = _compute_threshold(mad_values, threshold_factor)

    # Detect outliers
    outliers = _detect_outliers(normalized_data, threshold)
    outlier_indices = np.where(outliers.any(axis=1))[0]

    # Prepare results
    result = {
        "outlier_indices": outlier_indices,
        "normalized_data": normalized_data,
        "mad_values": mad_values,
        "threshold": threshold
    }

    metrics = {
        "n_outliers": len(outlier_indices),
        "outlier_ratio": len(outlier_indices) / data.shape[0]
    }

    params_used = {
        "normalization_method": normalization_method,
        "threshold_factor": threshold_factor
    }

    warnings = []

    return {
        "result": result,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings
    }
