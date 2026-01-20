"""
Quantix – Module detection_novelty
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# novelty_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def novelty_detection_fit(
    X: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    threshold_func: Optional[Callable[[np.ndarray], float]] = None,
    contamination: float = 0.1,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a novelty detection model to the training data.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features)
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    threshold_func : Optional[Callable[[np.ndarray], float]]
        Function to compute the novelty threshold. If None, uses contamination-based threshold.
    contamination : float
        Proportion of outliers in the data (used if threshold_func is None)
    random_state : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'model': The fitted model parameters
        - 'threshold': Computed novelty threshold
        - 'metrics': Dictionary of evaluation metrics
        - 'warnings': List of warning messages

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> result = novelty_detection_fit(X, distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data if specified
    X_normalized = _apply_normalization(X, normalizer)

    # Compute distance matrix
    distance_matrix = _compute_distance_matrix(X_normalized, distance_metric)

    # Fit model and compute threshold
    model = _fit_model(distance_matrix, random_state)
    threshold = _compute_threshold(model, distance_matrix, threshold_func, contamination)

    # Compute metrics
    metrics = _compute_metrics(distance_matrix, threshold)

    return {
        'model': model,
        'threshold': threshold,
        'metrics': metrics,
        'warnings': []
    }

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or infinite values")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _compute_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    """Compute pairwise distance matrix."""
    if metric == 'euclidean':
        return _compute_euclidean_distance(X)
    elif metric == 'manhattan':
        return _compute_manhattan_distance(X)
    elif metric == 'cosine':
        return _compute_cosine_distance(X)
    elif metric == 'minkowski':
        return _compute_minkowski_distance(X)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _fit_model(distance_matrix: np.ndarray, random_state: Optional[int]) -> Dict:
    """Fit novelty detection model."""
    if random_state is not None:
        np.random.seed(random_state)
    # Simple implementation: use median distance as reference
    return {'reference_distances': np.median(distance_matrix, axis=1)}

def _compute_threshold(
    model: Dict,
    distance_matrix: np.ndarray,
    threshold_func: Optional[Callable],
    contamination: float
) -> float:
    """Compute novelty threshold."""
    if threshold_func is not None:
        return threshold_func(distance_matrix)
    # Default implementation: use contamination-based threshold
    sorted_distances = np.sort(distance_matrix.flatten())
    n_outliers = int(contamination * len(sorted_distances))
    return sorted_distances[-n_outliers]

def _compute_metrics(distance_matrix: np.ndarray, threshold: float) -> Dict:
    """Compute evaluation metrics."""
    binary_scores = (distance_matrix > threshold).astype(int)
    return {
        'mean_distance': np.mean(distance_matrix),
        'std_distance': np.std(distance_matrix),
        'outlier_score_mean': np.mean(binary_scores),
        'outlier_score_std': np.std(binary_scores)
    }

# Distance metric implementations
def _compute_euclidean_distance(X: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix."""
    return np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2))

def _compute_manhattan_distance(X: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance matrix."""
    return np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)

def _compute_cosine_distance(X: np.ndarray) -> np.ndarray:
    """Compute Cosine distance matrix."""
    dot_products = X @ X.T
    norms = np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis]
    return 1 - (dot_products / (norms @ norms.T))

def _compute_minkowski_distance(X: np.ndarray, p: float = 3) -> np.ndarray:
    """Compute Minkowski distance matrix."""
    return np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :])**p, axis=2)**(1/p)

################################################################################
# anomaly_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def validate_input(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or infinite values")

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
    elif method != "none":
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm

def compute_distance(
    x: np.ndarray,
    X_ref: np.ndarray,
    metric: str = "euclidean",
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """Compute distance between a point and reference data."""
    if custom_distance is not None:
        return np.mean([custom_distance(x, xi) for xi in X_ref])

    if metric == "euclidean":
        return np.mean(np.linalg.norm(X_ref - x, axis=1))
    elif metric == "manhattan":
        return np.mean(np.sum(np.abs(X_ref - x), axis=1))
    elif metric == "cosine":
        return np.mean(1 - np.sum(X_ref * x, axis=1) / (np.linalg.norm(X_ref, axis=1) * np.linalg.norm(x)))
    elif metric == "minkowski":
        p = 3
        return np.mean(np.sum(np.abs(X_ref - x)**p, axis=1)**(1/p))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def anomaly_detection_fit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    threshold: float = 0.95,
    normalization_method: str = "standard",
    distance_metric: str = "euclidean",
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit anomaly detection model and detect anomalies in test data.

    Parameters:
    -----------
    X_train : np.ndarray
        Training data used to establish normal behavior.
    X_test : np.ndarray
        Test data where anomalies will be detected.
    threshold : float, optional
        Threshold for anomaly detection (default: 0.95).
    normalization_method : str, optional
        Normalization method (default: "standard").
    distance_metric : str, optional
        Distance metric to use (default: "euclidean").
    custom_normalizer : callable, optional
        Custom normalization function.
    custom_distance : callable, optional
        Custom distance function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_input(X_train)
    validate_input(X_test)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test data must have the same number of features")

    # Normalize data
    X_train_norm = normalize_data(X_train, normalization_method, custom_normalizer)
    X_test_norm = normalize_data(X_test, normalization_method, custom_normalizer)

    # Compute distances
    distances = np.array([compute_distance(x, X_train_norm, distance_metric, custom_distance) for x in X_test_norm])

    # Determine threshold
    quantile = np.quantile(distances, threshold)
    anomalies = distances > quantile

    # Calculate metrics
    anomaly_score = np.mean(distances)
    num_anomalies = np.sum(anomalies)

    return {
        "result": {
            "anomaly_scores": distances,
            "is_anomaly": anomalies
        },
        "metrics": {
            "mean_distance": float(anomaly_score),
            "num_anomalies": int(num_anomalies)
        },
        "params_used": {
            "normalization_method": normalization_method,
            "distance_metric": distance_metric
        },
        "warnings": []
    }

# Example usage:
"""
X_train = np.random.randn(100, 5)
X_test = np.random.randn(20, 5)

result = anomaly_detection_fit(
    X_train=X_train,
    X_test=X_test,
    threshold=0.95,
    normalization_method="standard",
    distance_metric="euclidean"
)
"""

################################################################################
# change_point_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def change_point_detection_fit(
    data: np.ndarray,
    *,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Detect change points in a time series data.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    normalizer : Callable[[np.ndarray], np.ndarray]
        Function to normalize the data. Default is identity function.
    metric : str
        Metric to use for change point detection. Options: 'mse', 'mae', 'r2'.
    distance : str
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    penalty : Optional[str]
        Penalty to use. Options: None, 'l1', 'l2'.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 10, 11, 12])
    >>> result = change_point_detection_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, penalty)

    # Normalize data
    normalized_data = normalizer(data)

    # Choose metric and distance functions
    metric_func, distance_func = _choose_metric_and_distance(metric, distance, custom_metric, custom_distance)

    # Detect change points
    if solver == 'closed_form':
        result = _closed_form_solver(normalized_data, metric_func, distance_func)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(normalized_data, metric_func, distance_func, penalty, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(data, result['change_points'], metric_func)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if hasattr(normalizer, '__name__') else 'custom',
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'penalty': penalty,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    normalizer: Callable[[np.ndarray], np.ndarray],
    metric: str,
    distance: str,
    solver: str,
    penalty: Optional[str]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

    valid_metrics = ['mse', 'mae', 'r2']
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}.")

    valid_distances = ['euclidean', 'manhattan', 'cosine']
    if distance not in valid_distances:
        raise ValueError(f"Distance must be one of {valid_distances}.")

    valid_solvers = ['closed_form', 'gradient_descent']
    if solver not in valid_solvers:
        raise ValueError(f"Solver must be one of {valid_solvers}.")

    valid_penalties = [None, 'l1', 'l2']
    if penalty not in valid_penalties:
        raise ValueError(f"Penalty must be one of {valid_penalties}.")

def _choose_metric_and_distance(
    metric: str,
    distance: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> tuple:
    """Choose metric and distance functions."""
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        if metric == 'mse':
            metric_func = _mse_metric
        elif metric == 'mae':
            metric_func = _mae_metric
        elif metric == 'r2':
            metric_func = _r2_metric

    if custom_distance is not None:
        distance_func = custom_distance
    else:
        if distance == 'euclidean':
            distance_func = _euclidean_distance
        elif distance == 'manhattan':
            distance_func = _manhattan_distance
        elif distance == 'cosine':
            distance_func = _cosine_distance

    return metric_func, distance_func

def _mse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error metric."""
    return np.mean((y_true - y_pred) ** 2)

def _mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error metric."""
    return np.mean(np.abs(y_true - y_pred))

def _r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _closed_form_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Union[list, int]]:
    """Closed form solver for change point detection."""
    n = len(data)
    change_points = []
    current_segment_start = 0

    for i in range(1, n):
        segment = data[current_segment_start:i+1]
        if metric_func(data[:i+1], np.mean(segment)) < metric_func(data[:current_segment_start+1], np.mean(data[current_segment_start:i+1])):
            change_points.append(i)
            current_segment_start = i

    return {
        'change_points': change_points,
        'num_segments': len(change_points) + 1
    }

def _gradient_descent_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Union[list, int]]:
    """Gradient descent solver for change point detection."""
    n = len(data)
    change_points = []
    current_segment_start = 0
    prev_metric = float('inf')

    for _ in range(max_iter):
        for i in range(current_segment_start + 1, n):
            segment = data[current_segment_start:i+1]
            current_metric = metric_func(data[:i+1], np.mean(segment))

            if penalty == 'l1':
                current_metric += abs(i - current_segment_start)
            elif penalty == 'l2':
                current_metric += (i - current_segment_start) ** 2

            if current_metric < prev_metric - tol:
                change_points.append(i)
                current_segment_start = i
                prev_metric = current_metric
                break

    return {
        'change_points': change_points,
        'num_segments': len(change_points) + 1
    }

def _calculate_metrics(
    data: np.ndarray,
    change_points: list,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for change point detection."""
    segments = np.split(data, change_points)
    segment_means = [np.mean(segment) for segment in segments]
    predicted_data = np.concatenate([np.full_like(segment, mean) for segment, mean in zip(segments, segment_means)])
    return {
        'metric_value': metric_func(data, predicted_data),
        'num_segments': len(segments)
    }

################################################################################
# concept_drift_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def concept_drift_detection_fit(
    X_ref: np.ndarray,
    X_test: np.ndarray,
    *,
    metric: Union[str, Callable] = 'mse',
    distance: Union[str, Callable] = 'euclidean',
    threshold: float = 0.05,
    normalize: bool = True,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Detect concept drift between reference and test data.

    Parameters:
    -----------
    X_ref : np.ndarray
        Reference data matrix of shape (n_samples, n_features)
    X_test : np.ndarray
        Test data matrix of shape (n_samples, n_features)
    metric : str or callable
        Metric to use for drift detection. Options: 'mse', 'mae', 'r2', or custom callable.
    distance : str or callable
        Distance metric for feature space. Options: 'euclidean', 'manhattan', 'cosine', or custom callable.
    threshold : float
        Threshold for drift detection (p-value).
    normalize : bool
        Whether to normalize the data before computation.
    custom_metric : callable, optional
        Custom metric function if not using built-in options.
    custom_distance : callable, optional
        Custom distance function if not using built-in options.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(X_ref, X_test)

    # Normalize data if required
    if normalize:
        X_ref = _normalize_data(X_ref)
        X_test = _normalize_data(X_test)

    # Select metric function
    if isinstance(metric, str):
        metric_func = _get_metric_function(metric)
    else:
        if custom_metric is None:
            raise ValueError("custom_metric must be provided when metric is a callable")
        metric_func = custom_metric

    # Select distance function
    if isinstance(distance, str):
        distance_func = _get_distance_function(distance)
    else:
        if custom_distance is None:
            raise ValueError("custom_distance must be provided when distance is a callable")
        distance_func = custom_distance

    # Compute drift statistic
    drift_statistic = _compute_drift_statistic(X_ref, X_test, metric_func, distance_func)

    # Determine if drift is detected
    is_drift = drift_statistic > threshold

    return {
        'result': {
            'is_drift_detected': is_drift,
            'drift_statistic': drift_statistic
        },
        'metrics': {
            'metric_used': metric if isinstance(metric, str) else 'custom',
            'distance_used': distance if isinstance(distance, str) else 'custom'
        },
        'params_used': {
            'threshold': threshold,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_inputs(X_ref: np.ndarray, X_test: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X_ref, np.ndarray) or not isinstance(X_test, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X_ref.shape[1] != X_test.shape[1]:
        raise ValueError("Reference and test data must have the same number of features")
    if np.any(np.isnan(X_ref)) or np.any(np.isnan(X_test)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(X_ref)) or np.any(np.isinf(X_test)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize data using standard scaling."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

def _get_metric_function(metric: str) -> Callable:
    """Get the appropriate metric function based on input string."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_distance_function(distance: str) -> Callable:
    """Get the appropriate distance function based on input string."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if distance not in distances:
        raise ValueError(f"Unknown distance: {distance}")
    return distances[distance]

def _compute_drift_statistic(
    X_ref: np.ndarray,
    X_test: np.ndarray,
    metric_func: Callable,
    distance_func: Callable
) -> float:
    """Compute the drift statistic between reference and test data."""
    # Calculate metric difference
    metric_ref = metric_func(X_ref)
    metric_test = metric_func(X_test)
    metric_diff = abs(metric_ref - metric_test)

    # Calculate distance between feature distributions
    dist_ref = _compute_feature_distances(X_ref, distance_func)
    dist_test = _compute_feature_distances(X_test, distance_func)

    # Combine metrics into a single drift statistic
    return metric_diff * np.mean(dist_ref + dist_test)

def _compute_feature_distances(X: np.ndarray, distance_func: Callable) -> np.ndarray:
    """Compute pairwise distances between features."""
    n_features = X.shape[1]
    distances = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i+1, n_features):
            distances[i, j] = distance_func(X[:, i], X[:, j])
    return distances + distances.T

def _mean_squared_error(X: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((X - X.mean(axis=0))**2)

def _mean_absolute_error(X: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(X - X.mean(axis=0)))

def _r_squared(X: np.ndarray) -> float:
    """Compute R-squared."""
    ss_total = np.sum((X - X.mean())**2)
    if ss_total == 0:
        return 1.0
    return 1 - np.sum((X - X.mean(axis=0))**2) / ss_total

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.sqrt(np.sum((a - b)**2))

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot_product / (norm_a * norm_b + 1e-8))

################################################################################
# outlier_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def outlier_detection_fit(
    data: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0,
    metric: Union[str, Callable] = 'euclidean',
    normalization: str = 'standard',
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Detect outliers in a dataset using various statistical methods.

    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Method for outlier detection ('zscore', 'iqr', 'mahalanobis')
    threshold : float, optional
        Threshold for outlier detection
    metric : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') or custom callable
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_metric : callable, optional
        Custom distance metric function

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.random.randn(100, 5)
    >>> result = outlier_detection_fit(data, method='zscore', threshold=3.0)
    """
    # Validate inputs
    _validate_inputs(data, method, metric, normalization)

    # Normalize data
    normalized_data = _apply_normalization(data, normalization)

    # Calculate outlier scores
    if method == 'zscore':
        scores = _calculate_zscores(normalized_data)
    elif method == 'iqr':
        scores = _calculate_iqr_scores(normalized_data)
    elif method == 'mahalanobis':
        scores = _calculate_mahalanobis_scores(normalized_data, metric, custom_metric)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Detect outliers
    is_outlier = scores > threshold

    # Calculate metrics
    metrics = _calculate_metrics(data, is_outlier)

    return {
        'result': {
            'outliers': data[is_outlier],
            'indices': np.where(is_outlier)[0],
            'scores': scores
        },
        'metrics': metrics,
        'params_used': {
            'method': method,
            'threshold': threshold,
            'metric': metric if isinstance(metric, str) else 'custom',
            'normalization': normalization
        },
        'warnings': _check_warnings(data, is_outlier)
    }

def _validate_inputs(
    data: np.ndarray,
    method: str,
    metric: Union[str, Callable],
    normalization: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    valid_methods = ['zscore', 'iqr', 'mahalanobis']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    valid_normalizations = ['none', 'standard', 'minmax', 'robust']
    if normalization not in valid_normalizations:
        raise ValueError(f"Normalization must be one of {valid_normalizations}")

    if isinstance(metric, str):
        valid_metrics = ['euclidean', 'manhattan', 'cosine']
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}")

def _apply_normalization(
    data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == 'minmax':
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif method == 'robust':
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _calculate_zscores(
    data: np.ndarray
) -> np.ndarray:
    """Calculate z-scores for each sample."""
    return np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))

def _calculate_iqr_scores(
    data: np.ndarray
) -> np.ndarray:
    """Calculate IQR-based scores for each sample."""
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    return np.abs((data - (q1 + q3)/2) / iqr)

def _calculate_mahalanobis_scores(
    data: np.ndarray,
    metric: str = 'euclidean',
    custom_metric: Optional[Callable] = None
) -> np.ndarray:
    """Calculate Mahalanobis distances for each sample."""
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)

    if custom_metric is not None:
        return _apply_custom_metric(data, inv_cov, custom_metric)

    if metric == 'euclidean':
        return np.sqrt(np.sum((data - np.mean(data, axis=0)) @ inv_cov * (data - np.mean(data, axis=0)), axis=1))
    elif metric == 'manhattan':
        return np.sum(np.abs((data - np.mean(data, axis=0)) @ inv_cov), axis=1)
    elif metric == 'cosine':
        return 1 - np.sum((data - np.mean(data, axis=0)) @ inv_cov * (data - np.mean(data, axis=0)), axis=1) / (
            np.linalg.norm((data - np.mean(data, axis=0)) @ inv_cov, axis=1) *
            np.linalg.norm(data - np.mean(data, axis=0), axis=1)
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _apply_custom_metric(
    data: np.ndarray,
    inv_cov: np.ndarray,
    metric_func: Callable
) -> np.ndarray:
    """Apply custom distance metric."""
    centered_data = data - np.mean(data, axis=0)
    transformed_data = centered_data @ inv_cov
    return np.array([metric_func(x, y) for x, y in zip(centered_data, transformed_data)])

def _calculate_metrics(
    data: np.ndarray,
    is_outlier: np.ndarray
) -> Dict:
    """Calculate various metrics about the outliers."""
    n_outliers = np.sum(is_outlier)
    outlier_ratio = n_outliers / len(data)

    return {
        'n_samples': len(data),
        'n_features': data.shape[1],
        'n_outliers': n_outliers,
        'outlier_ratio': outlier_ratio
    }

def _check_warnings(
    data: np.ndarray,
    is_outlier: np.ndarray
) -> list:
    """Check for potential issues and return warnings."""
    warnings = []

    if np.sum(is_outlier) == 0:
        warnings.append("No outliers detected")
    elif np.sum(is_outlier) / len(data) > 0.2:
        warnings.append("High proportion of outliers detected")

    if np.any(np.isinf(data)):
        warnings.append("Data contains infinite values")

    return warnings

################################################################################
# novel_class_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def novel_class_detection_fit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    threshold_func: Optional[Callable[[np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    contamination: float = 0.1,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a novelty class detection model.

    Parameters
    ----------
    X_train : np.ndarray
        Training data of shape (n_samples, n_features).
    X_test : np.ndarray
        Test data of shape (n_samples, n_features).
    normalizer : Optional[Callable], default=None
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : str, default='euclidean'
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    threshold_func : Optional[Callable], default=None
        Function to compute the threshold for novelty detection.
    custom_metric : Optional[Callable], default=None
        Custom metric function to evaluate the model.
    contamination : float, default=0.1
        Expected proportion of outliers in the data.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> X_train = np.random.randn(100, 5)
    >>> X_test = np.random.randn(20, 5)
    >>> result = novel_class_detection_fit(X_train, X_test)
    """
    # Validate inputs
    _validate_inputs(X_train, X_test)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X_train = normalizer(X_train)
        X_test = normalizer(X_test)

    # Compute distances
    distances = _compute_distances(X_train, X_test, distance_metric=distance_metric)

    # Compute threshold if a function is provided
    if threshold_func is not None:
        threshold = threshold_func(distances)
    else:
        threshold = np.percentile(distances, 100 * (1 - contamination))

    # Detect novelty
    is_novel = distances > threshold

    # Compute metrics if a custom metric is provided
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(X_train, X_test)

    # Prepare output
    result = {
        'result': is_novel,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'contamination': contamination
        },
        'warnings': []
    }

    return result

def _validate_inputs(X_train: np.ndarray, X_test: np.ndarray) -> None:
    """Validate the input data."""
    if not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray):
        raise TypeError("X_train and X_test must be numpy arrays.")
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2-dimensional.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features.")
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        raise ValueError("X_train and X_test must not contain NaN values.")
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        raise ValueError("X_train and X_test must not contain infinite values.")

def _compute_distances(
    X_train: np.ndarray,
    X_test: np.ndarray,
    distance_metric: str = 'euclidean'
) -> np.ndarray:
    """Compute distances between training and test data."""
    if distance_metric == 'euclidean':
        return np.linalg.norm(X_test[:, np.newaxis] - X_train, axis=2)
    elif distance_metric == 'manhattan':
        return np.sum(np.abs(X_test[:, np.newaxis] - X_train), axis=2)
    elif distance_metric == 'cosine':
        return 1 - np.dot(X_test, X_train.T) / (
            np.linalg.norm(X_test, axis=1)[:, np.newaxis] *
            np.linalg.norm(X_train, axis=1)[np.newaxis, :]
        )
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

################################################################################
# data_stream_novelty
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def data_stream_novelty_fit(
    X: np.ndarray,
    window_size: int = 100,
    threshold: float = 3.0,
    metric: str = 'mse',
    distance: str = 'euclidean',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Fit a novelty detection model on a data stream.

    Parameters
    ----------
    X : np.ndarray
        Input data stream of shape (n_samples, n_features).
    window_size : int, optional
        Size of the sliding window for novelty detection, by default 100.
    threshold : float, optional
        Threshold for novelty detection, by default 3.0.
    metric : str, optional
        Metric to use for novelty detection ('mse', 'mae'), by default 'mse'.
    distance : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine'), by default 'euclidean'.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Custom normalizer function, by default None.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        Dictionary containing the results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, window_size)

    # Initialize results dictionary
    results = {
        'result': {},
        'metrics': {},
        'params_used': {
            'window_size': window_size,
            'threshold': threshold,
            'metric': metric,
            'distance': distance
        },
        'warnings': []
    }

    # Normalize data if normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Initialize sliding window
    window = np.zeros((window_size, X.shape[1]))
    novelty_scores = []

    for i in range(len(X)):
        # Update sliding window
        if i < window_size:
            window[i] = X[i]
        else:
            window = np.roll(window, -1, axis=0)
            window[-1] = X[i]

        # Compute novelty score
        if i >= window_size:
            score = _compute_novelty_score(
                X[i],
                window[:-1],
                metric,
                distance,
                custom_metric,
                custom_distance
            )
            novelty_scores.append(score)

            # Detect novelty
            if score > threshold:
                results['result'][i] = {
                    'novelty_score': score,
                    'is_novel': True
                }
            else:
                results['result'][i] = {
                    'novelty_score': score,
                    'is_novel': False
                }

    # Compute metrics if there are novelty scores
    if novelty_scores:
        results['metrics']['mean_novelty_score'] = np.mean(novelty_scores)
        results['metrics']['max_novelty_score'] = np.max(novelty_scores)

    return results

def _validate_inputs(X: np.ndarray, window_size: int) -> None:
    """
    Validate the input data and parameters.

    Parameters
    ----------
    X : np.ndarray
        Input data stream.
    window_size : int
        Size of the sliding window.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or inf values.")

def _compute_novelty_score(
    point: np.ndarray,
    window: np.ndarray,
    metric: str,
    distance: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> float:
    """
    Compute the novelty score for a given point.

    Parameters
    ----------
    point : np.ndarray
        The point to evaluate.
    window : np.ndarray
        The sliding window of previous points.
    metric : str
        Metric to use for novelty detection.
    distance : str
        Distance metric to use.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function, by default None.

    Returns
    -------
    float
        The novelty score.
    """
    if custom_metric is not None:
        return custom_metric(point, window)
    elif metric == 'mse':
        return _compute_mse_novelty_score(point, window)
    elif metric == 'mae':
        return _compute_mae_novelty_score(point, window)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_mse_novelty_score(point: np.ndarray, window: np.ndarray) -> float:
    """
    Compute the MSE novelty score.

    Parameters
    ----------
    point : np.ndarray
        The point to evaluate.
    window : np.ndarray
        The sliding window of previous points.

    Returns
    -------
    float
        The MSE novelty score.
    """
    errors = np.sum((window - point) ** 2, axis=1)
    return np.mean(errors)

def _compute_mae_novelty_score(point: np.ndarray, window: np.ndarray) -> float:
    """
    Compute the MAE novelty score.

    Parameters
    ----------
    point : np.ndarray
        The point to evaluate.
    window : np.ndarray
        The sliding window of previous points.

    Returns
    -------
    float
        The MAE novelty score.
    """
    errors = np.abs(window - point)
    return np.mean(np.sum(errors, axis=1))

def _compute_distance(
    point: np.ndarray,
    window: np.ndarray,
    distance: str,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """
    Compute the distance between a point and each point in the window.

    Parameters
    ----------
    point : np.ndarray
        The point to evaluate.
    window : np.ndarray
        The sliding window of previous points.
    distance : str
        Distance metric to use.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function, by default None.

    Returns
    -------
    np.ndarray
        Array of distances.
    """
    if custom_distance is not None:
        return np.array([custom_distance(point, p) for p in window])
    elif distance == 'euclidean':
        return np.linalg.norm(window - point, axis=1)
    elif distance == 'manhattan':
        return np.sum(np.abs(window - point), axis=1)
    elif distance == 'cosine':
        return 1 - np.dot(window, point) / (np.linalg.norm(window, axis=1) * np.linalg.norm(point))
    else:
        raise ValueError(f"Unknown distance: {distance}")

################################################################################
# static_data_novelty
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def static_data_novelty_fit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    threshold_func: Optional[Callable[[np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    alpha: float = 0.05
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a static data novelty detection model.

    Parameters
    ----------
    X_train : np.ndarray
        Training data of shape (n_samples, n_features)
    X_test : np.ndarray
        Test data of shape (m_samples, n_features)
    normalizer : Optional[Callable]
        Function to normalize data. If None, no normalization is applied.
    distance_metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski')
    threshold_func : Optional[Callable]
        Function to compute novelty threshold. If None, uses default method.
    custom_metric : Optional[Callable]
        Custom metric function. If None, uses default distance metric.
    alpha : float
        Significance level for novelty detection

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Novelty scores for test data
        - 'metrics': Computed metrics
        - 'params_used': Parameters used in the computation
        - 'warnings': Any warnings generated

    Example
    -------
    >>> X_train = np.random.rand(100, 5)
    >>> X_test = np.random.rand(20, 5)
    >>> result = static_data_novelty_fit(X_train, X_test)
    """
    # Validate inputs
    _validate_inputs(X_train, X_test)

    # Normalize data if specified
    if normalizer is not None:
        X_train = normalizer(X_train)
        X_test = normalizer(X_test)

    # Compute novelty scores
    novelty_scores = _compute_novelty_scores(
        X_train, X_test,
        distance_metric=distance_metric,
        custom_metric=custom_metric
    )

    # Compute threshold if function provided, otherwise use default
    if threshold_func is not None:
        threshold = threshold_func(novelty_scores)
    else:
        threshold = _default_threshold(novelty_scores, alpha=alpha)

    # Compute metrics
    metrics = {
        'mean_novelty_score': np.mean(novelty_scores),
        'std_novelty_score': np.std(novelty_scores)
    }

    # Prepare output
    result = {
        'result': novelty_scores,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'alpha': alpha
        },
        'warnings': []
    }

    return result

def _validate_inputs(X_train: np.ndarray, X_test: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray):
        raise TypeError("Input data must be numpy arrays")
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test data must have the same number of features")
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("Training data contains NaN or infinite values")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        raise ValueError("Test data contains NaN or infinite values")

def _compute_novelty_scores(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    distance_metric: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> np.ndarray:
    """Compute novelty scores for test data."""
    if custom_metric is not None:
        return np.array([custom_metric(x, X_train) for x in X_test])

    distances = _compute_distances(X_test, X_train, metric=distance_metric)
    return np.min(distances, axis=1)

def _compute_distances(
    X_test: np.ndarray,
    X_train: np.ndarray,
    *,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Compute pairwise distances between test and training data."""
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    if metric == 'euclidean':
        distances = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train) ** 2, axis=2))
    elif metric == 'manhattan':
        distances = np.sum(np.abs(X_test[:, np.newaxis, :] - X_train), axis=2)
    elif metric == 'cosine':
        dot_products = np.sum(X_test[:, np.newaxis, :] * X_train, axis=2)
        norms = np.linalg.norm(X_test, axis=1)[:, np.newaxis] * np.linalg.norm(X_train, axis=1)
        distances = 1 - dot_products / norms
    elif metric == 'minkowski':
        p = 3  # default Minkowski exponent
        distances = np.sum(np.abs(X_test[:, np.newaxis, :] - X_train) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return distances

def _default_threshold(scores: np.ndarray, *, alpha: float = 0.05) -> float:
    """Compute default novelty threshold."""
    return np.percentile(scores, 100 * (1 - alpha))

################################################################################
# supervised_novelty_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """Validate input data dimensions and types."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input arrays must not contain NaN values")
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Input arrays must not contain infinite values")

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

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray,
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
        return 1 - (ss_res / (ss_tot + 1e-8))
    elif metric == 'logloss':
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_distance(X: np.ndarray, method: str = 'euclidean',
                     p: Optional[float] = None) -> Callable:
    """Return distance function based on specified method."""
    if method == 'euclidean':
        return lambda a, b: np.linalg.norm(a - b)
    elif method == 'manhattan':
        return lambda a, b: np.sum(np.abs(a - b))
    elif method == 'cosine':
        return lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    elif method == 'minkowski':
        if p is None:
            raise ValueError("Minkowski distance requires p parameter")
        return lambda a, b: np.sum(np.abs(a - b) ** p) ** (1 / p)
    else:
        raise ValueError(f"Unknown distance method: {method}")

def _fit_model(X: np.ndarray, y: np.ndarray,
               solver: str = 'closed_form',
               regularization: str = 'none',
               alpha: float = 1.0,
               max_iter: int = 1000,
               tol: float = 1e-4) -> Dict[str, np.ndarray]:
    """Fit model using specified solver and regularization."""
    if solver == 'closed_form':
        X_tx = np.dot(X.T, X)
        if regularization == 'l1':
            raise ValueError("L1 regularization not supported with closed form solver")
        elif regularization == 'l2':
            X_tx += alpha * np.eye(X.shape[1])
        elif regularization == 'elasticnet':
            raise ValueError("ElasticNet not supported with closed form solver")
        return {'weights': np.linalg.solve(X_tx, np.dot(X.T, y))}
    elif solver == 'gradient_descent':
        weights = np.zeros(X.shape[1])
        learning_rate = 0.01
        for _ in range(max_iter):
            gradient = np.dot(X.T, (np.dot(X, weights) - y)) / X.shape[0]
            if regularization == 'l1':
                gradient += alpha * np.sign(weights)
            elif regularization == 'l2':
                gradient += 2 * alpha * weights
            elif regularization == 'elasticnet':
                gradient += alpha * (np.sign(weights) + 2 * weights)
            new_weights = weights - learning_rate * gradient
            if np.linalg.norm(new_weights - weights) < tol:
                break
            weights = new_weights
        return {'weights': weights}
    else:
        raise ValueError(f"Unknown solver: {solver}")

def supervised_novelty_detection_fit(X: np.ndarray, y: np.ndarray,
                                    normalization: str = 'standard',
                                    metric: Union[str, Callable] = 'mse',
                                    distance_method: str = 'euclidean',
                                    p: Optional[float] = None,
                                    solver: str = 'closed_form',
                                    regularization: str = 'none',
                                    alpha: float = 1.0,
                                    max_iter: int = 1000,
                                    tol: float = 1e-4) -> Dict:
    """
    Fit a supervised novelty detection model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    metric : str or callable, optional
        Metric to evaluate performance ('mse', 'mae', 'r2', 'logloss')
    distance_method : str, optional
        Distance method for novelty detection ('euclidean', 'manhattan',
        'cosine', 'minkowski')
    p : float, optional
        Parameter for Minkowski distance
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent')
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet')
    alpha : float, optional
        Regularization strength
    max_iter : int, optional
        Maximum number of iterations for iterative solvers
    tol : float, optional
        Tolerance for convergence

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': fitted model parameters
        - 'metrics': computed metrics
        - 'params_used': parameters used for fitting
        - 'warnings': any warnings generated during fitting

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, size=100)
    >>> result = supervised_novelty_detection_fit(X, y,
    ...                                         normalization='standard',
    ...                                         metric='r2')
    """
    warnings = []

    # Validate inputs
    _validate_inputs(X, y)

    # Normalize data
    X_norm = _normalize_data(X, normalization)

    # Fit model
    params_used = {
        'normalization': normalization,
        'metric': metric if not callable(metric) else 'custom',
        'distance_method': distance_method,
        'p': p,
        'solver': solver,
        'regularization': regularization,
        'alpha': alpha,
        'max_iter': max_iter,
        'tol': tol
    }

    model = _fit_model(X_norm, y, solver=solver,
                      regularization=regularization,
                      alpha=alpha,
                      max_iter=max_iter,
                      tol=tol)

    # Compute predictions and metrics
    y_pred = np.dot(X_norm, model['weights'])
    primary_metric = _compute_metric(y, y_pred, metric)

    # Compute novelty scores (distance to decision boundary)
    distance_func = _compute_distance(X_norm, distance_method, p)
    novelty_scores = np.abs(y_pred - 0.5) / (np.linalg.norm(model['weights']) + 1e-8)

    return {
        'result': {
            'model_parameters': model,
            'novelty_scores': novelty_scores
        },
        'metrics': {
            primary_metric if isinstance(metric, str) else 'custom_metric': primary_metric
        },
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# unsupervised_novelty_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def unsupervised_novelty_detection_fit(
    X: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: str = 'euclidean',
    novelty_threshold: float = 0.95,
    contamination: Optional[float] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Fit an unsupervised novelty detection model.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : str, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    novelty_threshold : float, optional
        Threshold for novelty detection (0-1).
    contamination : Optional[float], optional
        Expected proportion of outliers in the data. If None, uses novelty_threshold.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Overrides distance_metric if provided.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - 'result': Array of novelty scores.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = unsupervised_novelty_detection_fit(X, distance_metric='euclidean')
    """
    # Validate inputs
    _validate_inputs(X)

    # Normalize data if normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Compute distance matrix
    distance_matrix = _compute_distance_matrix(
        X_normalized, distance_metric, custom_distance
    )

    # Compute novelty scores
    novelty_scores = _compute_novelty_scores(distance_matrix, contamination)

    # Apply threshold to get binary novelty detection
    is_novel = novelty_scores >= novelty_threshold

    # Compute metrics
    metrics = _compute_metrics(novelty_scores, is_novel)

    # Prepare output
    result = {
        'result': novelty_scores,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'novelty_threshold': novelty_threshold,
            'contamination': contamination
        },
        'warnings': []
    }

    return result

def _validate_inputs(X: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Input X must not contain NaN or infinite values.")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return X
    try:
        return normalizer(X)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _compute_distance_matrix(
    X: np.ndarray,
    distance_metric: str,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> np.ndarray:
    """Compute the distance matrix."""
    if custom_distance is not None:
        return _compute_custom_distance_matrix(X, custom_distance)

    distance_metric = distance_metric.lower()
    if distance_metric == 'euclidean':
        return np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
    elif distance_metric == 'manhattan':
        return np.abs(X[:, np.newaxis] - X).sum(axis=2)
    elif distance_metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def _compute_custom_distance_matrix(
    X: np.ndarray,
    custom_distance: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Compute distance matrix using custom distance function."""
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance_matrix[i, j] = custom_distance(X[i], X[j])
    return distance_matrix

def _compute_novelty_scores(
    distance_matrix: np.ndarray,
    contamination: Optional[float]
) -> np.ndarray:
    """Compute novelty scores from distance matrix."""
    if contamination is not None and (contamination < 0 or contamination > 1):
        raise ValueError("Contamination must be between 0 and 1.")

    # Compute local density
    k = max(5, int(np.sqrt(distance_matrix.shape[0])))
    local_density = np.zeros(distance_matrix.shape[0])
    for i in range(distance_matrix.shape[0]):
        nearest_neighbors = np.partition(distance_matrix[i], k)[:k]
        local_density[i] = 1 / (nearest_neighbors.sum() + 1e-6)

    # Compute novelty scores
    novelty_scores = np.zeros(distance_matrix.shape[0])
    for i in range(distance_matrix.shape[0]):
        novelty_scores[i] = np.mean(local_density) / (local_density[i] + 1e-6)

    if contamination is not None:
        threshold = np.quantile(novelty_scores, 1 - contamination)
        novelty_scores = novelty_scores / threshold

    return novelty_scores

def _compute_metrics(
    novelty_scores: np.ndarray,
    is_novel: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for novelty detection."""
    return {
        'mean_novelty_score': float(np.mean(novelty_scores)),
        'std_novelty_score': float(np.std(novelty_scores)),
        'novelty_rate': float(np.mean(is_novel))
    }

################################################################################
# semi_supervised_novelty_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, Tuple

def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Labels array of shape (n_samples,) if semi-supervised.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        Validated X and y.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    if y is not None and (not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != X.shape[0]):
        raise ValueError("y must be a 1D numpy array with the same number of samples as X.")
    if np.any(np.isnan(X)) or (y is not None and np.any(np.isnan(y))):
        raise ValueError("Input arrays must not contain NaN values.")
    return X, y

def normalize_data(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize input data.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    method : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
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

def compute_distance_matrix(X: np.ndarray, distance_metric: str = 'euclidean', custom_metric: Optional[Callable] = None) -> np.ndarray:
    """
    Compute distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    distance_metric : str
        Distance metric: 'euclidean', 'manhattan', 'cosine', or 'minkowski'.
    custom_metric : Optional[Callable]
        Custom distance metric function.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples).
    """
    if custom_metric is not None:
        return np.array([[custom_metric(x1, x2) for x2 in X] for x1 in X])

    if distance_metric == 'euclidean':
        return np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    elif distance_metric == 'manhattan':
        return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
    elif distance_metric == 'cosine':
        return 1 - np.dot(X, X.T) / (np.linalg.norm(X, axis=1)[:, np.newaxis] * np.linalg.norm(X, axis=1))
    elif distance_metric == 'minkowski':
        p = 3
        return np.sum(np.abs(X[:, np.newaxis] - X) ** p, axis=2) ** (1/p)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def semi_supervised_novelty_detection_fit(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalization_method: str = 'standard',
    distance_metric: str = 'euclidean',
    custom_distance_metric: Optional[Callable] = None,
    solver: str = 'gradient_descent',
    regularization: str = 'none',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Perform semi-supervised novelty detection.

    Parameters
    ----------
    X : np.ndarray
        Input features array of shape (n_samples, n_features).
    y : Optional[np.ndarray]
        Labels array of shape (n_samples,) if semi-supervised.
    normalization_method : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    distance_metric : str
        Distance metric: 'euclidean', 'manhattan', 'cosine', or 'minkowski'.
    custom_distance_metric : Optional[Callable]
        Custom distance metric function.
    solver : str
        Solver method: 'gradient_descent', 'newton', or 'coordinate_descent'.
    regularization : str
        Regularization method: 'none', 'l1', 'l2', or 'elasticnet'.
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    custom_metric : Optional[Callable]
        Custom metric function.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    X, y = validate_inputs(X, y)
    X_normalized = normalize_data(X, normalization_method)
    distance_matrix = compute_distance_matrix(X_normalized, distance_metric, custom_distance_metric)

    # Placeholder for the actual novelty detection logic
    result = np.zeros(X.shape[0])
    metrics = {}
    params_used = {
        'normalization_method': normalization_method,
        'distance_metric': distance_metric,
        'solver': solver,
        'regularization': regularization,
        'tol': tol,
        'max_iter': max_iter
    }
    warnings = []

    return {
        'result': result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': warnings
    }

# Example usage:
# result = semi_supervised_novelty_detection_fit(X_train, y_train)

################################################################################
# autoencoder_novelty
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def autoencoder_novelty_fit(
    X: np.ndarray,
    encoder_architecture: list = [64, 32],
    decoder_architecture: list = [32, 64],
    activation: str = 'relu',
    output_activation: str = 'sigmoid',
    loss_metric: Union[str, Callable] = 'mse',
    solver: str = 'adam',
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    tol: float = 1e-4,
    validation_split: float = 0.2,
    normalize: bool = True,
    custom_normalizer: Optional[Callable] = None,
    verbose: bool = False
) -> Dict:
    """
    Fit an autoencoder model for novelty detection.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    encoder_architecture : list
        List of integers specifying the number of neurons in each layer of the encoder.
    decoder_architecture : list
        List of integers specifying the number of neurons in each layer of the decoder.
    activation : str
        Activation function for hidden layers ('relu', 'tanh', etc.).
    output_activation : str
        Activation function for the output layer ('sigmoid', 'linear', etc.).
    loss_metric : Union[str, Callable]
        Loss metric to optimize ('mse', 'mae', custom callable).
    solver : str
        Optimization algorithm ('adam', 'sgd', etc.).
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size for training.
    epochs : int
        Number of training epochs.
    tol : float
        Tolerance for early stopping.
    validation_split : float
        Fraction of data to use for validation.
    normalize : bool
        Whether to normalize the input data.
    custom_normalizer : Optional[Callable]
        Custom normalization function if normalize=True and custom_normalizer is provided.
    verbose : bool
        Whether to print training progress.

    Returns:
    --------
    Dict
        Dictionary containing the fitted model, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(X, encoder_architecture, decoder_architecture)

    # Normalize data if required
    X_normalized, normalizer = _normalize_data(X, normalize, custom_normalizer)

    # Initialize model
    model = _initialize_autoencoder(
        encoder_architecture, decoder_architecture,
        activation, output_activation
    )

    # Train model
    history = _train_autoencoder(
        model, X_normalized,
        loss_metric, solver, learning_rate,
        batch_size, epochs, tol, validation_split,
        verbose
    )

    # Compute metrics
    metrics = _compute_metrics(model, X_normalized, loss_metric)

    # Prepare output
    result = {
        "model": model,
        "normalizer": normalizer,
        "metrics": metrics,
        "params_used": {
            "encoder_architecture": encoder_architecture,
            "decoder_architecture": decoder_architecture,
            "activation": activation,
            "output_activation": output_activation,
            "loss_metric": loss_metric,
            "solver": solver,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "tol": tol,
            "validation_split": validation_split,
            "normalize": normalize
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    encoder_architecture: list,
    decoder_architecture: list
) -> None:
    """Validate input data and architecture."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X must not contain NaN or Inf values.")
    if len(encoder_architecture) == 0 or len(decoder_architecture) == 0:
        raise ValueError("Encoder and decoder architectures must not be empty.")
    if encoder_architecture[-1] != decoder_architecture[0]:
        raise ValueError("The last layer of the encoder must match the first layer of the decoder.")

def _normalize_data(
    X: np.ndarray,
    normalize: bool,
    custom_normalizer: Optional[Callable]
) -> tuple:
    """Normalize the input data."""
    if not normalize:
        return X, None
    if custom_normalizer is not None:
        X_normalized = custom_normalizer(X)
        return X_normalized, custom_normalizer
    # Default normalization (standard scaling)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, lambda x: (x - mean) / std

def _initialize_autoencoder(
    encoder_architecture: list,
    decoder_architecture: list,
    activation: str,
    output_activation: str
) -> Dict:
    """Initialize the autoencoder model."""
    # Placeholder for actual model initialization
    model = {
        "encoder": {"layers": encoder_architecture, "activation": activation},
        "decoder": {"layers": decoder_architecture, "activation": activation, "output_activation": output_activation}
    }
    return model

def _train_autoencoder(
    model: Dict,
    X: np.ndarray,
    loss_metric: Union[str, Callable],
    solver: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    tol: float,
    validation_split: float,
    verbose: bool
) -> Dict:
    """Train the autoencoder model."""
    # Placeholder for actual training logic
    history = {
        "loss": [],
        "val_loss": []
    }
    return history

def _compute_metrics(
    model: Dict,
    X: np.ndarray,
    loss_metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for the trained model."""
    # Placeholder for actual metric computation
    if loss_metric == 'mse':
        reconstruction_error = np.mean((X - X) ** 2)
    elif loss_metric == 'mae':
        reconstruction_error = np.mean(np.abs(X - X))
    else:
        reconstruction_error = loss_metric(X, X)
    return {
        "reconstruction_error": reconstruction_error
    }

################################################################################
# one_class_svm
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def one_class_svm_fit(
    X: np.ndarray,
    kernel: str = 'rbf',
    nu: float = 0.1,
    gamma: Union[str, float] = 'scale',
    degree: int = 3,
    coef0: float = 0.0,
    tol: float = 1e-3,
    max_iter: int = -1,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: str = 'euclidean',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    solver: str = 'libsvm',
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a One-Class SVM model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    kernel : str, optional
        Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
    nu : float, optional
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        Should be in the range (0, 1]. Default is 0.1.
    gamma : str or float, optional
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Default is 'scale'.
    degree : int, optional
        Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Default is 3.
    coef0 : float, optional
        Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'. Default is 0.0.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-3.
    max_iter : int, optional
        Hard limit on iterations within solver. Default is -1 (no limit).
    normalizer : Callable, optional
        Function to normalize the input data. Default is None.
    metric : str, optional
        Distance metric for kernel computation. Default is 'euclidean'.
    custom_metric : Callable, optional
        Custom distance metric function. Default is None.
    solver : str, optional
        Solver to use for the optimization problem. Default is 'libsvm'.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        A dictionary containing the fitted model parameters and metrics.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> result = one_class_svm_fit(X, kernel='rbf', nu=0.1)
    """
    # Validate inputs
    _validate_inputs(X, kernel, nu, gamma, degree, coef0, tol, max_iter)

    # Normalize data if a normalizer is provided
    if normalizer is not None:
        X = normalizer(X)

    # Choose kernel function
    if kernel == 'rbf':
        kernel_func = _rbf_kernel
    elif kernel == 'linear':
        kernel_func = _linear_kernel
    elif kernel == 'poly':
        kernel_func = _poly_kernel
    elif kernel == 'sigmoid':
        kernel_func = _sigmoid_kernel
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    # Compute the kernel matrix
    K = _compute_kernel_matrix(X, kernel_func, gamma=gamma, degree=degree, coef0=coef0)

    # Solve the optimization problem
    alpha = _solve_oc_svm(K, nu=nu, tol=tol, max_iter=max_iter, solver=solver)

    # Compute the support vectors
    sv = _compute_support_vectors(X, alpha)

    # Compute the offset
    rho = _compute_offset(K[np.ix_(sv, sv)], alpha[sv])

    # Compute metrics
    metrics = _compute_metrics(X, K, alpha, rho)

    return {
        'result': {
            'alpha': alpha,
            'support_vectors': sv,
            'offset': rho
        },
        'metrics': metrics,
        'params_used': {
            'kernel': kernel,
            'nu': nu,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

def _validate_inputs(
    X: np.ndarray,
    kernel: str,
    nu: float,
    gamma: Union[str, float],
    degree: int,
    coef0: float,
    tol: float,
    max_iter: int
) -> None:
    """Validate the input parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if nu <= 0 or nu > 1:
        raise ValueError("nu must be in the range (0, 1].")
    if kernel == 'poly' and degree < 1:
        raise ValueError("degree must be >= 1 for polynomial kernel.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")

def _rbf_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, gamma: float = 1.0) -> np.ndarray:
    """Compute the RBF kernel."""
    if Y is None:
        Y = X
    pairwise_sq_dists = _compute_pairwise_distances(X, Y)
    return np.exp(-gamma * pairwise_sq_dists)

def _linear_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the linear kernel."""
    if Y is None:
        Y = X
    return np.dot(X, Y.T)

def _poly_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, degree: int = 3, gamma: float = 1.0, coef0: float = 0.0) -> np.ndarray:
    """Compute the polynomial kernel."""
    if Y is None:
        Y = X
    K = _linear_kernel(X, Y)
    return (gamma * K + coef0) ** degree

def _sigmoid_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, gamma: float = 1.0, coef0: float = 0.0) -> np.ndarray:
    """Compute the sigmoid kernel."""
    if Y is None:
        Y = X
    K = np.dot(X, Y.T)
    return np.tanh(gamma * K + coef0)

def _compute_pairwise_distances(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute pairwise distances between rows of X and Y."""
    if Y is None:
        Y = X
    sum_X = np.sum(np.square(X), axis=1)
    sum_Y = np.sum(np.square(Y), axis=1)
    dists = np.add(np.add(-2 * np.dot(X, Y.T), sum_X).T, sum_Y)
    return dists

def _compute_kernel_matrix(X: np.ndarray, kernel_func: Callable, **kernel_params) -> np.ndarray:
    """Compute the kernel matrix."""
    return kernel_func(X, **kernel_params)

def _solve_oc_svm(
    K: np.ndarray,
    nu: float = 0.1,
    tol: float = 1e-3,
    max_iter: int = -1,
    solver: str = 'libsvm'
) -> np.ndarray:
    """Solve the one-class SVM optimization problem."""
    n_samples = K.shape[0]
    H = np.outer(np.ones(n_samples), np.ones(n_samples)) - K
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.hstack((np.zeros(n_samples), (1 / nu) * np.ones(n_samples)))
    A = np.ones((1, n_samples))
    b = 1

    if solver == 'libsvm':
        from cvxopt import solvers, matrix
        P = matrix(H)
        q = matrix(np.zeros(n_samples))
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        solvers.options['show_progress'] = False
        if max_iter > 0:
            solvers.options['maxiters'] = max_iter
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution['x']).flatten()
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return alpha

def _compute_support_vectors(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute the support vectors."""
    return X[alpha > 1e-5]

def _compute_offset(K_sv: np.ndarray, alpha_sv: np.ndarray) -> float:
    """Compute the offset."""
    return np.mean(alpha_sv * (K_sv.diagonal() - 1))

def _compute_metrics(X: np.ndarray, K: np.ndarray, alpha: np.ndarray, rho: float) -> Dict[str, float]:
    """Compute various metrics for the model."""
    decision_values = np.sum(alpha[:, None] * K, axis=0) - rho
    accuracy = np.mean(decision_values >= 0)
    return {
        'accuracy': accuracy,
        'decision_values_mean': np.mean(decision_values),
        'decision_values_std': np.std(decision_values)
    }

################################################################################
# isolation_forest
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def isolation_forest_fit(
    X: np.ndarray,
    contamination: float = 0.1,
    max_samples: int = 256,
    n_estimators: int = 100,
    max_features: int = 1.0,
    bootstrap: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit an Isolation Forest model to the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    contamination : float, optional
        Expected proportion of outliers in the data (default: 0.1).
    max_samples : int, optional
        Number of samples to draw to train each tree (default: 256).
    n_estimators : int, optional
        Number of trees in the ensemble (default: 100).
    max_features : int or float, optional
        Number of features to consider when splitting a node (default: 1.0).
    bootstrap : bool, optional
        Whether to use bootstrap sampling (default: False).
    n_jobs : int, optional
        Number of parallel jobs to run (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    verbose : int, optional
        Verbosity level (default: 0).

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": Anomaly scores for each sample.
        - "metrics": Dictionary of metrics (e.g., average path length).
        - "params_used": Parameters used for fitting.
        - "warnings": Any warnings encountered during fitting.

    Example
    -------
    >>> X = np.random.randn(100, 5)
    >>> result = isolation_forest_fit(X, contamination=0.1)
    """
    # Validate inputs
    _validate_inputs(X, contamination, max_samples, n_estimators, max_features)

    # Set random state
    rng = np.random.RandomState(random_state)

    # Initialize warnings
    warnings = []

    # Fit the isolation forest
    anomaly_scores = _fit_isolation_forest(
        X, contamination, max_samples, n_estimators,
        max_features, bootstrap, rng
    )

    # Calculate metrics
    metrics = _calculate_metrics(X, anomaly_scores)

    # Prepare output
    result = {
        "result": anomaly_scores,
        "metrics": metrics,
        "params_used": {
            "contamination": contamination,
            "max_samples": max_samples,
            "n_estimators": n_estimators,
            "max_features": max_features,
            "bootstrap": bootstrap
        },
        "warnings": warnings
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    contamination: float,
    max_samples: int,
    n_estimators: int,
    max_features: Union[int, float]
) -> None:
    """Validate the inputs for isolation forest."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D numpy array.")
    if contamination <= 0 or contamination >= 1:
        raise ValueError("contamination must be in (0, 1).")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if max_features <= 0:
        raise ValueError("max_features must be positive.")

def _fit_isolation_forest(
    X: np.ndarray,
    contamination: float,
    max_samples: int,
    n_estimators: int,
    max_features: Union[int, float],
    bootstrap: bool,
    rng: np.random.RandomState
) -> np.ndarray:
    """Fit the isolation forest model."""
    n_samples, n_features = X.shape

    # Determine number of features to use
    if isinstance(max_features, float):
        max_features = int(max_features * n_features)

    # Initialize anomaly scores
    anomaly_scores = np.zeros(n_samples)

    for _ in range(n_estimators):
        # Sample data
        if bootstrap:
            sample_idx = rng.choice(n_samples, size=max_samples, replace=True)
        else:
            sample_idx = rng.choice(n_samples, size=min(max_samples, n_samples), replace=False)

        X_sample = X[sample_idx]

        # Build tree
        tree = _build_isolation_tree(
            X_sample, max_features, rng
        )

        # Compute path lengths for all samples
        path_lengths = _compute_path_lengths(
            X, tree, max_features, rng
        )

        # Update anomaly scores
        anomaly_scores += _compute_anomaly_scores(
            path_lengths, contamination
        )

    # Normalize anomaly scores
    anomaly_scores = 2 ** (-anomaly_scores / np.mean(anomaly_scores))

    return anomaly_scores

def _build_isolation_tree(
    X: np.ndarray,
    max_features: int,
    rng: np.random.RandomState
) -> Dict:
    """Build an isolation tree."""
    n_samples, n_features = X.shape

    # Base case: if all samples have the same feature value, return a leaf
    if n_samples == 0 or np.all(X[:, 0] == X[0, 0]):
        return {"leaf": True, "samples": X}

    # Select a random feature
    feature_idx = rng.choice(n_features, size=max_features, replace=False)

    # Select a random split value
    min_val = np.min(X[:, feature_idx], axis=1)
    max_val = np.max(X[:, feature_idx], axis=1)
    split_value = rng.uniform(min_val, max_val)

    # Split the data
    left_idx = X[:, feature_idx] < split_value
    right_idx = ~left_idx

    # Recursively build left and right subtrees
    left_subtree = _build_isolation_tree(
        X[left_idx], max_features, rng
    ) if np.any(left_idx) else {"leaf": True, "samples": X[left_idx]}

    right_subtree = _build_isolation_tree(
        X[right_idx], max_features, rng
    ) if np.any(right_idx) else {"leaf": True, "samples": X[right_idx]}

    return {
        "feature": feature_idx,
        "split_value": split_value,
        "left": left_subtree,
        "right": right_subtree
    }

def _compute_path_lengths(
    X: np.ndarray,
    tree: Dict,
    max_features: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """Compute path lengths for all samples in the tree."""
    n_samples = X.shape[0]
    path_lengths = np.zeros(n_samples)

    for i in range(n_samples):
        x = X[i]
        current_node = tree
        path_length = 0

        while not current_node.get("leaf", False):
            feature_idx = current_node["feature"]
            split_value = current_node["split_value"]

            if x[feature_idx] < split_value:
                current_node = current_node["left"]
            else:
                current_node = current_node["right"]

            path_length += 1

        path_lengths[i] = path_length

    return path_lengths

def _compute_anomaly_scores(
    path_lengths: np.ndarray,
    contamination: float
) -> np.ndarray:
    """Compute anomaly scores from path lengths."""
    mean_path_length = np.mean(path_lengths)
    std_path_length = np.std(path_lengths)

    # Avoid division by zero
    if std_path_length == 0:
        return np.zeros_like(path_lengths)

    normalized_scores = (path_lengths - mean_path_length) / std_path_length
    anomaly_scores = 2 ** (-normalized_scores)

    return anomaly_scores

def _calculate_metrics(
    X: np.ndarray,
    anomaly_scores: np.ndarray
) -> Dict[str, float]:
    """Calculate metrics for the isolation forest."""
    return {
        "average_path_length": np.mean(anomaly_scores),
        "std_path_length": np.std(anomaly_scores)
    }

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
        raise ValueError("Input must not contain NaN or Inf values")

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
    custom_metric_func: Optional[Callable] = None
) -> Dict:
    """
    Compute the Local Outlier Factor (LOF) of each sample.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    n_neighbors : int, optional
        Number of neighbors to consider. Default is 20.
    metric : str or callable, optional
        Distance metric to use. Default is 'euclidean'.
    algorithm : str, optional
        Algorithm for nearest neighbors search ('auto', 'ball_tree', 'kd_tree', 'brute'). Default is 'auto'.
    leaf_size : int, optional
        Leaf size for tree-based algorithms. Default is 30.
    p : float, optional
        Power parameter for Minkowski distance. Default is 2.0.
    custom_metric_func : callable, optional
        Custom distance metric function.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Array of LOF scores.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': List of warnings encountered.

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> result = local_outlier_factor_fit(X)
    """
    # Validate input
    validate_input(X)

    # Set up parameters
    params_used = {
        'n_neighbors': n_neighbors,
        'metric': metric,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'p': p
    }

    # Initialize warnings
    warnings = []

    # Set up distance metric
    if custom_metric_func is not None:
        distance_metric = custom_metric_func
    elif metric == 'euclidean':
        distance_metric = lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        distance_metric = lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        distance_metric = lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif metric == 'minkowski':
        distance_metric = lambda x, y: np.sum(np.abs(x - y) ** p) ** (1 / p)
    else:
        raise ValueError("Unsupported metric")

    # Compute k-distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm,
                         leaf_size=leaf_size, metric=distance_metric)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # Compute reachability distance
    k_distances = distances[:, -1]
    reachability_distance = np.zeros_like(k_distances)

    for i in range(len(X)):
        neighbors = nn.kneighbors([X[i]], n_neighbors=n_neighbors, return_distance=True)[1][0]
        for j in neighbors:
            if distances[j, -1] > k_distances[i]:
                reachability_distance[i] += distances[i, j]
            else:
                reachability_distance[i] += k_distances[j]

    # Compute local reachability density
    lrd = 1 / (np.mean(reachability_distance) + 1e-10)

    # Compute LOF
    lof_scores = np.zeros(len(X))
    for i in range(len(X)):
        neighbors = nn.kneighbors([X[i]], n_neighbors=n_neighbors, return_distance=True)[1][0]
        neighbor_lrd = lrd[neighbors]
        lof_scores[i] = np.mean(neighbor_lrd) / (lrd[i] + 1e-10)

    # Prepare output
    result = {
        'result': lof_scores,
        'metrics': {'lof_scores': lof_scores},
        'params_used': params_used,
        'warnings': warnings
    }

    return result

def _compute_k_distance(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Compute the k-distance for each sample."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    return distances[:, -1]

def _compute_reachability_distance(k_distances: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute the reachability distance for each sample."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=len(X))
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    reachability_distance = np.zeros_like(k_distances)

    for i in range(len(X)):
        neighbors = nn.kneighbors([X[i]], n_neighbors=len(X), return_distance=True)[1][0]
        for j in neighbors:
            if distances[j, -1] > k_distances[i]:
                reachability_distance[i] += distances[i, j]
            else:
                reachability_distance[i] += k_distances[j]

    return reachability_distance

def _compute_local_reachability_density(reachability_distance: np.ndarray) -> np.ndarray:
    """Compute the local reachability density for each sample."""
    return 1 / (np.mean(reachability_distance) + 1e-10)

def _compute_lof_scores(lrd: np.ndarray, X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Compute the LOF scores for each sample."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    lof_scores = np.zeros(len(X))

    for i in range(len(X)):
        neighbors = nn.kneighbors([X[i]], n_neighbors=n_neighbors, return_distance=True)[1][0]
        neighbor_lrd = lrd[neighbors]
        lof_scores[i] = np.mean(neighbor_lrd) / (lrd[i] + 1e-10)

    return lof_scores

################################################################################
# k_nearest_neighbors_novelty
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def k_nearest_neighbors_novelty_fit(
    X_train: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance_metric: Union[str, Callable] = 'euclidean',
    k: int = 5,
    novelty_threshold: Optional[float] = None
) -> Dict:
    """
    Fit a k-nearest neighbors novelty detection model.

    Parameters:
    -----------
    X_train : np.ndarray
        Training data of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]]
        Function to normalize the data. If None, no normalization is applied.
    distance_metric : Union[str, Callable]
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable.
    k : int
        Number of nearest neighbors to consider.
    novelty_threshold : Optional[float]
        Threshold for novelty detection. If None, the median distance of
        the k-nearest neighbors is used.

    Returns:
    --------
    Dict
        Dictionary containing the fitted model and related information.
    """
    # Validate inputs
    _validate_inputs(X_train, k)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X_train, normalizer)

    # Initialize the model
    model = {
        'X_normalized': X_normalized,
        'distance_metric': distance_metric,
        'k': k
    }

    # Set novelty threshold if not provided
    if novelty_threshold is None:
        model['novelty_threshold'] = _compute_median_distance(X_normalized, distance_metric, k)
    else:
        model['novelty_threshold'] = novelty_threshold

    return {
        'result': model,
        'metrics': {},
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance_metric': distance_metric,
            'k': k,
            'novelty_threshold': model['novelty_threshold']
        },
        'warnings': []
    }

def _validate_inputs(X_train: np.ndarray, k: int) -> None:
    """Validate the input data and parameters."""
    if not isinstance(X_train, np.ndarray):
        raise TypeError("X_train must be a numpy array.")
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train must not contain NaN or infinite values.")

def _apply_normalization(X: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(X)
    return X

def _compute_median_distance(
    X: np.ndarray,
    distance_metric: Union[str, Callable],
    k: int
) -> float:
    """Compute the median distance of the k-nearest neighbors."""
    distances = _compute_pairwise_distances(X, distance_metric)
    k_nearest_distances = np.partition(distances, k, axis=1)[:, :k]
    return float(np.median(k_nearest_distances))

def _compute_pairwise_distances(
    X: np.ndarray,
    distance_metric: Union[str, Callable]
) -> np.ndarray:
    """Compute pairwise distances between samples."""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    if isinstance(distance_metric, str):
        if distance_metric == 'euclidean':
            for i in range(n_samples):
                distances[i, :] = np.linalg.norm(X[i] - X, axis=1)
        elif distance_metric == 'manhattan':
            for i in range(n_samples):
                distances[i, :] = np.sum(np.abs(X[i] - X), axis=1)
        elif distance_metric == 'cosine':
            for i in range(n_samples):
                distances[i, :] = 1 - np.dot(X[i], X.T) / (
                    np.linalg.norm(X[i]) * np.linalg.norm(X, axis=1)
                )
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    else:
        for i in range(n_samples):
            distances[i, :] = distance_metric(X[i], X)

    return distances

# Example usage:
"""
X_train = np.random.rand(100, 5)
model = k_nearest_neighbors_novelty_fit(X_train, normalizer=None, distance_metric='euclidean', k=5)
"""

################################################################################
# gaussian_mixture_model
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def gaussian_mixture_model_fit(
    data: np.ndarray,
    n_components: int = 1,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    distance_metric: str = 'euclidean',
    normalization: str = 'standard',
    solver: str = 'expectation_maximization',
    regularization: Optional[str] = None,
    metric: str = 'loglikelihood',
    custom_distance: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Gaussian Mixture Model for novelty detection.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features).
    n_components : int, optional
        Number of mixture components.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence.
    random_state : Optional[int], optional
        Random seed for reproducibility.
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine', 'minkowski').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    solver : str, optional
        Solver to use ('expectation_maximization', 'gradient_descent').
    regularization : Optional[str], optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    metric : str, optional
        Metric to evaluate ('loglikelihood', 'mse', 'mae').
    custom_distance : Optional[Callable], optional
        Custom distance function.
    custom_metric : Optional[Callable], optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> result = gaussian_mixture_model_fit(data, n_components=3)
    """
    # Validate inputs
    _validate_inputs(data, n_components)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Initialize parameters
    params = _initialize_parameters(normalized_data, n_components, random_state)

    # Fit model
    if solver == 'expectation_maximization':
        params = _expectation_maximization(
            normalized_data, params, max_iter, tol,
            distance_metric, custom_distance
        )
    elif solver == 'gradient_descent':
        params = _gradient_descent(
            normalized_data, params, max_iter, tol,
            distance_metric, custom_distance
        )

    # Apply regularization if specified
    if regularization is not None:
        params = _apply_regularization(params, regularization)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, params, metric, custom_metric)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'n_components': n_components,
            'max_iter': max_iter,
            'tol': tol,
            'distance_metric': distance_metric,
            'normalization': normalization,
            'solver': solver,
            'regularization': regularization,
            'metric': metric
        },
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, n_components: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if n_components <= 0:
        raise ValueError("Number of components must be positive.")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
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
        raise ValueError(f"Unknown normalization method: {method}")

def _initialize_parameters(data: np.ndarray, n_components: int, random_state: Optional[int]) -> Dict:
    """Initialize parameters for the Gaussian Mixture Model."""
    np.random.seed(random_state)
    n_samples, n_features = data.shape
    weights = np.ones(n_components) / n_components
    means = data[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.array([np.cov(data.T) for _ in range(n_components)])
    return {'weights': weights, 'means': means, 'covariances': covariances}

def _expectation_maximization(
    data: np.ndarray,
    params: Dict,
    max_iter: int,
    tol: float,
    distance_metric: str,
    custom_distance: Optional[Callable]
) -> Dict:
    """Expectation-Maximization algorithm for Gaussian Mixture Model."""
    n_samples, _ = data.shape
    prev_loglikelihood = -np.inf

    for _ in range(max_iter):
        # E-step
        responsibilities = _e_step(data, params, distance_metric, custom_distance)

        # M-step
        params = _m_step(data, responsibilities, params)

        # Calculate loglikelihood
        current_loglikelihood = _calculate_loglikelihood(data, params)
        if np.abs(current_loglikelihood - prev_loglikelihood) < tol:
            break
        prev_loglikelihood = current_loglikelihood

    return params

def _e_step(
    data: np.ndarray,
    params: Dict,
    distance_metric: str,
    custom_distance: Optional[Callable]
) -> np.ndarray:
    """E-step of the EM algorithm."""
    n_samples, _ = data.shape
    n_components = params['weights'].shape[0]
    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        if custom_distance is not None:
            distances = custom_distance(data, params['means'][k])
        else:
            distances = _compute_distance(data, params['means'][k], distance_metric)
        responsibilities[:, k] = params['weights'][k] * _gaussian_pdf(data, params['means'][k], params['covariances'][k])

    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return responsibilities

def _m_step(
    data: np.ndarray,
    responsibilities: np.ndarray,
    params: Dict
) -> Dict:
    """M-step of the EM algorithm."""
    n_components = responsibilities.shape[1]
    weights = np.mean(responsibilities, axis=0)
    means = np.dot(responsibilities.T, data) / np.sum(responsibilities, axis=0, keepdims=True).T
    covariances = np.array([
        np.cov(data.T, aweights=responsibilities[:, k])
        for k in range(n_components)
    ])
    return {'weights': weights, 'means': means, 'covariances': covariances}

def _gradient_descent(
    data: np.ndarray,
    params: Dict,
    max_iter: int,
    tol: float,
    distance_metric: str,
    custom_distance: Optional[Callable]
) -> Dict:
    """Gradient Descent algorithm for Gaussian Mixture Model."""
    n_samples, _ = data.shape
    prev_loglikelihood = -np.inf

    for _ in range(max_iter):
        # Calculate gradients
        gradients = _calculate_gradients(data, params, distance_metric, custom_distance)

        # Update parameters
        params = _update_parameters(params, gradients)

        # Calculate loglikelihood
        current_loglikelihood = _calculate_loglikelihood(data, params)
        if np.abs(current_loglikelihood - prev_loglikelihood) < tol:
            break
        prev_loglikelihood = current_loglikelihood

    return params

def _calculate_gradients(
    data: np.ndarray,
    params: Dict,
    distance_metric: str,
    custom_distance: Optional[Callable]
) -> Dict:
    """Calculate gradients for gradient descent."""
    n_samples, _ = data.shape
    n_components = params['weights'].shape[0]
    responsibilities = np.zeros((n_samples, n_components))

    for k in range(n_components):
        if custom_distance is not None:
            distances = custom_distance(data, params['means'][k])
        else:
            distances = _compute_distance(data, params['means'][k], distance_metric)
        responsibilities[:, k] = params['weights'][k] * _gaussian_pdf(data, params['means'][k], params['covariances'][k])

    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    # Calculate gradients for weights, means, and covariances
    weight_gradients = np.mean(responsibilities, axis=0) - params['weights']
    mean_gradients = np.dot(responsibilities.T, data) / np.sum(responsibilities, axis=0, keepdims=True).T - params['means']
    covariance_gradients = np.array([
        np.cov(data.T, aweights=responsibilities[:, k]) - params['covariances'][k]
        for k in range(n_components)
    ])

    return {
        'weights': weight_gradients,
        'means': mean_gradients,
        'covariances': covariance_gradients
    }

def _update_parameters(
    params: Dict,
    gradients: Dict,
    learning_rate: float = 0.01
) -> Dict:
    """Update parameters using gradients."""
    params['weights'] += learning_rate * gradients['weights']
    params['means'] += learning_rate * gradients['means']
    params['covariances'] += learning_rate * gradients['covariances']

    # Normalize weights
    params['weights'] /= np.sum(params['weights'])

    return params

def _apply_regularization(
    params: Dict,
    method: str
) -> Dict:
    """Apply regularization to the parameters."""
    if method == 'l1':
        params['weights'] = np.maximum(params['weights'], 0)
    elif method == 'l2':
        params['means'] /= (1 + np.linalg.norm(params['means'], axis=1, keepdims=True))
        params['covariances'] /= (1 + np.linalg.norm(params['covariances'], axis=(1, 2), keepdims=True))
    elif method == 'elasticnet':
        params['weights'] = np.maximum(params['weights'], 0)
        params['means'] /= (1 + np.linalg.norm(params['means'], axis=1, keepdims=True))
        params['covariances'] /= (1 + np.linalg.norm(params['covariances'], axis=(1, 2), keepdims=True))
    return params

def _calculate_metrics(
    data: np.ndarray,
    params: Dict,
    metric: str,
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics for the model."""
    if custom_metric is not None:
        return {'custom': custom_metric(data, params)}

    if metric == 'loglikelihood':
        return {'loglikelihood': _calculate_loglikelihood(data, params)}
    elif metric == 'mse':
        return {'mse': _calculate_mse(data, params)}
    elif metric == 'mae':
        return {'mae': _calculate_mae(data, params)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_loglikelihood(data: np.ndarray, params: Dict) -> float:
    """Calculate the loglikelihood of the data."""
    n_samples = data.shape[0]
    loglikelihood = 0.0

    for k in range(params['weights'].shape[0]):
        loglikelihood += params['weights'][k] * _gaussian_pdf(data, params['means'][k], params['covariances'][k])

    return np.sum(np.log(loglikelihood + 1e-10))

def _calculate_mse(data: np.ndarray, params: Dict) -> float:
    """Calculate the mean squared error."""
    n_samples = data.shape[0]
    mse = 0.0

    for k in range(params['weights'].shape[0]):
        mse += params['weights'][k] * np.mean((data - params['means'][k])**2)

    return mse

def _calculate_mae(data: np.ndarray, params: Dict) -> float:
    """Calculate the mean absolute error."""
    n_samples = data.shape[0]
    mae = 0.0

    for k in range(params['weights'].shape[0]):
        mae += params['weights'][k] * np.mean(np.abs(data - params['means'][k]))

    return mae

def _compute_distance(
    data: np.ndarray,
    mean: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute distance between data points and mean."""
    if metric == 'euclidean':
        return np.linalg.norm(data - mean, axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(data - mean), axis=1)
    elif metric == 'cosine':
        return 1 - np.dot(data, mean) / (np.linalg.norm(data, axis=1) * np.linalg.norm(mean))
    elif metric == 'minkowski':
        return np.sum(np.abs(data - mean)**2, axis=1)**(1/3)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

def _gaussian_pdf(
    data: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray
) -> np.ndarray:
    """Calculate the Gaussian probability density function."""
    n_features = data.shape[1]
    det_covariance = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance)

    exponent = -0.5 * np.sum(
        (data - mean) @ inv_covariance * (data - mean),
        axis=1
    )

    return np.exp(exponent) / ((2 * np.pi)**(n_features/2) * det_covariance)

################################################################################
# autoencoder_reconstruction_error
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def autoencoder_reconstruction_error_fit(
    X: np.ndarray,
    encoder_architecture: list = [64, 32],
    decoder_architecture: list = [32, 64],
    activation: str = 'relu',
    output_activation: str = 'linear',
    loss_metric: Union[str, Callable] = 'mse',
    solver: str = 'adam',
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    validation_split: float = 0.1,
    normalization: str = 'standard',
    custom_metric: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Fit an autoencoder model and compute reconstruction error for novelty detection.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    encoder_architecture : list
        List of integers specifying the architecture of the encoder.
    decoder_architecture : list
        List of integers specifying the architecture of the decoder.
    activation : str
        Activation function for hidden layers ('relu', 'sigmoid', etc.).
    output_activation : str
        Activation function for the output layer.
    loss_metric : Union[str, Callable]
        Loss metric to use ('mse', 'mae', etc.) or custom callable.
    solver : str
        Optimization algorithm ('adam', 'sgd', etc.).
    learning_rate : float
        Learning rate for the optimizer.
    epochs : int
        Number of training epochs.
    batch_size : Optional[int]
        Batch size for training. If None, uses full batch.
    validation_split : float
        Fraction of data to use for validation.
    normalization : str
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_metric : Optional[Callable]
        Custom metric function.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': Reconstruction error for each sample.
        - 'metrics': Training and validation metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 10)
    >>> result = autoencoder_reconstruction_error_fit(X)
    """
    # Validate inputs
    _validate_inputs(X, normalization)

    # Normalize data
    X_normalized = _normalize_data(X, normalization)

    # Initialize autoencoder model
    model = _initialize_autoencoder(
        encoder_architecture, decoder_architecture,
        activation, output_activation
    )

    # Compile model with loss and metrics
    _compile_model(model, loss_metric, solver, learning_rate)

    # Train model
    history = _train_model(
        model, X_normalized,
        epochs, batch_size, validation_split
    )

    # Compute reconstruction error
    reconstructions = model.predict(X_normalized)
    errors = _compute_reconstruction_error(
        X_normalized, reconstructions,
        loss_metric if isinstance(loss_metric, str) else None
    )

    # Compute custom metrics if provided
    metrics = _compute_metrics(history, custom_metric)

    # Prepare output dictionary
    result_dict = {
        'result': errors,
        'metrics': metrics,
        'params_used': {
            'encoder_architecture': encoder_architecture,
            'decoder_architecture': decoder_architecture,
            'activation': activation,
            'output_activation': output_activation,
            'loss_metric': loss_metric,
            'solver': solver,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'normalization': normalization
        },
        'warnings': []
    }

    return result_dict

def _validate_inputs(X: np.ndarray, normalization: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a numpy array.")
    if len(X.shape) != 2:
        raise ValueError("Input X must be a 2D array.")
    if np.any(np.isnan(X)):
        raise ValueError("Input X contains NaN values.")
    if normalization not in ['none', 'standard', 'minmax', 'robust']:
        raise ValueError("Invalid normalization method.")

def _normalize_data(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize data based on the specified method."""
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
        raise ValueError("Invalid normalization method.")

def _initialize_autoencoder(
    encoder_arch: list,
    decoder_arch: list,
    activation: str,
    output_activation: str
) -> object:
    """Initialize and return an autoencoder model."""
    # This is a placeholder for the actual model initialization
    # In practice, you would use a library like TensorFlow or PyTorch
    class Autoencoder:
        def __init__(self, encoder_arch, decoder_arch, activation, output_activation):
            self.encoder_arch = encoder_arch
            self.decoder_arch = decoder_arch
            self.activation = activation
            self.output_activation = output_activation

        def predict(self, X):
            # Placeholder for prediction logic
            return np.random.rand(X.shape[0], X.shape[1])

    return Autoencoder(encoder_arch, decoder_arch, activation, output_activation)

def _compile_model(model: object, loss_metric: Union[str, Callable], solver: str, learning_rate: float) -> None:
    """Compile the autoencoder model with specified loss and optimizer."""
    # Placeholder for model compilation logic
    pass

def _train_model(
    model: object,
    X: np.ndarray,
    epochs: int,
    batch_size: Optional[int],
    validation_split: float
) -> Dict:
    """Train the autoencoder model and return training history."""
    # Placeholder for training logic
    return {'loss': [], 'val_loss': []}

def _compute_reconstruction_error(
    X: np.ndarray,
    reconstructions: np.ndarray,
    metric: Optional[str] = None
) -> np.ndarray:
    """Compute reconstruction error using specified metric."""
    if metric == 'mse':
        return np.mean((X - reconstructions) ** 2, axis=1)
    elif metric == 'mae':
        return np.mean(np.abs(X - reconstructions), axis=1)
    else:
        raise ValueError("Unsupported metric for reconstruction error.")

def _compute_metrics(history: Dict, custom_metric: Optional[Callable] = None) -> Dict:
    """Compute training and validation metrics."""
    metrics = {
        'training_loss': history['loss'],
        'validation_loss': history['val_loss']
    }
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(history)
    return metrics

################################################################################
# novelty_score_thresholding
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    novelty_scores: np.ndarray,
    threshold_method: str = 'percentile',
    percentile_value: float = 95.0,
    custom_threshold: Optional[float] = None
) -> None:
    """
    Validate input parameters for novelty score thresholding.

    Parameters
    ----------
    novelty_scores : np.ndarray
        Array of novelty scores to threshold.
    threshold_method : str, optional
        Method to determine the threshold ('percentile' or 'custom').
    percentile_value : float, optional
        Percentile value to use when threshold_method is 'percentile'.
    custom_threshold : float, optional
        Custom threshold value to use when threshold_method is 'custom'.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(novelty_scores, np.ndarray):
        raise ValueError("novelty_scores must be a numpy array")
    if novelty_scores.ndim != 1:
        raise ValueError("novelty_scores must be a 1D array")
    if np.any(np.isnan(novelty_scores)):
        raise ValueError("novelty_scores contains NaN values")
    if np.any(np.isinf(novelty_scores)):
        raise ValueError("novelty_scores contains infinite values")

    if threshold_method not in ['percentile', 'custom']:
        raise ValueError("threshold_method must be either 'percentile' or 'custom'")

    if threshold_method == 'percentile':
        if not 0 <= percentile_value <= 100:
            raise ValueError("percentile_value must be between 0 and 100")
    elif threshold_method == 'custom':
        if custom_threshold is None:
            raise ValueError("custom_threshold must be provided when threshold_method is 'custom'")
        if not isinstance(custom_threshold, (int, float)):
            raise ValueError("custom_threshold must be a numeric value")

def _calculate_percentile_threshold(
    novelty_scores: np.ndarray,
    percentile_value: float
) -> float:
    """
    Calculate threshold based on percentile value.

    Parameters
    ----------
    novelty_scores : np.ndarray
        Array of novelty scores.
    percentile_value : float
        Percentile value to use for threshold calculation.

    Returns
    ------
    float
        Calculated threshold.
    """
    return np.percentile(novelty_scores, percentile_value)

def _apply_threshold(
    novelty_scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Apply threshold to novelty scores.

    Parameters
    ----------
    novelty_scores : np.ndarray
        Array of novelty scores.
    threshold : float
        Threshold value to apply.

    Returns
    ------
    np.ndarray
        Binary array indicating novelty (1 if score > threshold, else 0).
    """
    return (novelty_scores > threshold).astype(int)

def novelty_score_thresholding_fit(
    novelty_scores: np.ndarray,
    threshold_method: str = 'percentile',
    percentile_value: float = 95.0,
    custom_threshold: Optional[float] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit novelty score thresholding model and return results.

    Parameters
    ----------
    novelty_scores : np.ndarray
        Array of novelty scores to threshold.
    threshold_method : str, optional
        Method to determine the threshold ('percentile' or 'custom').
    percentile_value : float, optional
        Percentile value to use when threshold_method is 'percentile'.
    custom_threshold : float, optional
        Custom threshold value to use when threshold_method is 'custom'.

    Returns
    ------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - 'result': Binary array indicating novelty
        - 'threshold': Calculated or provided threshold value
        - 'metrics': Dictionary of metrics (currently empty)
        - 'params_used': Dictionary of parameters used
        - 'warnings': List of warnings (currently empty)

    Examples
    --------
    >>> novelty_scores = np.array([0.1, 0.5, 0.9, 1.2, 1.5])
    >>> result = novelty_score_thresholding_fit(novelty_scores)
    """
    _validate_inputs(novelty_scores, threshold_method, percentile_value, custom_threshold)

    params_used = {
        'threshold_method': threshold_method,
        'percentile_value': percentile_value if threshold_method == 'percentile' else None,
        'custom_threshold': custom_threshold if threshold_method == 'custom' else None
    }

    if threshold_method == 'percentile':
        threshold = _calculate_percentile_threshold(novelty_scores, percentile_value)
    else:
        threshold = custom_threshold

    novelty_result = _apply_threshold(novelty_scores, threshold)

    return {
        'result': novelty_result,
        'threshold': threshold,
        'metrics': {},
        'params_used': params_used,
        'warnings': []
    }

################################################################################
# adaptive_novelty_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def adaptive_novelty_detection_fit(
    X: np.ndarray,
    metric: str = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit an adaptive novelty detection model.

    Parameters
    ----------
    X : np.ndarray
        Training data of shape (n_samples, n_features).
    metric : str or callable, optional
        Metric to use for novelty detection. Options: 'mse', 'mae', 'r2'.
    distance : str or callable, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str, optional
        Solver to use. Options: 'closed_form', 'gradient_descent'.
    normalizer : callable, optional
        Function to normalize the data.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_distance : callable, optional
        Custom distance function if not using built-in distances.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Fitted model parameters.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used during fitting.
        - 'warnings': Any warnings encountered.

    Example
    -------
    >>> X = np.random.rand(100, 5)
    >>> result = adaptive_novelty_detection_fit(X, metric='mse', distance='euclidean')
    """
    # Validate inputs
    _validate_inputs(X, metric, distance, normalizer)

    # Normalize data if a normalizer is provided
    X_normalized = _apply_normalization(X, normalizer)

    # Choose metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Fit model based on solver choice
    if solver == 'closed_form':
        params = _fit_closed_form(X_normalized, distance_func)
    elif solver == 'gradient_descent':
        params = _fit_gradient_descent(
            X_normalized, distance_func, tol=tol, max_iter=max_iter,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(X_normalized, params, metric_func)

    # Prepare output
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    X: np.ndarray,
    metric: str,
    distance: str,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if metric not in ['mse', 'mae', 'r2'] and not callable(metric):
        raise ValueError("Invalid metric")
    if distance not in ['euclidean', 'manhattan', 'cosine'] and not callable(distance):
        raise ValueError("Invalid distance")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("normalizer must be a callable or None")

def _apply_normalization(
    X: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return X
    return normalizer(X)

def _get_metric_function(
    metric: str,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on input."""
    if custom_metric is not None:
        return custom_metric

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    metric_functions = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    return metric_functions[metric]

def _get_distance_function(
    distance: str,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on input."""
    if custom_distance is not None:
        return custom_distance

    def euclidean(a, b):
        return np.linalg.norm(a - b)

    def manhattan(a, b):
        return np.sum(np.abs(a - b))

    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    distance_functions = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'cosine': cosine
    }

    return distance_functions[distance]

def _fit_closed_form(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, np.ndarray]:
    """Fit model using closed-form solution."""
    # Example: Compute centroid
    centroid = np.mean(X, axis=0)
    return {'centroid': centroid}

def _fit_gradient_descent(
    X: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Fit model using gradient descent."""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    centroid = np.random.rand(n_features)

    for _ in range(max_iter):
        gradients = np.zeros_like(centroid)
        for x in X:
            grad = 2 * (distance_func(x, centroid) / distance_func(x, centroid)) * (x - centroid)
            gradients += grad
        gradients /= n_samples

        new_centroid = centroid - 0.01 * gradients
        if np.linalg.norm(new_centroid - centroid) < tol:
            break
        centroid = new_centroid

    return {'centroid': centroid}

def _compute_metrics(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Compute metrics for the fitted model."""
    centroid = params['centroid']
    distances = np.array([metric_func(x, centroid) for x in X])
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances)
    }
