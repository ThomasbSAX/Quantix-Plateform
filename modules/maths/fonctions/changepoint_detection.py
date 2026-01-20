"""
Quantix – Module changepoint_detection
Généré automatiquement
Date: 2026-01-06
"""

################################################################################
# methods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def methods_fit(
    data: np.ndarray,
    method: str = 'binary_segmentation',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main function for changepoint detection methods.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    method : str, optional
        Changepoint detection method (default: 'binary_segmentation').
    normalizer : Callable, optional
        Normalization function (default: None).
    metric : str or Callable, optional
        Metric for evaluation (default: 'mse').
    distance : str or Callable, optional
        Distance metric (default: 'euclidean').
    solver : str, optional
        Solver method (default: 'closed_form').
    regularization : str, optional
        Regularization type (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_iter : int, optional
        Maximum iterations (default: 1000).
    custom_params : Dict, optional
        Additional parameters for the method (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data, method, normalizer, metric, distance, solver)

    # Apply normalization if specified
    normalized_data = _apply_normalization(data, normalizer)

    # Select method and compute changepoints
    if method == 'binary_segmentation':
        result, metrics = _binary_segmentation(
            normalized_data,
            metric=metric,
            distance=distance,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    elif method == 'pruned_exact_line_time':
        result, metrics = _pruned_exact_line_time(
            normalized_data,
            metric=metric,
            distance=distance,
            solver=solver,
            regularization=regularization,
            tol=tol,
            max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "method": method,
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric if isinstance(metric, str) else metric.__name__,
            "distance": distance if isinstance(distance, str) else distance.__name__,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    method: str,
    normalizer: Optional[Callable],
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if method not in ['binary_segmentation', 'pruned_exact_line_time']:
        raise ValueError(f"Unknown method: {method}")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None.")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError(f"Unknown metric: {metric}")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError(f"Unknown distance: {distance}")
    if solver not in ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']:
        raise ValueError(f"Unknown solver: {solver}")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _binary_segmentation(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Binary segmentation method for changepoint detection."""
    # Implementation of binary segmentation
    # This is a placeholder for the actual implementation
    result = {"changepoints": [0.5], "segments": [[0, 1]]}
    metrics = {"metric_value": 0.5}

    return result, metrics

def _pruned_exact_line_time(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Pruned exact line time method for changepoint detection."""
    # Implementation of pruned exact line time
    # This is a placeholder for the actual implementation
    result = {"changepoints": [0.3, 0.7], "segments": [[0, 1], [1, 2]]}
    metrics = {"metric_value": 0.3}

    return result, metrics

################################################################################
# algorithms
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def algorithms_fit(
    data: np.ndarray,
    method: str = 'binary_segmentation',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main function for changepoint detection algorithms.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    method : str, optional
        Changepoint detection method (default: 'binary_segmentation').
    normalizer : callable, optional
        Normalization function (default: None).
    metric : str or callable, optional
        Metric for evaluation (default: 'mse').
    distance : str or callable, optional
        Distance metric (default: 'euclidean').
    solver : str, optional
        Solver method (default: 'closed_form').
    penalty : str or None, optional
        Regularization type (default: None).
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_iter : int, optional
        Maximum iterations (default: 1000).
    custom_params : dict or None, optional
        Additional parameters for the method (default: None).

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data, method, normalizer, metric, distance, solver, penalty)

    # Apply normalization if specified
    normalized_data = _apply_normalization(data, normalizer)

    # Select method and compute changepoints
    if method == 'binary_segmentation':
        result = _binary_segmentation(normalized_data, metric, distance, solver, penalty, tol, max_iter)
    elif method == 'pruned_exact_line_time':
        result = _pruned_exact_line_time(normalized_data, metric, distance, solver, penalty, tol, max_iter)
    elif method == 'windowed':
        result = _windowed(normalized_data, metric, distance, solver, penalty, tol, max_iter)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics
    metrics = _compute_metrics(data, result['changepoints'], metric)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric if isinstance(metric, str) else 'custom',
            'distance': distance if isinstance(distance, str) else 'custom',
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
    method: str,
    normalizer: Optional[Callable],
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    penalty: Optional[str]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")
    if method not in ['binary_segmentation', 'pruned_exact_line_time', 'windowed']:
        raise ValueError("Unknown method")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError("Unknown metric")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("Unknown distance")
    if solver not in ['closed_form', 'gradient_descent', 'newton']:
        raise ValueError("Unknown solver")
    if penalty is not None and penalty not in ['l1', 'l2']:
        raise ValueError("Unknown penalty")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _binary_segmentation(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Binary segmentation algorithm for changepoint detection."""
    # Implementation of binary segmentation
    changepoints = []
    current_data = data.copy()

    while True:
        # Find the best changepoint
        best_cp = _find_best_changepoint(current_data, metric, distance, solver)

        if best_cp is None or len(changepoints) >= max_iter:
            break

        changepoints.append(best_cp)
        # Split data and continue
        current_data = np.split(current_data, [best_cp])

    return {
        'changepoints': sorted(changepoints),
        'method_params': {
            'solver': solver,
            'penalty': penalty
        }
    }

def _find_best_changepoint(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str
) -> Optional[int]:
    """Find the best changepoint in the data."""
    # Implementation of finding the best changepoint
    min_cost = float('inf')
    best_cp = None

    for i in range(1, len(data)):
        left, right = data[:i], data[i:]
        cost = _compute_cost(left, right, metric, distance)

        if cost < min_cost:
            min_cost = cost
            best_cp = i

    return best_cp if min_cost != float('inf') else None

def _compute_cost(
    left: np.ndarray,
    right: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable]
) -> float:
    """Compute the cost of a changepoint."""
    if isinstance(metric, str):
        if metric == 'mse':
            cost = np.mean((left - left.mean())**2) + np.mean((right - right.mean())**2)
        elif metric == 'mae':
            cost = np.mean(np.abs(left - left.mean())) + np.mean(np.abs(right - right.mean()))
        elif metric == 'r2':
            cost = 1 - (np.var(left) + np.var(right)) / np.var(np.concatenate([left, right]))
    else:
        cost = metric(left, right)

    return cost

def _pruned_exact_line_time(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Pruned exact line time algorithm for changepoint detection."""
    # Implementation of pruned exact line time
    changepoints = []
    current_data = data.copy()

    while True:
        # Find the best changepoint
        best_cp = _find_best_changepoint(current_data, metric, distance, solver)

        if best_cp is None or len(changepoints) >= max_iter:
            break

        changepoints.append(best_cp)
        # Split data and continue
        current_data = np.split(current_data, [best_cp])

    return {
        'changepoints': sorted(changepoints),
        'method_params': {
            'solver': solver,
            'penalty': penalty
        }
    }

def _windowed(
    data: np.ndarray,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> Dict[str, Any]:
    """Windowed algorithm for changepoint detection."""
    # Implementation of windowed
    changepoints = []
    window_size = max(1, len(data) // 10)

    for i in range(window_size, len(data) - window_size):
        left = data[i-window_size:i]
        right = data[i:i+window_size]
        cost = _compute_cost(left, right, metric, distance)

        if cost > tol:
            changepoints.append(i)
            i += window_size  # Skip ahead

    return {
        'changepoints': sorted(changepoints),
        'method_params': {
            'solver': solver,
            'penalty': penalty
        }
    }

def _compute_metrics(
    data: np.ndarray,
    changepoints: list,
    metric: Union[str, Callable]
) -> Dict[str, float]:
    """Compute metrics for the changepoint detection results."""
    segments = np.split(data, changepoints)
    metrics = {}

    if isinstance(metric, str):
        if metric == 'mse':
            for i, seg in enumerate(segments):
                metrics[f'segment_{i}_mse'] = np.mean((seg - seg.mean())**2)
        elif metric == 'mae':
            for i, seg in enumerate(segments):
                metrics[f'segment_{i}_mae'] = np.mean(np.abs(seg - seg.mean()))
        elif metric == 'r2':
            for i, seg in enumerate(segments):
                metrics[f'segment_{i}_r2'] = 1 - (np.var(seg) / np.var(data))
    else:
        for i, seg in enumerate(segments):
            metrics[f'segment_{i}_custom'] = metric(seg, seg.mean())

    return metrics

################################################################################
# statistical_tests
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def statistical_tests_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    test_statistic: Union[str, Callable[[np.ndarray], float]] = "mean_diff",
    significance_level: float = 0.05,
    min_segment_length: int = 1,
    custom_normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_test_statistic: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Detect changepoints in time series data using statistical tests.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate segments. Default is "mse".
    test_statistic : Union[str, Callable[[np.ndarray], float]], optional
        Test statistic to use. Default is "mean_diff".
    significance_level : float, optional
        Significance level for hypothesis tests. Default is 0.05.
    min_segment_length : int, optional
        Minimum length of segments to consider. Default is 1.
    custom_normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Custom normalization function. Overrides normalizer if provided.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Overrides metric if provided.
    custom_test_statistic : Optional[Callable[[np.ndarray], float]], optional
        Custom test statistic function. Overrides test_statistic if provided.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, min_segment_length)

    # Use custom functions if provided
    normalizer = custom_normalizer if custom_normalizer is not None else normalizer
    metric_func = _get_metric_function(custom_metric, metric)
    test_statistic_func = _get_test_statistic_function(custom_test_statistic, test_statistic)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Detect changepoints
    changepoints = _detect_changepoints(
        normalized_data,
        metric_func=metric_func,
        test_statistic_func=test_statistic_func,
        significance_level=significance_level,
        min_segment_length=min_segment_length
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, changepoints)

    # Prepare results
    result = {
        "changepoints": changepoints,
        "result": {"data": normalized_data, "changepoints": changepoints},
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric_func.__name__ if isinstance(metric_func, Callable) else metric,
            "test_statistic": test_statistic_func.__name__ if isinstance(test_statistic_func, Callable) else test_statistic,
            "significance_level": significance_level,
            "min_segment_length": min_segment_length
        },
        "warnings": _check_warnings(data, changepoints)
    }

    return result

def _validate_inputs(data: np.ndarray, min_segment_length: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if min_segment_length < 1:
        raise ValueError("min_segment_length must be at least 1.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data must not contain NaN or infinite values.")

def _apply_normalization(data: np.ndarray, normalizer: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(data)
    return data

def _get_metric_function(custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]], metric: Union[str, Callable]) -> Callable:
    """Get the metric function based on user input."""
    if custom_metric is not None:
        return custom_metric
    if isinstance(metric, str):
        return _get_builtin_metric(metric)
    return metric

def _get_test_statistic_function(custom_test_statistic: Optional[Callable[[np.ndarray], float]], test_statistic: Union[str, Callable]) -> Callable:
    """Get the test statistic function based on user input."""
    if custom_test_statistic is not None:
        return custom_test_statistic
    if isinstance(test_statistic, str):
        return _get_builtin_test_statistic(test_statistic)
    return test_statistic

def _detect_changepoints(
    data: np.ndarray,
    *,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    test_statistic_func: Callable[[np.ndarray], float],
    significance_level: float,
    min_segment_length: int
) -> np.ndarray:
    """Detect changepoints in the data."""
    n = len(data)
    changepoints = []

    for i in range(min_segment_length, n - min_segment_length):
        segment1 = data[:i]
        segment2 = data[i:]

        if len(segment1) < min_segment_length or len(segment2) < min_segment_length:
            continue

        stat = test_statistic_func(np.concatenate([segment1, segment2]))
        p_value = _calculate_p_value(stat, significance_level)

        if p_value < significance_level:
            changepoints.append(i)

    return np.array(changepoints, dtype=int)

def _calculate_metrics(data: np.ndarray, changepoints: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for the detected changepoints."""
    if len(changepoints) == 0:
        return {"num_changepoints": 0}

    segments = np.split(data, changepoints)
    metrics = {
        "num_changepoints": len(changepoints),
        "segment_lengths": [len(seg) for seg in segments],
        "mean_segment_length": np.mean([len(seg) for seg in segments])
    }
    return metrics

def _check_warnings(data: np.ndarray, changepoints: np.ndarray) -> list:
    """Check for potential warnings."""
    warnings = []
    if len(changepoints) == 0:
        warnings.append("No changepoints detected.")
    if np.any(np.diff(changepoints) == 0):
        warnings.append("Duplicate changepoints detected.")
    return warnings

def _get_builtin_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get a built-in metric function."""
    metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    return metrics[metric]

def _get_builtin_test_statistic(test_statistic: str) -> Callable[[np.ndarray], float]:
    """Get a built-in test statistic function."""
    stats = {
        "mean_diff": _mean_difference_test_statistic
    }
    if test_statistic not in stats:
        raise ValueError(f"Unknown test statistic: {test_statistic}")
    return stats[test_statistic]

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

def _mean_difference_test_statistic(data: np.ndarray) -> float:
    """Calculate mean difference test statistic."""
    return np.mean(data)

def _calculate_p_value(stat: float, significance_level: float) -> float:
    """Calculate p-value based on test statistic and significance level."""
    return 1.0 - significance_level  # Placeholder for actual p-value calculation

################################################################################
# online_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_inputs(data: np.ndarray) -> None:
    """Validate input data for online changepoint detection.

    Args:
        data: Input time series data as numpy array.

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values")

def normalize_data(data: np.ndarray, method: str = 'none') -> np.ndarray:
    """Normalize data using specified method.

    Args:
        data: Input time series data.
        method: Normalization method ('none', 'standard', 'minmax', 'robust').

    Returns:
        Normalized data.
    """
    if method == 'none':
        return data
    elif method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def compute_statistic(data: np.ndarray, window_size: int = 10,
                     statistic_func: Callable = np.mean) -> float:
    """Compute changepoint detection statistic for current window.

    Args:
        data: Input time series data.
        window_size: Size of the sliding window.
        statistic_func: Function to compute statistic (default: mean).

    Returns:
        Computed statistic value.
    """
    if len(data) < window_size:
        raise ValueError("Data length is smaller than window size")
    return statistic_func(data[-window_size:])

def detect_changepoint(current_stat: float, baseline_stat: float,
                      threshold_func: Callable = lambda x: 2 * np.std(x)) -> bool:
    """Detect changepoint based on current and baseline statistics.

    Args:
        current_stat: Current window statistic.
        baseline_stat: Baseline statistic (from previous windows).
        threshold_func: Function to compute detection threshold.

    Returns:
        Boolean indicating if changepoint was detected.
    """
    return abs(current_stat - baseline_stat) > threshold_func(baseline_stat)

def online_detection_fit(data: np.ndarray,
                        window_size: int = 10,
                        statistic_func: Callable = np.mean,
                        threshold_func: Callable = lambda x: 2 * np.std(x),
                        normalize_method: str = 'none',
                        min_distance: int = 1) -> Dict:
    """Online changepoint detection algorithm.

    Args:
        data: Input time series data.
        window_size: Size of the sliding window for statistics computation.
        statistic_func: Function to compute window statistics (default: mean).
        threshold_func: Function to compute detection threshold.
        normalize_method: Data normalization method.
        min_distance: Minimum distance between detected changepoints.

    Returns:
        Dictionary containing detection results, metrics and parameters used.
    """
    # Validate inputs
    validate_inputs(data)

    # Initialize variables
    results = []
    changepoints = []
    baseline_stat = None
    last_changepoint = 0

    # Normalize data if needed
    normalized_data = normalize_data(data, normalize_method)

    # Process each new data point
    for i in range(window_size, len(normalized_data)):
        # Compute current window statistic
        current_stat = compute_statistic(normalized_data[:i+1], window_size, statistic_func)

        # Initialize baseline if not set
        if baseline_stat is None:
            baseline_stat = current_stat

        # Detect changepoint
        if detect_changepoint(current_stat, baseline_stat, threshold_func):
            # Check minimum distance constraint
            if i - last_changepoint >= min_distance:
                changepoints.append(i)
                results.append({
                    'index': i,
                    'current_stat': current_stat,
                    'baseline_stat': baseline_stat
                })
                last_changepoint = i
            # Update baseline to current statistic after detection
            baseline_stat = current_stat

    return {
        'result': results,
        'changepoints': changepoints,
        'metrics': {
            'total_changepoints': len(changepoints),
            'last_changepoint_index': changepoints[-1] if changepoints else None
        },
        'params_used': {
            'window_size': window_size,
            'statistic_func': statistic_func.__name__ if hasattr(statistic_func, '__name__') else 'custom',
            'threshold_func': threshold_func.__name__ if hasattr(threshold_func, '__name__') else 'custom',
            'normalize_method': normalize_method,
            'min_distance': min_distance
        },
        'warnings': []
    }

# Example usage:
"""
data = np.random.randn(100)
result = online_detection_fit(
    data=data,
    window_size=20,
    statistic_func=np.median,
    threshold_func=lambda x: 3 * np.std(x),
    normalize_method='standard',
    min_distance=5
)
"""

################################################################################
# offline_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def offline_detection_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]:
    """
    Detect changepoints in offline data using various statistical methods.

    Parameters
    ----------
    data : np.ndarray
        Input time series data of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate changepoints. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for changepoint detection. Can be 'euclidean', 'manhattan',
        'cosine', 'minkowski', or a custom callable.
    solver : str, optional
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    penalty : Optional[str], optional
        Penalty type for regularization. Can be 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]
        Dictionary containing:
        - "result": Detected changepoints.
        - "metrics": Computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> result = offline_detection_fit(data, normalizer=np.std, metric='mse')
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, penalty)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Compute changepoints based on the chosen solver
    if solver == 'closed_form':
        changepoints = _closed_form_solver(normalized_data, metric, distance)
    elif solver == 'gradient_descent':
        changepoints = _gradient_descent_solver(normalized_data, metric, distance,
                                               penalty=penalty, tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        changepoints = _newton_solver(normalized_data, metric, distance,
                                     penalty=penalty, tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        changepoints = _coordinate_descent_solver(normalized_data, metric, distance,
                                                 penalty=penalty, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute metrics
    metrics = _compute_metrics(normalized_data, changepoints, metric, custom_metric)

    # Prepare output
    result = {
        "result": changepoints,
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "penalty": penalty,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    penalty: Optional[str]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None.")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError("Metric must be 'mse', 'mae', 'r2', or a custom callable.")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError("Distance must be 'euclidean', 'manhattan', 'cosine', 'minkowski', or a custom callable.")
    if solver not in ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']:
        raise ValueError("Solver must be 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.")
    if penalty is not None and penalty not in ['l1', 'l2', 'elasticnet']:
        raise ValueError("Penalty must be 'l1', 'l2', 'elasticnet', or None.")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _closed_form_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> np.ndarray:
    """Closed form solution for changepoint detection."""
    # Placeholder for actual implementation
    return np.array([0])

def _gradient_descent_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    *,
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for changepoint detection."""
    # Placeholder for actual implementation
    return np.array([0])

def _newton_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    *,
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton's method solver for changepoint detection."""
    # Placeholder for actual implementation
    return np.array([0])

def _coordinate_descent_solver(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    *,
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Coordinate descent solver for changepoint detection."""
    # Placeholder for actual implementation
    return np.array([0])

def _compute_metrics(
    data: np.ndarray,
    changepoints: np.ndarray,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Dict[str, float]:
    """Compute metrics for changepoint detection."""
    if custom_metric is not None:
        return {"custom_metric": custom_metric(data, changepoints)}

    metrics = {}
    if metric == 'mse':
        metrics['mse'] = np.mean((data - changepoints) ** 2)
    elif metric == 'mae':
        metrics['mae'] = np.mean(np.abs(data - changepoints))
    elif metric == 'r2':
        ss_res = np.sum((data - changepoints) ** 2)
        ss_tot = np.sum((data - np.mean(data, axis=0)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# parametric_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def parametric_methods_fit(
    data: np.ndarray,
    method: str = 'ols',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    normalizer: Optional[Callable] = None,
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    **kwargs
) -> Dict:
    """
    Detect change points in time series data using parametric methods.

    Parameters
    ----------
    data : np.ndarray
        Input time series data of shape (n_samples, n_features)
    method : str, optional
        Parametric method to use ('ols', 'glm')
    metric : Union[str, Callable], optional
        Metric to optimize ('mse', 'mae', 'r2') or custom callable
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent')
    normalizer : Optional[Callable], optional
        Normalization function to apply
    penalty : Optional[str], optional
        Penalty type ('l1', 'l2', 'elasticnet')
    tol : float, optional
        Tolerance for convergence
    max_iter : int, optional
        Maximum number of iterations
    custom_metric : Optional[Callable], optional
        Custom metric function if needed
    **kwargs :
        Additional parameters for the specific method

    Returns
    -------
    Dict
        Dictionary containing:
        - 'result': change point locations
        - 'metrics': computed metrics
        - 'params_used': parameters used
        - 'warnings': any warnings

    Example
    -------
    >>> data = np.random.randn(100, 2)
    >>> result = parametric_methods_fit(data, method='ols', metric='mse')
    """
    # Validate inputs
    _validate_inputs(data, method, solver)

    # Normalize data if needed
    normalized_data = _apply_normalization(data, normalizer)

    # Choose method implementation
    if method == 'ols':
        result = _ols_method(normalized_data, metric, solver, penalty, tol, max_iter)
    elif method == 'glm':
        result = _glm_method(normalized_data, metric, solver, penalty, tol, max_iter)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Prepare output
    return {
        'result': result['change_points'],
        'metrics': result['metrics'],
        'params_used': {
            'method': method,
            'metric': metric,
            'solver': solver,
            'normalizer': normalizer.__name__ if normalizer else None,
            'penalty': penalty
        },
        'warnings': result.get('warnings', [])
    }

def _validate_inputs(data: np.ndarray, method: str, solver: str) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    if method not in ['ols', 'glm']:
        raise ValueError(f"Method must be either 'ols' or 'glm', got {method}")
    if solver not in ['closed_form', 'gradient_descent']:
        raise ValueError(f"Solver must be either 'closed_form' or 'gradient_descent', got {solver}")

def _apply_normalization(data: np.ndarray, normalizer: Optional[Callable]) -> np.ndarray:
    """Apply normalization to data if specified."""
    if normalizer is None:
        return data
    try:
        return normalizer(data)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _ols_method(
    data: np.ndarray,
    metric: Union[str, Callable],
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Ordinary Least Squares change point detection."""
    # Implementation of OLS method
    metrics = _compute_metrics(data, metric)
    change_points = _find_change_points_ols(data, solver, penalty, tol, max_iter)

    return {
        'change_points': change_points,
        'metrics': metrics,
        'warnings': []
    }

def _glm_method(
    data: np.ndarray,
    metric: Union[str, Callable],
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> Dict:
    """Generalized Linear Model change point detection."""
    # Implementation of GLM method
    metrics = _compute_metrics(data, metric)
    change_points = _find_change_points_glm(data, solver, penalty, tol, max_iter)

    return {
        'change_points': change_points,
        'metrics': metrics,
        'warnings': []
    }

def _compute_metrics(data: np.ndarray, metric: Union[str, Callable]) -> Dict:
    """Compute metrics for change point detection."""
    if callable(metric):
        return {'custom_metric': metric(data)}
    elif metric == 'mse':
        return {'mse': np.mean((data - np.mean(data, axis=0))**2)}
    elif metric == 'mae':
        return {'mae': np.mean(np.abs(data - np.median(data, axis=0)))}
    elif metric == 'r2':
        return {'r2': 1 - np.var(data) / (np.mean(data)**2)}
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _find_change_points_ols(
    data: np.ndarray,
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Find change points using OLS method."""
    # Placeholder implementation
    return np.array([50, 75])  # Example change points

def _find_change_points_glm(
    data: np.ndarray,
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Find change points using GLM method."""
    # Placeholder implementation
    return np.array([30, 60])  # Example change points

def standard_normalizer(data: np.ndarray) -> np.ndarray:
    """Standard normalizer."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def minmax_normalizer(data: np.ndarray) -> np.ndarray:
    """Min-Max normalizer."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val + 1e-8)

################################################################################
# non_parametric_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def non_parametric_methods_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, np.ndarray, str]]:
    """
    Detect changepoints in time series data using non-parametric methods.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to use for changepoint detection. Default is 'mse'.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric to use. Default is 'euclidean'.
    solver : str, optional
        Solver method to use. Default is 'closed_form'.
    regularization : Optional[str], optional
        Regularization method to use. Default is None.
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
    Dict[str, Union[Dict, np.ndarray, str]]
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> result = non_parametric_methods_fit(data, normalizer=np.std, metric='mae')
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Detect changepoints based on the solver
    if solver == 'closed_form':
        result = _closed_form_solver(normalized_data, metric_func, distance_func)
    elif solver == 'gradient_descent':
        result = _gradient_descent_solver(normalized_data, metric_func, distance_func,
                                         regularization=regularization, tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, result['changepoints'], metric_func)

    # Prepare output
    output = {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
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
    if custom_metric is not None:
        return custom_metric
    if isinstance(metric, str):
        if metric == 'mse':
            return _mean_squared_error
        elif metric == 'mae':
            return _mean_absolute_error
        elif metric == 'r2':
            return _r_squared
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return metric

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the input."""
    if custom_distance is not None:
        return custom_distance
    if isinstance(distance, str):
        if distance == 'euclidean':
            return _euclidean_distance
        elif distance == 'manhattan':
            return _manhattan_distance
        elif distance == 'cosine':
            return _cosine_distance
        elif distance == 'minkowski':
            return _minkowski_distance
        else:
            raise ValueError(f"Unknown distance: {distance}")
    return distance

def _closed_form_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Union[np.ndarray, str]]:
    """Closed form solver for changepoint detection."""
    # Implement closed form solution here
    changepoints = np.array([len(data) // 2])  # Placeholder
    return {'changepoints': changepoints, 'method': 'closed_form'}

def _gradient_descent_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Union[np.ndarray, str]]:
    """Gradient descent solver for changepoint detection."""
    # Implement gradient descent solution here
    changepoints = np.array([len(data) // 3, len(data) * 2 // 3])  # Placeholder
    return {'changepoints': changepoints, 'method': 'gradient_descent'}

def _calculate_metrics(
    data: np.ndarray,
    changepoints: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the changepoint detection."""
    # Split data into segments based on changepoints
    segments = np.split(data, np.unique(changepoints))

    # Calculate metrics for each segment
    metrics = {}
    for i, segment in enumerate(segments):
        if len(segment) > 1:
            metrics[f'segment_{i}_metric'] = metric_func(segment, np.mean(segment))

    return metrics

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

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

################################################################################
# bayesian_approaches
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def bayesian_approaches_fit(
    data: np.ndarray,
    *,
    prior: str = 'uniform',
    likelihood: str = 'gaussian',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'closed_form',
    normalize: bool = True,
    custom_prior: Optional[Callable] = None,
    custom_likelihood: Optional[Callable] = None,
    custom_metric: Optional[Callable] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """
    Detect changepoints using Bayesian approaches.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    prior : str, optional
        Prior distribution type ('uniform', 'beta', 'gamma').
    likelihood : str, optional
        Likelihood function type ('gaussian', 'poisson').
    metric : str or callable, optional
        Metric to evaluate changepoints ('mse', 'mae', custom callable).
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent').
    normalize : bool, optional
        Whether to normalize the data.
    custom_prior : callable, optional
        Custom prior function if not using built-in types.
    custom_likelihood : callable, optional
        Custom likelihood function if not using built-in types.
    custom_metric : callable, optional
        Custom metric function if not using built-in types.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns:
    --------
    Dict containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data, normalize)

    # Normalize data if required
    normalized_data = _normalize_data(data) if normalize else data

    # Set up prior and likelihood functions
    prior_func = _get_prior_function(prior, custom_prior)
    likelihood_func = _get_likelihood_function(likelihood, custom_likelihood)

    # Set up metric function
    metric_func = _get_metric_function(metric, custom_metric)

    # Set up solver
    changepoints = _solve_changepoints(
        normalized_data,
        prior_func,
        likelihood_func,
        metric_func,
        solver,
        tol,
        max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, changepoints, metric_func)

    # Prepare output
    result = {
        'result': {'changepoints': changepoints},
        'metrics': metrics,
        'params_used': {
            'prior': prior if custom_prior is None else 'custom',
            'likelihood': likelihood if custom_likelihood is None else 'custom',
            'metric': metric if custom_metric is None else 'custom',
            'solver': solver,
            'normalize': normalize
        },
        'warnings': _check_warnings(normalized_data, changepoints)
    }

    return result

def _validate_inputs(data: np.ndarray, normalize: bool) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data using standard scaling."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-8)

def _get_prior_function(prior: str, custom_prior: Optional[Callable]) -> Callable:
    """Get prior function based on input parameters."""
    if custom_prior is not None:
        return custom_prior
    if prior == 'uniform':
        return lambda x: np.ones_like(x)
    elif prior == 'beta':
        return lambda x, alpha=1.0, beta=1.0: (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))
    elif prior == 'gamma':
        return lambda x, k=1.0, theta=1.0: (x ** (k - 1)) * np.exp(-x / theta)
    else:
        raise ValueError(f"Unknown prior type: {prior}")

def _get_likelihood_function(likelihood: str, custom_likelihood: Optional[Callable]) -> Callable:
    """Get likelihood function based on input parameters."""
    if custom_likelihood is not None:
        return custom_likelihood
    if likelihood == 'gaussian':
        return lambda x, mu, sigma: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    elif likelihood == 'poisson':
        return lambda x, mu: (mu ** x) * np.exp(-mu) / np.math.factorial(x)
    else:
        raise ValueError(f"Unknown likelihood type: {likelihood}")

def _get_metric_function(metric: Union[str, Callable], custom_metric: Optional[Callable]) -> Callable:
    """Get metric function based on input parameters."""
    if custom_metric is not None:
        return custom_metric
    if metric == 'mse':
        return lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    else:
        raise ValueError(f"Unknown metric type: {metric}")

def _solve_changepoints(
    data: np.ndarray,
    prior_func: Callable,
    likelihood_func: Callable,
    metric_func: Callable,
    solver: str,
    tol: float,
    max_iter: int
) -> List[int]:
    """Solve for changepoints using specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(data, prior_func, likelihood_func)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(data, prior_func, likelihood_func, metric_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver type: {solver}")

def _solve_closed_form(data: np.ndarray, prior_func: Callable, likelihood_func: Callable) -> List[int]:
    """Solve changepoints using closed form solution."""
    # Placeholder for actual implementation
    return [len(data) // 2]

def _solve_gradient_descent(
    data: np.ndarray,
    prior_func: Callable,
    likelihood_func: Callable,
    metric_func: Callable,
    tol: float,
    max_iter: int
) -> List[int]:
    """Solve changepoints using gradient descent."""
    # Placeholder for actual implementation
    return [len(data) // 2]

def _calculate_metrics(
    data: np.ndarray,
    changepoints: List[int],
    metric_func: Callable
) -> Dict:
    """Calculate metrics for changepoint detection."""
    # Placeholder for actual implementation
    return {'metric_value': 0.5}

def _check_warnings(data: np.ndarray, changepoints: List[int]) -> List[str]:
    """Check for potential warnings."""
    warnings = []
    if len(changepoints) == 0:
        warnings.append("No changepoints detected")
    if len(changepoints) > len(data) // 2:
        warnings.append("Large number of changepoints detected")
    return warnings

################################################################################
# penalty_based_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def penalty_based_methods_fit(
    data: np.ndarray,
    metric: str = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    penalty: str = 'l1',
    solver: str = 'closed_form',
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect change points using penalty-based methods.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    metric : str, optional
        Metric to use for change point detection ('mse', 'mae', 'r2').
    normalizer : Callable, optional
        Function to normalize the data.
    penalty : str, optional
        Penalty type ('l1', 'l2', 'elasticnet').
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton').
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
    validate_inputs(data, metric, normalizer, penalty, solver)

    # Normalize data if specified
    normalized_data = apply_normalization(data, normalizer)

    # Choose metric
    if custom_metric is not None:
        metric_func = custom_metric
    else:
        metric_func = get_metric_function(metric)

    # Choose penalty function
    penalty_func = get_penalty_function(penalty, **kwargs)

    # Choose solver
    solver_func = get_solver_function(solver, tol, max_iter)

    # Detect change points
    change_points = solver_func(normalized_data, metric_func, penalty_func)

    # Calculate metrics
    metrics = calculate_metrics(data, change_points, metric_func)

    # Prepare output
    result = {
        'result': {'change_points': change_points},
        'metrics': metrics,
        'params_used': {
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'penalty': penalty,
            'solver': solver
        },
        'warnings': []
    }

    return result

def validate_inputs(
    data: np.ndarray,
    metric: str,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    penalty: str,
    solver: str
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

    valid_metrics = ['mse', 'mae', 'r2']
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}.")

    valid_penalties = ['l1', 'l2', 'elasticnet']
    if penalty not in valid_penalties:
        raise ValueError(f"Penalty must be one of {valid_penalties}.")

    valid_solvers = ['closed_form', 'gradient_descent', 'newton']
    if solver not in valid_solvers:
        raise ValueError(f"Solver must be one of {valid_solvers}.")

def apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    return normalizer(data)

def get_metric_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the metric function based on the specified metric."""
    metrics = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r_squared
    }
    return metrics[metric]

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def get_penalty_function(penalty: str, **kwargs) -> Callable[[np.ndarray], float]:
    """Get the penalty function based on the specified penalty."""
    penalties = {
        'l1': lambda x: np.sum(np.abs(x)),
        'l2': lambda x: np.sum(x ** 2),
        'elasticnet': lambda x, l1_ratio=0.5: l1_ratio * np.sum(np.abs(x)) + (1 - l1_ratio) * np.sum(x ** 2)
    }
    return penalties[penalty]

def get_solver_function(
    solver: str,
    tol: float,
    max_iter: int
) -> Callable[[np.ndarray, Callable, Callable], np.ndarray]:
    """Get the solver function based on the specified solver."""
    solvers = {
        'closed_form': closed_form_solver,
        'gradient_descent': gradient_descent_solver,
        'newton': newton_solver
    }
    return solvers[solver]

def closed_form_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    penalty_func: Callable[[np.ndarray], float]
) -> np.ndarray:
    """Closed form solver for change point detection."""
    # Placeholder implementation
    return np.array([], dtype=int)

def gradient_descent_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    penalty_func: Callable[[np.ndarray], float],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Gradient descent solver for change point detection."""
    # Placeholder implementation
    return np.array([], dtype=int)

def newton_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    penalty_func: Callable[[np.ndarray], float],
    tol: float = 1e-4,
    max_iter: int = 1000
) -> np.ndarray:
    """Newton solver for change point detection."""
    # Placeholder implementation
    return np.array([], dtype=int)

def calculate_metrics(
    data: np.ndarray,
    change_points: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the detected change points."""
    # Placeholder implementation
    return {'metric_value': 0.0}

################################################################################
# segmentation
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def segmentation_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[List[int], Dict[str, float], Dict[str, str], List[str]]]:
    """
    Detects changepoints in a time series using segmentation.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate segmentation quality. Default is "mse".
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for segmentation. Default is "euclidean".
    solver : str, optional
        Solver to use for optimization. Default is "closed_form".
    regularization : Optional[str], optional
        Type of regularization to apply. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.

    Returns:
    --------
    Dict[str, Union[List[int], Dict[str, float], Dict[str, str], List[str]]]
        Dictionary containing the results of segmentation.
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, regularization)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Choose metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for segmentation
    changepoints = _solve_segmentation(
        normalized_data,
        metric_func=metric_func,
        distance_func=distance_func,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, changepoints, metric_func)

    # Prepare output
    result = {
        "result": {"changepoints": changepoints},
        "metrics": metrics,
        "params_used": {
            "normalizer": normalizer.__name__ if normalizer else None,
            "metric": metric,
            "distance": distance,
            "solver": solver,
            "regularization": regularization
        },
        "warnings": []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validates the inputs for segmentation."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Data contains infinite values.")

    valid_solvers = ["closed_form", "gradient_descent", "newton", "coordinate_descent"]
    if solver not in valid_solvers:
        raise ValueError(f"Solver must be one of {valid_solvers}.")

    valid_regularizations = [None, "l1", "l2", "elasticnet"]
    if regularization not in valid_regularizations:
        raise ValueError(f"Regularization must be one of {valid_regularizations}.")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Applies normalization to the data if a normalizer is provided."""
    if normalizer is None:
        return data
    return normalizer(data)

def _get_metric_function(
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns the metric function based on the input."""
    if custom_metric is not None:
        return custom_metric
    valid_metrics = {
        "mse": _mean_squared_error,
        "mae": _mean_absolute_error,
        "r2": _r_squared,
        "logloss": _log_loss
    }
    if isinstance(metric, str):
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {list(valid_metrics.keys())}.")
        return valid_metrics[metric]
    raise ValueError("Metric must be a string or a callable.")

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns the distance function based on the input."""
    if custom_distance is not None:
        return custom_distance
    valid_distances = {
        "euclidean": _euclidean_distance,
        "manhattan": _manhattan_distance,
        "cosine": _cosine_distance,
        "minkowski": _minkowski_distance
    }
    if isinstance(distance, str):
        if distance not in valid_distances:
            raise ValueError(f"Distance must be one of {list(valid_distances.keys())}.")
        return valid_distances[distance]
    raise ValueError("Distance must be a string or a callable.")

def _solve_segmentation(
    data: np.ndarray,
    *,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> List[int]:
    """Solves for segmentation using the specified solver."""
    if solver == "closed_form":
        return _closed_form_solution(data, metric_func, distance_func)
    elif solver == "gradient_descent":
        return _gradient_descent_solution(data, metric_func, distance_func, tol, max_iter)
    elif solver == "newton":
        return _newton_solution(data, metric_func, distance_func, tol, max_iter)
    elif solver == "coordinate_descent":
        return _coordinate_descent_solution(data, metric_func, distance_func, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

def _closed_form_solution(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> List[int]:
    """Computes the closed form solution for segmentation."""
    # Placeholder for actual implementation
    return []

def _gradient_descent_solution(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> List[int]:
    """Computes the gradient descent solution for segmentation."""
    # Placeholder for actual implementation
    return []

def _newton_solution(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> List[int]:
    """Computes the Newton solution for segmentation."""
    # Placeholder for actual implementation
    return []

def _coordinate_descent_solution(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> List[int]:
    """Computes the coordinate descent solution for segmentation."""
    # Placeholder for actual implementation
    return []

def _calculate_metrics(
    data: np.ndarray,
    changepoints: List[int],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculates the metrics for the segmentation."""
    # Placeholder for actual implementation
    return {}

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the R-squared metric."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the log loss."""
    # Placeholder for actual implementation
    return 0.0

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

################################################################################
# windowing_techniques
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def windowing_techniques_fit(
    data: np.ndarray,
    window_size: int,
    metric: Union[str, Callable] = 'mse',
    normalizer: Optional[Callable] = None,
    distance: Union[str, Callable] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_params: Optional[Dict] = None,
    custom_distance_params: Optional[Dict] = None
) -> Dict:
    """
    Detect change points using windowing techniques.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    window_size : int
        Size of the sliding window.
    metric : str or callable, optional
        Metric to use for change point detection ('mse', 'mae', 'r2', or custom callable).
    normalizer : callable, optional
        Normalization function to apply to the data.
    distance : str or callable, optional
        Distance metric ('euclidean', 'manhattan', 'cosine', 'minkowski', or custom callable).
    solver : str, optional
        Solver to use ('closed_form', 'gradient_descent', 'newton', 'coordinate_descent').
    regularization : str, optional
        Regularization type ('none', 'l1', 'l2', 'elasticnet').
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric_params : dict, optional
        Parameters for custom metric function.
    custom_distance_params : dict, optional
        Parameters for custom distance function.

    Returns:
    --------
    Dict containing:
        - 'result': List of detected change points.
        - 'metrics': Dictionary of computed metrics.
        - 'params_used': Dictionary of parameters used.
        - 'warnings': List of warnings.

    Example:
    --------
    >>> data = np.random.randn(100)
    >>> result = windowing_techniques_fit(data, window_size=10)
    """
    # Validate inputs
    _validate_inputs(data, window_size)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Compute change points
    change_points = _compute_change_points(
        normalized_data,
        window_size,
        metric,
        distance,
        solver,
        regularization,
        tol,
        max_iter,
        custom_metric_params,
        custom_distance_params
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, change_points, metric)

    # Prepare output
    result = {
        'result': change_points,
        'metrics': metrics,
        'params_used': {
            'window_size': window_size,
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

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
    normalizer: Optional[Callable]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    try:
        return normalizer(data)
    except Exception as e:
        raise ValueError(f"Normalization failed: {str(e)}")

def _compute_change_points(
    data: np.ndarray,
    window_size: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric_params: Optional[Dict],
    custom_distance_params: Optional[Dict]
) -> List[int]:
    """Compute change points using the specified method."""
    # Initialize change points list
    change_points = []

    # Get metric function
    metric_func = _get_metric_function(metric, custom_metric_params)

    # Get distance function
    distance_func = _get_distance_function(distance, custom_distance_params)

    # Iterate over the data with sliding window
    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        next_window = data[i + 1:i + window_size + 1]

        # Compute metric for current and next window
        current_metric = metric_func(window)
        next_metric = metric_func(next_window)

        # Compute distance between windows
        dist = distance_func(window, next_window)

        # Check for change point based on metric and distance
        if _is_change_point(current_metric, next_metric, dist, tol):
            change_points.append(i + window_size // 2)

    return change_points

def _get_metric_function(
    metric: Union[str, Callable],
    params: Optional[Dict]
) -> Callable:
    """Get the metric function based on the specified metric."""
    if callable(metric):
        return metric
    elif metric == 'mse':
        return lambda x: np.mean((x - np.mean(x))**2)
    elif metric == 'mae':
        return lambda x: np.mean(np.abs(x - np.mean(x)))
    elif metric == 'r2':
        return lambda x: 1 - np.sum((x - np.mean(x))**2) / np.sum((x - np.mean(x))**2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(
    distance: Union[str, Callable],
    params: Optional[Dict]
) -> Callable:
    """Get the distance function based on the specified distance."""
    if callable(distance):
        return distance
    elif distance == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif distance == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif distance == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif distance == 'minkowski':
        return lambda x, y: np.sum(np.abs(x - y)**params.get('p', 2))**(1/params.get('p', 2))
    else:
        raise ValueError(f"Unknown distance: {distance}")

def _is_change_point(
    current_metric: float,
    next_metric: float,
    distance: float,
    tol: float
) -> bool:
    """Determine if a change point is detected."""
    return abs(current_metric - next_metric) > tol and distance > tol

def _compute_metrics(
    data: np.ndarray,
    change_points: List[int],
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for the detected change points."""
    metric_func = _get_metric_function(metric, None)
    metrics = {}

    if change_points:
        segments = np.split(data, change_points)
        for i, segment in enumerate(segments):
            metrics[f'segment_{i}_metric'] = metric_func(segment)

    return metrics

################################################################################
# change_point_localization
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def change_point_localization_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[Dict, float, str]]:
    """
    Detects change points in a time series data using various configurable methods.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to use for change point detection. Can be "mse", "mae", "r2", or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric to use. Can be "euclidean", "manhattan", "cosine", or a custom callable.
    solver : str, optional
        Solver to use. Can be "closed_form", "gradient_descent", "newton", or "coordinate_descent".
    regularization : Optional[str], optional
        Regularization method. Can be "l1", "l2", or "elasticnet".
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function.

    Returns
    -------
    Dict[str, Union[Dict, float, str]]
        A dictionary containing the results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 10, 11, 12])
    >>> result = change_point_localization_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, regularization)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for change points
    if solver == "closed_form":
        result = _solve_closed_form(normalized_data, metric_func, distance_func)
    elif solver == "gradient_descent":
        result = _solve_gradient_descent(normalized_data, metric_func, distance_func, tol, max_iter)
    elif solver == "newton":
        result = _solve_newton(normalized_data, metric_func, distance_func, tol, max_iter)
    elif solver == "coordinate_descent":
        result = _solve_coordinate_descent(normalized_data, metric_func, distance_func, tol, max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Apply regularization if specified
    if regularization is not None:
        result = _apply_regularization(result, normalized_data, regularization)

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
        "warnings": _check_warnings(normalized_data, result)
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validates the input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data must not contain NaN or inf values.")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None.")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2"]:
        raise ValueError("Metric must be 'mse', 'mae', 'r2', or a custom callable.")
    if isinstance(distance, str) and distance not in ["euclidean", "manhattan", "cosine"]:
        raise ValueError("Distance must be 'euclidean', 'manhattan', 'cosine', or a custom callable.")
    if solver not in ["closed_form", "gradient_descent", "newton", "coordinate_descent"]:
        raise ValueError("Solver must be 'closed_form', 'gradient_descent', 'newton', or 'coordinate_descent'.")
    if regularization is not None and regularization not in ["l1", "l2", "elasticnet"]:
        raise ValueError("Regularization must be 'l1', 'l2', 'elasticnet', or None.")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Applies normalization to the data if a normalizer is provided."""
    if normalizer is not None:
        return normalizer(data)
    return data

def _get_metric_function(
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns the metric function based on the input."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    elif metric == "mse":
        return _mean_squared_error
    elif metric == "mae":
        return _mean_absolute_error
    elif metric == "r2":
        return _r_squared
    else:
        raise ValueError("Unknown metric.")

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns the distance function based on the input."""
    if callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance
    elif distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_distance
    else:
        raise ValueError("Unknown distance.")

def _solve_closed_form(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, Union[np.ndarray, float]]:
    """Solves for change points using a closed-form solution."""
    # Placeholder for actual implementation
    return {"change_points": np.array([len(data) // 2]), "value": metric_func(data[:len(data)//2], data[len(data)//2:])}

def _solve_gradient_descent(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> Dict[str, Union[np.ndarray, float]]:
    """Solves for change points using gradient descent."""
    # Placeholder for actual implementation
    return {"change_points": np.array([len(data) // 2]), "value": metric_func(data[:len(data)//2], data[len(data)//2:])}

def _solve_newton(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> Dict[str, Union[np.ndarray, float]]:
    """Solves for change points using Newton's method."""
    # Placeholder for actual implementation
    return {"change_points": np.array([len(data) // 2]), "value": metric_func(data[:len(data)//2], data[len(data)//2:])}

def _solve_coordinate_descent(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> Dict[str, Union[np.ndarray, float]]:
    """Solves for change points using coordinate descent."""
    # Placeholder for actual implementation
    return {"change_points": np.array([len(data) // 2]), "value": metric_func(data[:len(data)//2], data[len(data)//2:])}

def _apply_regularization(
    result: Dict[str, Union[np.ndarray, float]],
    data: np.ndarray,
    regularization: str
) -> Dict[str, Union[np.ndarray, float]]:
    """Applies regularization to the result."""
    # Placeholder for actual implementation
    return result

def _calculate_metrics(
    data: np.ndarray,
    result: Dict[str, Union[np.ndarray, float]],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculates metrics for the result."""
    # Placeholder for actual implementation
    return {"metric_value": metric_func(data[:len(data)//2], data[len(data)//2:])}

def _check_warnings(
    data: np.ndarray,
    result: Dict[str, Union[np.ndarray, float]]
) -> list:
    """Checks for warnings and returns them."""
    # Placeholder for actual implementation
    return []

def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the R-squared value."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

################################################################################
# multiple_change_points
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def multiple_change_points_fit(
    data: np.ndarray,
    *,
    n_changepoints: int = 1,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    penalty: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, any]]]:
    """
    Detect multiple change points in time series data.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data of shape (n_samples, n_features)
    n_changepoints : int
        Number of change points to detect (default: 1)
    normalizer : Optional[Callable]
        Function to normalize data (default: None)
    metric : Union[str, Callable]
        Metric to evaluate change points ('mse', 'mae', 'r2', or custom callable)
    distance : Union[str, Callable]
        Distance metric ('euclidean', 'manhattan', 'cosine', or custom callable)
    solver : str
        Solver method ('closed_form', 'gradient_descent', 'newton')
    penalty : Optional[str]
        Regularization type ('l1', 'l2', 'elasticnet')
    tol : float
        Tolerance for convergence (default: 1e-4)
    max_iter : int
        Maximum iterations (default: 1000)
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    Dict containing:
        - 'result': Detected change points
        - 'metrics': Evaluation metrics
        - 'params_used': Parameters used
        - 'warnings': Any warnings generated

    Example:
    --------
    >>> data = np.random.randn(100, 3)
    >>> result = multiple_change_points_fit(data, n_changepoints=2)
    """
    # Validate inputs
    _validate_inputs(data, n_changepoints)

    # Initialize parameters
    params_used = {
        'n_changepoints': n_changepoints,
        'normalizer': normalizer.__name__ if normalizer else None,
        'metric': metric if isinstance(metric, str) else 'custom',
        'distance': distance if isinstance(distance, str) else 'custom',
        'solver': solver,
        'penalty': penalty,
        'tol': tol,
        'max_iter': max_iter
    }

    # Normalize data if specified
    if normalizer is not None:
        try:
            data = normalizer(data)
        except Exception as e:
            return {
                'result': None,
                'metrics': {},
                'params_used': params_used,
                'warnings': f'Normalization failed: {str(e)}'
            }

    # Select metric function
    metric_func = _get_metric_function(metric)

    # Select distance function
    distance_func = _get_distance_function(distance)

    # Solve for change points
    changepoints, metrics = _solve_change_points(
        data,
        n_changepoints=n_changepoints,
        metric_func=metric_func,
        distance_func=distance_func,
        solver=solver,
        penalty=penalty,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )

    return {
        'result': changepoints,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(data: np.ndarray, n_changepoints: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim not in (1, 2):
        raise ValueError("Data must be 1D or 2D array")
    if n_changepoints <= 0:
        raise ValueError("n_changepoints must be positive")
    if n_changepoints >= len(data):
        raise ValueError("n_changepoints cannot be greater than or equal to data length")

def _get_metric_function(metric: Union[str, Callable]) -> Callable:
    """Get metric function based on input."""
    metrics = {
        'mse': _mean_squared_error,
        'mae': _mean_absolute_error,
        'r2': _r_squared
    }
    if isinstance(metric, str):
        return metrics.get(metric.lower(), _custom_metric)
    return metric

def _get_distance_function(distance: Union[str, Callable]) -> Callable:
    """Get distance function based on input."""
    distances = {
        'euclidean': _euclidean_distance,
        'manhattan': _manhattan_distance,
        'cosine': _cosine_distance
    }
    if isinstance(distance, str):
        return distances.get(distance.lower(), _custom_distance)
    return distance

def _solve_change_points(
    data: np.ndarray,
    *,
    n_changepoints: int,
    metric_func: Callable,
    distance_func: Callable,
    solver: str,
    penalty: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> tuple[np.ndarray, Dict[str, float]]:
    """Solve for change points using specified solver."""
    if solver == 'closed_form':
        return _solve_closed_form(data, n_changepoints)
    elif solver == 'gradient_descent':
        return _solve_gradient_descent(
            data, n_changepoints, metric_func, distance_func,
            penalty=penalty, tol=tol, max_iter=max_iter,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

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

def _custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Wrapper for custom metric function."""
    return metric_func(y_true, y_pred)

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _custom_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Wrapper for custom distance function."""
    return distance_func(a, b)

def _solve_closed_form(data: np.ndarray, n_changepoints: int) -> tuple[np.ndarray, Dict[str, float]]:
    """Solve for change points using closed form solution."""
    # Implement closed form solution
    changepoints = np.linspace(0, len(data)-1, n_changepoints+2)[1:-1].astype(int)
    metrics = {'metric': 0.0}  # Placeholder
    return changepoints, metrics

def _solve_gradient_descent(
    data: np.ndarray,
    n_changepoints: int,
    metric_func: Callable,
    distance_func: Callable,
    *,
    penalty: Optional[str],
    tol: float,
    max_iter: int,
    random_state: Optional[int]
) -> tuple[np.ndarray, Dict[str, float]]:
    """Solve for change points using gradient descent."""
    # Implement gradient descent solution
    if random_state is not None:
        np.random.seed(random_state)
    changepoints = np.sort(np.random.choice(len(data)-2, n_changepoints, replace=False))
    metrics = {'metric': 0.0}  # Placeholder
    return changepoints, metrics

################################################################################
# single_change_point
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def single_change_point_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "mse",
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[int, float, Dict, str]]:
    """
    Detect a single change point in time series data.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the change point. Default is "mse".
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for the solver. Default is "euclidean".
    solver : str, optional
        Solver to use for change point detection. Default is "closed_form".
    regularization : Optional[str], optional
        Regularization method. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.

    Returns
    -------
    Dict[str, Union[int, float, Dict, str]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 10, 11, 12])
    >>> result = single_change_point_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, regularization)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Solve for the change point
    if solver == "closed_form":
        change_point = _solve_closed_form(normalized_data, metric_func)
    elif solver == "gradient_descent":
        change_point = _solve_gradient_descent(
            normalized_data, metric_func, distance_func,
            tol=tol, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, change_point, metric_func)

    # Prepare the result dictionary
    result = {
        "result": {"change_point": change_point},
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

    return result

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate the inputs for single change point detection."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data must not contain NaN or infinite values.")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None.")
    if isinstance(metric, str) and metric not in ["mse", "mae", "r2", "logloss"]:
        raise ValueError("Unknown metric.")
    if isinstance(distance, str) and distance not in ["euclidean", "manhattan", "cosine", "minkowski"]:
        raise ValueError("Unknown distance.")

def _apply_normalization(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data if a normalizer is provided."""
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
    if metric == "mse":
        return _mean_squared_error
    elif metric == "mae":
        return _mean_absolute_error
    elif metric == "r2":
        return _r_squared
    elif metric == "logloss":
        return _log_loss
    else:
        raise ValueError("Unknown metric.")

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on the input."""
    if callable(distance):
        return distance
    if custom_distance is not None:
        return custom_distance
    if distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_distance
    elif distance == "minkowski":
        return _minkowski_distance
    else:
        raise ValueError("Unknown distance.")

def _solve_closed_form(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> int:
    """Solve for the change point using closed form solution."""
    n = len(data)
    costs = np.zeros(n - 1)
    for i in range(1, n):
        cost = metric_func(data[:i], data[i:])
        costs[i - 1] = cost
    return np.argmin(costs) + 1

def _solve_gradient_descent(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tol: float,
    max_iter: int
) -> int:
    """Solve for the change point using gradient descent."""
    n = len(data)
    change_point = n // 2
    for _ in range(max_iter):
        gradient = _compute_gradient(data, change_point, metric_func, distance_func)
        new_change_point = int(change_point - gradient)
        if new_change_point < 1:
            new_change_point = 1
        if new_change_point >= n - 1:
            new_change_point = n - 2
        if abs(new_change_point - change_point) < tol:
            break
        change_point = new_change_point
    return change_point

def _compute_gradient(
    data: np.ndarray,
    change_point: int,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Compute the gradient for gradient descent."""
    left = data[:change_point]
    right = data[change_point:]
    metric_left = metric_func(left, np.mean(left))
    metric_right = metric_func(right, np.mean(right))
    return distance_func(np.array([metric_left]), np.array([metric_right]))[0]

def _calculate_metrics(
    data: np.ndarray,
    change_point: int,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the change point."""
    left = data[:change_point]
    right = data[change_point:]
    metric_value = metric_func(left, right)
    return {"metric_value": metric_value}

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
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
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

def _minkowski_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** 3) ** (1/3)

################################################################################
# adaptive_windowing
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def adaptive_windowing_fit(
    data: np.ndarray,
    min_window_size: int = 5,
    max_window_size: int = 100,
    metric: Union[str, Callable[[np.ndarray], float]] = 'mse',
    distance: str = 'euclidean',
    solver: str = 'closed_form',
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect changepoints using adaptive windowing method.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    min_window_size : int, optional
        Minimum window size for detection (default: 5).
    max_window_size : int, optional
        Maximum window size for detection (default: 100).
    metric : str or callable, optional
        Metric to use for changepoint detection ('mse', 'mae', 'r2') or custom callable (default: 'mse').
    distance : str, optional
        Distance metric ('euclidean', 'manhattan', 'cosine') (default: 'euclidean').
    solver : str, optional
        Solver method ('closed_form', 'gradient_descent') (default: 'closed_form').
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust') (default: None).
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_iter : int, optional
        Maximum iterations for solver (default: 1000).
    random_state : int, optional
        Random seed for reproducibility (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used and warnings.
    """
    # Validate inputs
    _validate_inputs(data, min_window_size, max_window_size)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Select metric function
    if callable(metric):
        metric_func = metric
    else:
        metric_func = _get_metric_function(metric)

    # Select distance function
    distance_func = _get_distance_function(distance)

    # Select solver function
    solver_func = _get_solver_function(solver, random_state)

    # Detect changepoints
    changepoints = _detect_changepoints(
        normalized_data,
        min_window_size,
        max_window_size,
        metric_func,
        distance_func,
        solver_func,
        tol,
        max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, changepoints, metric_func)

    # Prepare output
    result = {
        'result': {'changepoints': changepoints},
        'metrics': metrics,
        'params_used': {
            'min_window_size': min_window_size,
            'max_window_size': max_window_size,
            'metric': metric if not callable(metric) else 'custom',
            'distance': distance,
            'solver': solver,
            'normalization': normalization
        },
        'warnings': []
    }

    return result

def _validate_inputs(
    data: np.ndarray,
    min_window_size: int,
    max_window_size: int
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Data must be 1-dimensional")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    if min_window_size <= 0:
        raise ValueError("min_window_size must be positive")
    if max_window_size <= min_window_size:
        raise ValueError("max_window_size must be greater than min_window_size")

def _apply_normalization(
    data: np.ndarray,
    method: Optional[str]
) -> np.ndarray:
    """Apply specified normalization to data."""
    if method is None or method == 'none':
        return data
    elif method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _get_metric_function(
    metric_name: str
) -> Callable[[np.ndarray], float]:
    """Get the specified metric function."""
    if metric_name == 'mse':
        return _mean_squared_error
    elif metric_name == 'mae':
        return _mean_absolute_error
    elif metric_name == 'r2':
        return _r_squared
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

def _get_distance_function(
    distance_name: str
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the specified distance function."""
    if distance_name == 'euclidean':
        return _euclidean_distance
    elif distance_name == 'manhattan':
        return _manhattan_distance
    elif distance_name == 'cosine':
        return _cosine_distance
    else:
        raise ValueError(f"Unknown distance: {distance_name}")

def _get_solver_function(
    solver_name: str,
    random_state: Optional[int]
) -> Callable:
    """Get the specified solver function."""
    if solver_name == 'closed_form':
        return _closed_form_solver
    elif solver_name == 'gradient_descent':
        return lambda **kwargs: _gradient_descent_solver(random_state, **kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

def _detect_changepoints(
    data: np.ndarray,
    min_window_size: int,
    max_window_size: int,
    metric_func: Callable[[np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver_func: Callable,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Detect changepoints using adaptive windowing."""
    n = len(data)
    changepoints = []

    for window_size in range(min_window_size, max_window_size + 1):
        for i in range(n - window_size + 1):
            window = data[i:i+window_size]
            # Here you would implement the actual detection logic
            # This is a placeholder for the actual implementation
            if _should_detect_changepoint(window, metric_func):
                changepoints.append(i + window_size // 2)

    # Remove duplicate changepoints and sort
    changepoints = sorted(list(set(changepoints)))
    return np.array(changepoints)

def _should_detect_changepoint(
    window: np.ndarray,
    metric_func: Callable[[np.ndarray], float]
) -> bool:
    """Determine if a changepoint should be detected in the given window."""
    # Placeholder for actual detection logic
    return metric_func(window) > 0.5

def _calculate_metrics(
    data: np.ndarray,
    changepoints: np.ndarray,
    metric_func: Callable[[np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the detected changepoints."""
    return {
        'num_changepoints': len(changepoints),
        'mean_metric': np.mean([metric_func(data[max(0, cp-5):min(len(data), cp+5)])
                              for cp in changepoints])
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
    return 1 - (ss_res / (ss_tot + 1e-8))

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _closed_form_solver(**kwargs) -> Any:
    """Closed form solver placeholder."""
    return {}

def _gradient_descent_solver(
    random_state: Optional[int],
    **kwargs
) -> Any:
    """Gradient descent solver placeholder."""
    if random_state is not None:
        np.random.seed(random_state)
    return {}

################################################################################
# fixed_windowing
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def fixed_windowing_fit(
    data: np.ndarray,
    window_size: int,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric_params: Optional[Dict] = None
) -> Dict:
    """
    Detect change points using fixed windowing approach.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    window_size : int
        Size of the sliding window for change point detection.
    metric : str or callable, optional
        Metric to use for comparison between windows. Options: 'mse', 'mae', 'r2'.
    normalizer : callable, optional
        Function to normalize data before processing.
    distance : str or callable, optional
        Distance metric for window comparison. Options: 'euclidean', 'manhattan', 'cosine'.
    solver : str, optional
        Solver to use for optimization. Options: 'closed_form', 'gradient_descent'.
    regularization : str, optional
        Regularization type. Options: None, 'l1', 'l2'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric_params : dict, optional
        Parameters for custom metric function.

    Returns:
    --------
    Dict containing:
        - 'result': Detected change points.
        - 'metrics': Computed metrics.
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during processing.

    Example:
    --------
    >>> data = np.random.randn(100)
    >>> result = fixed_windowing_fit(data, window_size=10)
    """
    # Validate inputs
    _validate_inputs(data, window_size)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Compute change points
    change_points = _compute_change_points(
        normalized_data,
        window_size,
        metric,
        distance,
        solver,
        regularization,
        tol,
        max_iter,
        custom_metric_params
    )

    # Compute metrics
    metrics = _compute_metrics(normalized_data, change_points, metric)

    # Prepare output
    result = {
        'result': change_points,
        'metrics': metrics,
        'params_used': {
            'window_size': window_size,
            'metric': metric,
            'normalizer': normalizer.__name__ if normalizer else None,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': []
    }

    return result

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
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]]
) -> np.ndarray:
    """Apply normalization to the data."""
    if normalizer is None:
        return data
    return normalizer(data)

def _compute_change_points(
    data: np.ndarray,
    window_size: int,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric_params: Optional[Dict]
) -> np.ndarray:
    """Compute change points using the specified method."""
    # Initialize variables
    n = len(data)
    change_points = []

    # Define metric function
    metric_func = _get_metric_function(metric, custom_metric_params)

    # Define distance function
    distance_func = _get_distance_function(distance)

    # Iterate through the data with sliding window
    for i in range(window_size, n - window_size):
        # Split data into two windows
        window1 = data[i - window_size:i]
        window2 = data[i:i + window_size]

        # Compute metric between windows
        current_metric = metric_func(window1, window2)

        # Check for change point based on threshold
        if current_metric > tol:
            change_points.append(i)

    return np.array(change_points)

def _get_metric_function(
    metric: Union[str, Callable],
    custom_metric_params: Optional[Dict]
) -> Callable:
    """Get the metric function based on the specified metric."""
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

def _get_distance_function(
    distance: Union[str, Callable]
) -> Callable:
    """Get the distance function based on the specified distance."""
    if callable(distance):
        return distance
    elif distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    else:
        raise ValueError(f"Unknown distance: {distance}")

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

def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance."""
    return np.linalg.norm(a - b)

def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance."""
    return np.sum(np.abs(a - b))

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _compute_metrics(
    data: np.ndarray,
    change_points: np.ndarray,
    metric: Union[str, Callable]
) -> Dict:
    """Compute metrics for the detected change points."""
    metrics = {}
    if callable(metric):
        metrics['custom_metric'] = _compute_custom_metrics(data, change_points, metric)
    else:
        metrics['mse'] = _compute_mse(data, change_points) if metric == 'mse' else None
        metrics['mae'] = _compute_mae(data, change_points) if metric == 'mae' else None
        metrics['r2'] = _compute_r2(data, change_points) if metric == 'r2' else None
    return metrics

def _compute_custom_metrics(
    data: np.ndarray,
    change_points: np.ndarray,
    metric_func: Callable
) -> float:
    """Compute custom metrics for the detected change points."""
    # Placeholder for custom metric computation
    return 0.0

def _compute_mse(data: np.ndarray, change_points: np.ndarray) -> float:
    """Compute MSE for the detected change points."""
    # Placeholder for MSE computation
    return 0.0

def _compute_mae(data: np.ndarray, change_points: np.ndarray) -> float:
    """Compute MAE for the detected change points."""
    # Placeholder for MAE computation
    return 0.0

def _compute_r2(data: np.ndarray, change_points: np.ndarray) -> float:
    """Compute R-squared for the detected change points."""
    # Placeholder for R-squared computation
    return 0.0

################################################################################
# dynamic_programming
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def dynamic_programming_fit(
    data: np.ndarray,
    metric: Union[str, Callable[[np.ndarray], float]] = "mse",
    penalty: Optional[Callable[[int, int], float]] = None,
    normalize: bool = True,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None,
    custom_penalty: Optional[Callable[[int, int], float]] = None
) -> Dict[str, Union[np.ndarray, Dict[str, float], str]]:
    """
    Detect changepoints in a time series using dynamic programming.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    metric : str or callable, optional
        Metric to use for changepoint detection. Options: "mse", "mae", custom callable.
    penalty : callable, optional
        Penalty function for changepoints. If None, no penalty is applied.
    normalize : bool, optional
        Whether to normalize the data before processing.
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.
    custom_penalty : callable, optional
        Custom penalty function if not using built-in penalties.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict[str, float], str]]
        Dictionary containing:
        - "result": Array of changepoint indices.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during processing.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 10, 11, 12])
    >>> result = dynamic_programming_fit(data)
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1-dimensional array.")

    # Normalize data if required
    if normalize:
        data = (data - np.mean(data)) / np.std(data)

    # Determine metric function
    if custom_metric is not None:
        metric_func = custom_metric
    elif isinstance(metric, str):
        if metric == "mse":
            metric_func = _compute_mse
        elif metric == "mae":
            metric_func = _compute_mae
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metric_func = metric

    # Determine penalty function
    if custom_penalty is not None:
        penalty_func = custom_penalty
    else:
        penalty_func = penalty

    # Compute changepoints using dynamic programming
    changepoints, metrics = _dynamic_programming_core(
        data,
        metric_func,
        penalty_func
    )

    # Prepare output dictionary
    result = {
        "result": changepoints,
        "metrics": metrics,
        "params_used": {
            "metric": metric if isinstance(metric, str) else "custom",
            "penalty": "none" if penalty is None and custom_penalty is None else "custom",
            "normalize": normalize
        },
        "warnings": []
    }

    return result

def _dynamic_programming_core(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray], float],
    penalty_func: Optional[Callable[[int, int], float]]
) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Core dynamic programming algorithm for changepoint detection.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    metric_func : callable
        Function to compute the metric for a segment.
    penalty_func : callable or None
        Function to compute the penalty for a changepoint.

    Returns
    -------
    tuple[np.ndarray, Dict[str, float]]
        Tuple containing:
        - Array of changepoint indices.
        - Dictionary of computed metrics.
    """
    n = len(data)
    cost_matrix = np.zeros((n, n))
    path_matrix = np.zeros((n, n), dtype=int)

    # Initialize cost matrix
    for i in range(n):
        for j in range(i, n):
            segment = data[i:j+1]
            cost_matrix[i, j] = metric_func(segment)
            if penalty_func is not None and i > 0:
                cost_matrix[i, j] += penalty_func(i, j)

    # Dynamic programming to find optimal path
    for i in range(1, n):
        for j in range(i, n):
            min_cost = float('inf')
            best_k = 0
            for k in range(i-1, -1, -1):
                current_cost = cost_matrix[k, i-1] + cost_matrix[i, j]
                if penalty_func is not None:
                    current_cost += penalty_func(i, j)
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_k = k
            cost_matrix[i, j] = min_cost
            path_matrix[i, j] = best_k

    # Backtrack to find changepoints
    changepoints = []
    i, j = 0, n-1
    while i < j:
        k = path_matrix[i, j]
        if k != 0:
            changepoints.append(k)
        i = k
        j -= 1

    # Compute metrics
    metrics = {
        "total_cost": cost_matrix[n-1, n-1],
        "num_changepoints": len(changepoints)
    }

    return np.array(changepoints), metrics

def _compute_mse(segment: np.ndarray) -> float:
    """
    Compute Mean Squared Error for a segment.

    Parameters
    ----------
    segment : np.ndarray
        Segment of the time series.

    Returns
    -------
    float
        Mean Squared Error.
    """
    return np.mean((segment - np.mean(segment)) ** 2)

def _compute_mae(segment: np.ndarray) -> float:
    """
    Compute Mean Absolute Error for a segment.

    Parameters
    ----------
    segment : np.ndarray
        Segment of the time series.

    Returns
    -------
    float
        Mean Absolute Error.
    """
    return np.mean(np.abs(segment - np.mean(segment)))

################################################################################
# pruning_techniques
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def pruning_techniques_fit(
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
) -> Dict[str, Any]:
    """
    Main function for applying pruning techniques in changepoint detection.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the pruning. Default is "mse".
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for pruning. Default is "euclidean".
    solver : str, optional
        Solver to use for optimization. Default is "closed_form".
    regularization : Optional[str], optional
        Type of regularization to apply. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    validate_inputs(data)

    # Normalize data if a normalizer is provided
    normalized_data = data if normalizer is None else normalizer(data)

    # Set metric and distance functions
    metric_func = get_metric_function(metric, custom_metric)
    distance_func = get_distance_function(distance, custom_distance)

    # Set solver function
    solver_func = get_solver_function(solver)

    # Estimate parameters using the chosen solver
    params = solver_func(
        normalized_data,
        metric_func=metric_func,
        distance_func=distance_func,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = calculate_metrics(normalized_data, params, metric_func)

    # Prepare the result dictionary
    result = {
        "result": params,
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

    return result

def validate_inputs(data: np.ndarray) -> None:
    """
    Validate the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

def get_metric_function(
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Get the metric function based on the input.

    Parameters:
    -----------
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Metric to evaluate the pruning.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom metric function.

    Returns:
    --------
    Callable[[np.ndarray, np.ndarray], float]
        The metric function.
    """
    if custom_metric is not None:
        return custom_metric
    elif isinstance(metric, str):
        if metric == "mse":
            return mean_squared_error
        elif metric == "mae":
            return mean_absolute_error
        elif metric == "r2":
            return r_squared
        elif metric == "logloss":
            return log_loss
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return metric

def get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Get the distance function based on the input.

    Parameters:
    -----------
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]]
        Distance metric for pruning.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]]
        Custom distance function.

    Returns:
    --------
    Callable[[np.ndarray, np.ndarray], float]
        The distance function.
    """
    if custom_distance is not None:
        return custom_distance
    elif isinstance(distance, str):
        if distance == "euclidean":
            return euclidean_distance
        elif distance == "manhattan":
            return manhattan_distance
        elif distance == "cosine":
            return cosine_distance
        elif distance == "minkowski":
            return minkowski_distance
        else:
            raise ValueError(f"Unknown distance: {distance}")
    else:
        return distance

def get_solver_function(solver: str) -> Callable[..., Dict[str, Any]]:
    """
    Get the solver function based on the input.

    Parameters:
    -----------
    solver : str
        Solver to use for optimization.

    Returns:
    --------
    Callable[..., Dict[str, Any]]
        The solver function.
    """
    if solver == "closed_form":
        return closed_form_solver
    elif solver == "gradient_descent":
        return gradient_descent_solver
    elif solver == "newton":
        return newton_solver
    elif solver == "coordinate_descent":
        return coordinate_descent_solver
    else:
        raise ValueError(f"Unknown solver: {solver}")

def calculate_metrics(
    data: np.ndarray,
    params: Dict[str, Any],
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """
    Calculate the metrics for the pruning results.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    params : Dict[str, Any]
        Parameters estimated by the solver.
    metric_func : Callable[[np.ndarray, np.ndarray], float]
        Metric function to evaluate the pruning.

    Returns:
    --------
    Dict[str, float]
        Dictionary containing the calculated metrics.
    """
    # Example: Calculate the metric for the pruning results
    # This is a placeholder and should be replaced with actual logic
    metric_value = metric_func(data, params["predictions"])
    return {"metric": metric_value}

# Example of a normalizer function
def standard_normalizer(data: np.ndarray) -> np.ndarray:
    """
    Standard normalizer function.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.

    Returns:
    --------
    np.ndarray
        Normalized data.
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# Example of a metric function
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error metric.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns:
    --------
    float
        Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

# Example of a distance function
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance function.

    Parameters:
    -----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.

    Returns:
    --------
    float
        Euclidean distance.
    """
    return np.linalg.norm(a - b)

# Example of a solver function
def closed_form_solver(
    data: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Closed form solver function.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    metric_func : Callable[[np.ndarray, np.ndarray], float]
        Metric function to evaluate the pruning.
    distance_func : Callable[[np.ndarray, np.ndarray], float]
        Distance function for pruning.
    regularization : Optional[str], optional
        Type of regularization to apply. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the estimated parameters.
    """
    # Placeholder for actual solver logic
    return {"predictions": data}  # Example output

# Additional metric, distance, and solver functions can be added here

################################################################################
# cost_functions
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def cost_functions_fit(
    data: np.ndarray,
    normalizations: Dict[str, Callable] = None,
    metrics: Union[str, Callable] = "mse",
    distances: Union[str, Callable] = "euclidean",
    solver: str = "closed_form",
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Compute cost functions for changepoint detection.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalizations : Dict[str, Callable], optional
        Dictionary of normalization functions. Default is None.
    metrics : Union[str, Callable], optional
        Metric to use. Can be 'mse', 'mae', 'r2', or a custom callable.
    distances : Union[str, Callable], optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'cosine',
        or a custom callable.
    solver : str, optional
        Solver to use. Can be 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization type. Can be 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable], optional
        Custom distance function. Default is None.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, normalizations, metrics, distances)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalizations)

    # Compute cost function based on solver
    if solver == "closed_form":
        result = _closed_form_solver(normalized_data, metrics, distances)
    elif solver == "gradient_descent":
        result = _gradient_descent_solver(normalized_data, metrics, distances,
                                         regularization, tol, max_iter)
    elif solver == "newton":
        result = _newton_solver(normalized_data, metrics, distances,
                               regularization, tol, max_iter)
    elif solver == "coordinate_descent":
        result = _coordinate_descent_solver(normalized_data, metrics, distances,
                                           regularization, tol, max_iter)
    else:
        raise ValueError("Invalid solver specified.")

    # Compute metrics
    metrics_result = _compute_metrics(result, normalized_data,
                                     custom_metric=custom_metric)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics_result,
        "params_used": {
            "normalizations": normalizations,
            "metrics": metrics,
            "distances": distances,
            "solver": solver,
            "regularization": regularization,
            "tol": tol,
            "max_iter": max_iter
        },
        "warnings": []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    normalizations: Dict[str, Callable],
    metrics: Union[str, Callable],
    distances: Union[str, Callable]
) -> None:
    """
    Validate input data and parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalizations : Dict[str, Callable]
        Dictionary of normalization functions.
    metrics : Union[str, Callable]
        Metric to use.
    distances : Union[str, Callable]
        Distance metric to use.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")

def _apply_normalization(
    data: np.ndarray,
    normalizations: Dict[str, Callable]
) -> np.ndarray:
    """
    Apply normalization to data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalizations : Dict[str, Callable]
        Dictionary of normalization functions.

    Returns:
    --------
    np.ndarray
        Normalized data.
    """
    if normalizations is None:
        return data

    for norm_name, norm_func in normalizations.items():
        if norm_name == "standard":
            data = (data - np.mean(data)) / np.std(data)
        elif norm_name == "minmax":
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        elif norm_name == "robust":
            data = (data - np.median(data)) / (np.percentile(data, 75) -
                                              np.percentile(data, 25))
        else:
            data = norm_func(data)

    return data

def _closed_form_solver(
    data: np.ndarray,
    metrics: Union[str, Callable],
    distances: Union[str, Callable]
) -> np.ndarray:
    """
    Closed form solver for cost function.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    metrics : Union[str, Callable]
        Metric to use.
    distances : Union[str, Callable]
        Distance metric to use.

    Returns:
    --------
    np.ndarray
        Result of the closed form solver.
    """
    # Placeholder for actual implementation
    return np.zeros_like(data)

def _gradient_descent_solver(
    data: np.ndarray,
    metrics: Union[str, Callable],
    distances: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """
    Gradient descent solver for cost function.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    metrics : Union[str, Callable]
        Metric to use.
    distances : Union[str, Callable]
        Distance metric to use.
    regularization : Optional[str]
        Regularization type.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    np.ndarray
        Result of the gradient descent solver.
    """
    # Placeholder for actual implementation
    return np.zeros_like(data)

def _newton_solver(
    data: np.ndarray,
    metrics: Union[str, Callable],
    distances: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """
    Newton solver for cost function.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    metrics : Union[str, Callable]
        Metric to use.
    distances : Union[str, Callable]
        Distance metric to use.
    regularization : Optional[str]
        Regularization type.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    np.ndarray
        Result of the Newton solver.
    """
    # Placeholder for actual implementation
    return np.zeros_like(data)

def _coordinate_descent_solver(
    data: np.ndarray,
    metrics: Union[str, Callable],
    distances: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """
    Coordinate descent solver for cost function.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    metrics : Union[str, Callable]
        Metric to use.
    distances : Union[str, Callable]
        Distance metric to use.
    regularization : Optional[str]
        Regularization type.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    np.ndarray
        Result of the coordinate descent solver.
    """
    # Placeholder for actual implementation
    return np.zeros_like(data)

def _compute_metrics(
    result: np.ndarray,
    data: np.ndarray,
    custom_metric: Optional[Callable] = None
) -> Dict:
    """
    Compute metrics for the result.

    Parameters:
    -----------
    result : np.ndarray
        Result of the solver.
    data : np.ndarray
        Input data array.
    custom_metric : Optional[Callable]
        Custom metric function.

    Returns:
    --------
    Dict
        Dictionary of computed metrics.
    """
    metrics = {}

    if custom_metric is not None:
        metrics["custom"] = custom_metric(result, data)
    else:
        if isinstance(metrics, str):
            if metrics == "mse":
                metrics["mse"] = np.mean((result - data) ** 2)
            elif metrics == "mae":
                metrics["mae"] = np.mean(np.abs(result - data))
            elif metrics == "r2":
                ss_res = np.sum((data - result) ** 2)
                ss_tot = np.sum((data - np.mean(data)) ** 2)
                metrics["r2"] = 1 - (ss_res / ss_tot)

    return metrics

################################################################################
# likelihood_ratio_tests
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def likelihood_ratio_tests_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict:
    """
    Fit likelihood ratio tests for changepoint detection.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features).
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the fit. Can be 'mse', 'mae', 'r2', or a custom callable.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for changepoint detection. Can be 'euclidean', 'manhattan',
        'cosine', 'minkowski', or a custom callable.
    solver : str, optional
        Solver to use. Options are 'closed_form', 'gradient_descent', 'newton',
        or 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization type. Options are 'l1', 'l2', or 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function.

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalizer)

    # Choose metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Choose solver
    if solver == 'closed_form':
        params = _solve_closed_form(normalized_data, distance_func)
    elif solver == 'gradient_descent':
        params = _solve_gradient_descent(normalized_data, distance_func,
                                        regularization=regularization,
                                        tol=tol, max_iter=max_iter)
    elif solver == 'newton':
        params = _solve_newton(normalized_data, distance_func,
                              regularization=regularization,
                              tol=tol, max_iter=max_iter)
    elif solver == 'coordinate_descent':
        params = _solve_coordinate_descent(normalized_data, distance_func,
                                          regularization=regularization,
                                          tol=tol, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, params, metric_func)

    # Prepare results
    result = {
        'result': params,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
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

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if normalizer is not None and not callable(normalizer):
        raise TypeError("Normalizer must be a callable or None")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2']:
        raise ValueError("Metric must be 'mse', 'mae', 'r2' or a custom callable")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan',
                                                     'cosine', 'minkowski']:
        raise ValueError("Distance must be 'euclidean', 'manhattan', 'cosine', "
                         "'minkowski' or a custom callable")

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
    """Get the metric function based on input."""
    if callable(metric):
        return metric
    elif custom_metric is not None:
        return custom_metric
    else:
        if metric == 'mse':
            return _mean_squared_error
        elif metric == 'mae':
            return _mean_absolute_error
        elif metric == 'r2':
            return _r_squared
        else:
            raise ValueError(f"Unknown metric: {metric}")

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function based on input."""
    if callable(distance):
        return distance
    elif custom_distance is not None:
        return custom_distance
    else:
        if distance == 'euclidean':
            return _euclidean_distance
        elif distance == 'manhattan':
            return _manhattan_distance
        elif distance == 'cosine':
            return _cosine_distance
        elif distance == 'minkowski':
            return _minkowski_distance
        else:
            raise ValueError(f"Unknown distance: {distance}")

def _solve_closed_form(
    data: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict:
    """Solve using closed form solution."""
    # Placeholder for actual implementation
    return {'changepoints': []}

def _solve_gradient_descent(
    data: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Solve using gradient descent."""
    # Placeholder for actual implementation
    return {'changepoints': []}

def _solve_newton(
    data: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Solve using Newton's method."""
    # Placeholder for actual implementation
    return {'changepoints': []}

def _solve_coordinate_descent(
    data: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    regularization: Optional[str] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict:
    """Solve using coordinate descent."""
    # Placeholder for actual implementation
    return {'changepoints': []}

def _calculate_metrics(
    data: np.ndarray,
    params: Dict,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict:
    """Calculate metrics for the fitted model."""
    # Placeholder for actual implementation
    return {'metric_value': 0.0}

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

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

################################################################################
# cumulative_sum_control_chart
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def cumulative_sum_control_chart_fit(
    data: np.ndarray,
    target_mean: float = 0.0,
    h: float = 5.0,
    k: Optional[float] = None,
    normalizer: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    metric: str = 'mse',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    verbose: bool = False
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Fit a Cumulative Sum Control Chart (CUSUM) for change point detection.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    target_mean : float, optional
        Target mean for the process (default is 0.0).
    h : float, optional
        Decision interval threshold (default is 5.0).
    k : Optional[float], optional
        Allowable distance from target (default is None, which sets k = h/2).
    normalizer : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the data (default is identity function).
    metric : str, optional
        Metric to use for change point detection ('mse', 'mae') (default is 'mse').
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function (default is None).
    verbose : bool, optional
        Whether to print intermediate steps (default is False).

    Returns:
    --------
    Dict[str, Union[np.ndarray, float, Dict]]
        Dictionary containing:
        - "result": Array of cumulative sums.
        - "metrics": Dictionary of computed metrics.
        - "params_used": Dictionary of parameters used.
        - "warnings": List of warnings encountered.

    Example:
    --------
    >>> data = np.random.randn(100)
    >>> result = cumulative_sum_control_chart_fit(data, target_mean=0.5, h=4.0)
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")

    # Set default k if not provided
    if k is None:
        k = h / 2

    # Normalize data
    normalized_data = normalizer(data)

    # Compute cumulative sums
    cumsum = _compute_cumulative_sums(normalized_data, target_mean, k)

    # Detect change points
    change_points = _detect_change_points(cumsum, h)

    # Compute metrics
    metrics = {}
    if custom_metric is not None:
        metrics['custom'] = custom_metric(data, normalized_data)
    else:
        if metric == 'mse':
            metrics['mse'] = np.mean((normalized_data - target_mean) ** 2)
        elif metric == 'mae':
            metrics['mae'] = np.mean(np.abs(normalized_data - target_mean))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Prepare output
    result = {
        "result": cumsum,
        "change_points": change_points,
        "metrics": metrics,
        "params_used": {
            "target_mean": target_mean,
            "h": h,
            "k": k
        },
        "warnings": []
    }

    return result

def _compute_cumulative_sums(
    data: np.ndarray,
    target_mean: float,
    k: float
) -> np.ndarray:
    """
    Compute cumulative sums for CUSUM.

    Parameters:
    -----------
    data : np.ndarray
        Normalized input data.
    target_mean : float
        Target mean for the process.
    k : float
        Allowable distance from target.

    Returns:
    --------
    np.ndarray
        Array of cumulative sums.
    """
    cumsum_plus = np.zeros_like(data)
    cumsum_minus = np.zeros_like(data)

    for i in range(1, len(data)):
        cumsum_plus[i] = max(0, data[i] - (target_mean + k) + cumsum_plus[i-1])
        cumsum_minus[i] = max(0, (target_mean - k) - data[i] + cumsum_minus[i-1])

    return np.column_stack((cumsum_plus, cumsum_minus))

def _detect_change_points(
    cumsum: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Detect change points based on cumulative sums.

    Parameters:
    -----------
    cumsum : np.ndarray
        Array of cumulative sums.
    h : float
        Decision interval threshold.

    Returns:
    --------
    np.ndarray
        Array of change point indices.
    """
    cumsum_plus, cumsum_minus = cumsum[:, 0], cumsum[:, 1]
    change_points_plus = np.where(cumsum_plus > h)[0]
    change_points_minus = np.where(cumsum_minus > h)[0]

    return np.unique(np.concatenate((change_points_plus, change_points_minus)))

################################################################################
# exponential_weighted_moving_average
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def exponential_weighted_moving_average_fit(
    data: np.ndarray,
    alpha: float = 0.3,
    normalize: str = 'none',
    metric: Callable[[np.ndarray, np.ndarray], float] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Fit an exponential weighted moving average model to detect changepoints.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    alpha : float, optional
        Smoothing factor (0 < alpha <= 1).
    normalize : str, optional
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    metric : Callable, optional
        Metric function to evaluate the fit (e.g., mse, mae).
    custom_metric : Callable, optional
        Custom metric function if not using built-in metrics.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    data = _validate_input(data)
    alpha = _validate_alpha(alpha)

    # Normalize data if required
    normalized_data, normalization_params = _normalize_data(data, normalize)

    # Initialize EWMA
    ewma_values = np.zeros_like(normalized_data)
    ewma_values[0] = normalized_data[0]

    # Compute EWMA
    for i in range(1, len(normalized_data)):
        ewma_values[i] = alpha * normalized_data[i] + (1 - alpha) * ewma_values[i-1]

    # Compute residuals
    residuals = normalized_data - ewma_values

    # Compute metrics if provided
    metrics = {}
    if metric is not None:
        metrics['metric'] = metric(normalized_data, ewma_values)
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(normalized_data, ewma_values)

    # Detect changepoints based on residuals
    changepoints = _detect_changepoints(residuals, tol=tol)

    return {
        'result': {
            'ewma_values': ewma_values,
            'residuals': residuals,
            'changepoints': changepoints
        },
        'metrics': metrics,
        'params_used': {
            'alpha': alpha,
            'normalize': normalize
        },
        'warnings': []
    }

def _validate_input(data: np.ndarray) -> np.ndarray:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")
    return data

def _validate_alpha(alpha: float) -> float:
    """Validate alpha parameter."""
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be in the range (0, 1].")
    return alpha

def _normalize_data(data: np.ndarray, method: str) -> tuple[np.ndarray, Dict[str, Any]]:
    """Normalize data based on the specified method."""
    if method == 'none':
        return data, {}
    elif method == 'standard':
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
        raise ValueError(f"Unknown normalization method: {method}.")
    return normalized_data, {'method': method}

def _detect_changepoints(residuals: np.ndarray, tol: float = 1e-6) -> list[int]:
    """Detect changepoints based on residuals."""
    changepoints = []
    for i in range(1, len(residuals)):
        if abs(residuals[i] - residuals[i-1]) > tol:
            changepoints.append(i)
    return changepoints

# Example usage
if __name__ == "__main__":
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = exponential_weighted_moving_average_fit(data)
    print(result)

################################################################################
# kernel_change_point_detection
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def kernel_change_point_detection_fit(
    data: np.ndarray,
    *,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'mse',
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'euclidean',
    solver: str = 'closed_form',
    regularization: Optional[str] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Detect change points in time series data using kernel methods.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to normalize the data. Default is None.
    metric : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Metric to evaluate the change points. Default is 'mse'.
    distance : Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
        Distance metric for kernel computation. Default is 'euclidean'.
    solver : str, optional
        Solver to use for optimization. Default is 'closed_form'.
    regularization : Optional[str], optional
        Regularization type. Default is None.
    tol : float, optional
        Tolerance for convergence. Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    custom_metric : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom metric function. Default is None.
    custom_distance : Optional[Callable[[np.ndarray, np.ndarray], float]], optional
        Custom distance function. Default is None.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate inputs
    _validate_inputs(data, normalizer, metric, distance, solver, regularization)

    # Normalize data if a normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Prepare metric and distance functions
    metric_func = _get_metric_function(metric, custom_metric)
    distance_func = _get_distance_function(distance, custom_distance)

    # Detect change points
    change_points = _detect_change_points(
        normalized_data,
        metric_func=metric_func,
        distance_func=distance_func,
        solver=solver,
        regularization=regularization,
        tol=tol,
        max_iter=max_iter
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, change_points, metric_func)

    # Prepare output
    result = {
        'result': {'change_points': change_points},
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
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

def _validate_inputs(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]],
    metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    solver: str,
    regularization: Optional[str]
) -> None:
    """Validate input parameters."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values.")
    if normalizer is not None and not callable(normalizer):
        raise ValueError("Normalizer must be a callable function.")
    if isinstance(metric, str) and metric not in ['mse', 'mae', 'r2', 'logloss']:
        raise ValueError("Metric must be one of 'mse', 'mae', 'r2', 'logloss' or a custom callable.")
    if isinstance(distance, str) and distance not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        raise ValueError("Distance must be one of 'euclidean', 'manhattan', 'cosine', 'minkowski' or a custom callable.")
    if solver not in ['closed_form', 'gradient_descent', 'newton', 'coordinate_descent']:
        raise ValueError("Solver must be one of 'closed_form', 'gradient_descent', 'newton', 'coordinate_descent'.")
    if regularization is not None and regularization not in ['l1', 'l2', 'elasticnet']:
        raise ValueError("Regularization must be one of 'l1', 'l2', 'elasticnet' or None.")

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
    """Get the metric function."""
    if callable(metric):
        return metric
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
        raise ValueError("Invalid metric specified.")

def _get_distance_function(
    distance: Union[str, Callable[[np.ndarray, np.ndarray], float]],
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]]
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Get the distance function."""
    if callable(distance):
        return distance
    if custom_distance is not None:
        return custom_distance
    if distance == 'euclidean':
        return _euclidean_distance
    elif distance == 'manhattan':
        return _manhattan_distance
    elif distance == 'cosine':
        return _cosine_distance
    elif distance == 'minkowski':
        return _minkowski_distance
    else:
        raise ValueError("Invalid distance specified.")

def _detect_change_points(
    data: np.ndarray,
    *,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    solver: str,
    regularization: Optional[str],
    tol: float,
    max_iter: int
) -> np.ndarray:
    """Detect change points in the data."""
    # Placeholder for actual implementation
    return np.array([], dtype=int)

def _calculate_metrics(
    data: np.ndarray,
    change_points: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """Calculate metrics for the change points."""
    # Placeholder for actual implementation
    return {'metric_value': 0.0}

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
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
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

def _minkowski_distance(a: np.ndarray, b: np.ndarray, p: int = 3) -> float:
    """Calculate Minkowski distance."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

################################################################################
# neural_network_based_methods
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def neural_network_based_methods_fit(
    data: np.ndarray,
    normalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
    Detect change points in time series data using neural network based methods.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    normalizer : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function. Default is None.
    metric : Union[str, Callable], optional
        Metric to use for evaluation. Options: 'mse', 'mae', 'r2', custom callable.
    distance : Union[str, Callable], optional
        Distance metric. Options: 'euclidean', 'manhattan', 'cosine', 'minkowski', custom callable.
    solver : str, optional
        Solver to use. Options: 'gradient_descent', 'newton', 'coordinate_descent'.
    regularization : Optional[str], optional
        Regularization type. Options: 'l1', 'l2', 'elasticnet'.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    custom_metric : Optional[Callable], optional
        Custom metric function if not using built-in metrics.
    custom_distance : Optional[Callable], optional
        Custom distance function if not using built-in distances.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> result = neural_network_based_methods_fit(data, normalizer=np.std, metric='mse')
    """
    # Validate inputs
    _validate_inputs(data)

    # Normalize data if normalizer is provided
    normalized_data = _apply_normalization(data, normalizer)

    # Initialize neural network parameters
    params = _initialize_parameters(normalized_data.shape[0])

    # Choose solver and optimize
    if solver == 'gradient_descent':
        optimized_params = _gradient_descent(
            normalized_data, params, metric, distance,
            regularization, tol, max_iter, custom_metric, custom_distance
        )
    elif solver == 'newton':
        optimized_params = _newton_method(
            normalized_data, params, metric, distance,
            regularization, tol, max_iter, custom_metric, custom_distance
        )
    elif solver == 'coordinate_descent':
        optimized_params = _coordinate_descent(
            normalized_data, params, metric, distance,
            regularization, tol, max_iter, custom_metric, custom_distance
        )
    else:
        raise ValueError("Unsupported solver. Choose from 'gradient_descent', 'newton', 'coordinate_descent'.")

    # Detect change points
    change_points = _detect_change_points(normalized_data, optimized_params, distance)

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, optimized_params, metric)

    # Prepare results
    result = {
        'result': change_points,
        'metrics': metrics,
        'params_used': {
            'normalizer': normalizer.__name__ if normalizer else None,
            'metric': metric,
            'distance': distance,
            'solver': solver,
            'regularization': regularization,
            'tol': tol,
            'max_iter': max_iter
        },
        'warnings': _check_warnings(normalized_data, optimized_params)
    }

    return result

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

def _initialize_parameters(n: int) -> Dict:
    """Initialize neural network parameters."""
    return {
        'weights': np.random.randn(n, 1),
        'bias': np.zeros(1)
    }

def _gradient_descent(
    data: np.ndarray,
    params: Dict,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> Dict:
    """Perform gradient descent optimization."""
    # Implementation of gradient descent
    pass

def _newton_method(
    data: np.ndarray,
    params: Dict,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> Dict:
    """Perform Newton's method optimization."""
    # Implementation of Newton's method
    pass

def _coordinate_descent(
    data: np.ndarray,
    params: Dict,
    metric: Union[str, Callable],
    distance: Union[str, Callable],
    regularization: Optional[str],
    tol: float,
    max_iter: int,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> Dict:
    """Perform coordinate descent optimization."""
    # Implementation of coordinate descent
    pass

def _detect_change_points(
    data: np.ndarray,
    params: Dict,
    distance: Union[str, Callable]
) -> np.ndarray:
    """Detect change points in the data."""
    # Implementation of change point detection
    pass

def _calculate_metrics(
    data: np.ndarray,
    params: Dict,
    metric: Union[str, Callable]
) -> Dict:
    """Calculate evaluation metrics."""
    # Implementation of metric calculation
    pass

def _check_warnings(
    data: np.ndarray,
    params: Dict
) -> list:
    """Check for potential warnings."""
    # Implementation of warning checks
    pass

################################################################################
# deep_learning_approaches
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union, List

def deep_learning_approaches_fit(
    data: np.ndarray,
    window_size: int = 10,
    normalization: str = 'standard',
    metric: Union[str, Callable] = 'mse',
    solver: str = 'gradient_descent',
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict:
    """
    Detect change points in time series data using deep learning approaches.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    window_size : int, optional
        Size of the sliding window for change point detection, by default 10.
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust'), by default 'standard'.
    metric : Union[str, Callable], optional
        Metric to use for change point detection ('mse', 'mae', 'r2', custom callable), by default 'mse'.
    solver : str, optional
        Solver to use ('gradient_descent', 'newton'), by default 'gradient_descent'.
    learning_rate : float, optional
        Learning rate for gradient descent, by default 0.01.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    tol : float, optional
        Tolerance for convergence, by default 1e-4.
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.
    custom_distance : Optional[Callable], optional
        Custom distance function, by default None.

    Returns
    -------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> result = deep_learning_approaches_fit(data, window_size=5, metric='mae')
    """
    # Validate inputs
    _validate_inputs(data, window_size)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Initialize parameters
    params = {
        'window_size': window_size,
        'normalization': normalization,
        'metric': metric,
        'solver': solver,
        'learning_rate': learning_rate,
        'max_iter': max_iter,
        'tol': tol
    }

    # Detect change points
    change_points = _detect_change_points(
        normalized_data,
        window_size,
        metric,
        solver,
        learning_rate,
        max_iter,
        tol,
        custom_metric,
        custom_distance
    )

    # Calculate metrics
    metrics = _calculate_metrics(normalized_data, change_points, metric, custom_metric)

    # Prepare results
    result = {
        'result': change_points,
        'metrics': metrics,
        'params_used': params,
        'warnings': []
    }

    return result

def _validate_inputs(data: np.ndarray, window_size: int) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values.")
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if window_size > len(data):
        raise ValueError("Window size must be less than or equal to the length of data.")

def _normalize_data(data: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data based on the specified method."""
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == 'robust':
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _detect_change_points(
    data: np.ndarray,
    window_size: int,
    metric: Union[str, Callable],
    solver: str,
    learning_rate: float,
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> List[int]:
    """Detect change points in the data."""
    change_points = []
    n = len(data)

    for i in range(window_size, n - window_size):
        window = data[i - window_size:i + window_size]
        if solver == 'gradient_descent':
            change_point = _gradient_descent(
                window,
                metric,
                learning_rate,
                max_iter,
                tol,
                custom_metric,
                custom_distance
            )
        elif solver == 'newton':
            change_point = _newton_method(
                window,
                metric,
                max_iter,
                tol,
                custom_metric,
                custom_distance
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if change_point is not None:
            change_points.append(i + change_point)

    return change_points

def _gradient_descent(
    window: np.ndarray,
    metric: Union[str, Callable],
    learning_rate: float,
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> Optional[int]:
    """Perform gradient descent to find change point."""
    n = len(window)
    change_point = n // 2
    prev_loss = float('inf')

    for _ in range(max_iter):
        loss = _compute_loss(window, change_point, metric, custom_metric, custom_distance)
        if abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

        # Update change point using gradient descent
        gradient = _compute_gradient(window, change_point, metric, custom_metric, custom_distance)
        change_point = int(change_point - learning_rate * gradient)

        # Ensure change point is within bounds
        change_point = max(0, min(n - 1, change_point))

    return change_point if loss < tol else None

def _newton_method(
    window: np.ndarray,
    metric: Union[str, Callable],
    max_iter: int,
    tol: float,
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> Optional[int]:
    """Perform Newton's method to find change point."""
    n = len(window)
    change_point = n // 2
    prev_loss = float('inf')

    for _ in range(max_iter):
        loss = _compute_loss(window, change_point, metric, custom_metric, custom_distance)
        if abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

        # Update change point using Newton's method
        gradient = _compute_gradient(window, change_point, metric, custom_metric, custom_distance)
        hessian = _compute_hessian(window, change_point, metric, custom_metric, custom_distance)
        change_point = int(change_point - gradient / (hessian + 1e-8))

        # Ensure change point is within bounds
        change_point = max(0, min(n - 1, change_point))

    return change_point if loss < tol else None

def _compute_loss(
    window: np.ndarray,
    change_point: int,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> float:
    """Compute the loss for a given change point."""
    left = window[:change_point]
    right = window[change_point:]

    if custom_metric is not None:
        return custom_metric(left, right)
    elif metric == 'mse':
        return np.mean((left - np.mean(left))**2) + np.mean((right - np.mean(right))**2)
    elif metric == 'mae':
        return np.mean(np.abs(left - np.mean(left))) + np.mean(np.abs(right - np.mean(right)))
    elif metric == 'r2':
        return 1 - (np.sum((left - np.mean(left))**2) + np.sum((right - np.mean(right))**2)) / np.var(window)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_gradient(
    window: np.ndarray,
    change_point: int,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> float:
    """Compute the gradient for a given change point."""
    left = window[:change_point]
    right = window[change_point:]

    if custom_metric is not None:
        return _numerical_gradient(window, change_point, custom_metric)
    elif metric == 'mse':
        return _numerical_gradient(window, change_point, lambda x, y: np.mean((x - np.mean(x))**2) + np.mean((y - np.mean(y))**2))
    elif metric == 'mae':
        return _numerical_gradient(window, change_point, lambda x, y: np.mean(np.abs(x - np.mean(x))) + np.mean(np.abs(y - np.mean(y))))
    elif metric == 'r2':
        return _numerical_gradient(window, change_point, lambda x, y: 1 - (np.sum((x - np.mean(x))**2) + np.sum((y - np.mean(y))**2)) / np.var(window))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _compute_hessian(
    window: np.ndarray,
    change_point: int,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    custom_distance: Optional[Callable]
) -> float:
    """Compute the Hessian for a given change point."""
    left = window[:change_point]
    right = window[change_point:]

    if custom_metric is not None:
        return _numerical_hessian(window, change_point, custom_metric)
    elif metric == 'mse':
        return _numerical_hessian(window, change_point, lambda x, y: np.mean((x - np.mean(x))**2) + np.mean((y - np.mean(y))**2))
    elif metric == 'mae':
        return _numerical_hessian(window, change_point, lambda x, y: np.mean(np.abs(x - np.mean(x))) + np.mean(np.abs(y - np.mean(y))))
    elif metric == 'r2':
        return _numerical_hessian(window, change_point, lambda x, y: 1 - (np.sum((x - np.mean(x))**2) + np.sum((y - np.mean(y))**2)) / np.var(window))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _numerical_gradient(
    window: np.ndarray,
    change_point: int,
    metric_func: Callable
) -> float:
    """Compute numerical gradient."""
    h = 1e-5
    loss_plus = _compute_loss(window, change_point + h, 'mse', None, None)
    loss_minus = _compute_loss(window, change_point - h, 'mse', None, None)
    return (loss_plus - loss_minus) / (2 * h)

def _numerical_hessian(
    window: np.ndarray,
    change_point: int,
    metric_func: Callable
) -> float:
    """Compute numerical Hessian."""
    h = 1e-5
    gradient_plus = _numerical_gradient(window, change_point + h, metric_func)
    gradient_minus = _numerical_gradient(window, change_point - h, metric_func)
    return (gradient_plus - gradient_minus) / (2 * h)

def _calculate_metrics(
    data: np.ndarray,
    change_points: List[int],
    metric: Union[str, Callable],
    custom_metric: Optional[Callable]
) -> Dict:
    """Calculate metrics for the detected change points."""
    metrics = {}

    if custom_metric is not None:
        metrics['custom_metric'] = _compute_custom_metrics(data, change_points, custom_metric)
    else:
        if metric == 'mse':
            metrics['mse'] = _compute_mse(data, change_points)
        elif metric == 'mae':
            metrics['mae'] = _compute_mae(data, change_points)
        elif metric == 'r2':
            metrics['r2'] = _compute_r2(data, change_points)

    return metrics

def _compute_mse(
    data: np.ndarray,
    change_points: List[int]
) -> float:
    """Compute MSE for the detected change points."""
    total_mse = 0.0
    n = len(data)

    if not change_points:
        return float('inf')

    prev_point = 0
    for point in change_points:
        segment = data[prev_point:point]
        total_mse += np.mean((segment - np.mean(segment))**2)
        prev_point = point

    segment = data[prev_point:]
    total_mse += np.mean((segment - np.mean(segment))**2)

    return total_mse / len(change_points)

def _compute_mae(
    data: np.ndarray,
    change_points: List[int]
) -> float:
    """Compute MAE for the detected change points."""
    total_mae = 0.0
    n = len(data)

    if not change_points:
        return float('inf')

    prev_point = 0
    for point in change_points:
        segment = data[prev_point:point]
        total_mae += np.mean(np.abs(segment - np.mean(segment)))
        prev_point = point

    segment = data[prev_point:]
    total_mae += np.mean(np.abs(segment - np.mean(segment)))

    return total_mae / len(change_points)

def _compute_r2(
    data: np.ndarray,
    change_points: List[int]
) -> float:
    """Compute R2 for the detected change points."""
    total_r2 = 0.0
    n = len(data)

    if not change_points:
        return float('inf')

    prev_point = 0
    for point in change_points:
        segment = data[prev_point:point]
        total_r2 += 1 - (np.sum((segment - np.mean(segment))**2) / np.var(data))
        prev_point = point

    segment = data[prev_point:]
    total_r2 += 1 - (np.sum((segment - np.mean(segment))**2) / np.var(data))

    return total_r2 / len(change_points)

def _compute_custom_metrics(
    data: np.ndarray,
    change_points: List[int],
    custom_metric: Callable
) -> float:
    """Compute custom metrics for the detected change points."""
    total_metric = 0.0
    n = len(data)

    if not change_points:
        return float('inf')

    prev_point = 0
    for point in change_points:
        left = data[prev_point:point]
        right = data[point:]
        total_metric += custom_metric(left, right)
        prev_point = point

    return total_metric / len(change_points)
