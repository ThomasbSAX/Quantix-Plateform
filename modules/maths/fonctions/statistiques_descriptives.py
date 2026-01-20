"""
Quantix – Module statistiques_descriptives
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# mesure_tendance_centrale
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

def _normalize_data(
    data: np.ndarray,
    normalization: str = "none",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Normalize data based on the specified method."""
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
    elif custom_normalization is not None:
        return custom_normalization(data)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_measure(
    data: np.ndarray,
    measure_type: str = "mean",
    custom_measure: Optional[Callable[[np.ndarray], float]] = None
) -> float:
    """Compute the central tendency measure."""
    if custom_measure is not None:
        return custom_measure(data)
    elif measure_type == "mean":
        return np.mean(data)
    elif measure_type == "median":
        return np.median(data)
    elif measure_type == "mode":
        values, counts = np.unique(data, return_counts=True)
        return float(values[np.argmax(counts)])
    else:
        raise ValueError(f"Unknown measure type: {measure_type}")

def _compute_metrics(
    data: np.ndarray,
    measure_value: float,
    metrics: Union[str, list] = "all",
    custom_metric: Optional[Callable[[np.ndarray, float], Dict[str, float]]] = None
) -> Dict[str, float]:
    """Compute metrics for the central tendency measure."""
    if custom_metric is not None:
        return custom_metric(data, measure_value)

    metrics_dict = {}
    if metrics == "all" or "mse" in metrics:
        mse = np.mean((data - measure_value) ** 2)
        metrics_dict["mse"] = mse
    if metrics == "all" or "mae" in metrics:
        mae = np.mean(np.abs(data - measure_value))
        metrics_dict["mae"] = mae
    if metrics == "all" or "r2" in metrics:
        ss_total = np.sum((data - np.mean(data)) ** 2)
        ss_residual = np.sum((data - measure_value) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        metrics_dict["r2"] = r2
    return metrics_dict

def mesure_tendance_centrale_fit(
    data: np.ndarray,
    measure_type: str = "mean",
    normalization: str = "none",
    metrics: Union[str, list] = "all",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_measure: Optional[Callable[[np.ndarray], float]] = None,
    custom_metric: Optional[Callable[[np.ndarray, float], Dict[str, float]]] = None
) -> Dict:
    """Compute central tendency measures and metrics for descriptive statistics.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    measure_type : str, optional
        Type of central tendency measure to compute ("mean", "median", "mode").
    normalization : str, optional
        Normalization method to apply ("none", "standard", "minmax", "robust").
    metrics : str or list, optional
        Metrics to compute ("all", "mse", "mae", "r2").
    custom_normalization : callable, optional
        Custom normalization function.
    custom_measure : callable, optional
        Custom central tendency measure function.
    custom_metric : callable, optional
        Custom metric computation function.

    Returns:
    --------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)
    normalized_data = _normalize_data(data, normalization, custom_normalization)
    measure_value = _compute_measure(normalized_data, measure_type, custom_measure)
    metrics_dict = _compute_metrics(data, measure_value, metrics, custom_metric)

    return {
        "result": measure_value,
        "metrics": metrics_dict,
        "params_used": {
            "measure_type": measure_type,
            "normalization": normalization,
            "metrics": metrics
        },
        "warnings": []
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = mesure_tendance_centrale_fit(data)

################################################################################
# moyenne
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for mean calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(data).any():
        raise ValueError("Input contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input contains infinite values")

def _compute_mean(data: np.ndarray) -> float:
    """Compute the arithmetic mean of the data."""
    return np.mean(data)

def _compute_standardized_mean(data: np.ndarray) -> float:
    """Compute the standardized mean (z-score)."""
    return _compute_mean(data) / np.std(data)

def _compute_robust_mean(data: np.ndarray) -> float:
    """Compute the robust mean (median)."""
    return np.median(data)

def _compute_minmax_mean(data: np.ndarray) -> float:
    """Compute the min-max normalized mean."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return 0.0
    normalized_data = (data - data_min) / (data_max - data_min)
    return _compute_mean(normalized_data)

def moyenne_fit(
    data: np.ndarray,
    normalization: str = "none",
    metric: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, float], float]] = None
) -> Dict[str, Any]:
    """
    Compute the mean of data with various options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization : str, optional (default="none")
        Normalization method: "none", "standard", "minmax", or "robust".
    metric : str, optional
        Metric to compute: "mse", "mae", or None.
    custom_metric : Callable, optional
        Custom metric function that takes (data, mean) and returns a float.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    normalization_methods = {
        "none": _compute_mean,
        "standard": _compute_standardized_mean,
        "minmax": _compute_minmax_mean,
        "robust": _compute_robust_mean
    }

    if normalization not in normalization_methods:
        raise ValueError(f"Normalization method must be one of: {list(normalization_methods.keys())}")

    mean_func = normalization_methods[normalization]
    result = mean_func(data)

    metrics = {}
    if metric is not None:
        if metric == "mse":
            metrics["mse"] = np.mean((data - result) ** 2)
        elif metric == "mae":
            metrics["mae"] = np.mean(np.abs(data - result))
        else:
            raise ValueError("Metric must be 'mse' or 'mae'")

    if custom_metric is not None:
        metrics["custom"] = custom_metric(data, result)

    warnings = []
    if normalization == "standard" and np.std(data) == 0:
        warnings.append("Standard deviation is zero, standardized mean is undefined")

    return {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "normalization": normalization,
            "metric": metric,
            "custom_metric": custom_metric is not None
        },
        "warnings": warnings
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = moyenne_fit(data, normalization="standard", metric="mse")

################################################################################
# mediane
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for median calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(data).any():
        raise ValueError("Input contains NaN values")
    if not np.isfinite(data).all():
        raise ValueError("Input contains infinite values")

def _compute_median(data: np.ndarray) -> float:
    """Compute the median of a 1D array."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return float(sorted_data[n // 2])
    else:
        return float((sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2)

def mediane_fit(
    data: np.ndarray,
    method: str = "default",
    custom_metric: Optional[Callable[[np.ndarray, float], float]] = None,
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Compute the median of a dataset with configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Method for median calculation. Currently only "default" is supported.
    custom_metric : Callable, optional
        Custom metric function to evaluate the median. Must take (data, median) and return a float.

    Returns
    -------
    dict
        Dictionary containing:
        - "result": computed median
        - "metrics": dictionary of metrics (if custom_metric is provided)
        - "params_used": parameters used in the calculation
        - "warnings": list of warnings

    Examples
    --------
    >>> data = np.array([1, 3, 5, 7, 9])
    >>> mediane_fit(data)
    {
        'result': 5.0,
        'metrics': {},
        'params_used': {'method': 'default'},
        'warnings': []
    }
    """
    _validate_input(data)

    params_used = {
        "method": method,
    }

    warnings: list = []

    if method != "default":
        raise ValueError(f"Unsupported method: {method}")

    median = _compute_median(data)

    metrics = {}
    if custom_metric is not None:
        try:
            metrics["custom"] = custom_metric(data, median)
        except Exception as e:
            warnings.append(f"Custom metric calculation failed: {str(e)}")

    return {
        "result": median,
        "metrics": metrics,
        "params_used": params_used,
        "warnings": warnings,
    }

################################################################################
# mode
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def mode_fit(
    data: np.ndarray,
    *,
    distance_metric: str = 'euclidean',
    custom_distance: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """
    Compute the mode of a dataset using specified distance metric.

    Parameters:
    -----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'cosine')
    custom_distance : Callable, optional
        Custom distance function if not using built-in metrics
    tolerance : float, optional
        Convergence tolerance for mode calculation
    max_iterations : int, optional
        Maximum number of iterations

    Returns:
    --------
    Dict with keys:
        - 'result': computed mode
        - 'metrics': distance metrics information
        - 'params_used': parameters used in computation
        - 'warnings': any warnings encountered

    Example:
    --------
    >>> data = np.array([[1, 2], [1, 4], [5, 8]])
    >>> result = mode_fit(data)
    """
    # Validate inputs
    _validate_inputs(data, distance_metric, custom_distance)

    # Initialize parameters dictionary
    params_used = {
        'distance_metric': distance_metric,
        'tolerance': tolerance,
        'max_iterations': max_iterations
    }

    # Select distance function
    if custom_distance is not None:
        distance_func = custom_distance
    else:
        distance_func = _get_distance_function(distance_metric)

    # Compute mode
    mode_result, distances = _compute_mode(data, distance_func, tolerance, max_iterations)

    # Calculate metrics
    metrics = _calculate_metrics(distances)

    return {
        'result': mode_result,
        'metrics': metrics,
        'params_used': params_used,
        'warnings': []
    }

def _validate_inputs(
    data: np.ndarray,
    distance_metric: str,
    custom_distance: Optional[Callable]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")

    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional")

    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")

    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

    valid_metrics = ['euclidean', 'manhattan', 'cosine']
    if distance_metric not in valid_metrics and custom_distance is None:
        raise ValueError(f"Invalid distance metric. Choose from {valid_metrics} or provide custom_distance")

def _get_distance_function(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return distance function based on metric name."""
    if metric == 'euclidean':
        return lambda x, y: np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x, y: 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

def _compute_mode(
    data: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    tolerance: float,
    max_iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mode using Lloyd's algorithm."""
    # Initialize with random point
    current_mode = data[np.random.randint(0, len(data))]

    for _ in range(max_iterations):
        # Assign each point to nearest mode
        distances = np.array([distance_func(current_mode, x) for x in data])
        closest_indices = np.argmin(distances, axis=0)

        # Update mode to mean of assigned points
        new_mode = np.mean(data[closest_indices], axis=0)

        # Check for convergence
        if distance_func(current_mode, new_mode) < tolerance:
            break

        current_mode = new_mode

    return current_mode, distances

def _calculate_metrics(distances: np.ndarray) -> Dict[str, float]:
    """Calculate metrics from distance calculations."""
    return {
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances)
    }

################################################################################
# quartiles
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for quartiles computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data must not contain NaN or infinite values.")

def _compute_quartiles(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """Compute quartiles using the specified interpolation method."""
    sorted_data = np.sort(data)
    n = len(sorted_data)

    def _linear_interpolation(pos: float) -> float:
        """Linear interpolation for quartile computation."""
        if pos < 1:
            return sorted_data[0]
        elif pos > n:
            return sorted_data[-1]
        else:
            k = int(pos)
            d = pos - k
            return (1 - d) * sorted_data[k - 1] + d * sorted_data[k]

    def _nearest_interpolation(pos: float) -> float:
        """Nearest neighbor interpolation for quartile computation."""
        k = int(round(pos))
        return sorted_data[max(0, min(k, n - 1))]

    interpolation_methods = {
        'linear': _linear_interpolation,
        'nearest': _nearest_interpolation
    }

    if method not in interpolation_methods:
        raise ValueError(f"Interpolation method must be one of {list(interpolation_methods.keys())}")

    quartiles = np.array([0.25, 0.5, 0.75])
    positions = (n - 1) * quartiles
    return np.array([interpolation_methods[method](pos) for pos in positions])

def quartiles_fit(
    data: np.ndarray,
    method: str = 'linear',
    custom_interpolation: Optional[Callable[[float], float]] = None
) -> Dict[str, Union[np.ndarray, Dict, str]]:
    """
    Compute quartiles of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data for which to compute quartiles.
    method : str, optional
        Interpolation method to use ('linear' or 'nearest'). Default is 'linear'.
    custom_interpolation : Callable[[float], float], optional
        Custom interpolation function. If provided, overrides the method parameter.

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': Computed quartiles (Q1, median, Q3).
        - 'metrics': Dictionary of metrics (currently empty for quartiles).
        - 'params_used': Parameters used in the computation.
        - 'warnings': Any warnings generated during computation.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> quartiles_fit(data)
    {
        'result': array([2. , 3. , 4. ]),
        'metrics': {},
        'params_used': {'method': 'linear', 'custom_interpolation': None},
        'warnings': []
    }
    """
    _validate_input(data)

    if custom_interpolation is not None:
        method = 'custom'

    params_used = {
        'method': method,
        'custom_interpolation': custom_interpolation
    }

    warnings = []

    if method == 'custom':
        if not callable(custom_interpolation):
            raise TypeError("Custom interpolation must be a callable function.")
        quartiles = _compute_quartiles(data, method='linear')
        sorted_data = np.sort(data)
        n = len(sorted_data)
        positions = (n - 1) * np.array([0.25, 0.5, 0.75])
        quartiles = np.array([custom_interpolation(pos) for pos in positions])
    else:
        quartiles = _compute_quartiles(data, method=method)

    return {
        'result': quartiles,
        'metrics': {},
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# ecart_interquartile
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for interquartile range calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _calculate_quartiles(data: np.ndarray, method: str = 'linear') -> Dict[str, float]:
    """Calculate quartiles using specified interpolation method."""
    if method not in ['linear', 'lower', 'higher', 'midpoint', 'nearest']:
        raise ValueError("Invalid interpolation method")

    sorted_data = np.sort(data)
    n = len(sorted_data)

    def get_quartile(pos: float) -> float:
        k = (n - 1) * pos
        f = np.floor(k)
        c = np.ceil(k)

        if method == 'linear':
            d0 = sorted_data[int(f)] * (c - k)
            d1 = sorted_data[int(c)] * (k - f)
            return d0 + d1
        elif method == 'lower':
            return sorted_data[int(f)]
        elif method == 'higher':
            return sorted_data[int(c)]
        elif method == 'midpoint':
            return 0.5 * (sorted_data[int(f)] + sorted_data[int(c)])
        elif method == 'nearest':
            return sorted_data[np.round(k).astype(int)]

    q1 = get_quartile(0.25)
    q3 = get_quartile(0.75)

    return {'q1': q1, 'q3': q3}

def ecart_interquartile_fit(
    data: np.ndarray,
    interpolation_method: str = 'linear',
    custom_quartile_func: Optional[Callable[[np.ndarray], Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Calculate the interquartile range (IQR) of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to calculate the IQR.
    interpolation_method : str, optional
        Interpolation method for quartile calculation ('linear', 'lower',
        'higher', 'midpoint', 'nearest'). Default is 'linear'.
    custom_quartile_func : callable, optional
        Custom function to calculate quartiles. Must return a dict with 'q1' and 'q3'.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'result': float, the calculated IQR
        - 'metrics': dict with quartile values
        - 'params_used': dict of parameters used
        - 'warnings': list of warnings (if any)

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> ecart_interquartile_fit(data)
    {
        'result': 2.0,
        'metrics': {'q1': 1.75, 'q3': 4.25},
        'params_used': {'interpolation_method': 'linear', 'custom_quartile_func': None},
        'warnings': []
    }
    """
    _validate_input(data)

    params_used = {
        'interpolation_method': interpolation_method,
        'custom_quartile_func': custom_quartile_func is not None
    }

    warnings = []

    if custom_quartile_func is not None:
        try:
            quartiles = custom_quartile_func(data)
        except Exception as e:
            raise ValueError(f"Custom quartile function failed: {str(e)}")
    else:
        quartiles = _calculate_quartiles(data, interpolation_method)

    iqr = quartiles['q3'] - quartiles['q1']

    return {
        'result': iqr,
        'metrics': quartiles,
        'params_used': params_used,
        'warnings': warnings
    }

################################################################################
# variance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def variance_fit(
    data: np.ndarray,
    axis: int = 0,
    ddof: int = 1,
    normalization: str = "none",
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute the variance of a dataset with configurable normalization.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    axis : int, optional
        Axis along which to compute the variance (default is 0).
    ddof : int, optional
        Delta degrees of freedom (default is 1 for sample variance).
    normalization : str, optional
        Type of normalization to apply before computing variance.
        Options: "none", "standard", "minmax", "robust" (default is "none").
    custom_normalization : Callable[[np.ndarray], np.ndarray], optional
        Custom normalization function to apply (overrides `normalization` if provided).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the computed variance and metadata.

    Examples:
    ---------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> result = variance_fit(data)
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if axis not in (0, 1):
        raise ValueError("Axis must be either 0 or 1.")
    if ddof < 0:
        raise ValueError("Delta degrees of freedom (ddof) must be non-negative.")

    # Apply normalization if specified
    normalized_data = _apply_normalization(data, normalization, custom_normalization)

    # Compute variance
    result = {
        "result": _compute_variance(normalized_data, axis=axis, ddof=ddof),
        "params_used": {
            "axis": axis,
            "ddof": ddof,
            "normalization": normalization if custom_normalization is None else "custom",
        },
        "warnings": [],
    }

    return result

def _apply_normalization(
    data: np.ndarray,
    normalization: str,
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    """
    Apply normalization to the input data based on specified method.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization : str
        Type of normalization to apply.
    custom_normalization : Callable[[np.ndarray], np.ndarray], optional
        Custom normalization function.

    Returns:
    --------
    np.ndarray
        Normalized data array.
    """
    if custom_normalization is not None:
        return custom_normalization(data)

    if normalization == "standard":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / std
    elif normalization == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif normalization == "robust":
        median = np.median(data, axis=0)
        iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
        return (data - median) / iqr
    else:
        return data

def _compute_variance(
    data: np.ndarray,
    axis: int = 0,
    ddof: int = 1,
) -> np.ndarray:
    """
    Compute the variance of the input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    axis : int, optional
        Axis along which to compute the variance (default is 0).
    ddof : int, optional
        Delta degrees of freedom (default is 1 for sample variance).

    Returns:
    --------
    np.ndarray
        Variance of the data along the specified axis.
    """
    mean = np.mean(data, axis=axis)
    squared_diff = (data - mean) ** 2
    return np.sum(squared_diff, axis=axis) / (data.shape[axis] - ddof)

################################################################################
# ecart_type
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for standard deviation calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")

def _compute_standard_deviation(
    data: np.ndarray,
    ddof: int = 1,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """Compute standard deviation with optional custom metric."""
    if custom_metric is not None:
        return {"result": custom_metric(data), "metrics": {}}

    mean = np.mean(data)
    squared_diffs = (data - mean) ** 2
    variance = np.sum(squared_diffs) / (len(data) - ddof)
    std_dev = np.sqrt(variance)

    return {
        "result": std_dev,
        "metrics": {"variance": variance, "mean": mean}
    }

def ecart_type_fit(
    data: np.ndarray,
    ddof: int = 1,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the standard deviation of a dataset with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to compute the standard deviation.
    ddof : int, optional
        Delta degrees of freedom. The divisor used in the calculation is N - ddof,
        where N is the number of elements. Default is 1 (sample standard deviation).
    custom_metric : Callable[[np.ndarray], float], optional
        Custom function to compute a metric instead of the standard deviation.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> ecart_type_fit(data)
    {
        'result': 1.5811388300841898,
        'metrics': {'variance': 2.5, 'mean': 3.0},
        'params_used': {'ddof': 1},
        'warnings': []
    }
    """
    _validate_input(data)

    result = _compute_standard_deviation(data, ddof, custom_metric)

    return {
        "result": result["result"],
        "metrics": result["metrics"],
        "params_used": {"ddof": ddof},
        "warnings": []
    }

################################################################################
# etendue
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for range calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.isnan(data).any():
        raise ValueError("Input contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input contains infinite values")

def _compute_range(data: np.ndarray) -> float:
    """Compute the range of a dataset."""
    return np.max(data) - np.min(data)

def etendue_fit(
    data: np.ndarray,
    *,
    normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the range of a dataset with optional normalization.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalize : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Normalization function to apply before computation. If None, no normalization is applied.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str]]]
        Dictionary containing:
        - "result": computed range
        - "metrics": empty dict (no metrics for range)
        - "params_used": dictionary of parameters used
        - "warnings": empty dict (no warnings for range)

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> etendue_fit(data)
    {
        'result': 4.0,
        'metrics': {},
        'params_used': {'normalize': None},
        'warnings': {}
    }
    """
    _validate_input(data)

    # Apply normalization if provided
    normalized_data = data.copy()
    if normalize is not None:
        try:
            normalized_data = normalize(normalized_data)
        except Exception as e:
            raise ValueError(f"Normalization failed: {str(e)}")

    # Compute range
    result = _compute_range(normalized_data)

    return {
        "result": float(result),
        "metrics": {},
        "params_used": {"normalize": normalize},
        "warnings": {}
    }

################################################################################
# asymetrie
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for skewness calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or infinite values.")

def _calculate_skewness(data: np.ndarray, method: str = 'moment') -> float:
    """Calculate skewness using the specified method."""
    if method == 'moment':
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    elif method == 'median':
        median = np.median(data)
        skewness = 3 * (np.mean(data) - median) / np.std(data, ddof=1)
    else:
        raise ValueError(f"Unknown skewness method: {method}")
    return float(skewness)

def _calculate_metrics(data: np.ndarray, skewness: float) -> Dict[str, float]:
    """Calculate additional metrics related to skewness."""
    return {
        'skewness': skewness,
        'kurtosis': _calculate_kurtosis(data),
    }

def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of the data."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float(kurtosis)

def asymetrie_fit(
    data: np.ndarray,
    method: str = 'moment',
    custom_metric: Optional[Callable[[np.ndarray, float], Dict[str, float]]] = None,
    **kwargs
) -> Dict[str, Union[Dict[str, float], Dict[str, str], list]]:
    """
    Calculate skewness of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data for which skewness is to be calculated.
    method : str, optional
        Method to calculate skewness ('moment' or 'median'), by default 'moment'.
    custom_metric : Optional[Callable], optional
        Custom metric function, by default None.

    Returns
    -------
    Dict[str, Union[Dict[str, float], Dict[str, str], list]]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = asymetrie_fit(data)
    """
    _validate_input(data)

    skewness = _calculate_skewness(data, method)
    metrics = _calculate_metrics(data, skewness) if custom_metric is None else custom_metric(data, skewness)

    return {
        'result': {'skewness': skewness},
        'metrics': metrics,
        'params_used': {
            'method': method,
            'custom_metric': custom_metric.__name__ if custom_metric else None
        },
        'warnings': []
    }

################################################################################
# aplatissement
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for flatness computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_flatness_metric(data: np.ndarray, metric: str) -> float:
    """Compute flatness metric based on user choice."""
    if metric == 'mse':
        return np.mean((data - np.mean(data))**2)
    elif metric == 'mae':
        return np.mean(np.abs(data - np.mean(data)))
    elif metric == 'r2':
        ss_total = np.sum((data - np.mean(data))**2)
        return 1 - (ss_total / ss_total) if ss_total != 0 else 0.0
    elif callable(metric):
        return metric(data)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _normalize_data(data: np.ndarray, normalization: str) -> np.ndarray:
    """Normalize data based on user choice."""
    if normalization == 'none':
        return data
    elif normalization == 'standard':
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    elif normalization == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    elif normalization == 'robust':
        return (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

def aplatissement_compute(
    data: np.ndarray,
    metric: Union[str, Callable] = 'mse',
    normalization: str = 'none',
    custom_metric: Optional[Callable] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Compute flatness of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data to compute flatness for.
    metric : str or callable, optional
        Metric to use for flatness computation ('mse', 'mae', 'r2' or custom callable).
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust').
    custom_metric : callable, optional
        Custom metric function if not using built-in metrics.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings.

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0])
    >>> result = aplatissement_compute(data)
    """
    _validate_inputs(data)

    # Normalize data
    normalized_data = _normalize_data(data, normalization)

    # Compute flatness metric
    if custom_metric is not None:
        flatness = _compute_flatness_metric(normalized_data, custom_metric)
    else:
        flatness = _compute_flatness_metric(normalized_data, metric)

    return {
        'result': flatness,
        'metrics': {'flatness': flatness},
        'params_used': {
            'metric': metric if not custom_metric else 'custom',
            'normalization': normalization
        },
        'warnings': []
    }

################################################################################
# skewness
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for skewness calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _compute_skewness(data: np.ndarray, bias: bool = False) -> float:
    """Compute the skewness of a dataset."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0 if bias else 1)

    if std == 0:
        return 0.0

    skewness = (n / ((n - 1) if not bias else n)) * np.sum(((data - mean) / std) ** 3)
    return skewness

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data using the specified method."""
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
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def skewness_fit(
    data: np.ndarray,
    normalize_method: str = "none",
    bias_corrected: bool = False,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the skewness of a dataset with various options.

    Parameters:
    -----------
    data : np.ndarray
        Input data.
    normalize_method : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    bias_corrected : bool, optional
        Whether to apply bias correction.
    custom_metric : Callable[[np.ndarray], float], optional
        Custom metric function.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    normalized_data = _normalize_data(data, normalize_method)
    skewness_value = _compute_skewness(normalized_data, bias=bias_corrected)

    metrics = {}
    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(normalized_data)

    result = {
        "result": skewness_value,
        "metrics": metrics,
        "params_used": {
            "normalize_method": normalize_method,
            "bias_corrected": bias_corrected
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = skewness_fit(data, normalize_method="standard", bias_corrected=True)

################################################################################
# kurtosis
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for kurtosis calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def _compute_kurtosis(data: np.ndarray, fisher: bool = True) -> float:
    """Compute the kurtosis of a dataset."""
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    if std_dev == 0:
        raise ValueError("Standard deviation is zero, cannot compute kurtosis")

    fourth_moment = np.mean((data - mean) ** 4)
    squared_std_dev = std_dev ** 2

    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (
        (n - 1) * (n - 2) / ((n - 1) ** 2)) * fourth_moment / (squared_std_dev ** 2) - 3 * ((n - 1) ** 2 / ((n - 2) * (n - 3)))

    if fisher:
        kurtosis -= 3

    return kurtosis

def _compute_metrics(data: np.ndarray, kurtosis: float) -> Dict[str, float]:
    """Compute additional metrics related to kurtosis."""
    return {
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis if not fisher else kurtosis + 3
    }

def kurtosis_fit(
    data: np.ndarray,
    fisher: bool = True,
    normalize: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray, float], Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Compute the kurtosis of a dataset with various options.

    Parameters:
    -----------
    data : np.ndarray
        Input data for which to compute kurtosis.
    fisher : bool, optional
        Whether to use Fisher's definition of kurtosis (default: True).
    normalize : str, optional
        Normalization method to apply before computation. Options: 'standard', 'minmax', 'robust'.
    custom_metric : Callable, optional
        Custom metric function to compute additional metrics.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = kurtosis_fit(data)
    """
    _validate_input(data)

    if normalize == 'standard':
        data = (data - np.mean(data)) / np.std(data)
    elif normalize == 'minmax':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif normalize == 'robust':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        data = (data - q1) / iqr

    kurtosis_value = _compute_kurtosis(data, fisher)
    metrics = _compute_metrics(data, kurtosis_value)

    if custom_metric is not None:
        metrics.update(custom_metric(data, kurtosis_value))

    return {
        "result": kurtosis_value,
        "metrics": metrics,
        "params_used": {
            "fisher": fisher,
            "normalize": normalize
        },
        "warnings": []
    }

################################################################################
# coefficient_variation
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for coefficient of variation calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_coefficient_variation(data: np.ndarray, ddof: int = 0) -> float:
    """Compute the coefficient of variation."""
    mean = np.mean(data)
    std = np.std(data, ddof=ddof)
    if mean == 0:
        raise ValueError("Mean of data is zero, coefficient of variation undefined")
    return std / mean

def coefficient_variation_fit(
    data: np.ndarray,
    ddof: int = 0,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the coefficient of variation with optional custom metrics.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    ddof : int, optional
        Delta degrees of freedom for standard deviation calculation (default 0).
    custom_metric : Callable, optional
        Custom metric function that takes data as input and returns a float.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = coefficient_variation_fit(data)
    """
    _validate_inputs(data)

    # Compute main result
    try:
        cv = _compute_coefficient_variation(data, ddof)
    except ValueError as e:
        return {
            "result": None,
            "metrics": {},
            "params_used": {"ddof": ddof},
            "warnings": str(e)
        }

    # Compute custom metric if provided
    metrics = {}
    if custom_metric is not None:
        try:
            metrics["custom_metric"] = custom_metric(data)
        except Exception as e:
            metrics["custom_metric_error"] = str(e)

    return {
        "result": cv,
        "metrics": metrics,
        "params_used": {"ddof": ddof},
        "warnings": []
    }

################################################################################
# percentiles
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for percentiles computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize data according to specified method."""
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
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_percentiles(data: np.ndarray, percentiles: Union[list, np.ndarray], method: str = "linear") -> Dict[str, float]:
    """Compute percentiles using specified interpolation method."""
    if not isinstance(percentiles, (list, np.ndarray)):
        raise TypeError("Percentiles must be a list or numpy array")
    if method not in ["linear", "lower", "higher", "midpoint", "nearest"]:
        raise ValueError(f"Unknown interpolation method: {method}")

    sorted_data = np.sort(data)
    n = len(sorted_data)

    result = {}
    for p in percentiles:
        if not 0 <= p <= 100:
            raise ValueError(f"Percentile value must be between 0 and 100, got {p}")
        rank = (n - 1) * (p / 100)
        lower = int(np.floor(rank))
        upper = int(np.ceil(rank))

        if method == "linear":
            value = sorted_data[lower] + (sorted_data[upper] - sorted_data[lower]) * (rank - lower)
        elif method == "lower":
            value = sorted_data[lower]
        elif method == "higher":
            value = sorted_data[upper] if upper < n else sorted_data[-1]
        elif method == "midpoint":
            value = (sorted_data[lower] + sorted_data[upper]) / 2
        elif method == "nearest":
            value = sorted_data[int(round(rank))] if int(round(rank)) < n else sorted_data[-1]

        result[f"p_{int(p)}"] = float(value)

    return result

def percentiles_fit(
    data: np.ndarray,
    percentiles: Union[list, np.ndarray],
    normalization: str = "none",
    interpolation_method: str = "linear"
) -> Dict[str, Union[Dict[str, float], Dict[str, str], Dict[str, str], list]]:
    """
    Compute percentiles for given data with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1-dimensional)
    percentiles : Union[list, np.ndarray]
        List of percentile values to compute (0-100)
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust")
    interpolation_method : str, optional
        Interpolation method for percentile computation

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> percentiles_fit(data, [25, 50, 75], normalization="standard")
    """
    # Validate input
    _validate_input(data)

    # Normalize data if needed
    normalized_data = _normalize_data(data, normalization)
    norm_method_used = normalization

    # Compute percentiles
    result = _compute_percentiles(normalized_data, percentiles, interpolation_method)

    # Prepare output
    output = {
        "result": result,
        "metrics": {},
        "params_used": {
            "normalization": norm_method_used,
            "interpolation_method": interpolation_method
        },
        "warnings": []
    }

    return output

################################################################################
# distribution_frequences
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for frequency distribution."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_bins(data: np.ndarray, bins: Union[int, np.ndarray], range: Optional[tuple] = None) -> tuple:
    """Compute bins for frequency distribution."""
    if isinstance(bins, int):
        return np.histogram_bin_edges(data, bins=bins, range=range)
    elif isinstance(bins, np.ndarray):
        if len(bins) < 2:
            raise ValueError("Custom bins must have at least two elements")
        return np.sort(bins)
    else:
        raise TypeError("Bins must be either an integer or a numpy array")

def _normalize_frequencies(frequencies: np.ndarray, method: str = 'count') -> np.ndarray:
    """Normalize frequencies according to specified method."""
    if method == 'count':
        return frequencies
    elif method == 'density':
        total = np.sum(frequencies)
        if total > 0:
            return frequencies / (total * np.diff(bins)[0])
        else:
            return frequencies
    elif method == 'percent':
        total = np.sum(frequencies)
        if total > 0:
            return frequencies / total * 100
        else:
            return frequencies
    elif method == 'probability':
        total = np.sum(frequencies)
        if total > 0:
            return frequencies / total
        else:
            return frequencies
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _compute_metrics(frequencies: np.ndarray, bins: np.ndarray, metrics: Dict[str, Callable]) -> Dict[str, float]:
    """Compute specified metrics for frequency distribution."""
    results = {}
    for name, metric_func in metrics.items():
        try:
            if name == 'mean':
                results[name] = np.average(bins[:-1], weights=frequencies)
            elif name == 'median':
                cumulative = np.cumsum(frequencies)
                median_idx = np.searchsorted(cumulative, cumulative[-1] / 2)
                results[name] = bins[median_idx]
            elif name == 'std':
                mean_val = np.average(bins[:-1], weights=frequencies)
                variance = np.average((bins[:-1] - mean_val)**2, weights=frequencies)
                results[name] = np.sqrt(variance)
            else:
                results[name] = metric_func(frequencies, bins)
        except Exception as e:
            results[name] = np.nan
    return results

def distribution_frequences_fit(
    data: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    range: Optional[tuple] = None,
    normalize: str = 'count',
    metrics: Dict[str, Callable] = None,
    custom_metrics: Optional[Dict[str, Callable]] = None
) -> Dict:
    """
    Compute frequency distribution with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array
    bins : int or np.ndarray, optional
        Number of bins or custom bin edges (default: 10)
    range : tuple, optional
        Range of the bins (default: None)
    normalize : str, optional
        Normalization method ('count', 'density', 'percent', 'probability') (default: 'count')
    metrics : Dict[str, Callable], optional
        Dictionary of built-in metrics to compute (default: None)
    custom_metrics : Dict[str, Callable], optional
        Dictionary of custom metrics to compute (default: None)

    Returns:
    --------
    Dict
        Dictionary containing results, metrics, parameters used, and warnings

    Example:
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = distribution_frequences_fit(data, bins=20, normalize='density')
    """
    # Validate input
    _validate_input(data)

    # Initialize default metrics if not provided
    if metrics is None:
        metrics = {
            'mean': lambda x, y: np.average(y[:-1], weights=x),
            'median': lambda x, y: np.median(np.repeat(y[:-1], x)),
            'std': lambda x, y: np.std(np.repeat(y[:-1], x))
        }

    # Compute bins
    bins = _compute_bins(data, bins, range)

    # Compute frequency distribution
    frequencies, _ = np.histogram(data, bins=bins)

    # Normalize frequencies
    normalized_frequencies = _normalize_frequencies(frequencies, normalize)

    # Compute metrics
    computed_metrics = _compute_metrics(normalized_frequencies, bins, metrics)

    # Add custom metrics if provided
    if custom_metrics is not None:
        for name, metric_func in custom_metrics.items():
            try:
                computed_metrics[name] = metric_func(normalized_frequencies, bins)
            except Exception as e:
                computed_metrics[name] = np.nan

    # Prepare results
    result = {
        'result': {
            'bins': bins,
            'frequencies': normalized_frequencies
        },
        'metrics': computed_metrics,
        'params_used': {
            'bins': bins if isinstance(bins, int) else list(bins),
            'range': range,
            'normalize': normalize
        },
        'warnings': []
    }

    return result

################################################################################
# histogramme
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for histogram computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_bins(data: np.ndarray, bins: Union[int, np.ndarray], range: Optional[tuple] = None) -> tuple:
    """Compute bin edges and indices for histogram."""
    if isinstance(bins, int):
        if range is None:
            bin_edges = np.histogram_bin_edges(data, bins=bins)
        else:
            bin_edges = np.linspace(range[0], range[1], bins + 1)
    else:
        bin_edges = np.asarray(bins)

    indices = np.digitize(data, bin_edges) - 1
    return bin_edges, indices

def _apply_normalization(counts: np.ndarray, normalization: str) -> np.ndarray:
    """Apply normalization to histogram counts."""
    if normalization == "none":
        return counts
    elif normalization == "standard":
        return (counts - np.mean(counts)) / np.std(counts)
    elif normalization == "minmax":
        return (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
    elif normalization == "robust":
        return (counts - np.median(counts)) / (np.percentile(counts, 75) - np.percentile(counts, 25))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def _compute_metrics(counts: np.ndarray, bin_edges: np.ndarray, metric_funcs: Dict[str, Callable]) -> Dict[str, float]:
    """Compute metrics for histogram."""
    metrics = {}
    bin_widths = np.diff(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for name, func in metric_funcs.items():
        if name == "mean":
            metrics[name] = np.sum(counts * bin_centers) / np.sum(counts)
        elif name == "std":
            metrics[name] = np.sqrt(np.sum(counts * (bin_centers - metrics["mean"])**2) / np.sum(counts))
        elif name == "custom":
            metrics[name] = func(counts, bin_edges)
        else:
            raise ValueError(f"Unknown metric: {name}")

    return metrics

def histogramme_fit(
    data: np.ndarray,
    bins: Union[int, np.ndarray] = 10,
    range: Optional[tuple] = None,
    normalization: str = "none",
    metrics: Dict[str, Union[str, Callable]] = None,
    **kwargs
) -> Dict:
    """
    Compute histogram with configurable parameters.

    Parameters:
    -----------
    data : np.ndarray
        Input data for histogram computation.
    bins : int or np.ndarray, optional
        Number of bins or bin edges. Default is 10.
    range : tuple, optional
        Range of the bins. If None, range is determined from data.
    normalization : str, optional
        Normalization method. Options: "none", "standard", "minmax", "robust". Default is "none".
    metrics : Dict[str, Union[str, Callable]], optional
        Dictionary of metrics to compute. Keys are metric names, values are either built-in names or callables.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns:
    --------
    Dict
        Dictionary containing:
        - "result": dict with bin edges and counts
        - "metrics": computed metrics
        - "params_used": parameters used in computation
        - "warnings": any warnings generated

    Example:
    --------
    >>> data = np.random.normal(0, 1, 1000)
    >>> result = histogramme_fit(data, bins=20, normalization="standard", metrics={"mean": "mean"})
    """
    # Validate inputs
    _validate_inputs(data)

    # Initialize warnings and parameters used
    warnings = []
    params_used = {
        "bins": bins,
        "range": range,
        "normalization": normalization
    }

    # Compute bin edges and indices
    bin_edges, indices = _compute_bins(data, bins, range)

    # Compute counts
    counts, _ = np.histogram(data, bins=bin_edges)

    # Apply normalization
    normalized_counts = _apply_normalization(counts, normalization)
    if normalization != "none":
        warnings.append(f"Normalization '{normalization}' applied to counts")

    # Compute metrics if requested
    metrics_result = {}
    if metrics is not None:
        metric_funcs = {
            name: (lambda x, y: _compute_metrics(x, y, {name: func})[name]
                  if callable(func) else name)
            for name, func in metrics.items()
        }
        metrics_result = _compute_metrics(normalized_counts, bin_edges, metric_funcs)

    # Prepare result
    result = {
        "result": {
            "bin_edges": bin_edges,
            "counts": counts,
            "normalized_counts": normalized_counts
        },
        "metrics": metrics_result,
        "params_used": params_used,
        "warnings": warnings
    }

    return result

################################################################################
# boxplot
################################################################################

import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for boxplot computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")

def compute_boxplot_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics for boxplot."""
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Handle outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return {
        'q1': q1,
        'median': median,
        'q3': q3,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': data[(data < lower_bound) | (data > upper_bound)]
    }

def boxplot_fit(
    data: np.ndarray,
    normalization: str = 'none',
    custom_normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metrics: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'none',
    custom_metrics: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute boxplot statistics with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data for boxplot computation.
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    custom_normalization : callable, optional
        Custom normalization function.
    metrics : str or callable, optional
        Metrics to compute ('none', 'mse', 'mae', 'r2') or custom callable.
    custom_metrics : callable, optional
        Custom metrics function.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate input data
    validate_input(data)

    # Apply normalization if specified
    normalized_data = data.copy()
    if normalization == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
    elif normalization == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    elif normalization == 'robust':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        normalized_data = (data - q1) / iqr
    elif custom_normalization is not None:
        normalized_data = custom_normalization(data)

    # Compute boxplot statistics
    stats = compute_boxplot_statistics(normalized_data)

    # Compute metrics if specified
    metrics_result = {}
    if isinstance(metrics, str):
        if metrics == 'mse':
            residuals = normalized_data - stats['median']
            metrics_result['mse'] = np.mean(residuals ** 2)
        elif metrics == 'mae':
            residuals = normalized_data - stats['median']
            metrics_result['mae'] = np.mean(np.abs(residuals))
        elif metrics == 'r2':
            residuals = normalized_data - stats['median']
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((normalized_data - np.mean(normalized_data)) ** 2)
            metrics_result['r2'] = 1 - (ss_res / ss_tot)
    elif callable(metrics):
        metrics_result['custom'] = metrics(normalized_data, stats['median'])
    elif custom_metrics is not None:
        metrics_result['custom'] = custom_metrics(normalized_data, stats['median'])

    # Prepare output dictionary
    result = {
        'result': stats,
        'metrics': metrics_result if metrics_result else None,
        'params_used': {
            'normalization': normalization,
            'metrics': metrics if isinstance(metrics, str) else 'custom'
        },
        'warnings': []
    }

    return result

# Example usage:
# data = np.random.normal(0, 1, 100)
# result = boxplot_fit(data, normalization='standard', metrics='mse')

################################################################################
# tableau_croissance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(
    data: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: Union[str, Callable] = "euclidean"
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")

    normalization_options = ["none", "standard", "minmax", "robust"]
    if normalize not in normalization_options:
        raise ValueError(f"Normalization must be one of {normalization_options}")

    metric_options = ["mse", "mae", "r2", "logloss"]
    if isinstance(metric, str) and metric not in metric_options:
        raise ValueError(f"Metric must be one of {metric_options} or a callable")

    distance_options = ["euclidean", "manhattan", "cosine", "minkowski"]
    if isinstance(distance, str) and distance not in distance_options:
        raise ValueError(f"Distance must be one of {distance_options} or a callable")

def _normalize_data(
    data: np.ndarray,
    normalize: str = "none"
) -> np.ndarray:
    """Normalize data based on specified method."""
    if normalize == "none":
        return data
    elif normalize == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalize == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalize == "robust":
        median = np.median(data, axis=0)
        iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        return (data - median) / iqr
    else:
        raise ValueError("Invalid normalization method")

def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = "mse"
) -> float:
    """Compute specified metric between true and predicted values."""
    if isinstance(metric, str):
        if metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        elif metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == "logloss":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        return metric(y_true, y_pred)

def _compute_distance(
    a: np.ndarray,
    b: np.ndarray,
    distance: Union[str, Callable] = "euclidean",
    p: float = 2.0
) -> float:
    """Compute specified distance between two vectors."""
    if isinstance(distance, str):
        if distance == "euclidean":
            return np.linalg.norm(a - b)
        elif distance == "manhattan":
            return np.sum(np.abs(a - b))
        elif distance == "cosine":
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        elif distance == "minkowski":
            return np.sum(np.abs(a - b) ** p) ** (1 / p)
    else:
        return distance(a, b)

def tableau_croissance_fit(
    data: np.ndarray,
    normalize: str = "none",
    metric: Union[str, Callable] = "mse",
    distance: Union[str, Callable] = "euclidean",
    p: float = 2.0,
    custom_metric: Optional[Callable] = None,
    custom_distance: Optional[Callable] = None
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Compute growth table statistics for input data.

    Parameters:
    -----------
    data : np.ndarray
        Input data as 2D numpy array.
    normalize : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    metric : str or callable, optional
        Metric to compute ("mse", "mae", "r2", "logloss") or custom callable.
    distance : str or callable, optional
        Distance metric ("euclidean", "manhattan", "cosine", "minkowski") or custom callable.
    p : float, optional
        Parameter for Minkowski distance (default 2.0).
    custom_metric : callable, optional
        Custom metric function if not using built-in options.
    custom_distance : callable, optional
        Custom distance function if not using built-in options.

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_inputs(data, normalize, metric, distance)

    # Normalize data
    normalized_data = _normalize_data(data, normalize)

    # Compute growth statistics
    diffs = np.diff(normalized_data, axis=0)
    growth_rates = diffs / normalized_data[:-1]

    # Compute metrics
    if isinstance(metric, str):
        metric_value = _compute_metric(normalized_data[:-1], normalized_data[1:], metric)
    else:
        metric_value = _compute_metric(normalized_data[:-1], normalized_data[1:], custom_metric)

    # Compute distances
    if isinstance(distance, str):
        distance_value = _compute_distance(normalized_data[0], normalized_data[-1], distance, p)
    else:
        distance_value = _compute_distance(normalized_data[0], normalized_data[-1], custom_distance)

    return {
        "result": {
            "growth_rates": growth_rates,
            "differences": diffs
        },
        "metrics": {
            "metric_value": metric_value,
            "distance_value": distance_value
        },
        "params_used": {
            "normalize": normalize,
            "metric": metric if isinstance(metric, str) else "custom",
            "distance": distance if isinstance(distance, str) else "custom",
            "p": p
        },
        "warnings": []
    }

# Example usage:
"""
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = tableau_croissance_fit(data, normalize="standard", metric="mse")
print(result)
"""

################################################################################
# correlation
################################################################################

import numpy as np
from typing import Callable, Dict, Union, Optional

def correlation_fit(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
    normalize_X: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    normalize_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Calculate correlation between features and target.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    method : str, optional
        Correlation method ('pearson', 'spearman', 'kendall')
    normalize_X : callable, optional
        Function to normalize features (e.g., standard_scaler)
    normalize_y : callable, optional
        Function to normalize target (e.g., standard_scaler)
    custom_metric : callable, optional
        Custom correlation metric function

    Returns
    -------
    dict
        Dictionary containing:
        - 'result': correlation coefficients
        - 'metrics': additional metrics if applicable
        - 'params_used': parameters used in computation
        - 'warnings': any warnings generated

    Examples
    --------
    >>> X = np.random.rand(10, 3)
    >>> y = np.random.rand(10)
    >>> result = correlation_fit(X, y)
    """
    # Validate inputs
    _validate_inputs(X, y)

    # Apply normalizations if specified
    X_norm = normalize_X(X) if normalize_X else X
    y_norm = normalize_y(y) if normalize_y else y

    # Calculate correlation based on method
    if custom_metric:
        correlations = np.array([custom_metric(X_norm[:, i], y_norm) for i in range(X.shape[1])])
    else:
        correlations = _calculate_correlation(X_norm, y_norm, method)

    # Prepare output
    result_dict = {
        'result': correlations,
        'metrics': {},
        'params_used': {
            'method': method,
            'normalize_X': normalize_X.__name__ if normalize_X else None,
            'normalize_y': normalize_y.__name__ if normalize_y else None
        },
        'warnings': []
    }

    return result_dict

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

def _calculate_correlation(X: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """Calculate correlation coefficients."""
    n_features = X.shape[1]
    correlations = np.zeros(n_features)

    for i in range(n_features):
        x = X[:, i]
        if method == 'pearson':
            correlations[i] = np.corrcoef(x, y)[0, 1]
        elif method == 'spearman':
            correlations[i] = np.corrcoef(_rank_data(x), _rank_data(y))[0, 1]
        elif method == 'kendall':
            correlations[i] = _kendall_tau(x, y)
        else:
            raise ValueError(f"Unknown method: {method}")

    return correlations

def _rank_data(x: np.ndarray) -> np.ndarray:
    """Convert data to ranks."""
    return np.argsort(np.argsort(x))

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Kendall's tau correlation."""
    n = len(x)
    concordant = discordant = 0

    for i in range(n):
        for j in range(i+1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1

    return (concordant - discordant) / ((n * (n-1)) / 2)

def standard_scaler(data: np.ndarray) -> np.ndarray:
    """Standard normalization (z-score)."""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def minmax_scaler(data: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

################################################################################
# covariance
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def covariance_fit(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    normalize: str = 'none',
    method: str = 'closed_form',
    custom_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate the covariance between two variables or a variable and each column of X.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    Y : Optional[np.ndarray]
        Target variable or matrix of shape (n_samples,) or (n_samples, n_targets).
        If None, computes covariance between each pair of features in X.
    normalize : str
        Normalization method: 'none', 'standard', 'minmax', or 'robust'.
    method : str
        Method to compute covariance: 'closed_form' (default), or other methods if implemented.
    custom_metric : Optional[Callable]
        Custom metric function to evaluate the covariance. Must take two arrays and return a float.
    **kwargs
        Additional keyword arguments for specific methods.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - "result": Computed covariance values.
        - "metrics": Metrics if custom_metric is provided.
        - "params_used": Parameters used in the computation.
        - "warnings": Any warnings generated during computation.

    Example:
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> result = covariance_fit(X)
    """
    # Validate inputs
    X, Y, warnings = _validate_inputs(X, Y)

    # Normalize data if required
    X_norm, Y_norm = _normalize_data(X, Y, normalize)

    # Compute covariance
    if Y is None:
        cov_result = _compute_covariance_matrix(X_norm)
    else:
        cov_result = _compute_covariance_vector(X_norm, Y_norm)

    # Compute metrics if custom metric is provided
    metrics = {}
    if custom_metric is not None:
        metrics = _compute_metrics(cov_result, X_norm, Y_norm, custom_metric)

    # Prepare output
    output = {
        "result": cov_result,
        "metrics": metrics,
        "params_used": {
            "normalize": normalize,
            "method": method,
            "custom_metric": custom_metric.__name__ if custom_metric else None
        },
        "warnings": warnings
    }

    return output

def _validate_inputs(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None
) -> tuple[np.ndarray, Optional[np.ndarray], list]:
    """
    Validate input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix.
    Y : Optional[np.ndarray]
        Target variable or matrix.

    Returns:
    --------
    tuple
        Validated X, Y, and warnings.
    """
    warnings = []

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if Y is not None and not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array or None.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if Y is not None and Y.ndim > 2:
        raise ValueError("Y must be a 1D or 2D array.")

    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    if np.any(np.isnan(X)):
        warnings.append("NaN values found in X and ignored.")

    if Y is not None and np.any(np.isnan(Y)):
        warnings.append("NaN values found in Y and ignored.")

    return X, Y, warnings

def _normalize_data(
    X: np.ndarray,
    Y: Optional[np.ndarray],
    normalize: str
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Normalize data based on the specified method.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix.
    Y : Optional[np.ndarray]
        Target variable or matrix.
    normalize : str
        Normalization method.

    Returns:
    --------
    tuple
        Normalized X and Y.
    """
    if normalize == 'standard':
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        if Y is not None:
            Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif normalize == 'minmax':
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        if Y is not None:
            Y = (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))
    elif normalize == 'robust':
        X = (X - np.median(X, axis=0)) / (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0))
        if Y is not None:
            Y = (Y - np.median(Y, axis=0)) / (np.percentile(Y, 75, axis=0) - np.percentile(Y, 25, axis=0))
    # 'none' does nothing

    return X, Y

def _compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix of X.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix.

    Returns:
    --------
    np.ndarray
        Covariance matrix.
    """
    n = X.shape[0]
    return np.cov(X, rowvar=False)

def _compute_covariance_vector(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Compute the covariance between X and Y.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix.
    Y : np.ndarray
        Target variable or matrix.

    Returns:
    --------
    np.ndarray
        Covariance vector.
    """
    if Y.ndim == 1:
        return np.cov(X, Y, rowvar=False)[0:-1, -1]
    else:
        return np.cov(X, Y, rowvar=False)[0:-Y.shape[1], -Y.shape[1]:]

def _compute_metrics(
    cov_result: np.ndarray,
    X: np.ndarray,
    Y: Optional[np.ndarray],
    custom_metric: Callable[[np.ndarray, np.ndarray], float]
) -> Dict[str, float]:
    """
    Compute custom metrics for the covariance result.

    Parameters:
    -----------
    cov_result : np.ndarray
        Covariance result.
    X : np.ndarray
        Input data matrix.
    Y : Optional[np.ndarray]
        Target variable or matrix.
    custom_metric : Callable
        Custom metric function.

    Returns:
    --------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    if Y is None:
        raise ValueError("Y must be provided to compute metrics.")

    return {"custom_metric": custom_metric(X, Y)}
