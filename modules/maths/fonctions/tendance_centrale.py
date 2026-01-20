"""
Quantix – Module tendance_centrale
Généré automatiquement
Date: 2026-01-09
"""

################################################################################
# moyenne
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for mean calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.any(np.isnan(data)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input contains infinite values")

def _apply_normalization(
    data: np.ndarray,
    normalization: str = "none",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Apply normalization to the input data."""
    if custom_norm is not None:
        return custom_norm(data)

    normalized_data = data.copy()
    if normalization == "standard":
        mean = np.mean(normalized_data)
        std = np.std(normalized_data)
        normalized_data = (normalized_data - mean) / std
    elif normalization == "minmax":
        min_val = np.min(normalized_data)
        max_val = np.max(normalized_data)
        normalized_data = (normalized_data - min_val) / (max_val - min_val)
    elif normalization == "robust":
        median = np.median(normalized_data)
        iqr = np.percentile(normalized_data, 75) - np.percentile(normalized_data, 25)
        normalized_data = (normalized_data - median) / iqr

    return normalized_data

def _compute_mean(
    data: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute the mean of the data."""
    if weights is not None:
        if len(weights) != len(data):
            raise ValueError("Weights must have the same length as data")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        return np.sum(data * weights) / np.sum(weights)
    else:
        return np.mean(data)

def _compute_metrics(
    data: np.ndarray,
    mean_val: float,
    metric_names: Optional[list] = None
) -> Dict[str, float]:
    """Compute various metrics based on the mean."""
    if metric_names is None:
        metric_names = ["mse", "mae"]

    metrics = {}
    residuals = data - mean_val

    if "mse" in metric_names:
        metrics["mse"] = np.mean(residuals ** 2)
    if "mae" in metric_names:
        metrics["mae"] = np.mean(np.abs(residuals))
    if "r2" in metric_names:
        ss_total = np.sum((data - np.mean(data)) ** 2)
        ss_residual = np.sum(residuals ** 2)
        metrics["r2"] = 1 - (ss_residual / ss_total)

    return metrics

def moyenne_fit(
    data: np.ndarray,
    normalization: str = "none",
    custom_norm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
    metrics: Optional[list] = None,
    **kwargs
) -> Dict[str, Union[float, Dict[str, float], dict, list]]:
    """
    Compute the mean of the input data with various options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Type of normalization to apply ("none", "standard", "minmax", "robust").
    custom_norm : callable, optional
        Custom normalization function.
    weights : np.ndarray, optional
        Weights for weighted mean calculation.
    metrics : list, optional
        List of metric names to compute ("mse", "mae", "r2").

    Returns
    -------
    dict
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    normalized_data = _apply_normalization(data, normalization, custom_norm)
    mean_val = _compute_mean(normalized_data, weights)

    metrics_result = {}
    if metrics is not None:
        metrics_result = _compute_metrics(normalized_data, mean_val, metrics)

    return {
        "result": mean_val,
        "metrics": metrics_result,
        "params_used": {
            "normalization": normalization,
            "weights": weights is not None
        },
        "warnings": []
    }

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = moyenne_fit(data, normalization="standard", metrics=["mse", "mae"])

################################################################################
# mediane
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for median calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.any(np.isnan(data)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input contains infinite values")

def sort_and_count(data: np.ndarray) -> tuple:
    """Sort data and count elements."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    return sorted_data, n

def find_median(sorted_data: np.ndarray, n: int) -> float:
    """Calculate median based on sorted data and count."""
    if n % 2 == 1:
        return float(sorted_data[n // 2])
    else:
        return float((sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2)

def mediane_fit(
    data: np.ndarray,
    normalize: Optional[str] = None,
    distance_metric: str = "euclidean",
    custom_distance: Optional[Callable] = None,
    solver: str = "closed_form"
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]:
    """
    Calculate the median of a dataset with configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    normalize : str, optional
        Normalization method. Options: "none", "standard", "minmax", "robust".
    distance_metric : str, optional
        Distance metric for custom calculations. Options: "euclidean", "manhattan", "cosine", "minkowski".
    custom_distance : Callable, optional
        Custom distance function.
    solver : str, optional
        Solver method. Options: "closed_form", "gradient_descent", "newton", "coordinate_descent".

    Returns
    -------
    Dict[str, Union[float, Dict[str, float], Dict[str, str], list]]
        Dictionary containing:
        - "result": calculated median
        - "metrics": dictionary of metrics (empty for median)
        - "params_used": dictionary of parameters used
        - "warnings": list of warnings

    Examples
    --------
    >>> data = np.array([1, 3, 5, 7, 9])
    >>> mediane_fit(data)
    {
        'result': 5.0,
        'metrics': {},
        'params_used': {'normalize': 'none', 'distance_metric': 'euclidean', 'solver': 'closed_form'},
        'warnings': []
    }
    """
    # Validate input
    validate_input(data)

    # Normalize data if specified
    normalized_data = data.copy()
    if normalize == "standard":
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
    elif normalize == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    elif normalize == "robust":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        normalized_data = (data - q1) / iqr

    # Sort data and count elements
    sorted_data, n = sort_and_count(normalized_data)

    # Calculate median based on the solver
    if solver == "closed_form":
        median = find_median(sorted_data, n)
    else:
        raise ValueError(f"Solver {solver} not implemented for median calculation")

    # Prepare output
    result = {
        "result": median,
        "metrics": {},
        "params_used": {
            "normalize": normalize if normalize else "none",
            "distance_metric": distance_metric,
            "solver": solver
        },
        "warnings": []
    }

    return result

################################################################################
# mode
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any, Union

def validate_input(data: np.ndarray) -> None:
    """Validate input data for mode computation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input must be a 1-dimensional array")
    if np.any(np.isnan(data)):
        raise ValueError("Input contains NaN values")

def compute_frequency(data: np.ndarray) -> Dict[Any, int]:
    """Compute frequency of each value in the data."""
    unique_values, counts = np.unique(data, return_counts=True)
    return dict(zip(unique_values, counts))

def find_mode(frequencies: Dict[Any, int]) -> Any:
    """Find the mode from frequency dictionary."""
    max_count = max(frequencies.values())
    modes = [k for k, v in frequencies.items() if v == max_count]
    return modes[0] if len(modes) == 1 else np.mean(modes)

def mode_fit(
    data: np.ndarray,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: abs(x - y),
    normalize: bool = False,
    custom_metric: Optional[Callable[[np.ndarray, Any], float]] = None
) -> Dict[str, Any]:
    """
    Compute the mode of a dataset with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    distance_metric : Callable, optional
        Function to compute distance between values (default: absolute difference).
    normalize : bool, optional
        Whether to normalize the data before computation (default: False).
    custom_metric : Callable, optional
        Custom metric function to evaluate the mode (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    # Validate input
    validate_input(data)

    # Normalize data if required
    if normalize:
        data = (data - np.mean(data)) / np.std(data)

    # Compute frequency
    frequencies = compute_frequency(data)

    # Find mode
    result = find_mode(frequencies)

    # Compute metrics if custom metric is provided
    metrics = {}
    if custom_metric is not None:
        metrics['custom_metric'] = custom_metric(data, result)

    # Prepare output
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "distance_metric": distance_metric.__name__ if hasattr(distance_metric, '__name__') else "custom",
            "normalize": normalize,
            "custom_metric": custom_metric.__name__ if custom_metric else None
        },
        "warnings": []
    }

    return output

# Example usage:
# data = np.array([1, 2, 2, 3, 4])
# result = mode_fit(data)

################################################################################
# quantile
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray, quantile: float) -> None:
    """Validate input data and quantile value."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if len(data.shape) != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values.")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1.")

def _compute_quantile(data: np.ndarray, quantile: float) -> float:
    """Compute the quantile using numpy's percentile function."""
    return np.percentile(data, quantile * 100)

def _compute_quantile_custom(
    data: np.ndarray,
    quantile: float,
    method: Callable[[np.ndarray, float], float]
) -> float:
    """Compute the quantile using a custom method."""
    return method(data, quantile)

def _compute_metrics(
    data: np.ndarray,
    quantile_value: float,
    metric_funcs: Dict[str, Callable[[np.ndarray, float], float]]
) -> Dict[str, float]:
    """Compute metrics for the quantile estimation."""
    metrics = {}
    for name, func in metric_funcs.items():
        try:
            metrics[name] = func(data, quantile_value)
        except Exception as e:
            metrics[name] = np.nan
    return metrics

def quantile_fit(
    data: np.ndarray,
    quantile: float = 0.5,
    method: str = "numpy",
    custom_method: Optional[Callable[[np.ndarray, float], float]] = None,
    metrics: Dict[str, Callable[[np.ndarray, float], float]] = None
) -> Dict[str, Union[float, Dict[str, float], Dict[str, str]]]:
    """
    Compute the quantile of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data.
    quantile : float, optional
        Quantile to compute (default is 0.5 for median).
    method : str, optional
        Method to use ('numpy' or 'custom').
    custom_method : Callable, optional
        Custom method to compute quantile.
    metrics : Dict[str, Callable], optional
        Dictionary of metric functions to compute.

    Returns:
    --------
    Dict containing:
    - result: computed quantile value
    - metrics: computed metrics
    - params_used: parameters used
    - warnings: any warnings generated

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> quantile_fit(data, quantile=0.5)
    """
    _validate_inputs(data, quantile)

    warnings = []
    params_used = {
        "quantile": quantile,
        "method": method
    }

    if method == "numpy":
        result = _compute_quantile(data, quantile)
    elif method == "custom" and custom_method is not None:
        result = _compute_quantile_custom(data, quantile, custom_method)
    else:
        raise ValueError("Invalid method or missing custom method.")

    metrics_result = {}
    if metrics is not None:
        metrics_result = _compute_metrics(data, result, metrics)

    return {
        "result": result,
        "metrics": metrics_result,
        "params_used": params_used,
        "warnings": warnings
    }

################################################################################
# ecart_interquartile
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def _validate_inputs(data: np.ndarray) -> None:
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
            return (sorted_data[int(f)] + sorted_data[int(c)]) / 2
        else:  # nearest
            return sorted_data[int(round(k))]

    q1 = get_quartile(0.25)
    q3 = get_quartile(0.75)

    return {'q1': q1, 'q3': q3}

def ecart_interquartile_fit(
    data: np.ndarray,
    interpolation_method: str = 'linear',
    custom_quartile_func: Optional[Callable[[np.ndarray], Dict[str, float]]] = None
) -> Dict[str, Union[float, Dict[str, float], str]]:
    """
    Calculate the interquartile range (IQR) of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1D)
    interpolation_method : str, optional
        Interpolation method for quartile calculation (default: 'linear')
    custom_quartile_func : callable, optional
        Custom function to calculate quartiles (must return dict with 'q1' and 'q3')

    Returns:
    --------
    Dict containing:
        - result: float, the interquartile range
        - metrics: dict with quartiles and method used
        - params_used: dict of parameters actually used
        - warnings: list of any warnings

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> ecart_interquartile_fit(data)
    """
    _validate_inputs(data)

    warnings = []

    if custom_quartile_func is not None:
        try:
            quartiles = custom_quartile_func(data)
        except Exception as e:
            warnings.append(f"Custom quartile function failed: {str(e)}")
            quartiles = _calculate_quartiles(data, interpolation_method)
    else:
        quartiles = _calculate_quartiles(data, interpolation_method)

    iqr = quartiles['q3'] - quartiles['q1']

    return {
        'result': iqr,
        'metrics': {
            'quartiles': quartiles,
            'method': interpolation_method if custom_quartile_func is None else 'custom'
        },
        'params_used': {
            'interpolation_method': interpolation_method,
            'custom_quartile_func': custom_quartile_func is not None
        },
        'warnings': warnings
    }
