"""
Quantix – Module dispersion_variabilite
Généré automatiquement
Date: 2026-01-07
"""

################################################################################
# ecart_type
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for standard deviation calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _compute_standard_deviation(
    data: np.ndarray,
    ddof: int = 1,
    axis: Optional[int] = None
) -> np.ndarray:
    """Compute standard deviation with given degrees of freedom."""
    mean = np.mean(data, axis=axis)
    squared_diff = (data - mean) ** 2
    variance = np.sum(squared_diff, axis=axis) / (len(data) - ddof)
    return np.sqrt(variance)

def _apply_normalization(
    data: np.ndarray,
    normalization: str = "none",
    **kwargs
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if normalization == "standard":
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
    elif normalization == "none":
        return data
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

def ecart_type_fit(
    data: np.ndarray,
    normalization: str = "none",
    ddof: int = 1,
    axis: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute standard deviation with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    normalization : str, optional
        Normalization method ("none", "standard", "minmax", "robust").
    ddof : int, optional
        Delta degrees of freedom for variance calculation.
    axis : Optional[int], optional
        Axis along which to compute standard deviation.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    # Validate input
    _validate_input(data)

    # Apply normalization if specified
    normalized_data = _apply_normalization(data, normalization)

    # Compute standard deviation
    std_dev = _compute_standard_deviation(normalized_data, ddof=ddof, axis=axis)

    # Prepare output
    result = {
        "result": std_dev,
        "metrics": {
            "variance": np.var(normalized_data, ddof=ddof, axis=axis),
            "mean": np.mean(normalized_data, axis=axis)
        },
        "params_used": {
            "normalization": normalization,
            "ddof": ddof,
            "axis": axis
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = ecart_type_fit(data, normalization="standard")

################################################################################
# variance
################################################################################

import numpy as np
from typing import Callable, Dict, Optional, Union

def variance_fit(
    data: np.ndarray,
    axis: int = 0,
    ddof: int = 1,
    normalization: str = "none",
    metric: Union[str, Callable] = "mse",
    custom_metric: Optional[Callable] = None,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict, str]]:
    """
    Calculate the variance of a dataset with configurable options.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    axis : int, optional
        Axis along which to compute the variance (default is 0).
    ddof : int, optional
        Delta degrees of freedom (default is 1 for sample variance).
    normalization : str, optional
        Type of normalization to apply ('none', 'standard', 'minmax', 'robust').
    metric : Union[str, Callable], optional
        Metric to use for variance calculation ('mse', 'mae', or custom callable).
    custom_metric : Optional[Callable], optional
        Custom metric function if metric is 'custom'.
    weights : Optional[np.ndarray], optional
        Weights for weighted variance calculation.

    Returns
    -------
    Dict[str, Union[float, Dict, str]]
        Dictionary containing:
        - "result": computed variance
        - "metrics": dictionary of additional metrics
        - "params_used": parameters used in the calculation
        - "warnings": any warnings generated

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> variance_fit(data)
    {
        'result': 1.0,
        'metrics': {'mse': 1.0},
        'params_used': {
            'axis': 0,
            'ddof': 1,
            'normalization': 'none',
            'metric': 'mse'
        },
        'warnings': []
    }
    """
    # Validate inputs
    _validate_inputs(data, axis, ddof, weights)

    # Normalize data if specified
    normalized_data = _apply_normalization(data, normalization)

    # Calculate variance based on metric choice
    result, metrics = _calculate_variance(
        normalized_data,
        axis=axis,
        ddof=ddof,
        metric=metric,
        custom_metric=custom_metric,
        weights=weights
    )

    # Prepare output dictionary
    output = {
        "result": result,
        "metrics": metrics,
        "params_used": {
            "axis": axis,
            "ddof": ddof,
            "normalization": normalization,
            "metric": metric
        },
        "warnings": []
    }

    return output

def _validate_inputs(
    data: np.ndarray,
    axis: int,
    ddof: int,
    weights: Optional[np.ndarray]
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for input array")
    if ddof < 0:
        raise ValueError("Delta degrees of freedom (ddof) must be non-negative")
    if weights is not None:
        if len(weights) != data.shape[axis]:
            raise ValueError("Weights must have the same length as the axis dimension")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

def _apply_normalization(
    data: np.ndarray,
    normalization: str
) -> np.ndarray:
    """Apply specified normalization to the data."""
    if normalization == "standard":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif normalization == "minmax":
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    elif normalization == "robust":
        return (data - np.median(data, axis=0)) / (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0))
    return data

def _calculate_variance(
    data: np.ndarray,
    axis: int,
    ddof: int,
    metric: Union[str, Callable],
    custom_metric: Optional[Callable],
    weights: Optional[np.ndarray]
) -> tuple:
    """Calculate variance using specified metric."""
    metrics = {}

    if metric == "mse":
        result = np.mean((data - np.mean(data, axis=axis, keepdims=True))**2, axis=axis)
        metrics["mse"] = result
    elif metric == "mae":
        result = np.mean(np.abs(data - np.mean(data, axis=axis, keepdims=True)), axis=axis)
        metrics["mae"] = result
    elif metric == "custom" and custom_metric is not None:
        if weights is None:
            result = np.mean(custom_metric(data, axis=axis), axis=axis)
        else:
            result = np.average(custom_metric(data, axis=axis), axis=axis, weights=weights)
        metrics["custom"] = result
    else:
        # Default to sample variance calculation
        if weights is None:
            result = np.var(data, axis=axis, ddof=ddof)
        else:
            result = np.average((data - np.mean(data, axis=axis, keepdims=True))**2, axis=axis, weights=weights)
        metrics["sample_variance"] = result

    return result, metrics

################################################################################
# etendue
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(data: np.ndarray) -> None:
    """Validate input data for range calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")

def _compute_range(data: np.ndarray) -> float:
    """Compute the range of the data."""
    return np.max(data) - np.min(data)

def etendue_fit(
    data: np.ndarray,
    normalization: Optional[str] = None,
    custom_metric: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the range of a dataset with optional normalization.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1D)
    normalization : str, optional
        Normalization method ('none', 'standard', 'minmax', 'robust')
    custom_metric : callable, optional
        Custom metric function that takes a numpy array and returns a float

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = etendue_fit(data)
    """
    _validate_inputs(data)

    # Normalization
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

    # Compute range
    result_range = _compute_range(normalized_data)

    # Custom metric
    metrics = {}
    if custom_metric is not None:
        try:
            metrics['custom_metric'] = custom_metric(normalized_data)
        except Exception as e:
            metrics['custom_metric'] = f"Error: {str(e)}"
            warnings.append("Custom metric calculation failed")

    # Prepare output
    output = {
        'result': result_range,
        'metrics': metrics,
        'params_used': {
            'normalization': normalization,
            'custom_metric': custom_metric.__name__ if custom_metric else None
        },
        'warnings': []
    }

    return output

################################################################################
# interquartile_range
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

def _calculate_quartiles(data: np.ndarray, method: str = 'linear') -> tuple:
    """Calculate quartiles using specified interpolation method."""
    if method not in ['linear', 'lower', 'higher', 'midpoint', 'nearest']:
        raise ValueError("Invalid interpolation method for quartile calculation")

    sorted_data = np.sort(data)
    n = len(sorted_data)

    def _get_quartile(pos: float) -> float:
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

    q1 = _get_quartile(0.25)
    q3 = _get_quartile(0.75)

    return q1, q3

def interquartile_range_compute(
    data: np.ndarray,
    interpolation_method: str = 'linear',
    custom_quartile_func: Optional[Callable[[np.ndarray], tuple]] = None
) -> Dict[str, Any]:
    """
    Calculate the interquartile range (IQR) of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1D)
    interpolation_method : str, optional
        Interpolation method for quartile calculation ('linear', 'lower',
        'higher', 'midpoint', 'nearest')
    custom_quartile_func : callable, optional
        Custom function to calculate quartiles (must return q1, q3)

    Returns:
    --------
    dict
        Dictionary containing:
        - result: float, the calculated IQR
        - metrics: dict with additional statistics
        - params_used: dict of parameters used
        - warnings: list of any warnings

    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> result = interquartile_range_compute(data)
    """
    # Validate input
    _validate_input(data)

    # Prepare output dictionary
    result_dict: Dict[str, Any] = {
        'result': None,
        'metrics': {},
        'params_used': {
            'interpolation_method': interpolation_method,
            'custom_quartile_func': custom_quartile_func is not None
        },
        'warnings': []
    }

    # Calculate quartiles
    if custom_quartile_func is not None:
        q1, q3 = custom_quartile_func(data)
    else:
        q1, q3 = _calculate_quartiles(data, interpolation_method)

    # Calculate IQR
    iqr = q3 - q1

    # Store results
    result_dict['result'] = float(iqr)
    result_dict['metrics']['q1'] = float(q1)
    result_dict['metrics']['q3'] = float(q3)

    return result_dict

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

def _compute_mean(data: np.ndarray, normalization: str) -> float:
    """Compute mean with optional normalization."""
    if normalization == "robust":
        return np.median(data)
    return np.mean(data)

def _compute_std(data: np.ndarray, normalization: str) -> float:
    """Compute standard deviation with optional normalization."""
    if normalization == "robust":
        return np.median(np.abs(data - np.median(data)))
    return np.std(data, ddof=1)

def coefficient_variation_fit(
    data: np.ndarray,
    normalization: str = "none",
    custom_mean_func: Optional[Callable[[np.ndarray], float]] = None,
    custom_std_func: Optional[Callable[[np.ndarray], float]] = None
) -> Dict[str, Any]:
    """
    Compute the coefficient of variation with configurable options.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (1D)
    normalization : str, optional
        Normalization method ('none', 'standard', 'robust')
    custom_mean_func : callable, optional
        Custom function to compute mean (must take np.ndarray and return float)
    custom_std_func : callable, optional
        Custom function to compute standard deviation (must take np.ndarray and return float)

    Returns:
    --------
    dict
        Dictionary containing results, metrics, parameters used and warnings

    Example:
    --------
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> result = coefficient_variation_fit(data)
    """
    _validate_inputs(data)

    # Parameter validation
    if normalization not in ["none", "standard", "robust"]:
        raise ValueError("Normalization must be 'none', 'standard' or 'robust'")

    # Compute mean
    if custom_mean_func is not None:
        mean = custom_mean_func(data)
    else:
        mean = _compute_mean(data, normalization)

    # Compute standard deviation
    if custom_std_func is not None:
        std = custom_std_func(data)
    else:
        std = _compute_std(data, normalization)

    # Compute coefficient of variation
    if mean == 0:
        cv = np.nan
        warnings = ["Mean is zero, coefficient of variation is undefined"]
    else:
        cv = std / mean
        warnings = []

    # Prepare output dictionary
    result_dict: Dict[str, Any] = {
        "result": cv,
        "metrics": {
            "mean": mean,
            "std": std
        },
        "params_used": {
            "normalization": normalization,
            "custom_mean_func": custom_mean_func is not None,
            "custom_std_func": custom_std_func is not None
        },
        "warnings": warnings
    }

    return result_dict

################################################################################
# moyenne_ecart_absolu
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_inputs(
    data: np.ndarray,
    axis: int = 0
) -> None:
    """Validate input data and parameters."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if axis not in (0, 1):
        raise ValueError("Axis must be either 0 or 1.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _compute_mean_absolute_deviation(
    data: np.ndarray,
    axis: int = 0
) -> float:
    """Compute the mean absolute deviation along the specified axis."""
    mean = np.mean(data, axis=axis)
    abs_deviation = np.abs(data - mean)
    mad = np.mean(abs_deviation, axis=axis)
    return float(mad)

def _compute_median_absolute_deviation(
    data: np.ndarray,
    axis: int = 0
) -> float:
    """Compute the median absolute deviation along the specified axis."""
    median = np.median(data, axis=axis)
    abs_deviation = np.abs(data - median)
    mad = np.median(abs_deviation, axis=axis)
    return float(mad)

def moyenne_ecart_absolu_fit(
    data: np.ndarray,
    method: str = 'mean',
    axis: int = 0
) -> Dict[str, Any]:
    """
    Compute the mean or median absolute deviation of the data.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    method : str, optional
        Method to compute absolute deviation ('mean' or 'median'). Default is 'mean'.
    axis : int, optional
        Axis along which to compute the absolute deviation. Default is 0.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.

    Examples:
    ---------
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> result = moyenne_ecart_absolu_fit(data, method='mean')
    """
    _validate_inputs(data, axis)

    if method == 'mean':
        result = _compute_mean_absolute_deviation(data, axis)
    elif method == 'median':
        result = _compute_median_absolute_deviation(data, axis)
    else:
        raise ValueError("Method must be either 'mean' or 'median'.")

    metrics = {
        'method': method,
        'axis': axis
    }

    return {
        'result': result,
        'metrics': metrics,
        'params_used': {
            'method': method,
            'axis': axis
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
        raise TypeError("Input data must be a numpy array")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional")
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values")

def _calculate_skewness(data: np.ndarray, bias_corrected: bool = True) -> float:
    """Calculate skewness of the data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1 if bias_corrected else 0)

    if std == 0:
        return 0.0

    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    return skewness

def _calculate_metrics(skewness: float) -> Dict[str, float]:
    """Calculate additional metrics based on skewness."""
    return {
        "skewness": skewness,
        "absolute_skewness": abs(skewness),
    }

def skewness_fit(
    data: np.ndarray,
    bias_corrected: bool = True,
    custom_metric: Optional[Callable[[float], float]] = None
) -> Dict[str, Any]:
    """
    Calculate skewness of the data with optional custom metrics.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    bias_corrected : bool, optional
        Whether to apply bias correction (default is True).
    custom_metric : Callable[[float], float], optional
        Custom metric function that takes skewness as input.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    skewness = _calculate_skewness(data, bias_corrected)
    metrics = _calculate_metrics(skewness)

    if custom_metric is not None:
        metrics["custom_metric"] = custom_metric(skewness)

    result = {
        "result": skewness,
        "metrics": metrics,
        "params_used": {
            "bias_corrected": bias_corrected,
            "custom_metric": custom_metric is not None
        },
        "warnings": []
    }

    return result

# Example usage:
# data = np.array([1, 2, 3, 4, 5])
# result = skewness_fit(data)

################################################################################
# kurtosis
################################################################################

import numpy as np
from typing import Callable, Optional, Dict, Any

def _validate_input(data: np.ndarray) -> None:
    """Validate input data for kurtosis calculation."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")

def _compute_kurtosis(data: np.ndarray, fisher: bool = True) -> float:
    """Compute the kurtosis of a dataset."""
    mean = np.mean(data)
    std = np.std(data, ddof=1) if len(data) > 1 else 0.0
    if std == 0:
        return float('nan')
    n = len(data)
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    if fisher:
        kurtosis -= 3
    return float(kurtosis)

def _compute_metrics(data: np.ndarray, kurtosis_value: float) -> Dict[str, float]:
    """Compute additional metrics related to kurtosis."""
    return {
        'kurtosis': kurtosis_value,
        'excess_kurtosis': kurtosis_value if fisher else kurtosis_value + 3
    }

def kurtosis_fit(
    data: np.ndarray,
    fisher: bool = True,
    normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute the kurtosis of a dataset with optional normalization.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    fisher : bool, optional
        Whether to use Fisher's definition of kurtosis (default: True).
    normalize : Callable[[np.ndarray], np.ndarray], optional
        Function to normalize the data before computation (default: None).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the result, metrics, parameters used, and warnings.
    """
    _validate_input(data)

    if normalize is not None:
        data = normalize(data.copy())

    kurtosis_value = _compute_kurtosis(data, fisher)
    metrics = _compute_metrics(data, kurtosis_value)

    return {
        'result': kurtosis_value,
        'metrics': metrics,
        'params_used': {
            'fisher': fisher,
            'normalize': normalize.__name__ if normalize else None
        },
        'warnings': []
    }

# Example usage:
# data = np.random.normal(0, 1, 1000)
# result = kurtosis_fit(data, fisher=True)
